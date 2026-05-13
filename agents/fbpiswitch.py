from typing import Any
import os
import json
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent
from utils.networks import GCActor, GCValue
from agents.fbpiswitch_nonhierarchical import FBpiSwitchNonHierarchicalAgent

class FBpiSwitchAgent(flax.struct.PyTreeNode):
    """
        FB pi-Switch agent trained from frozen parameters of learned representation and low-level policy.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    
    def load_agent_from_frozen(self, FLAGS, config, example_batch):
        flags_path = os.path.join(config['frozen_path'], "flags.json")
        with open(flags_path, "r") as f:
            saved_flags = json.load(f)
            
        frozen_config = saved_flags['agent']
        FBpiSwitchNonHierarchical = FBpiSwitchNonHierarchicalAgent.create(FLAGS.seed, example_batch, frozen_config)
        FBpiSwitchNonHierarchical = restore_agent(FBpiSwitchNonHierarchical, config['frozen_path'], config['restore_epoch'])
        
        # Unfreeze parameters to modify
        agent_params = flax.core.unfreeze(self.network.params)
        frozen_params = flax.core.unfreeze(FBpiSwitchNonHierarchical.network.params)
        
        for module_name in frozen_params.keys():
            if module_name in ['modules_backward_repr', 'modules_forward_repr', 'modules_actor']:
                agent_params[module_name] = frozen_params[module_name]
                
        # Freeze back
        agent = self.replace(network=self.network.replace(params=agent_params))      
        return agent
    
    def normalize_z(self, z):
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.config['latent_dim'])
    
    def successor_measure_extract(self, observations, z_goals, z_intents, module='forward_repr'):
        # module is either "forward_repr" or "target_forward_repr"
        z_intents = self.normalize_z(z_intents)
        forward_reps = self.network.select(module)(observations, z_intents, goal_encoded=True)
        return jnp.sum(forward_reps * z_goals[None, :, :], axis=-1)
    


    def high_actor_loss(self, batch, grad_params, rng): 
        """Compute the high-level actor loss."""
        obs = batch['observations']
        subgoals = batch['high_actor_targets']   
        goals = batch['high_actor_goals']
        new_rng, rng = jax.random.split(self.rng)
        
        latents_norm, latents = jax.lax.stop_gradient(self.sample_latents(subgoals, rng, 0.0))
        z_goals_norm, z_goals = jax.lax.stop_gradient(self.sample_latents(goals, new_rng, self.config['critic_latent_mix_prob']))

        Msww = self.successor_measure_extract(obs, latents, latents)
        Mwww = self.successor_measure_extract(subgoals, latents, latents)       
        Vwrr = self.successor_measure_extract(subgoals, z_goals, z_goals)
        Vrstar = self.successor_measure_extract(obs, z_goals, z_goals)
        Vswr = self.successor_measure_extract(obs, z_goals, latents)

        adv = Vswr + Msww/Mwww*Vwrr - Vrstar
        exp_a = jnp.exp(jnp.clip(adv * self.config['high_alpha'], max=5.0))

        dist = self.network.select('high_actor')(obs, z_goals_norm, params=grad_params)
        log_prob = dist.log_prob(latents_norm)
        actor_loss = -(exp_a * log_prob).mean()
        
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'adv_std': adv.std(),
            'adv_min': adv.min(),
            'adv_max': adv.max(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - latents_norm) ** 2),
            'mse_z_latents': jnp.mean((z_goals_norm - latents_norm) ** 2),
            'Msww': Msww.mean(),
            'Vwrr': Vwrr.mean(),
            'Vrstar': Vrstar.mean(),
            'active_exps': (jnp.abs(exp_a) > 1e-3).mean(),
            'saturated': jnp.mean(adv * self.config['high_alpha'] >= 5.0),
        }
    


    
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, high_actor_rng = jax.random.split(rng, 2)

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v
            
        return high_actor_loss, info


    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        
        return self.replace(network=new_network, rng=new_rng), info


    @jax.jit
    def infer_latent(self, batch):
        """Infer the latent variable using rewards on downstream tasks."""
        observations = batch['observations']
        rewards = batch['rewards']
        weights = jax.nn.softmax(self.config['reward_temperature'] * rewards, axis=0)
        
        backward_reprs = self.network.select('backward_repr')(observations)
        
        # reward-weighted average
        latent = jnp.mean((weights * rewards)[..., None] * backward_reprs, axis=0)
        if self.config['normalize_latent']:
            latent = self.normalize_z(latent)

        return latent
    
    
    @jax.jit
    def sample_latents(self, targets, rng, latent_mix_prob):
        """Sample latent variables and intrinsic rewards."""
        batch_size = targets.shape[0]
        
        rng, latent_rng, mix_rng = jax.random.split(rng, 3)

        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']))
        norm_latents = self.normalize_z(latents)
        
        backward_reprs = self.network.select('backward_repr')(targets)
        norm_backward_reprs = self.normalize_z(backward_reprs)
        
        flags_latents = jax.random.uniform(mix_rng, (batch_size, 1)) < latent_mix_prob
        latents = jnp.where(flags_latents, latents, backward_reprs)
        norm_latents = jnp.where(flags_latents, norm_latents, norm_backward_reprs)

        return norm_latents, latents
    
    
    

    @jax.jit
    def sample_actions(self, observations, latents=None, seed=None, temperature=1.0):
        """Sample actions from the actor."""

        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, latents, goal_encoded=True, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        
        goal_reps = self.normalize_z(goal_reps)
        low_dist = self.network.select('actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        return jnp.clip(actions, -1, 1)

        
        
    @classmethod
    def create(cls, seed, ex_batch, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch['observations']
        ex_actions = ex_batch['actions']
        action_dim = ex_actions.shape[-1]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

        # Define networks.
        forward_repr_def = GCValue(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['forward_repr_layer_norm'],
            num_ensembles=2,
        )
        
        backward_repr_def = GCValue(
            hidden_dims=config['backward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            activations=getattr(nn, config['activation']),
            layer_norm=config['backward_repr_layer_norm'],
            num_ensembles=1,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
        )


        # Trainable new modules
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['latent_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=None,
        )
        
        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, None, None, True)),
            backward_repr=(backward_repr_def, (ex_observations,)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            
            # Trainable new modules
            high_actor=(high_actor_def, (ex_observations, ex_latents, True)),
        )
        
        def mask_fn(params):
            flat = flax.traverse_util.flatten_dict(params)
            mask = {}
            for k in flat:
                module_name = k[0]  # e.g. 'modules_forward_repr'
        
                if module_name in [
                    'modules_high_actor',
                ]:
                    mask[k] = 'train'
                else:
                    mask[k] = 'frozen'
        
            return flax.traverse_util.unflatten_dict(mask)

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        network_tx = optax.multi_transform(
            {
                'train': optax.adam(config['lr']),
                'frozen': optax.set_to_zero(),
            },
            mask_fn(network_params)
        )
         
        network = TrainState.create(network_def, network_params, tx=network_tx)    

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fbpiswitch',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            forward_repr_hidden_dims=(512, 512, 512),  # Forward representation network hidden dimensions.
            backward_repr_hidden_dims=(512, 512, 512),  # Backward representation network hidden dimension.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            forward_repr_layer_norm=True,  # Whether to use layer normalization for the forward representations.
            backward_repr_layer_norm=True,  # Whether to use layer normalization for the backward representations.
            activation='gelu',  # Activation function.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            alpha=3.0,  # Low-level AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actors.
            
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            # tanh_squash=True,  # Whether to use tanh squash for the actor.
            reward_temperature=0.0,  # Reward weight temperature.
            num_zero_shot_samples = 100_000,


            # FB pi-Switch specific hyperparameters.
            relabeling=True,  # Whether to relabel rewards.
            latent_dim=128,  # Embedding dimension for FB representation
            normalize_latent=True,  # Whether to normalize backward representations.
            orthonorm_coeff=1e-3,  # orthonormalization coefficient
            actor_latent_mix_prob=0.5,  # Probability to replace latents sampled from gaussian with backward representations.
            high_alpha=0.1,
            critic_latent_mix_prob=0.5,  # Probability to replace latents sampled from gaussian with backward representations.
            frozen_path="",
            restore_epoch=1_000_000,
        )
    )
    return config