from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, MLP, LengthNormalize, GCBilinearValue


class HIQLwithIntentionsAgent(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning (HIQL) without hierarchy, with bilinear structure and double value networks V(s,g,s')"""
    rng: Any
    network: Any
    config: Any = nonpytree_field()


    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)


    def value_loss(self, batch, grad_params):
        observations = batch['observations']
        next_observations = batch['next_observations']
        rewards = batch['rewards']
        value_goals = batch['value_goals']
        intention_goals = batch['intention_goals']
        intention_rewards = ((observations == intention_goals).all(axis=1)).astype(float)
        discount = self.config['discount']
        expectile = self.config['expectile']
        
        # TD-error
        (next_v1_gz, next_v2_gz) = self.network.select('target_value')(next_observations, value_goals, intention_goals)
        q1_gz = rewards + discount * next_v1_gz
        q2_gz = rewards + discount * next_v2_gz

        q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)
        (v1_gz, v2_gz) = self.network.select('value')(observations, value_goals, intention_goals, params=grad_params)
    
        # Direction-error 
        (next_v1_zz, next_v2_zz) = self.network.select('value')(next_observations, intention_goals, intention_goals, params=grad_params)
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
        q_zz = intention_rewards + discount * next_v_zz
        
        (v1_zz, v2_zz) = self.network.select('value')(observations, intention_goals, intention_goals, params=grad_params)
        v_zz = (v1_zz + v2_zz) / 2
        adv = jax.lax.stop_gradient(q_zz - v_zz)
        
        value_loss1 = self.expectile_loss(adv, q1_gz-v1_gz, expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2_gz-v2_gz, expectile).mean()
        value_loss = value_loss1 + value_loss2
        
        return value_loss, {
            'value_loss': value_loss,
            'v_gz max': v1_gz.max(),
            'v_gz min': v1_gz.min(),
            'v_zz': v_zz.mean(),
            'v_gz': v1_gz.mean(),
            'abs adv mean': jnp.abs(adv).mean(),
            'adv mean': adv.mean(),
            'adv max': adv.max(),
            'adv min': adv.min(),
            'accept prob': (adv >= 0).mean(),
            'reward mean': batch['rewards'].mean(),
            'q_gz max': q1_gz.max(),
        }

    

    def actor_loss(self, batch, grad_params):
        """Compute the actor loss (AWR)."""
        """ Here we use high_actor_goals since those are sampled uniformly in 
        trajectory after current state with probability 1-actor_p_randomgoal and
        random with probability actor_p_randomgoal
        """
        v1, v2 = self.network.select('value')(batch['observations'], batch['actor_goals'], batch['actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['actor_goals'], batch['actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = jax.lax.stop_gradient(nv - v)
        # exp_a = jnp.exp(jnp.clip(adv * self.config['alpha'], a_max=5.0))
        exp_a = jnp.exp(jnp.clip(adv * self.config['alpha'], max=5.0))

        goal_reps = self.network.select('goal_rep')(batch['actor_goals'], params=grad_params)
        dist = self.network.select('actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)

        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }



    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss
        return loss, info


    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params


        
    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from the actor."""
        goal_reps = self.network.select('goal_rep')(goals)
        dist = self.network.select('actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)

        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1) 


    @classmethod
    def create(cls, seed, ex_batch, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = ex_batch['observations']
        ex_actions = ex_batch['actions']
        action_dim = ex_actions.shape[-1]   
        ex_goals = ex_batch['value_goals']
        ex_latents = jnp.zeros([ex_observations.shape[0], config['latent_dim']])
        
        goal_rep_def = nn.Sequential([
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['latent_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            ), 
            LengthNormalize(),  
        ])
        
        value_encoder = GCEncoder(state_encoder=goal_rep_def)
        target_value_encoder = GCEncoder(state_encoder=goal_rep_def)

        value_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            latent_dim=config['latent_dim'],
            goal_encoder=value_encoder,
        )
        
        target_value_def = GCBilinearValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            latent_dim=config['latent_dim'],
            goal_encoder=target_value_encoder,
        )
         
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=None,
        )
    
        network_info = dict(
            goal_rep=(goal_rep_def, (ex_goals,)),
            value=(value_def, (ex_observations, ex_goals, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals, ex_goals)),
            actor=(actor_def, (ex_observations, ex_latents)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_tx = optax.adam(learning_rate=config['lr'])        
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = jax.tree_util.tree_map(lambda x: x, params['modules_value'])

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='iqlintentions',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            alpha=3.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            relabeling=True,  # Whether to relabel rewards.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.2,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.3,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            
            # IQL double specific hyperparameters.
            latent_dim = 128,  # Embedding dimension for FB representation
            # p_same_goals=0.5, # Probability of goal and intention being the same state
        )
    )
    return config