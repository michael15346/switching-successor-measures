import copy
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


class CausalMixerBlock(nn.Module):
    tokens_dim: int
    channels_dim: int
    num_tokens: int

    @nn.compact
    def __call__(self, x):
        res = x
        x = nn.LayerNorm()(x)
        x = jnp.swapaxes(x, 1, 2)
        x = nn.Dense(self.tokens_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.num_tokens)(x)
        x = jnp.swapaxes(x, 1, 2)
        x = x + res

        res = x
        x = nn.LayerNorm()(x)
        M_full = self.param('causal_matrix', nn.initializers.normal(stddev=0.1), (self.num_tokens, self.num_tokens))
        mask = jnp.tril(jnp.ones((self.num_tokens, self.num_tokens)))
        M = M_full * mask
        x = jnp.einsum('ij,bjc->bic', M, x)
        x = x + res

        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.channels_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.channels_dim)(x)
        x = x + res

        return x


class FBChainOfGoalsMixer(nn.Module):
    latent_dim: int
    num_subgoals: int
    tokens_dim: int
    channels_dim: int
    num_blocks: int

    @nn.compact
    def __call__(self, z_s, z_g, target_subgoals=None, train=False):
        batch_size = z_s.shape[0]
        H = self.num_subgoals

        subgoal_tokens = self.param('subgoal_tokens',
                                    nn.initializers.normal(stddev=0.1),
                                    (H, self.latent_dim))
        subgoal_tokens = jnp.broadcast_to(subgoal_tokens, (batch_size, H, self.latent_dim))

        seq = jnp.concatenate([z_s[:, None, :], z_g[:, None, :], subgoal_tokens], axis=1)

        predicted_subgoals = []

        for i in range(H):
            x = seq
            for _ in range(self.num_blocks):
                x = CausalMixerBlock(
                    tokens_dim=self.tokens_dim,
                    channels_dim=self.channels_dim,
                    num_tokens=2 + H
                )(x)

            pred_z = x[:, 2 + i, :]
            pred_z = nn.Dense(self.latent_dim)(pred_z)
            predicted_subgoals.append(pred_z)

            if train and target_subgoals is not None:
                seq = seq.at[:, 2 + i, :].set(target_subgoals[:, i, :])
            else:
                seq = seq.at[:, 2 + i, :].set(pred_z)

        return jnp.stack(predicted_subgoals, axis=1)


class FBPiSwitchCoGHPAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def load_agent_from_frozen(self, FLAGS, config, example_batch):
        flags_path = os.path.join(config['frozen_path'], "flags.json")
        with open(flags_path, "r") as f:
            saved_flags = json.load(f)
        frozen_config = saved_flags['agent']

        full_config = dict(FBpiSwitchNonHierarchicalAgent.get_config())
        full_config.update(frozen_config)

        FBpiSwitchNonHierarchical = FBpiSwitchNonHierarchicalAgent.create(FLAGS.seed, example_batch, full_config)
        FBpiSwitchNonHierarchical = restore_agent(FBpiSwitchNonHierarchical, config['frozen_path'],
                                                  config['restore_epoch'])

        agent_params = flax.core.unfreeze(self.network.params)
        frozen_params = flax.core.unfreeze(FBpiSwitchNonHierarchical.network.params)

        for module_name in frozen_params.keys():
            if module_name in ['modules_backward_repr', 'modules_forward_repr', 'modules_actor']:
                agent_params[module_name] = frozen_params[module_name]

        agent = self.replace(network=self.network.replace(params=agent_params))
        return agent

    def normalize_z(self, z):
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8) * jnp.sqrt(self.config['latent_dim'])

    def successor_measure_extract(self, observations, z_goals, z_intents, module='forward_repr'):
        z_intents = self.normalize_z(z_intents)
        forward_reps = self.network.select(module)(observations, z_intents, goal_encoded=True)
        return jnp.sum(forward_reps * z_goals[None, :, :], axis=-1)

    def high_sequence_actor_loss(self, batch, grad_params):
        obs = batch['observations']
        goals = batch['high_actor_goals']
        subgoals_seq = batch['subgoal_sequence']

        B = self.network.select('backward_repr')
        z_s = self.normalize_z(B(obs))
        z_g = self.normalize_z(B(goals))
        z_targets_seq = self.normalize_z(B(subgoals_seq))

        pred_z_seq = self.network.select('high_sequence_mixer')(
            z_s, z_g, z_targets_seq, train=True, params=grad_params
        )

        H = self.config['num_subgoals']
        gamma_h = self.config['subgoal_discount_factor']
        high_alpha = self.config['high_alpha']

        total_loss = 0.0
        info = {}

        for i in range(H):
            z_pred = pred_z_seq[:, i, :]
            w_i = subgoals_seq[:, i, :]
            z_target = z_targets_seq[:, i, :]

            Msww = self.successor_measure_extract(obs, z_target, z_target)
            Mwww = self.successor_measure_extract(w_i, z_target, z_target)
            Vwrr = self.successor_measure_extract(w_i, z_g, z_g)
            Vrstar = self.successor_measure_extract(obs, z_g, z_g)
            Vswr = self.successor_measure_extract(obs, z_g, z_target)

            adv = Vswr + (Msww / (Mwww + 1e-8)) * Vwrr - Vrstar
            adv = jax.lax.stop_gradient(adv)

            exp_a = jnp.exp(jnp.clip(adv * high_alpha, max=5.0))

            mse = jnp.sum((z_pred - z_target) ** 2, axis=-1)
            loss_i = jnp.mean(exp_a * mse)

            total_loss += (gamma_h ** (H - 1 - i)) * loss_i

            info[f'subgoal_{i}/adv'] = adv.mean()
            info[f'subgoal_{i}/mse'] = mse.mean()
            info[f'subgoal_{i}/loss'] = loss_i

        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, high_actor_rng = jax.random.split(rng, 2)
        high_actor_loss, high_actor_info = self.high_sequence_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v
        return high_actor_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        observations = batch['observations']
        rewards = batch['rewards']
        weights = jax.nn.softmax(self.config['reward_temperature'] * rewards, axis=0)
        backward_reprs = self.network.select('backward_repr')(observations)
        latent = jnp.mean((weights * rewards)[..., None] * backward_reprs, axis=0)
        if self.config['normalize_latent']:
            latent = self.normalize_z(latent)
        return latent

    @jax.jit
    def sample_actions(self, observations, goals, seed, temperature=1.0):
        """
        Generates the chain of subgoals and executes the nearest one (z_1).
        Replanning is handled by the evaluation loop calling this periodically.
        """
        is_single = (observations.ndim == 1)
        if is_single:
            observations = observations[None, ...]
            goals = goals[None, ...]

        z_s = self.normalize_z(self.network.select('backward_repr')(observations))

        z_g = self.normalize_z(goals)

        pred_z_seq = self.network.select('high_sequence_mixer')(
            z_s, z_g, None, train=False
        )

        z_1 = pred_z_seq[:, -1, :]
        z_1 = self.normalize_z(z_1)

        low_dist = self.network.select('actor')(observations, z_1, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        if is_single:
            actions = actions[0]
        return actions

    @classmethod
    def create(cls, seed, ex_batch, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_observations = ex_batch['observations']
        ex_actions = ex_batch['actions']
        action_dim = ex_actions.shape[-1]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']))

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

        high_sequence_mixer_def = FBChainOfGoalsMixer(
            latent_dim=config['latent_dim'],
            num_subgoals=config['num_subgoals'],
            tokens_dim=config['mixer_tokens_dim'],
            channels_dim=config['mixer_channels_dim'],
            num_blocks=config['mixer_num_blocks'],
        )

        z_s_dummy = jnp.ones((1, config['latent_dim']))
        z_g_dummy = jnp.ones((1, config['latent_dim']))
        target_dummy = jnp.ones((1, config['num_subgoals'], config['latent_dim']))

        network_info = dict(
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, None, None, True)),
            backward_repr=(backward_repr_def, (ex_observations,)),
            actor=(actor_def, (ex_observations, ex_latents, True)),
            high_sequence_mixer=(high_sequence_mixer_def, (z_s_dummy, z_g_dummy, target_dummy, True)),
        )

        def mask_fn(params):
            flat = flax.traverse_util.flatten_dict(params)
            mask = {}
            for k in flat:
                module_name = k[0]
                if module_name in ['modules_high_sequence_mixer']:
                    mask[k] = 'train'
                else:
                    mask[k] = 'frozen'
            return flax.traverse_util.unflatten_dict(mask)

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params = flax.core.unfreeze(network_params)
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
            agent_name='fbpiswitch_coghp',
            lr=3e-4,
            batch_size=2048,
            actor_hidden_dims=(512, 512, 512),
            forward_repr_hidden_dims=(512, 512, 512),
            backward_repr_hidden_dims=(512, 512, 512),
            actor_layer_norm=False,
            forward_repr_layer_norm=True,
            backward_repr_layer_norm=True,
            activation='gelu',
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            alpha=3.0,
            const_std=True,
            dataset_class='HGCDataset_CoGHP',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=False,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
            reward_temperature=0.0,
            num_zero_shot_samples=100_000,
            relabeling=True,
            latent_dim=128,
            normalize_latent=True,
            orthonorm_coeff=1e-3,
            actor_latent_mix_prob=0.5,
            high_alpha=0.1,
            critic_latent_mix_prob=0.5,
            frozen_path="",
            restore_epoch=1_000_000,
            num_subgoals=5,
            subgoal_steps=25,
            mixer_tokens_dim=64,
            mixer_channels_dim=128,
            mixer_num_blocks=3,
            subgoal_discount_factor=0.8
        )
    )
    return config