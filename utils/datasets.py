import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        if 'prev_actions' not in result:
            result['prev_actions'] = self._dict['actions'][np.minimum(np.maximum(idxs - 1, 0), self.size - 1)]
        if 'next_actions' not in result:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            # assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                new_dict = dict(observations=stacked_observations)
                if 'next_observations' in self.dataset:
                    stacked_next_observations = self.get_stacked_observations(
                        np.arange(self.size), key='next_observations')
                    new_dict['next_observations'] = stacked_next_observations
                self.dataset = Dataset(self.dataset.copy(new_dict))

    def sample(self, batch_size, idxs=None, relabeling=True, augmentation=True):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            relabeling: Whether to relabel reward. If True and 'relabeling' is also True in config, reward relabeling is applied.
            augmentation: Whether to augment image. If True, image augmentation is applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(np.minimum(idxs + 1, self.size - 1))
        if 'oracle_reps' in self.dataset:
            batch['goals'] = self.get_observations(idxs, key='oracle_reps')
            batch['next_goals'] =  self.get_observations(
                np.minimum(idxs + 1, self.size - 1), key='oracle_reps')
        else:
            batch['goals'] = self.get_observations(idxs)
            batch['next_goals'] =  self.get_observations(
                np.minimum(idxs + 1, self.size - 1))
        
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goal_observations'] = self.get_observations(value_goal_idxs)
        batch['actor_goal_observations'] = self.get_observations(actor_goal_idxs)
        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.get_observations(value_goal_idxs, key='oracle_reps')
            batch['actor_goals'] = self.get_observations(actor_goal_idxs, key='oracle_reps')
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
            batch['actor_goals'] = self.get_observations(actor_goal_idxs)

        if self.config['relabeling'] and relabeling:
            successes = (idxs == value_goal_idxs).astype(float)
            batch['masks'] = 1.0 - successes
            batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and augmentation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample
    ):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - self.config['discount'], size=batch_size
            ) - 1 # in [0, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Goals at the current state.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs, key='observations'):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset[key])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs, key='observations'):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset[key]))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """
    def sample_sphere(self, batch_size, dim, dtype=np.float32, scale_sqrt_d=True):
        z = np.random.randn(batch_size, dim)
        z /= (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        if scale_sqrt_d:
            z *= np.sqrt(dim)
        return z.astype(dtype)

    def sample_mask(self, batch_size, prob, dtype=bool):
        """Return a boolean mask of size batch_size, True with probability `prob`."""
        return (np.random.rand(batch_size) < prob).astype(dtype)

    def sample(self, batch_size, idxs=None, relabeling=True, augmentation=True):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            relabeling: Whether to relabel reward. If True and 'relabeling' is also True in config, reward relabeling is applied.
            augmentation: Whether to augment image. If True, image augmentation is applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)


        # Sample value goals.
        value_goal_idxs = self.sample_goals(idxs, self.config['value_p_curgoal'], self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'], self.config['value_geom_sample'])
        batch['value_goals'] = self.get_observations(value_goal_idxs)
                
        if self.config['relabeling'] and relabeling:
            successes = (idxs == value_goal_idxs).astype(float)
            batch['masks'] = 1.0 - successes
            batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        
        # Sample low-level actor goals
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if self.config['agent_name']=='hiql':
            low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
            mid_low_goal_idxs = np.minimum(idxs + 10, final_state_idxs)
        else:
            offsets = np.random.geometric(p=1-self.config['discount'], size=batch_size)  # in [1, inf)
            low_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
            mid_low_goal_idxs = np.minimum(idxs + offsets//2, final_state_idxs)

        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)
        batch['mid_low_actor_goals'] = self.get_observations(mid_low_goal_idxs)

        
        # Sample high-level actor goals
        if self.config['actor_geom_sample']:
            offsets = np.random.geometric(p=1-self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round((np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(int)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        
        if self.config['agent_name'] == 'hiql':
            high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)
            high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
            high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
            batch['high_actor_targets'] = self.get_observations(high_target_idxs)
        else:
            offsets = np.random.geometric(p=1-self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_target_idxs = np.minimum(idxs + offsets, final_state_idxs)
            high_random_target_idxs = np.minimum(idxs + offsets, final_state_idxs)
            high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
            batch['high_actor_targets'] = self.get_observations(high_target_idxs)


  
        return batch


@dataclasses.dataclass
class HGCDataset_CoGHP(HGCDataset):
    """Dataset class for hierarchical goal-conditioned RL with Chain-of-Goals Hierarchical Policy-like high-level sampling.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample_future_state(
            self,
            idxs,
            steps=None,
            geom=False,
            min_offset=1,
            key="observations",
            return_idxs=False,
    ):
        """
        Sample a future state from the same trajectory as each index.

        Args:
            idxs: Current dataset indices.
            steps: Fixed number of steps into the future. If None, sample randomly.
            geom: If True and steps is None, use geometric sampling.
            min_offset: Minimum future offset when randomly sampling.
            key: Observation key to return.
            return_idxs: If True, return future indices instead of observations.
        """
        idxs = np.asarray(idxs)
        batch_size = idxs.shape[0]

        # Find the end of the trajectory for each index.
        final_state_idxs = self.terminal_locs[
            np.searchsorted(self.terminal_locs, idxs)
        ]

        # Maximum allowed offset before hitting the trajectory end.
        max_offsets = np.maximum(final_state_idxs - idxs, 0)

        if steps is None:
            if geom:
                # Geometric future sampling, like the existing GCDataset code.
                offsets = np.random.geometric(
                    p=1.0 - self.config["discount"],
                    size=batch_size,
                ) - 1
                offsets = offsets + min_offset
            else:
                # Uniformly sample a future offset in [min_offset, max_offset].
                upper = np.maximum(max_offsets, min_offset)
                offsets = (
                                  np.random.rand(batch_size) * (upper - min_offset + 1)
                          ).astype(int) + min_offset

                # If already at terminal state, keep offset 0.
                offsets = np.where(max_offsets >= min_offset, offsets, 0)
        else:
            # Fixed future step.
            offsets = np.broadcast_to(np.asarray(steps), (batch_size,)).astype(int)

        # Do not go past the end of the trajectory.
        offsets = np.minimum(offsets, max_offsets)

        future_idxs = idxs + offsets
        future_idxs = np.minimum(future_idxs, final_state_idxs)

        if return_idxs:
            return future_idxs

        return self.get_observations(future_idxs, key=key)

    def sample_subgoal_sequence(
            self,
            idxs,
            num_subgoals=None,
            subgoal_steps=None,
            key="observations",
            far_to_near=True,
    ):
        """
        Sample a sequence of future states from the same trajectory.

        For CoGHP-style training, returns states at fixed k-step intervals:

            s_{t+k}, s_{t+2k}, ..., s_{t+Hk}

        clipped to the trajectory terminal.

        By default, returns them farthest-to-nearest:

            s_{t+Hk}, ..., s_{t+2k}, s_{t+k}

        because FBChainOfGoalsMixer uses the last token as the nearest subgoal.
        """
        idxs = np.asarray(idxs)

        if num_subgoals is None:
            num_subgoals = self.config.get("num_subgoals", 2)

        if subgoal_steps is None:
            subgoal_steps = self.config.get("subgoal_steps", 25)

        H = num_subgoals
        k = subgoal_steps

        final_state_idxs = self.terminal_locs[
            np.searchsorted(self.terminal_locs, idxs)
        ]

        # Nearest-to-farthest offsets: [k, 2k, ..., Hk]
        offsets = k * np.arange(1, H + 1)[None, :]

        # Future indices for each subgoal.
        future_idxs = np.minimum(
            idxs[:, None] + offsets,
            final_state_idxs[:, None],
        )

        if far_to_near:
            # Convert to [Hk, ..., 2k, k]
            future_idxs = future_idxs[:, ::-1]

        # Get observations for each subgoal position.
        future_obs_list = [
            self.get_observations(future_idxs[:, i], key=key)
            for i in range(H)
        ]

        # Stack into shape (batch, H, ...)
        return jax.tree_util.tree_map(
            lambda *xs: np.stack(xs, axis=1),
            *future_obs_list,
        )

    def sample(self, batch_size, idxs=None, relabeling=True, augmentation=True):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            relabeling: Whether to relabel reward. If True and 'relabeling' is also True in config, reward relabeling is applied.
            augmentation: Whether to augment image. If True, image augmentation is applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(idxs, self.config['value_p_curgoal'], self.config['value_p_trajgoal'],
                                            self.config['value_p_randomgoal'], self.config['value_geom_sample'])
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        if self.config['relabeling'] and relabeling:
            successes = (idxs == value_goal_idxs).astype(float)
            batch['masks'] = 1.0 - successes
            batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Sample low-level actor goals
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if self.config['agent_name'] == 'hiql':
            low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
            mid_low_goal_idxs = np.minimum(idxs + 10, final_state_idxs)
        else:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            low_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
            mid_low_goal_idxs = np.minimum(idxs + offsets // 2, final_state_idxs)

        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)
        batch['mid_low_actor_goals'] = self.get_observations(mid_low_goal_idxs)

        # Sample high-level actor goals
        if self.config['actor_geom_sample']:
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))).astype(int)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)

        offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
        high_traj_target_idxs = np.minimum(idxs + offsets, final_state_idxs)
        high_random_target_idxs = np.minimum(idxs + offsets, final_state_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)
        batch["goals"] = batch["high_actor_goals"]
        batch["subgoal_sequence"] = self.sample_subgoal_sequence(
            idxs,
            num_subgoals=self.config.get("num_subgoals", 2),
            subgoal_steps=self.config.get("subgoal_steps", 25),
        )
        return batch
