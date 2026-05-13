from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from utils.reward_configs import complex_rewards_maze, get_reward_cfg

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    task_info=None,
    inferred_latent=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    complex_task_name=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        inferred_latent: Latent to be used for evaluation (only for unsupervised algorithms).
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    
    qpos_xy_start_idx = 0
    qvel_xy_start_idx = 15
    
    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        
        options = dict(render_goal=should_render)
        if complex_task_name is not None:
            reward_cfg = get_reward_cfg(env, task_id, complex_task_name) 
            options['task_info'] = {'init_ij': reward_cfg['initial_state'], 'goal_ij': reward_cfg['argmax']}
            current_return=0
            current_discounted_return=0
        elif task_id is not None:
            options['task_id'] = task_id
            
        observation, info = env.reset(options=options)
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done, done_regions = False, False
        step = 0    
        render = []
        episode_info = {}
        

        while not done:
            if inferred_latent is not None:
                action = actor_fn(observation, inferred_latent, temperature=eval_temperature)
            else:
                action = actor_fn(observation, goal, temperature=eval_temperature)
            action = np.asarray(action)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            step += 1

            if complex_task_name is not None: # do not terminate general tasks
                done = truncated
            else:
                done = terminated or truncated

                
            if complex_task_name is not None:
                next_observation_dict = {   
                    "xy_pos": next_observation[None,qpos_xy_start_idx : qpos_xy_start_idx + 2],
                    "xy_vel": next_observation[None,qvel_xy_start_idx : qvel_xy_start_idx + 2],
                    }
                reward = complex_rewards_maze(env, next_observation_dict, task_id, complex_task_name)[0] # relabelling reward using true reward function
                

            if complex_task_name == 'regions' and not done_regions:
                if reward > 1: # reached highest reward state
                    done_regions = True
                    episode_info.update({
                        'to_goal_return': float(current_return),
                        'to_goal_discounted_return': float(current_discounted_return),
                        'in_goal_discounted_return': reward * agent.config['discount']**(step-1)/(1-agent.config['discount']),
                        'in_goal_return': reward * (1000 - step),
                        'goal_reached': 1.0,
                    })
                else:   # has not reached highest reward state yet
                    episode_info.update({
                        'to_goal_return': float(current_return),
                        'to_goal_discounted_return': float(current_discounted_return),
                        'in_goal_return': 0.0,
                        'in_goal_discounted_return': 0.0,
                        'goal_reached': 0.0,
                    })
                            
            if complex_task_name is not None:
                current_discounted_return += reward * agent.config['discount']**(step-1)
                current_return += reward
                
                episode_info['total_discounted_return'] = current_discounted_return
                episode_info['total_return'] = current_return
                info = {**info, **episode_info}

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)
                        
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.asarray(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajs, renders





