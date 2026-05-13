import numpy as np
X = -np.inf
L = 5.0 
G = 10.0
CONST_goal_tol = 0.5  # line 86 in ogbench/ogbench/locomaze/maze.py

BASE_MAPS = {
    "medium": [
        [X, X, X, X, X, X, X, X],
        [X, 0, 0, X, X, 0, 0, X],
        [X, 0, 0, X, 0, 0, 0, X],
        [X, X, 0, 0, 0, X, X, X],
        [X, 0, 0, X, 0, 0, 0, X],
        [X, 0, X, 0, 0, X, 0, X],
        [X, 0, 0, 0, X, 0, 0, X],
        [X, X, X, X, X, X, X, X],
        ],
    "large": [
        [X, X, X, X, X, X, X, X, X, X, X, X],
        [X, 0, 0, 0, 0, X, 0, 0, 0, 0, 0, X],
        [X, 0, X, X, 0, X, 0, X, 0, X, 0, X],
        [X, 0, 0, 0, 0, 0, 0, X, 0, 0, 0, X],
        [X, 0, X, X, X, X, 0, X, X, X, 0, X],
        [X, 0, 0, X, 0, X, 0, 0, 0, 0, 0, X],
        [X, X, 0, X, 0, X, 0, X, 0, X, X, X],
        [X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
        [X, X, X, X, X, X, X, X, X, X, X, X],
        ],
    "giant": [
        [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
        [X, 0, X, 0, 0, 0, 0, 0, 0, X, X, 0, 0, 0, 0, X],
        [X, 0, X, 0, X, X, 0, X, 0, X, 0, 0, X, X, 0, X],
        [X, 0, 0, 0, X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
        [X, 0, X, X, X, 0, X, X, X, X, X, X, 0, X, 0, X],
        [X, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X, 0, X],
        [X, X, X, 0, X, 0, X, 0, 0, X, 0, X, 0, X, X, X],
        [X, 0, 0, 0, X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
        [X, 0, X, 0, X, 0, X, X, X, X, X, X, 0, X, 0, X],
        [X, 0, X, X, X, 0, 0, 0, X, 0, 0, 0, X, X, 0, X],
        [X, 0, 0, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X],
        [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
        ],
    "teleport": [
        [X, X, X, X, X, X, X, X, X, X, X, X],
        [X, 0, 0, 0, 0, 0, X, 0, X, 0, 0, X],
        [X, X, 0, X, 0, 0, 0, X, 0, 0, X, X],
        [X, X, 0, X, X, X, 0, 0, 0, 0, 0, X],
        [X, 0, 0, 0, 0, X, 0, X, 0, X, 0, X],
        [X, 0, X, X, 0, X, 0, X, 0, X, 0, X],
        [X, 0, X, X, 0, X, 0, X, 0, X, 0, X],
        [X, 0, 0, 0, 0, X, 0, 0, 0, X, 0, X],
        [X, X, X, X, X, X, X, X, X, X, X, X],        
        ],
    }

REWARD_MAPS_REGIONS = {
        "large": {
            1: {
                "grid": [
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                        [X, 0,-1,-1,-1, X, 0, 0, 0, 0, 0, X],
                        [X, 1, X, X,-1, X, 0, X, 0, X, 0, X],
                        [X, 1, 1, 1, 0, 0, 0, X, 0, 0, 0, X],
                        [X, 0, X, X, X, X, 0, X, X, X, 0, X],
                        [X, 0, 0, X, 0, X, 0, 0, 0, 0, 0, X],
                        [X, X, 0, X, 0, X, 0, X, 0, X, X, X],
                        [X, 0, 0, X, 0, 0, 0, X, 0, 0, L, X],
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                    ],
                "argmax": (7, 10),
                "initial_state": (1, 1),
            },
            2: {
                "grid": [
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                        [X, L, 0, 0, 0, X, 0, 0, 0,-1,-1, X],
                        [X, 0, X, X, 0, X, 0, X, 1, X,-1, X],
                        [X, 0, 0, 0, 0, 0, 0, X, 1, 1, 0, X],
                        [X, 0, X, X, X, X,-1, X, X, X, 0, X],
                        [X, 0, 0, X, 0, X,-1,-1, 0, 0, 0, X],
                        [X, X, 0, X, 0, X, 0, X, 0, X, X, X],
                        [X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                    ],
                "argmax": (1, 1),
                "initial_state": (5, 10),
            },
            3: {
                "grid": [
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                        [X, 1, 1, 1, 1, X, 0, 0, 0, 0, L, X],
                        [X, 0, X, X, 1, X, 0, X, 0, X, 0, X],
                        [X, 0, 0, 0, 0, 0, 0, X, 0, 0, 0, X],
                        [X, 0, X, X, X, X, 0, X, X, X, 0, X],
                        [X, 0, 0, X, 0, X, 0, 0, 0, 0, 0, X],
                        [X, X, 0, X, 0, X, 0, X, 0, X, X, X],
                        [X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                    ],
                "argmax": (1, 10),
                "initial_state": (7, 1),
            },
            4: {
                "grid": [
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                        [X, 0, 0, 0, 0, X,-1,-1, 0, 0, 0, X],
                        [X, 0, X, X, 0, X,-1, X, 0, X, 1, X],
                        [X, 0, 0, 0, 0, 0,-1, X, 0, 0, 1, X],
                        [X, 0, X, X, X, X,-1, X, X, X, 1, X],
                        [X, 0, 0, X, G, X, 0, 0, 0, 1, 1, X],
                        [X, X, 0, X, 0, X, 0, X, 0, X, X, X],
                        [X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                    ],
                "argmax": (5, 4),
                "initial_state": (1, 10),
            },
            5: {
                "grid": [
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                        [X, 0, 0, 0, 0, X, 0, 0, 0, 0, 0, X],
                        [X, 0, X, X, 0, X, 0, X, L, X,-1, X],
                        [X, 0, 0, 0, 0, 0, 0, X, 0, 0,-1, X],
                        [X, 0, X, X, X, X, 0, X, X, X,-1, X],
                        [X, 0, 0, X, 0, X, 0, 0, 0, 0,-1, X],
                        [X, X, 0, X, 0, X, 0, X, 0, X, X, X],
                        [X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                        [X, X, X, X, X, X, X, X, X, X, X, X],
                    ],
                "argmax": (2, 8),
                "initial_state": (7, 10),
            },          
        },
        "giant": {
            1: {
                "grid":  [
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                         [X, 0, X, 0, 0, 0, 0,-1,-1, X, X, 0, 0, 0, 0, X],
                         [X, 0, X, 0, X, X, 0, X,-1, X, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, X, 0, 0, X,-1, 0, 0, X, 0, 0, 0, X],
                         [X, 0, X, X, X, 0, X, X, X, X, X, X, 1, X, 0, X],
                         [X, 0, 0, 0, X, G, 0, 0, X, 0, 1, 1, 1, X, 0, X],
                         [X, X, X, 0, X, 0, X, 0, 0, X, 0, X, 0, X, X, X],
                         [X, 0, 0, 0, X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                         [X, 0, X, 0, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, X, X, X, 0, 0, 0, X, 0, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X],
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                     ],
                "argmax": (5, 5),
                "initial_state": (1, 14),
            },
            2: {
                "grid":  [
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                         [X, 0, X, 0, 0, 0, 0, 0, 0, X, X, 0, 0, 0, 0, X],
                         [X, 0, X, 0, X, X, 0, X, 0, X, 0, 0, X, X, 0, X],
                         [X, 1, 1, 1, X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                         [X, 1, X, X, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 1, 0, 0, X,-1, 0, 0, X, 0, 0, 0, 0, X, 0, X],
                         [X, X, X, 0, X,-1, X, 0, 0, X, 0, X, 0, X, X, X],
                         [X, 0, 0, 0, X,-1, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                         [X, 0, X, 0, X,-1, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, X, X, X,-1, 0, 0, X, 0, 0, 0, X, X, 0, X],
                         [X, G, 0, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X],
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                     ],
                "argmax": (10, 1),
                "initial_state": (1, 6),
            }, 
            3: {
                "grid":  [
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                         [X, 0, X, 0, 0, 0, 0, 0, 0, X, X, 0, 0, 0, G, X],
                         [X, 0, X, 0, X, X, 0, X, 0, X, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                         [X, 0, X, X, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, 0, 0, X, 0, 0, 0, X, 0, 1, 1, 0, X, 0, X],
                         [X, X, X, 0, X, 0, X, 0, 1, X, 1, X, 0, X, X, X],
                         [X, 0, 0, 0, X, 0, 0, X, 1, 1, 1, X, 0, 0, 0, X],
                         [X, 0, X, 0, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, X, X, X, 0, 0, 0, X, 0, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X],
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                     ],
                "argmax": (1, 14),
                "initial_state": (5, 5),
            }, 
            4: {
                "grid":  [
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                         [X, 0, X, 0, 0, 0, 1, 1, 1, X, X, 0, 0, 0, 0, X],
                         [X, 0, X, 0, X, X, 0, X, 1, X, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, X, 0, 0, X, 1, 0, 0, X, G, 0, 0, X],
                         [X, 0, X, X, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, 0, 0, X, 0,-1,-1, X, 0, 0, 0, 0, X, 0, X],
                         [X, X, X, 0, X, 0, X,-1,-1, X, 0, X, 0, X, X, X],
                         [X, 0, 0, 0, X, 0, 0, X,-1,-1, 0, X, 0, 0, 0, X],
                         [X, 0, X, 0, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, X, X, X, 0, 0, 0, X, 0, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X],
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                     ],
                "argmax": (3, 12),
                "initial_state": (4, 5),
            }, 
            5: {
                "grid":  [
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                         [X, 0, X, 0, 0, 0, 0, 0, 0, X, X, 0, 0, 0, G, X],
                         [X, 0, X, 0, X, X, 0, X, 0, X, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, X, 0, 0, X, 0, 0, 0, X, 0, 0, 0, X],
                         [X, 0, X, X, X, 0, X, X, X, X, X, X,-1, X, 0, X],
                         [X, 0, 0, 0, X, 0,-1,-1, X, 0,-1,-1,-1, X, 0, X],
                         [X, X, X, 0, X, 0, X,-1,-1, X,-1, X, 0, X, X, X],
                         [X, 0, 0, 0, X, 0, 0, X,-1,-1,-1, X, 0, 0, 0, X],
                         [X, 0, X, 0, X, 0, X, X, X, X, X, X, 0, X, 0, X],
                         [X, 0, X, X, X, 0, 0, 0, X, 0, 0, 0, X, X, 0, X],
                         [X, 0, 0, 0, 0, 0, X, 0, 0, 0, X, 0, 0, 0, 0, X],
                         [X, X, X, X, X, X, X, X, X, X, X, X, X, X, X, X],
                     ],
                "argmax": (1, 14),
                "initial_state": (5, 5),
            },              
        },
    }



def get_reward_cfg(env, reward_id, task_type):
    loco_env_size = env.unwrapped._maze_type
    if task_type == 'base':
        return BASE_MAPS[loco_env_size]
    elif task_type == 'regions':
        return REWARD_MAPS_REGIONS[loco_env_size][reward_id]
    else:
        raise ValueError("Not proper task type selected!")


def complex_rewards_maze(env, observations, task_id, task_type):
    obs_pos = observations["xy_pos"]
    if task_type == 'regions':
        return reward_fn_regions(env, obs_pos, task_id)
    else:
        raise ValueError("Not proper task type selected!")


def reward_fn_regions(env, obs_pos, task_id):
    reward_cfg = get_reward_cfg(env, task_id, 'regions')
    reward_map = reward_cfg['grid']
    reward_map = np.array(reward_cfg['grid'])  
    ijs = np.array([env.unwrapped.xy_to_ij(tuple(c)) for c in obs_pos])
    rewards = reward_map[ijs[:, 0], ijs[:, 1]]
    
    goal_xy = env.unwrapped.ij_to_xy(reward_cfg['argmax'])
    dists = np.linalg.norm(obs_pos - goal_xy, axis=-1)
    successes = (dists <= CONST_goal_tol)
    rewards[successes] = G

    return rewards.reshape(obs_pos.shape[:-1])

