from flenv.src.env import Environment
from src.network.dqn import DeepQNetwork

dqn = DeepQNetwork(input_size=[50, 50, 4], action_size=8, num_episodes=10000, memory_frame_rate=3)
restored = dqn.restore_last_checkpoint(checkpoint=5)

if restored:
    print("Restored checkpoint!")

from collections import deque

state_stack = deque(maxlen=4)

import numpy as np

def resolve(env):
    # The model learns to return "1" for every state so it doesn't work due to sparse rewards
    state_stack.append(env.get_raster())
    if len(state_stack) < 4:
        return 7
    action = dqn.get_action_for_env(stacked_state=np.transpose(np.stack(state_stack)))
    #np.reshape(action, [1, 200, 200])
    print("Action: %s" % action)
    print(action, env.hash, "total reward: %s, num collisions: %s, collision percent: %s" % (env.total_reward, env.n_collisions, 100 * env.n_collisions/env.total_reward))
    return action


env = Environment(render=True, keyboard=False, scale=5, fov_size=25, actionResolver=resolve, framerate=250)

env.run()
