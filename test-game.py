from flenv.src.env import Environment
from src.network.dqn import DeepQNetwork

dqn = DeepQNetwork(input_size=[100, 100, 4], action_size=9, num_episodes=10000, memory_frame_rate=3)
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
        return np.random.randint(0, dqn.action_size)
    action = dqn.get_action_for_env(stacked_state=np.transpose(np.stack(state_stack)))
    #np.reshape(action, [1, 200, 200])
    #print(action, env.hash, "total reward: %s, num collisions: %s, collision percent: %s" % (env.total_reward, env.n_collisions, 100 * env.n_collisions/env.total_reward))
    return action

avg_collisions = 0
n_experiments = 5
for _ in range(0, n_experiments):
    env = Environment(render=True, keyboard=False, scale=5, fov_size=50, max_projectiles=20, actionResolver=resolve, framerate=25, max_age=1000)

    env.run()
    if env.total_reward > 0:
        avg_collisions += env.total_reward
        print(env.total_reward)

print("Total avg score: %s" % (avg_collisions / n_experiments))