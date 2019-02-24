from flenv.src.env import Environment
from src.network.dqn import DeepQNetwork

dqn = DeepQNetwork(input_size=[200, 200, 1], action_size=9, num_episodes=1000, memory_frame_rate=60)
restored = dqn.restore_last_checkpoint(checkpoint=5)

if restored:
    print("Restored checkpoint!")


def resolve(env):
    action = dqn.get_action_for_env(env)
    print(action, env.hash)
    return action


env = Environment(render=True, keyboard=False, scale=5, fov_size=100, actionResolver=resolve)

env.run()
