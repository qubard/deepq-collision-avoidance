from flenv.src.env import Environment
from src.network.dqn import DeepQNetwork

dqn = DeepQNetwork(input_size=[200, 200], action_size=9, num_episodes=1000, memory_frame_rate=3)
restored = dqn.restore_last_checkpoint(checkpoint=5)

if restored:
    print("Restored checkpoint!")


def resolve(env):
    # The model learns to return "1" for every state so it doesn't work due to sparse rewards
    action = dqn.get_action_for_env(env)
    print(action, env.hash)
    return action


env = Environment(render=True, keyboard=False, scale=5, fov_size=100, actionResolver=resolve)

env.run()
