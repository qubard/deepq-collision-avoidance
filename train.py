from src.network.dqn import DeepQNetwork

network = DeepQNetwork(input_size=[400, 400, 1], action_size=9, num_episodes=100000)
network.train()