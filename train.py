from src.network.dqn import DeepQNetwork

network = DeepQNetwork(input_size=[200, 200, 1], action_size=9, num_episodes=1000, memory_frame_rate=60)
network.train()