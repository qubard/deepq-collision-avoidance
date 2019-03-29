from src.network.dqn import DeepQNetwork

network = DeepQNetwork(input_size=[100, 100, 4], state_stack_size=4, action_size=9, num_episodes=10000, memory_frame_rate=3, device='gpu')
network.restore_last_checkpoint()
network.train(25)
