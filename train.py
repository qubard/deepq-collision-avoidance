from src.network.dqn import DeepQNetwork

network = DeepQNetwork(input_size=[100, 100, 8], state_stack_size=8, action_size=9, num_episodes=10000, memory_frame_rate=3, device='gpu')
network.restore_last_checkpoint()
network.train()
