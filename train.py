from src.network.ddqn import DuelingDeepQNetwork

network = DuelingDeepQNetwork(input_size=[100, 100, 4], learning_rate=0.01, state_stack_size=4, action_size=9, \
                              num_episodes=10000, memory_frame_rate=3, device='gpu')
network.restore_last_checkpoint()
network.train(0)