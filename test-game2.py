from flenv.src.env import Environment

env = Environment(render=True, keyboard=True, scale=5, fov_size=50, max_projectiles=40, framerate=25, max_age=1000)

env.run()

print(env.total_reward)