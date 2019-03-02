from flenv.src.env import Environment

env = Environment(render=True, keyboard=False, scale=5, fov_size=50, max_projectiles=20, framerate=25)

env.run()