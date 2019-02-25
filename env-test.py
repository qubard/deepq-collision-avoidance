from flenv.src.env import Environment

env = Environment(render=False, keyboard=False, scale=5, seed=0, fov_size=100)

env.run()

env = Environment(render=True, keyboard=False, scale=5, seed=0, fov_size=100)

env.run()
