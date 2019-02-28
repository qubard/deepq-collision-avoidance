from flenv.src.env import Environment

#env = Environment(render=False, keyboard=False, scale=5, seed=0, fov_size=100)

#env.run()

env = Environment(render=False, keyboard=False, scale=5, fov_size=25, max_age=150)

env.run()

import matplotlib.pyplot as plt
import numpy as np

raster = env.get_raster()
plt.imshow(raster)

plt.show()