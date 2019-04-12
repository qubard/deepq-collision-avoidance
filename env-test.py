from flenv.src.env import Environment

env = Environment(render=True, keyboard=True, framerate=1000, scale=5, seed=0, fov_size=50)

env.run()


#import matplotlib.pyplot as plt
#import numpy as np

#raster = env.get_raster()
#plt.imshow(raster)

#plt.show()