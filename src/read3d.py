from ezc3d import c3d

# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
# from utils import read_c3d, display_motion, display_point
# import numpy as np

path = "../dataset/000004_000103_75_236_R_003_972.c3d"
c = c3d(path)
point_data = c["data"]["points"]
print(point_data.shape)


# display_motion(path)
