from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from config import Confing
from utils import read_c3d,gen_shape_subspace,cal_magnitude,gen_shape_difference_subspace
from utils import gen_shape_principal_com_subspace,display_motion_score,gram_schmidt,display_motion_score_contribution
import numpy as np


path = "../dataset/07_01.c3d"
cfg = Confing()
tau = cfg.interval
data = read_c3d(path)
num_frame = data.shape[2]

data_title = path.split('/')[2].split('.')[0]

mag_list = []
frame_list = []
f = tau*2 // 2

contribution_list = []

for i in range(num_frame-tau*2):

    S1 = gen_shape_subspace(data[:,:,i],cfg)
    S2 = gen_shape_subspace(data[:,:,i+tau],cfg)
    S3 = gen_shape_subspace(data[:,:,i+tau*2],cfg)

    M = gen_shape_principal_com_subspace(S1,S3,cfg)

    D = gen_shape_difference_subspace(S2,M,cfg)

    P = D @ D.T
    V = P @ S2

    V = V / np.linalg.norm(V, axis=0)
    V = gram_schmidt(V)

    V = np.square(V)
    V = np.sum(V, axis=1)

    mag = cal_magnitude(S2,M)
    mag_list.append(mag)
    frame_list.append(f)
    f += 1

    contribution_list.append(V)

display_motion_score_contribution(path,frame_list,mag_list,contribution_list, f"../result/{data_title}_second.gif")
