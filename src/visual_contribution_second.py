import sys

# from ezc3d import c3d
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
from config import Config
from utils import (
    read_c3d,
    gen_shape_subspace,
    cal_magnitude,
    gen_shape_difference_subspace,
)
from utils import gen_shape_principal_com_subspace, gram_schmidt
from util.display import display_motion_score_contribution
from util.preprocess import remove_nan
import numpy as np
from tqdm import tqdm

# コマンドライン引数からパスを取得
if len(sys.argv) < 2:
    print("Usage: python xxx.py [data_path]")
    sys.exit(1)

path = sys.argv[1]

cfg = Config()
tau = cfg.interval
data = read_c3d(path)
data = remove_nan(data)
num_frame = data.shape[2]

data_title = path.split("/")[2].split(".")[0]

mag_list = []
frame_list = []
f = tau * 2 // 2

contribution_list = []

for i in tqdm(range(num_frame - tau * 2)):

    if np.isnan(data[:, :, i]).any() or np.isinf(data[:, :, i]).any():
        print("A contains NaN or inf values")

    S1 = gen_shape_subspace(data[:, :, i], cfg)
    S2 = gen_shape_subspace(data[:, :, i + tau], cfg)
    S3 = gen_shape_subspace(data[:, :, i + tau * 2], cfg)

    M = gen_shape_principal_com_subspace(S1, S3, cfg)

    D = gen_shape_difference_subspace(S2, M, cfg)

    P = D @ D.T
    V = P @ S2

    if np.isnan(V).any():
        print("V has NoN")
        exit(1)

    if np.isinf(V).any():
        print("V has Inf")
        exit(1)

    norm = np.linalg.norm(V, axis=0)
    if np.any(norm == 0):
        # print(V)
        # print("norm is zero!!")

        mag_list.append(0)
        frame_list.append(f)
        f += 1
        V = np.sum(V, axis=1)
        contribution_list.append(V.real)
    else:
        V = V / norm
        V = gram_schmidt(V)

        V = np.square(V)
        V = np.sum(V, axis=1)

        mag = cal_magnitude(S2, M)
        mag_list.append(mag)
        frame_list.append(f)
        f += 1

        contribution_list.append(V)

display_motion_score_contribution(
    path, frame_list, mag_list, contribution_list, f"../result/{data_title}_second.gif"
)
