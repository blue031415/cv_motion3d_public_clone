import sys

# from ezc3d import c3d
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
from config import Config
from utils import (
    read_c3d,
    gen_shape_subspace,
    # cal_magnitude,
    # gen_shape_difference_subspace,
)

# from utils import gen_shape_principal_com_subspace, gram_schmidt
from utils import along_geodesic, orth_decomposition_geodesic
from util.display import display_motion_some_score
from util.preprocess import remove_nan
import numpy as np
from tqdm import tqdm


# コマンドライン引数からパスを取得
if len(sys.argv) < 2:
    print("Usage: python xxx.py [data_path]")
    sys.exit(1)

path = sys.argv[1]
# path = "../dataset/07_01.c3d"

cfg = Config()
tau = cfg.interval
data = read_c3d(path)
data = remove_nan(data)
num_frame = data.shape[2]

data_title = path.split("/")[2].split(".")[0]

mag1_list = []
mag2_list = []
frame_list = []
f = tau * 2 // 2

contribution_list = []

for i in tqdm(range(num_frame - tau * 2)):

    if np.isnan(data[:, :, i]).any() or np.isinf(data[:, :, i]).any():
        print("A contains NaN or inf values")

    S1 = gen_shape_subspace(data[:, :, i], cfg)
    S2 = gen_shape_subspace(data[:, :, i + tau], cfg)
    S3 = gen_shape_subspace(data[:, :, i + tau * 2], cfg)

    mag1 = along_geodesic(S1, S2, S3, cfg)
    mag2 = orth_decomposition_geodesic(S1, S2, S3, cfg)

    mag1_list.append(mag1)
    mag2_list.append(mag2)

    frame_list.append(f)
    f += 1

mag_list = [mag1_list, mag2_list]
mag_list_label = ["along geodesic", "orth to geodesic"]

display_motion_some_score(
    path, frame_list, mag_list, mag_list_label, f"../result/{data_title}_geodestic.gif"
)
