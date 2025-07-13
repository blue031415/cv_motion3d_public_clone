import os

import numpy as np
from ezc3d import c3d

# import scipy
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# from scipy.linalg import eig
from util.preprocess import remove_nan


def display_motion(path):
    c = c3d(path)
    point_data = c["data"]["points"]  # (XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]

    x_range = np.nanmax(point_data[0, :, :]) - np.nanmin(point_data[0, :, :])
    y_range = np.nanmax(point_data[1, :, :]) - np.nanmin(point_data[1, :, :])
    z_range = np.nanmax(point_data[2, :, :]) - np.nanmin(point_data[2, :, :])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(data[0, :], data[1, :], data[2, :], s=5)
    ft = ax.set_title(f"frame num {0}")

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlim(np.nanmin(point_data[0, :, :]), np.nanmax(point_data[0, :, :]))
    ax.set_ylim(np.nanmin(point_data[1, :, :]), np.nanmax(point_data[1, :, :]))
    ax.set_zlim(np.nanmin(point_data[2, :, :]), np.nanmax(point_data[2, :, :]))

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])
        ft.set_text(f"frame num {frame}")
        return sc

    FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()


def display_motion_score(path, x, y, save_path):
    c = c3d(path)
    point_data = c["data"]["points"]  # (XYZ1, num_mark, num_frame)
    point_data = np.nan_to_num(point_data)
    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]

    title = os.path.basename(path)

    x_range = np.max(point_data[0, :, :]) - np.min(point_data[0, :, :])
    y_range = np.max(point_data[1, :, :]) - np.min(point_data[1, :, :])
    z_range = np.max(point_data[2, :, :]) - np.min(point_data[2, :, :])

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(data[0, :], data[1, :], data[2, :], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0, :, :]), np.max(point_data[0, :, :]))
    ax1.set_ylim(np.min(point_data[1, :, :]), np.max(point_data[1, :, :]))
    ax1.set_zlim(np.min(point_data[2, :, :]), np.max(point_data[2, :, :]))

    ax2 = fig.add_subplot(122)
    (line,) = ax2.plot([], [])
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title(title)
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])
        if x[0] <= frame and frame < x[-1]:
            line.set_data(x[0 : frame - x[0]], y[0 : frame - x[0]])
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    ani.save(save_path, writer="pillow", fps=20)


def display_motion_score_contribution(path, x, y, contribution, save_path):
    # config = Config()
    title = os.path.basename(path)

    c = c3d(path)
    point_data = c["data"]["points"]  # (XYZ1, num_mark, num_frame)
    point_data = point_data[0:3, :, :]
    point_data = remove_nan(point_data)

    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]
    cb = contribution[0]
    norm = colors.Normalize(vmin=min(cb), vmax=max(cb))

    x_range = np.max(point_data[0, :, :]) - np.min(point_data[0, :, :])
    y_range = np.max(point_data[1, :, :]) - np.min(point_data[1, :, :])
    z_range = np.max(point_data[2, :, :]) - np.min(point_data[2, :, :])

    fig = plt.figure(figsize=(11, 5))

    # fig.suptitle(f"{config.motion_description[title.split('.')[0]]} ({title})")
    fig.suptitle(f"({title})")
    gs = fig.add_gridspec(9, 11)
    ax1 = fig.add_subplot(gs[1:9, 0:7], projection="3d")

    sc = ax1.scatter(
        data[0, :], data[1, :], data[2, :], c=cb, norm=norm, cmap="jet", s=5
    )
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0, :, :]), np.max(point_data[0, :, :]))
    ax1.set_ylim(np.min(point_data[1, :, :]), np.max(point_data[1, :, :]))
    ax1.set_zlim(np.min(point_data[2, :, :]), np.max(point_data[2, :, :]))

    ax2 = fig.add_subplot(gs[1:8, 7:11])
    (line,) = ax2.plot([], [])
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title(f"Dissimilarity of {os.path.splitext(os.path.basename(path))[0]} DS")
    ax2.grid(True)

    cbar = plt.colorbar(sc, pad=0.2)
    cbar.set_ticks([])

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])

        if x[0] <= frame and frame < x[-1]:
            line.set_data(x[0 : frame - x[0] + 1], y[0 : frame - x[0] + 1])
            cb = contribution[frame - x[0] + 1]
            norm = colors.Normalize(vmin=min(cb), vmax=max(cb))
            sc.set_array(cb)
            sc.set_norm(norm)
        ft.set_text(f"frame num {frame}")
        cbar.set_ticks([])

        return sc, line, cbar

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    # plt.show()
    ani.save(save_path, writer="pillow", fps=20)


# 複数の波形を表示する
def display_motion_some_score(path, x, y, y_label, save_path):
    # cfg = Config()

    # yはarray
    # y[0] 値1
    # y[1] 値2
    # y_label[0] 値1のラベル
    # y_label[1] 値2のラベル

    title = os.path.basename(path)

    c = c3d(path)
    point_data = c["data"]["points"]  # (XYZ1, num_mark, num_frame)
    point_data = point_data[0:3, :, :]
    point_data = remove_nan(point_data)

    num_frame = point_data.shape[2]
    data = point_data[0:3, :, 0]

    x_range = np.max(point_data[0, :, :]) - np.min(point_data[0, :, :])
    y_range = np.max(point_data[1, :, :]) - np.min(point_data[1, :, :])
    z_range = np.max(point_data[2, :, :]) - np.min(point_data[2, :, :])

    fig = plt.figure(figsize=(11, 5))
    # fig.suptitle(f"{cfg.motion_description[title.split('.')[0]]} ({title})")
    fig.suptitle(f"({title})")
    gs = fig.add_gridspec(9, 11)
    ax1 = fig.add_subplot(gs[0:9, 0:6], projection="3d")

    sc = ax1.scatter(data[0, :], data[1, :], data[2, :], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0, :, :]), np.max(point_data[0, :, :]))
    ax1.set_ylim(np.min(point_data[1, :, :]), np.max(point_data[1, :, :]))
    ax1.set_zlim(np.min(point_data[2, :, :]), np.max(point_data[2, :, :]))

    ax2 = fig.add_subplot(gs[1:7, 7:11])

    line = []
    for i in range(len(y)):
        line.append(ax2.plot([], [], label=y_label[i]))

    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title(f"Dissimilarity of {os.path.splitext(os.path.basename(path))[0]} DS")
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3, :, frame]
        sc._offsets3d = (data[0, :], data[1, :], data[2, :])

        if x[0] <= frame and frame < x[-1]:
            for i in range(len(y)):
                line[i][0].set_data(x[0 : frame - x[0] + 1], y[i][0 : frame - x[0] + 1])
        ft.set_text(f"frame num {frame}")

        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)
    plt.legend()
    # plt.show()
    ani.save(save_path, writer="pillow", fps=20)
