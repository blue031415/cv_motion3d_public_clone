from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.linalg import eig
from config import Confing


def read_c3d(path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    data = point_data[0:3,:,:] #(XYZ, num_mark, num_frame)
    return data

def display_point(path, frame):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    data = point_data[0:3,:,frame]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ax.set_box_aspect([1, 1, 2])
    plt.show()

def display_motion(path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax.set_title(f"frame num {0}")

    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        ft.set_text(f"frame num {frame}")
        return sc

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()

def display_motion_score(path, x, y):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    ax2 = fig.add_subplot(122)
    line, = ax2.plot([], [])
    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        if x[0]<=frame:
            line.set_data(x[0:frame-x[0]],y[0:frame-x[0]])
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()



#1frame内のポイントデータから形状部分空間を作成する。
def gen_shape_subspace(data, cfg):
    #data shape is (3, num)
    X = data.T
    mv = np.mean(X, axis=0)
    Xc = X - mv
    U, S, V = np.linalg.svd(Xc)

    return U[:,0:cfg.subspace_dim]

def gen_shape_difference_subspace(S1,S2,cfg):
    # U, S, Vt = np.linalg.svd(S1.T @ S2)
    # S = np.diag(S)
    # I = np.eye(S.shape[0],S.shape[1])
    # D = ((S1 @ U) - (S2 @ Vt.T)) @ ((2 * (I - S))**(-0.5))
    G = S1 @ S1.T + S2 @ S2.T
    eigen_val, eigen_vec = eig(G)
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]
    return eigen_vec[:,idx]

def cal_magnitude(S1,S2):
    _, S, _ = np.linalg.svd(S1.T @ S2)
    mag = np.sum(2*(1 - S))
    return mag



if __name__ == "__main__": 
    path = "../dataset/01_01.c3d"
    cfg = Confing()
    data = read_c3d(path)
    display_motion(path)


