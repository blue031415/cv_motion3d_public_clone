from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.linalg import eig
from config import Confing
import scipy
from matplotlib import colors


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


def display_motion_score(path, x, y, save_path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(10,4))
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
    ax2.set_xlabel('frame')
    ax2.set_xlabel('value')
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        if x[0]<=frame and frame < x[-1]:
            line.set_data(x[0:frame-x[0]],y[0:frame-x[0]])
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()
    ani.save(save_path, writer='pillow', fps=20)


def display_motion_score_contribution(path, x, y, contribution, save_path):
    c = c3d(path)
    point_data = c['data']['points'] #(XYZ1, num_mark, num_frame)
    num_frame = point_data.shape[2]
    data = point_data[0:3,:,0]
    cb = contribution[0]
    norm = colors.Normalize(vmin=min(cb), vmax=max(cb))

    x_range = np.max(point_data[0,:,:]) - np.min(point_data[0,:,:])
    y_range = np.max(point_data[1,:,:]) - np.min(point_data[1,:,:])
    z_range = np.max(point_data[2,:,:]) - np.min(point_data[2,:,:])

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121, projection='3d')
    sc = ax1.scatter(data[0,:],data[1,:],data[2,:], c=cb, norm=norm, cmap='jet', s=5)
    ft = ax1.set_title(f"frame num {0}")

    ax1.set_box_aspect([x_range, y_range, z_range])
    ax1.set_xlim(np.min(point_data[0,:,:]),np.max(point_data[0,:,:]))
    ax1.set_ylim(np.min(point_data[1,:,:]),np.max(point_data[1,:,:]))
    ax1.set_zlim(np.min(point_data[2,:,:]),np.max(point_data[2,:,:]))

    ax2 = fig.add_subplot(122)
    line, = ax2.plot([], [])
    ax2.set_xlim(np.min(x),np.max(x))
    ax2.set_ylim(np.min(y),np.max(y))
    ax2.set_xlabel('frame')
    ax2.set_xlabel('value')
    ax2.grid(True)

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        
        if x[0]<=frame and frame < x[-1]:
            line.set_data(x[0:frame-x[0]+1],y[0:frame-x[0]+1])
            cb = contribution[frame-x[0]+1]
            norm = colors.Normalize(vmin=min(cb), vmax=max(cb))
            sc.set_array(cb)
            sc.set_norm(norm)
        ft.set_text(f"frame num {frame}")
        return sc, line

    ani = FuncAnimation(fig, update, frames=num_frame, interval=50, blit=False)

    plt.show()
    ani.save(save_path, writer='pillow', fps=20)




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


def gram_schmidt(arr):
    arr = np.array(arr, dtype=np.float64)
    k = arr.shape[1]
    u = arr[:,[0]]
    q = u / scipy.linalg.norm(u)

    for j in range(1, k):
        u = arr[:,[j]]
        for i in range(j):
            u -= np.dot(q[:,i], arr[:,j]) * q[:,[i]]
        qi = u / scipy.linalg.norm(u)
        q = np.append(q, qi, axis=1)
    return q


if __name__ == "__main__": 
    # score の例
    score = np.array([100, 2, 3, 4, 5, 6])

    # 散布図の座標データ（例）
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([2, 3, 4, 5, 6, 7])
    z = np.array([2, 3, 4, 5, 6, 7])
    norm = colors.Normalize(vmin=min(score), vmax=max(score))
    

    # plot
    ax=plt.figure(figsize=(3, 3)).add_subplot(111, projection='3d')
    scatter = ax.scatter(x,y,z, c=score,norm=norm, cmap='jet')
    plt.colorbar(scatter, label='Score')
    plt.show()






