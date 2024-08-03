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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
    ax.set_box_aspect([1, 1, 2])

    def update(frame):
        data = point_data[0:3,:,frame]
        sc._offsets3d = (data[0,:],data[1,:],data[2,:])
        return sc

    ani = FuncAnimation(fig, update, frames=num_frame, interval=20, blit=False)

    plt.show()


#1frame内のポイントデータから形状部分空間を作成する。
def gen_shape_subspace(data, cfg):
    #data shape is (3, num)
    X = data.T
    mv = np.mean(X, axis=0)
    Xc = X - mv
    U, S, V = np.linalg.svd(Xc)

    return U[:,0:cfg.subspace_dim]
    



if __name__ == "__main__": 
    path = "../dataset/01_01.c3d"
    cfg = Confing()
    data = read_c3d(path)
    display_motion(path)
    # print(data.shape)
    # U = gen_shape_subspace(data[:,:,0], cfg)
    # print(U.shape)


