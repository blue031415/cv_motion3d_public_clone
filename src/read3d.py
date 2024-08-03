from ezc3d import c3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

path = "../dataset/07_01.c3d"

c = c3d(path)
print(c['parameters']['POINT']['USED']['value'][0])
point_data = c['data']['points']
print(point_data.shape) #(XYZ1, 41, 316)
points_residuals = c['data']['meta_points']['residuals']
print(points_residuals.shape)
analog_data = c['data']['analogs']
print(analog_data.shape)

# data = point_data[0:3,:,1]
# print(data.shape)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data[0,:],data[1,:],data[2,:], s=5)
# ax.set_box_aspect([abs(max(data[0,:])-min(data[0,:])), abs(max(data[1,:])-min(data[1,:])), abs(max(data[2,:])-min(data[2,:]))])
# plt.show()

num_frame = 316

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