import numpy as np

# depth intrinsic matrix
K_depth = np.array([[619.829773,0,322.658264],[0,619.829773,238.828934],[0,0,1]])

depth_data = np.load('./point_cloud/depth_3.npy')

world = np.ones((480*640,3), np.float32)
world[:,:2] = np.mgrid[0:480, 0:640].T.reshape(-1,2)
xy_cood = np.dot(np.linalg.inv(K_depth), world.T)
xy_cood = xy_cood * depth_data.T.reshape(1, -1)
print(depth_data.T.reshape(1, -1).shape)

# save txt
np.savetxt('./point_cloud/pointcloud_10.txt', xy_cood.T)