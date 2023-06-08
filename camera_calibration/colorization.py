import numpy as np
import cv2

# intrinsic matrix (depth, rgb)
K_depth = np.array([[619.829773,0,322.658264],[0,619.829773,238.828934],[0,0,1]])
K_rgb = np.array([[610.65109937,0,319.94904282],[0,611.21087259,250.36298641],[0,0,1]])

# np.dot(Rt_rgb, inv(Rt_depth))
Rt_matrix = np.array([[9.99989745e-01, 9.70878181e-04, 4.42348896e-03, 1.52100554e+00],
 [-9.45451266e-04, 9.99983041e-01, -5.74662779e-03, -2.37160321e-01],
 [-4.42899322e-03, 5.74238667e-03, 9.99973704e-01, 4.42699898e-02],
 [0,0,0,1]])

depth_data = np.load('./point_cloud/depth_3.npy')
img = cv2.imread('./point_cloud/rgb_img_3.png')

depth_pixel = np.ones((480*640,3), np.float32)
depth_pixel[:,:2] = np.mgrid[0:480, 0:640].T.reshape(-1,2)
depth_cam = np.dot(np.linalg.inv(K_depth), depth_pixel.T)
depth_cam = depth_cam * depth_data.T.reshape(1, -1) # depth cam 좌표계
#-----------------------------------------------------------------------------
depth_cam = np.concatenate((depth_cam, np.ones((1, 307200), np.float32)), axis=0) # depth_cam.shape : (4, 307200)
rgb_cam = np.dot(Rt_matrix, depth_cam) # (4x4)*(4x307200)
rgb_cam = np.delete(rgb_cam, 3 ,0) # rgb_cam.shape: (3, 307200) # (x, y, z, 1) -> (x, y, z)
rgb_cam = rgb_cam/rgb_cam[2, :] # z로 나눠주기 
rgb_pixel = np.dot(K_rgb, rgb_cam) # (3x3)*(3x307200)
rgb_pixel_round = np.int16(np.round(rgb_pixel)) # 소수점으로 남은 pixel 좌표를 int로 변환
rgb_pixel_round = rgb_pixel_round.T

# pixel xy 조건 추가 : 0 이하, 480 또는 640 이상 0으로 변환
rgb_pixel_x = np.where(rgb_pixel_round[:, 0] > 479, 0, rgb_pixel_round[:, 0]) 
rgb_pixel_x = np.where(rgb_pixel_x <= 0, 0, rgb_pixel_x)
rgb_pixel_y = np.where(rgb_pixel_round[:, 1] > 640, 0, rgb_pixel_round[:, 1])
rgb_pixel_y = np.where(rgb_pixel_y <= 0, 0, rgb_pixel_y)
rgb_pixel_xy = np.concatenate((rgb_pixel_x.reshape(-1, 1), rgb_pixel_y.reshape(-1, 1)), axis=1) # rgb_pixel_xy.shape: (2, 307200) # x,y 좌표만 존재 


# rgb 값 추출
rgb_data = np.zeros((480*640, 3), np.int16) # shape: 307200, 3
rgb_data[:, :] = img[rgb_pixel_xy[:, 0], rgb_pixel_xy[:, 1], :] # img에서 x, y좌표에 해당하는 rgb 넣기

pointcloud = np.loadtxt('./point_cloud/pointcloud_3.txt')
colorization = np.concatenate((pointcloud, rgb_data), axis=1) 

# save txt
np.savetxt('./point_cloud/colorization_11.txt', colorization)