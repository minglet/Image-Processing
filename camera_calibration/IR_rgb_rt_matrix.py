import cv2
import numpy as np

# rgb & depth intrinsic matrix
K_depth = np.array([[619.829773,0,322.658264],[0,619.829773,238.828934],[0,0,1]])
K_rgb = np.array([[610.65109937,0,319.94904282],[0,611.21087259,250.36298641],[0,0,1]])

# rgb & depth distortion 
dist_depth = np.array([[-1.64263637e-01, 2.05108283e+00, 1.10395678e-03, -7.00348745e-04,-7.02665650e+00]])
dist_rgb = np.array([[1.20663576e-02,1.06840071e+00,2.59352819e-03,-1.35519345e-03,-4.15062861e+00]])

CIRCLEBOARD = (5, 9) # circle board 차원

objp = np.zeros((5*9, 3), np.float32)
objp[0]  = (0, 0, 0)
objp[1]  = (1.6, 0, 0)
objp[2]  = (3.2, 0, 0)
objp[3]  = (4.8, 0, 0)
objp[4]  = (6.4, 0, 0)
objp[5]  = (0.8, 0.8, 0)
objp[6]  = (2.4, 0.8, 0)
objp[7]  = (4.0, 0.8, 0)
objp[8]  = (5.6, 0.8, 0)
objp[9]  = (7.2, 0.8, 0)
objp[10] = (0, 1.6, 0)
objp[11] = (1.6, 1.6, 0)
objp[12] = (3.2, 1.6, 0)
objp[13] = (4.8, 1.6, 0)
objp[14] = (6.4 , 1.6, 0)
objp[15] = (0.8 , 2.4, 0)
objp[16] = (2.4, 2.4, 0)
objp[17] = (4.0 , 2.4, 0)
objp[18] = (5.6, 2.4, 0)
objp[19] = (7.2, 2.4, 0)
objp[20] = (0, 3.2, 0)
objp[21] = (1.6, 3.2, 0)
objp[22] = (3.2, 3.2, 0)
objp[23] = (4.8 , 3.2, 0)
objp[24] = (6.4 , 3.2, 0)
objp[25] = (0.8 , 4.0, 0)
objp[26] = (2.4, 4.0, 0)
objp[27] = (4.0, 4.0, 0)
objp[28] = (5.6, 4.0, 0)
objp[29] = (7.2, 4.0, 0)
objp[30] = (0, 4.8, 0)
objp[31] = (1.6 , 4.8, 0)
objp[32] = (3.2 , 4.8, 0)
objp[33] = (4.8 , 4.8, 0)
objp[34] = (6.4 , 4.8, 0)
objp[35] = (0.8, 5.6, 0)
objp[36] = (2.4, 5.6, 0)
objp[37] = (4.0, 5.6, 0)
objp[38] = (5.6, 5.6, 0)
objp[39] = (7.2, 5.6, 0)
objp[40] = (0 , 6.4, 0)
objp[41] = (1.6, 6.4, 0)
objp[42] = (3.2 , 6.4, 0)
objp[43] = (4.8 , 6.4, 0)
objp[44] = (6.4, 6.4, 0)

def get_Rt(path, mtx, dist):
    image = cv2.imread(path)
    # 그레이 스케일로 변환
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 이미지에서 원하는 개수의 코너가 발견되면 ret = true
    ret, corners = cv2.findCirclesGrid(gray_img,
                                    CIRCLEBOARD,
                                    None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    retval, rvec, tvec = cv2.solvePnP(objectPoints=objp, imagePoints=corners, cameraMatrix=mtx, distCoeffs=dist)
    rotVec,_ = cv2.Rodrigues(rvec)
    Rt = np.concatenate((rotVec, tvec), axis=1)
    Rt = np.concatenate((Rt, np.array([[0,0,0,1]])), axis=0)

    return Rt

result = []
for i in range(0, 20):
    Rt_depth = get_Rt(f'./rgb_depth_images/IR/IR_img_{i}.png', K_depth, dist_depth)

    # A 
    Rt_rgb = get_Rt(f'./rgb_depth_images/rgb/rgb_img_{i}.png', K_rgb, dist_rgb)
    mtx_rgb = np.dot(Rt_rgb, np.linalg.inv(Rt_depth))
    result.append(mtx_rgb)


average_matrix = np.zeros((4,4))

# 각 요소의 평균 계산
for matrix in result:
    average_matrix += matrix

average_matrix /= len(result)
print(average_matrix)
