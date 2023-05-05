import cv2
import numpy as np
import glob

# Circleboard의 차원 정의
CIRCLEBOARD = (5,9) # CircleBoard 행 circle과 열 cirle 갯수

# 알고리즘 종료 기준 설정: 알고리즘의 반복 횟수가 30이 되거나, 정확도가 0.001 이하일 때 종료
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 각 CicleBoard 이미지에 대한 3D 점 벡터를 저장할 리스트 생성
objpoints = []
# 각 CircleBoard 이미지에 대한 2D 점 벡터를 저장할 리스트 생성
imgpoints = [] 

# 3D 점의 세계 좌표 정의
# CircleBoard의 각 원의 중심 간의 거리 1.6cm
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

# 주어진 디렉터리에 저장된 개별 이미지의 경로 추출
images = glob.glob('./images/*.png')
for fname in images:
    img = cv2.imread(fname)
    # 그레이 스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Circle grid 코너 찾기
		# corners : 감지된 코너의 출력 배열
		# ret : 패턴이 감지되었는지 여부에 따라 출력은 true/false로 반환
		# 원하는 수의 패턴이 감지 되었으면 ret = True
    ret, corners = cv2.findCirclesGrid(gray,
                                    CIRCLEBOARD,
                                    None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # 코너 그리기 및 표시
        img = cv2.drawChessboardCorners(img, CIRCLEBOARD, corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
h, w = img.shape[:2] # 480, 640

# 알려진 3D 점(objpoints) 값과 감지된 코너의 해당 픽셀 좌표(imgpoints) 전달, 카메라 캘리브레이션 수행
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print("RMS: ", rms) # RMS

print("Camera matrix : ") # 내부 카메라 행렬
print(mtx)

print("\n dist : ") # 렌즈 왜곡 계수(Lens distortion coefficients)
print(dist)

print("\n rvecs : ") # 회전 벡터
print(rvecs)

print("\n tvecs : ") # 이동 벡터
print(tvecs)