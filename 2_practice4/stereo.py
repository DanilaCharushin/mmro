import cv2
import numpy as np

imgL = cv2.imread("L.jpg", 0)
imgR = cv2.imread("R.jpg", 0)

minDisparity = 1
numDisparities = 72
blockSize = 1
disp12MaxDiff = 1
uniquenessRatio = 1
speckleWindowSize = 1
speckleRange = 1

# Создание объекта алгоритма StereoSGBM
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    disp12MaxDiff=disp12MaxDiff,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
)
# Вычисл    ение диспарита с использованием алгоритма StereoSGBM
disparity = stereo.compute(imgL, imgR).astype(np.float32)
cv2.imshow("disparity", disparity)

disp_1 = (disparity / 16.0 - minDisparity) / numDisparities

# Отображение карты диспаратности
cv2.imshow("disp_1", disp_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
