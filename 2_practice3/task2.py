import cv2
import numpy as np

# Загрузите изображение
img = cv2.imread('dog.png')
# Преобразуйте изображение в массив NumPy и измените его форму
pixel_vals = img.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)
# Установите параметры для кластеризации k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 3
# Выполните кластеризацию k-means
_, labels, centers = cv2.kmeans(
    pixel_vals, k, None, criteria, 10,
    cv2.KMEANS_RANDOM_CENTERS
    )
# Преобразуйте метки в форму, которую можно использовать для создания маски сегментации
labels = labels.reshape((img.shape[0], img.shape[1]))

centers = np.uint8(centers)
newimage = centers[labels.flatten()]
newimage = newimage.reshape(img.shape)

cv2.imshow('newimage', newimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

k = 0
for center in centers:
    # select color and create mask
    #print(center)
    layer = newimage.copy()
    mask = cv2.inRange(layer, center, center)

    # apply mask to layer
    layer[mask == 0] = [0,0,0]
    #cv2.imshow('layer', layer)
    #cv2.waitKey(0)

    # save kmeans clustered image and layer
    cv2.imwrite("1_test{0}.jpg".format(k), layer)
    k = k + 1
