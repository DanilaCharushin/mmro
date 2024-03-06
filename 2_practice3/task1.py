import cv2
import numpy as np

# Загрузите изображение
img = cv2.imread('dog.png')
# Преобразуйте изображение в оттенки серого
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Примените пороговое значение для сегментации (можно настроить параметры)

_, thresh = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)

cv2.imshow('tresh', thresh)

# Примените морфологические операции для улучшения сегментации
kernel = np.ones((5, 5), np.uint8)
morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('morph_img', morph_img)

# Выполните сегментацию путем нахождения контуров объектов на изображении
contours, _ = cv2.findContours(
    morph_img, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)
# Нарисуйте контуры на исходном изображении
segmented_img = img.copy()
cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 2)
# Отобразите результат
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
