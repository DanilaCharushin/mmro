import cv2

filename = "people.png"
# Загрузка изображения в оттенках серого
img = cv2.imread(filename, 0)
cv2.imshow(filename, img)

params = [
    {
        "threshold1": 100,
        "threshold2": 100,
        "apertureSize": 3,
    },
    {
        "threshold1": 200,
        "threshold2": 200,
        "apertureSize": 3,
    }, {
        "threshold1": 500,
        "threshold2": 500,
        "apertureSize": 3,
    },
    {
        "threshold1": 100,
        "threshold2": 100,
        "apertureSize": 5,
    },
    {
        "threshold1": 100,
        "threshold2": 100,
        "apertureSize": 7,
    },
]

for index, param in enumerate(params, start=1):
    # Применение алгоритма Canny для обнаружения границ
    edges = cv2.Canny(img, **param)
    # Вывод изображения с границами
    cv2.imshow(f"edges_{index}_{filename}", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
