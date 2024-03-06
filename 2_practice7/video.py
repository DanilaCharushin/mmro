import cv2

# загрузка видеофайла и создание объекта видеопотока
video = cv2.VideoCapture('video.mp4')
# создание объекта фоновой вычитания
fgbg = cv2.createBackgroundSubtractorMOG2()

paused = False

while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        paused = not paused

    if not paused:
        # чтение кадра видеопотока
        ret, frame = video.read()
        # применение алгоритма вычитания фона
        fgmask = fgbg.apply(frame)
        # поиск контуров на двоичной карте переднего плана
        contours, hierarchy = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        max_contour = max(contours, key=cv2.contourArea)
        # получение координат и размеров прямоугольника, описывающего контур
        x, y, w, h = cv2.boundingRect(max_contour)
        # if w < 50:
        #     continue
        # отображение кадра со слежением за объектом
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # for contour in contours:
        #     извлечение контура с максимальной площадью

        cv2.imshow('frame', frame)
        # если нажата клавиша 'q', закрыть окно
        # освобождение ресурсов

video.release()
cv2.destroyAllWindows()
