import math
import os
import urllib.request as urlreq

import cv2
import numpy as np

DIRNAME = "data"

haarcascade_filename = "haarcascade_frontalface_alt2.xml"
haarcascade_path = f"{DIRNAME}/{haarcascade_filename}"
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

lbf_model_filename = "LFBmodel.yaml"
lbf_model_path = f"{DIRNAME}/{lbf_model_filename}"
lbf_model_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

if not os.path.isdir(DIRNAME):
    os.mkdir(DIRNAME)

files_to_download = [
    (haarcascade_filename, haarcascade_path, haarcascade_url),
    (lbf_model_filename, lbf_model_path, lbf_model_url),
]

for (filename, path, url) in files_to_download:
    if filename in os.listdir(DIRNAME):
        print(f"Файл {filename} уже существует")
    else:
        print(f"Файл {filename} отсутствует! Скачиваю...")
        urlreq.urlretrieve(url, path)
        print(f"Файл {filename} загружен!")

detector = cv2.CascadeClassifier(haarcascade_path)
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(lbf_model_path)

print("Подключение к веб-камере...")
webcam_cap = cv2.VideoCapture(0)

while True:
    _, frame = webcam_cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray)
    for (x, y, w, d) in faces:
        _, landmarks = landmark_detector.fit(gray, np.array(faces))

        for i in range(len(landmarks)):
            marks = landmarks[i][0]
            for j, (p_x, p_y) in enumerate(marks):
                cv2.putText(frame, str(j), (int(p_x), int(p_y)), 0, 0.4, (255, 0, 0), 2)

            cv2.line(
                frame, (int(marks[27][0]), int(marks[27][1])), (int(marks[8][0]), int(marks[8][1])), (0, 0, 255), 2
            )

            XL = (marks[45][0] + marks[42][0]) / 2
            YL = (marks[45][1] + marks[42][1]) / 2
            XR = (marks[39][0] + marks[36][0]) / 2
            YR = (marks[39][1] + marks[36][1]) / 2

            cv2.line(frame, (int(XL), int(YL)), (int(XR), int(YR)), (0, 0, 255), 2)

            DX = XR - XL
            DY = YR - YL
            L = math.sqrt(DX * DX + DY * DY)
            X1 = marks[27][0]
            Y1 = marks[27][1]
            X2 = marks[8][0]
            Y2 = marks[8][1]
            DX1 = abs(X1 - X2)
            DY1 = abs(Y1 - Y2)
            L1 = math.sqrt(DX1 * DX1 + DY1 * DY1)
            X0 = (XL + XR) / 2
            Y0 = (YL + YR) / 2
            sin_AL = DY / L
            cos_AL = DX / L
            X_User_0 = (marks[27][0] - X0) / L
            Y_User_0 = -(marks[27][1] - Y0) / L
            X_User27 = X_User_0 * cos_AL - Y_User_0 * sin_AL
            Y_User27 = X_User_0 * sin_AL + Y_User_0 * cos_AL
            X_User_0 = (marks[30][0] - X0) / L
            Y_User_0 = -(marks[30][1] - Y0) / L
            X_User30 = X_User_0 * cos_AL - Y_User_0 * sin_AL
            Y_User30 = X_User_0 * sin_AL + Y_User_0 * cos_AL

            if abs(X_User27 - X_User30) <= 0.1:
                value = abs(L1 / L)
                print(f"{value=}")

                cv2.putText(
                    frame, str(round(value, 2)), (int(marks[27][0]), int(marks[27][1]) - 100), 1, 2, (0, 255, 0), 2
                )

                if 1.7 <= value <= 1.9:
                    cv2.putText(
                        frame, "Danil", (int(marks[27][0]), int(marks[27][1])), 1, 2,
                        (0, 0, 255), 2
                    )
                elif 1.9 <= value <= 2.1:
                    cv2.putText(
                        frame, "Vitaly", (int(marks[27][0]), int(marks[27][1])), 1, 2,
                        (0, 255, 255), 2
                    )
                else:
                    cv2.putText(
                        frame, "Unknown", (int(marks[27][0]), int(marks[27][1])),
                        1, 2, (255, 0, 255), 2
                    )

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

webcam_cap.release()
cv2.destroyAllWindows()
