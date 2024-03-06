import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# mpl.use("TkAgg")  # simple fix, but not always works
mpl.use("MacOSX")  # good fix, works only on MacOSX

filename = "dog.png"


def rotate90(img):
    rotated90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("rotated90_" + filename, rotated90)


def rotateN(img, n):
    (rows, cols) = img.shape[:2]

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), n, 1)
    res = cv2.warpAffine(img, M, (cols, rows))

    cv2.imwrite(f"rotated{n}_" + filename, res)


def resize(img):
    half = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    bigger = cv2.resize(img, (1050, 1610))

    stretch_near = cv2.resize(
        img, (780, 540),
        interpolation=cv2.INTER_NEAREST
        )

    Titles = ["Original", "Half", "Bigger", "Interpolation Nearest"]
    images = [img, half, bigger, stretch_near]
    count = 4

    for i in range(count):
        plt.subplot(2, 3, i + 1)
        plt.title(Titles[i])
        plt.imshow(images[i])

    plt.savefig("resized_" + filename)


def removeColors(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("uncolored_" + filename, res)


def move(img):
    height, width = img.shape[:2]

    quarter_height, quarter_width = height / 4, width / 4

    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

    res = cv2.warpAffine(img, T, (width, height))
    cv2.imwrite("moved_" + filename, res)


def findEdges(img):
    res = cv2.Canny(img, 100, 200)
    cv2.imwrite("edged_" + filename, res)


def thresh(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)

    cv2.imwrite("Binary_Threshold_" + filename, thresh1)
    cv2.imwrite("Binary_Threshold_Inverted_" + filename, thresh2)
    cv2.imwrite("Truncated_Threshold_" + filename, thresh3)
    cv2.imwrite("Threshold_Set_to_0_" + filename, thresh4)
    cv2.imwrite("Threshold_Set_to_0_Inverted_" + filename, thresh5)


def adaptiveThresh(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

    cv2.imwrite("Adaptive_Mean_" + filename, thresh1)
    cv2.imwrite("Adaptive_Gaussian_" + filename, thresh2)


def otsu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite("Otsu_Threshold_" + filename, thresh1)


def blur(img):
    Gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imwrite("Gaussian_Blurring_" + filename, Gaussian)

    median = cv2.medianBlur(img, 5)
    cv2.imwrite("Median_Blurring_" + filename, median)

    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite("Bilateral_Blurring_" + filename, bilateral)


def filter(img):
    bilateral = cv2.bilateralFilter(img, 15, 100, 100)
    cv2.imwrite("Bilateral_" + filename, bilateral)


def contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 200)
    cv2.imwrite("Canny_Edges_After_Contouring_" + filename, edged)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    print("Number of Contours found = " + str(len(contours)))

    cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    cv2.imwrite("Contours_" + filename, img)


def erosion(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    cv2.imwrite("Erosion_" + filename, img_erosion)
    cv2.imwrite("Dilation_" + filename, img_dilation)


def match(query, train):
    query_img_bw = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors, trainDescriptors)

    final_img = cv2.drawMatches(query, queryKeypoints, train, trainKeypoints, matches[:20], None)

    final_img = cv2.resize(final_img, (1000, 650))

    cv2.imwrite("Matches_" + filename, final_img)


def draw():
    img = np.zeros((400, 400, 3), dtype="uint8")

    cv2.rectangle(img, (30, 30), (300, 200), (0, 255, 0), 5)

    cv2.imwrite("rectangle_on_dark.jpeg", img)


def sum(img1, img2):
    res = cv2.addWeighted(img1, 0.5, img2, 0.4, 0)
    cv2.imwrite("summed.jpeg", res)


def subtract(img1, img2):
    res = cv2.subtract(img1, img2)
    cv2.imwrite("subtracted.jpeg", res)


def bitwiseAnd(img1, img2):
    res = cv2.bitwise_and(img1, img2)
    cv2.imwrite("bitwise_and.jpeg", res)


def bitwiseOr(img1, img2):
    res = cv2.bitwise_or(img1, img2)
    cv2.imwrite("bitwise_or.jpeg", res)


def bitwiseXor(img1, img2):
    res = cv2.bitwise_xor(img1, img2)
    cv2.imwrite("bitwise_or.jpeg", res)


def bitwiseNot(img1, img2):
    res1 = cv2.bitwise_not(img1)
    cv2.imwrite("bitwise_not_duck.jpeg", res1)

    res2 = cv2.bitwise_not(img2)
    cv2.imwrite("bitwise_not_worm.jpeg", res2)


img = cv2.imread(filename, cv2.IMREAD_COLOR)
rotate90(img)
rotateN(img, 45)
resize(img)
removeColors(img)
move(img)
findEdges(img)
thresh(img)
adaptiveThresh(img)
otsu(img)
blur(img)
filter(img)
contours(img)
erosion(img)
match(img, img.copy())
draw()

duck = cv2.imread("duck.png", cv2.IMREAD_COLOR)
worm = cv2.imread("worm.png", cv2.IMREAD_COLOR)

duck = cv2.resize(duck, (233, 127))
worm = cv2.resize(worm, (233, 127))

sum(duck, worm)
subtract(duck, worm)
bitwiseAnd(duck, worm)
bitwiseOr(duck, worm)
bitwiseXor(duck, worm)
bitwiseNot(duck, worm)

plt.show()
