import cv2
import matplotlib as mpl

mpl.use("MacOSX")  # good fix, works only on MacOSX

img1 = cv2.imread('L.jpg', 0)
img2 = cv2.imread('R.jpg', 0)

# Initiate ORB detector
orb = cv2.ORB_create()

sift = cv2.SIFT_create()

# detect and compute the keypoints and descriptors with SIFT

# Нахождение особых точек и описаний признаков на изображении
kp1, des1 = orb.compute(img1, orb.detect(img1, None))
kp2, des2 = orb.compute(img2, orb.detect(img2, None))

# create BFMatcher object
bf = cv2.BFMatcher()

matches = bf.match(des1, des2)

# sort the matches based on distance
matches = sorted(matches, key=lambda val: val.distance)

# Draw first 50 matches.
out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
# out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

cv2.imshow("out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
