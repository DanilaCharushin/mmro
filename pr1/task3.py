# importing cv2
import cv2

image_path = "geeks.png"
# Using cv2.imread() method
# to read the image
img = cv2.imread(image_path)
# Filename
filename = "savedImage.jpg"
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, img)
# Reading and showing the saved image
img = cv2.imread(filename)
cv2.imshow("GeeksforGeeks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
