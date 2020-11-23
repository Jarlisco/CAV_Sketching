import cv2 as cv

img = cv.imread("images/frog.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print(img.shape)