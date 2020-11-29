import cv2 as cv
import numpy as np

# u = np.ones((3,3,2))
# v = np.ones((3,3,2)) 
# w = np.array([[1,2,3],[4,5,6],[7,8,9]])

# # print(u)
# up = np.einsum('...j,...k->...jk',u,u) 
# res = np.einsum('ij,ijkl->ijkl',w,up)
# print(res)

# ol = np.ones((3,3,2,2))
# ok = np.ones((2))

# res = np.einsum('...ij,j->...i',ol,ok) 

# aa = [1,2,3,4,5,6,7,8,9]
# print(aa[2:-2])

canvas = np.full((200,200), 255, np.uint8)

p1 = [100,100]
p2 = [50,100]

print(np.add(p1,2))

cv.line(canvas, (p1[1],p1[0]), (p2[1],p2[0]), 128,1)
cv.circle(canvas, (p1[1],p1[0]), 50, 128,1)
p3 = [-1,0]

while True:
    o1 = np.subtract(p2,p1)
    o2 = p3

    dot = np.dot(o1,o2) / (np.linalg.norm(o1) * np.linalg.norm(o2))
    det = np.linalg.det([o1,o2]) / (np.linalg.norm(o1) * np.linalg.norm(o2))
    angle = np.arctan2(det,dot)
    #angle = np.radians(270)

    print(angle)

    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    print(R)

    res = R.dot(o1)
    print(( int(p1[0] + res[0]) , int(p1[1] + res[1]) ))

    cv.line(canvas, (p1[1],p1[0]), ( int(p1[1] + res[1]) , int(p1[0] + res[0]) ), 128,1)

    cv.imshow("canvas", canvas)
    cv.waitKey(1)
    p3[0] = float(input())
    p3[1] = float(input())
