import cv2 as cv
import numpy as np
import random
from skimage.draw import line
from skimage import feature

# Ouverture image
img = cv.imread("images/test.PNG")
img = cv.cvtColor(img, cv.COLOR_BGR2YUV)

rows = img.shape[0]
cols = img.shape[1]


# On isole le channel Y
img_lumi = cv.split(img)[0]
img_lumi = cv.GaussianBlur(img_lumi,(5,5), 2)

Axx, Axy, Ayy  = feature.structure_tensor(img_lumi)
E1, E2 = feature.structure_tensor_eigvals( Axx, Axy, Ayy )
print(E1.shape)
T1 = np.stack([Axx,Axy],axis=-1)
T2 = np.stack([Axy,Ayy],axis=-1)
T = np.stack([T1,T2],axis=-1)

# Canvas
canvas = np.full((rows,cols), 255, np.uint8)

grid_i = np.linspace(10,rows,8,dtype=int)
grid_j = np.linspace(10,cols,12,dtype=int)

for i in grid_i:
    for j in grid_j:
        cv.ellipse(canvas, (i,j), (2,2), 0, 0, 180, 128,-1,8,0)

# # Angle set
# angles = np.linspace(0,180,7)

# # For each angles
# for angle in angles: 

#     # Compute vector field
#     vector_field = np.ndarray((rows,cols,2))

#     a = [np.cos(angle), np.sin(angle)]
#     for i in range(rows):
#         for j in range(cols):
#             vector_field[i,j] = np.dot(np.sqrt(T[i,j]), a)

#     coords = [(random.randrange(rows), random.randrange(cols)) for _ in range(1000)]
    
#     for p in coords:
#         x = p[1]
#         y = p[0]
#         xn = int(x + vector_field[i,j][0] * 10)
#         yn = int(y + vector_field[i,j][1] * 10)

#         cv.line(canvas,(x,y),(xn,yn),0)

# Show canvas
cv.imshow("canvas", canvas)

# Enregistrement
cv.imwrite("images/canvas.jpg",canvas)

cv.waitKey(0)