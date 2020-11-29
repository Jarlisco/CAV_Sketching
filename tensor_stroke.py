import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from skimage.draw import line
from skimage import feature



def structure_tensor_eigvectors(Axx, Axy, Ayy):
    
    det = (Axx - Ayy) ** 2 + 4 * Axy ** 2
    
    first = Axy * 2
    second_pos = Ayy - Axx + np.sqrt(det)
    second_neg = Ayy - Axx - np.sqrt(det)

    O1 =  np.stack([first,second_pos],axis=-1)
    O2 =  np.stack([first,second_neg],axis=-1)

    return O1, O2

def structure_tensor_eigvals(Axx, Axy, Ayy):
    
    det = (Axx - Ayy) ** 2 + 4 * Axy ** 2

    E1 = (Axx + Ayy) / 2 + np.sqrt(det) / 2
    E2 = (Axx + Ayy) / 2 - np.sqrt(det) / 2

    return E1, E2

def structure_tensor(Axx, Axy, Ayy):
    T1 = np.stack([Axx,Axy],axis=-1)
    T2 = np.stack([Axy,Ayy],axis=-1)
    T = np.stack([T1,T2],axis=-1)
    return T

def tensor_field(E1, E2, O1, O2, p1, p2):
    
    C1 = np.add(E1,E2) + 1. ** (-p1)
    C2 = np.add(E1,E2) + 1. ** (-p2)

    T1 = np.einsum('...j,...k->...jk',O1,O1)
    T2 = np.einsum('...j,...k->...jk',O2,O2)

    T = np.add(np.einsum('ij,ijkl->ijkl', C1, T1), np.einsum('ij,ijkl->ijkl', C2, T2))
    
    return T

def tensor_field_sqrt(T):
    T_sqrt = np.zeros(T.shape)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            T_sqrt[i,j] = sqrtm(T[i,j])

    return T_sqrt

def display_vector_field(vector_field, title_plot):

    height, width, _ = vector_field.shape

    X, Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)

    for j in range(X.shape[0]):
        for i in range(X.shape[1]):
            U[j, i] = vector_field[j][i][0] * 100
            V[j, i] = vector_field[j][i][1] * 100

    _, ax = plt.subplots()
    skip_scalar = 5
    skip = (slice(None, None, skip_scalar), slice(None, None, skip_scalar))
    ax.quiver(X[skip], Y[skip], U[skip], V[skip], units='xy' ,scale=0.5, color='blue')
    ax.set(aspect=1, title=title_plot)

    plt.gca().invert_yaxis()
    #plt.savefig(filename, bbox_inches='tight', dpi=200)
    # plt.ion()
    plt.show()
    plt.pause(0.005)

def draw_stroke(canvas, origin, direction, size):

    destination = [origin[0] + size, origin[1] + size]
    vec         = np.subtract(origin,destination)
    
    dot = np.dot( vec, direction ) / ( np.linalg.norm(vec) * np.linalg.norm(origin) )
    det = np.linalg.det([vec, direction]) / ( np.linalg.norm(vec) * np.linalg.norm(origin) )
    angle = np.arctan2(det, dot) + np.pi / 2.

    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    pix_to_move = R.dot(vec).astype(int)
    destination_point = (origin[0] + pix_to_move[0] , origin[1] + pix_to_move[1])

    cv.line(canvas, ( origin[1],origin[0] ), ( destination_point[1],destination_point[0] ) , 128,1)
    return destination_point




# Ouverture image
# img = cv.imread("images/frog.jpg")
img = cv.imread("images/test.PNG")
# img = cv.imread("images/lenna.png")

img = cv.cvtColor(img, cv.COLOR_BGR2YUV)

rows = img.shape[0]
cols = img.shape[1]


# On isole le channel Y
img_lumi = cv.split(img)[0]
img_lumi = cv.GaussianBlur(img_lumi,(3,3), 0)

Axx, Axy, Ayy = feature.structure_tensor(img_lumi)
Axx = cv.GaussianBlur(Axx,(5,5), 0)
Axy = cv.GaussianBlur(Axy,(5,5), 0)
Ayy = cv.GaussianBlur(Ayy,(5,5), 0)

E1, E2        = structure_tensor_eigvals( Axx, Axy, Ayy )
O1, O2        = structure_tensor_eigvectors( Axx, Axy, Ayy )
G             = structure_tensor( Axx, Axy, Ayy )
T             = tensor_field( E1, E2, O1, O2, 1.2, 0.5 )
T_sqrt        = tensor_field_sqrt(T)

print("Tensor field done !")

# print(T)
# print(T.shape)
# print(img_lumi.shape)

# Canvas
canvas = np.full((rows,cols), 255, np.uint8)

# I try to do cool looking ellipses
E1norm = E1 / np.max(E1)
E2norm = E2 / np.max(E2)


eig_lambda1 = E1norm
eig_lambda2 = E2norm

eig_eta = O1

shift = 12




for angle in [3.14/2]:

    a = np.array([np.cos(angle), np.sin(angle)])
    w = np.einsum('...ij,j->...i',T_sqrt,a)
    # w = w / np.max(w)

    display_vector_field(w[10:-10,10:-10],str(angle))


    coords = [(random.randrange(10,rows-10), random.randrange(10,cols-10)) for _ in range(2000)]

    for p in coords:
        
        current = p

        for _ in range(10):
            
            next_point = draw_stroke(canvas, current, w[current], 3)

            if next_point[0] >= rows or next_point[1] >= cols or next_point[0] < 0 or next_point[1] < 0 :
                break
            
            current = next_point


# for i in range(10,rows,shift):
#     for j in range(10,cols,shift):

#         axesLength = (int(np.log(1 + 100000*eig_lambda1[i,j])),3) #log(10000)=4
#         angleG = int(np.arctan2(eig_eta[i,j,1], eig_eta[i,j,0])  * 180. / np.pi)
#         angleI = int(np.arctan2(eig_eta[i,j,0], -eig_eta[i,j,1]) * 180. / np.pi)
#         cv.ellipse(canvas, (j,i), axesLength, angleI, 0, 360, 0,1)


# grid_i = np.linspace(10,rows-1,20,dtype=int)
# grid_j = np.linspace(10,cols-1,20,dtype=int)

# scale = 8

# E1sc = (E1 * scale).astype(int)
# E2sc = (E2 * scale).astype(int)
# O1sc = (O1 * scale).astype(int)
# O2sc = (O2 * scale).astype(int)
# print(E1sc)
# print(E2sc)
# print(E1sc[10,10],E2sc[10,10])
# for i in grid_i:
#     for j in grid_j:
#         print((E1sc[i,j],E2sc[i,j]))
#         cv.ellipse(canvas, (j,i), (E1sc[i,j] + 3,E2sc[i,j] + 3), 0, 0, 360, 128,-1,8,0)

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

# Show
# cv.imshow("e1", E1)
# cv.imshow("e2", E2)
# cv.imshow("e1norm", E1norm)
# cv.imshow("e2norm", E2norm)
cv.imshow("canvas", canvas)

# Enregistrement
cv.imwrite("images/canvas.jpg",canvas)

cv.waitKey(0)