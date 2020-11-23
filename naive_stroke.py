import cv2 as cv
import numpy as np
import random
from skimage.draw import line

# Traçage d'une stroke pour un point si l'intensité du gradient
# en ce point est supérieur à un seuil
def drawstroke(canvas, x, y, gradx, grady, intensity):

    if intensity > 0.1:

        # Calcul de l'isophote normé
        iso_mul = 5
        # iso_x = gradx / np.sqrt(gradx*gradx + grady*grady)
        # iso_y = grady / np.sqrt(gradx*gradx + grady*grady)

        iso_x = -gradx / np.sqrt(gradx*gradx + grady*grady)
        iso_y = -grady / np.sqrt(gradx*gradx + grady*grady)

        # Création des nouveaux extrémités de la stroke
        # xn = int(x + (iso_x * iso_mul))
        # yn = int(y + (iso_y * iso_mul))

        xm = int(x + (iso_x * iso_mul))
        ym = int(y + (iso_y * iso_mul))


        # Manière assez mystérieuse de tracer une ligne
        # ---------------------------------------------

        discrete_line = line(x,y,xm,ym)
        # Coordonnées des points qui composent la ligne
        coords = list(zip(discrete_line[1], discrete_line[0])) 

        for p in coords:
            if (p[0] < canvas.shape[0]) and (p[0] > 0) and (p[1] < canvas.shape[1]) and (p[1] > 0):
                if canvas[p] > 0:
                    canvas[p] = canvas[p] - 20

    return



# Ouverture image
img = cv.imread("images/Capture.PNG")
img = cv.cvtColor(img, cv.COLOR_BGR2YUV)

# On isole le channel Y
img_lumi = cv.split(img)[0]
img_lumi = cv.GaussianBlur(img_lumi,(5,5), 3)

# Calcul du gradient X et Y avec sobel
sobel_x = cv.Sobel(img_lumi,cv.CV_64F,1,0,ksize=3)
sobel_y = cv.Sobel(img_lumi,cv.CV_64F,0,1,ksize=3)

# Calcul et normalisation de la norme du gradient
intensity = np.sqrt(pow(sobel_x,2) + pow(sobel_y,2))
cv.normalize(intensity, intensity,0.,1.,cv.NORM_MINMAX)

# Création de la canvas
# ---------------------
canvas = np.full(img_lumi.shape, 255, np.uint8)
# Tirage des points
coords = [(random.randrange(img_lumi.shape[0]), random.randrange(img_lumi.shape[1])) for _ in range(10000)]
# Traçage d'une stroke pour chaque point
for p in coords:
    drawstroke(canvas, p[1], p[0], sobel_x[p[0]][p[1]], sobel_y[p[0]][p[1]], intensity[p[0]][p[1]])


# Affichage
cv.imshow("img_lumi",   img_lumi)
cv.imshow("sobel_x",    sobel_x)
cv.imshow("sobel_y",    sobel_y)
cv.imshow("intensity",  intensity)
cv.imshow("canvas",     canvas)

# Enregistrement
cv.imwrite("images/canvas.jpg",canvas)

cv.waitKey(0)