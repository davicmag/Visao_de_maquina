import cv2
import matplotlib.pyplot as plt
import numpy as np


seixosP = cv2.imread("Figuras_APS2/Fig1_SeixoP.bmp", cv2.IMREAD_GRAYSCALE)
if seixosP is None:
    print("File 1 not found. Bye!")
    exit(0)

seixosM = cv2.imread("Figuras_APS2/Fig1_SeixoM.bmp", cv2.IMREAD_GRAYSCALE)
if seixosM is None:
    print("File not 2 found. Bye!")
    exit(0)


seixosG = cv2.imread("Figuras_APS2/Fig1_SeixoG.bmp", cv2.IMREAD_GRAYSCALE)
if seixosG is None:
    print("File not 3 found. Bye!")
    exit(0)




(h, w) = seixosP.shape



img_bin = np.zeros((h,w), dtype = "uint8")


for i in range(h):
    for j in range(w):
        if seixosP[i,j] > 70:
            img_bin[i,j] = 255
  



# Mostrando a nova imagem gerada - binarizada (preto e branca)
plt.figure('Fig Bin')
plt.imshow(img_bin, cmap = 'gray')
plt.show()  



