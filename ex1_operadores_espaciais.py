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


fig_binP = np.where(seixosP > 100, 255, 0)
fig_binM = np.where(seixosM > 100, 255, 0)
fig_binG = np.where(seixosG > 100, 255, 0)

hp,wp = fig_binP.shape
hm,wm = fig_binM.shape
hg,wg = fig_binG.shape

ranhurasP = 0
ranhurasM = 0
ranhurasG = 0

for i in range (hp):
    for j in range (wp):
        if fig_binP[i,j] == 0:
            ranhurasP += 1

for i in range (hm):
    for j in range (wm):
        if fig_binM[i,j] == 0:
            ranhurasM += 1

for i in range (hg):
    for j in range (wg):
        if fig_binG[i,j] == 0:
            ranhurasG += 1


print(ranhurasP,ranhurasM,ranhurasG)








