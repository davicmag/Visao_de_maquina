import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("Visao_de_maquina/aps1/Figuras_APS1/Fig_APS1_2a.bmp", cv2.IMREAD_COLOR)
if img is None:
    print("File 1 not found. Bye!")
    exit(0)
img2 = cv2.imread("Visao_de_maquina/aps1/Figuras_APS1/Fig_APS1_2b.bmp", cv2.IMREAD_COLOR)
if img2 is None:
    print("File 2 not found. Bye!")
    exit(0)

[B,G,R] = cv2.split(img)
[B2,G2,R2] = cv2.split(img2)

(h, w, c) = img.shape
print("Height = ", h)
print("Width = ", w)
img_out = np.zeros((h, w, c), dtype=np.uint8)

for i in range(h):  
    for j in range(w):
            if G2[i, j] > 110 and R2[i, j] < 110 and B2[i, j] < 110:
                img_out[i, j] = [B[i, j],G[i, j],R[i, j]]
            else:
                img_out[i, j] = [B2[i, j],G2[i, j],R2[i, j]]
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
plt.figure('Fig Out')
plt.imshow(img_out) 
plt.show()