import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("aps1\Figuras_APS1\Fig_APS1_3a.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("File 1 not found. Bye!")
    exit(0)
img2 = cv2.imread("aps1\Figuras_APS1\Fig_APS1_3b.bmp", cv2.IMREAD_GRAYSCALE)
if img2 is None:
    print("File 2 not found. Bye!")
    exit(0)

(h, w) = img.shape
print("Height = ", h)
print("Width = ", w)
img_out = np.zeros((h, w), dtype=np.uint8)
soma_i = 0
n = h*w
for i in range(h):  
    for j in range(w):
            soma_i += int(img2[i, j])
for i in range(h):  
    for j in range(w):
         K = (soma_i/n)/img2[i, j]
         img_out[i, j] = img[i, j]* K
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
plt.figure('Fig Out')
plt.imshow(img_out, cmap='gray') 
plt.show()