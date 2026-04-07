import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("aps1\Figuras_APS1\Fig_APS1_1a.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("File 1 not found. Bye!")
    exit(0)

logo = cv2.imread("aps1\Figuras_APS1\Fig_APS1_1b.bmp", cv2.IMREAD_GRAYSCALE)
if logo is None:
    print("File 2 not found. Bye!")
    exit(0)

(h, w) = img.shape

img_bin = np.zeros((h, w), dtype=np.uint8)
conta_area = 0

for i in range(h):
    for j in range(w):
        if img[i, j] > 99:
            img_bin[i, j] = 255
        if logo[i, j] < 99:
           conta_area += 1

conta_veio = 0

fig_out = np.zeros((h, w), dtype=np.uint8)
for i in range(h):  
    for j in range(w):
        intens32 = int(img_bin[i, j]) + int(logo[i, j])
        fig_out[i, j] = np.clip(intens32, 0, 255)
        if fig_out[i, j] < 100:
            conta_veio += 1
porcentagem_veios = (conta_veio/conta_area)*100
print("Porcentagem do logo que veio = ", porcentagem_veios, "%")

plt.figure('Fig out')
plt.imshow(fig_out, cmap='gray')
plt.show()
