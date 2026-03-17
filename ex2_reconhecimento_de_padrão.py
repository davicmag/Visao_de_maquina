import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("Visao_de_maquina\Figuras_APS2\Fig2_Ferramentas_u8.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w = img.shape

panda = cv2.imread("Visao_de_maquina\Figuras_APS2\Fig2_Padrao_u8.bmp", cv2.IMREAD_GRAYSCALE)
if panda is None:
    print("Panda not found. Bye!")
    exit(0)
(m,n) = panda.shape

d1 = int((m)//2)
d2 = int((n)//2)

sad = np.zeros((h, w), dtype=np.int32)

for i in range(d1, h-d1):
    for j in range(d2, w-d2):
        secao_img = img[i-d1:i-d1+m, j-d2:j-d2+n]
        erro = abs(secao_img.astype(np.int32) - panda.astype(np.int32)).sum()
        sad[i, j] = erro


sub_sad = sad[d1:h-d1, d2:w-d2]
min_pos = np.unravel_index(np.argmin(sub_sad), sub_sad.shape)

cordx = min_pos[0] + d1
cordy = min_pos[1] + d2
smin = sad[cordx, cordy]

print(cordx, cordy)
print("SAD min = ", smin)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_final = cv2.rectangle(img_rgb, (cordy-d2, cordx-d1), (cordy+d2, cordx+d1), (255, 0, 0), 2)
f = plt.figure(figsize=(10, 5))
plt.imshow(img_final, cmap='gray') 
plt.show()