import cv2
import matplotlib.pyplot as plt
import numpy as np
from Find_N_MinMaxValues import Find_N_MaxValues

img = cv2.imread("Visao_de_maquina/aps2/Figuras_APS2/Fig2_Ferramentas_u8.bmp", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w = img.shape

symbol = cv2.imread("Visao_de_maquina/aps2/Figuras_APS2/Fig2_Padrao_u8.bmp", cv2.IMREAD_GRAYSCALE)
if symbol is None:
    print("Symbol not found. Bye!")
    exit(0)
(m,n) = symbol.shape

d1 = int((m)//2)
d2 = int((n)//2)

ncc = np.zeros((h, w), dtype=np.float32)

template = symbol.astype(np.float32) / 255.0
den2 = (template**2).sum()


for i in range(d1, h - d1):
    for j in range(d2, w - d2):

        secao_img = img[i-d1:i-d1+m, j-d2:j-d2+n]
        secao = secao_img.astype(np.float32) / 255.0

        numerador = (secao * template).sum()
        den1 = (secao**2).sum()
        
        denominador = np.sqrt(den1 * den2)
        ncc_val = numerador / denominador 

        ncc[i, j] = (ncc_val + 1) / 2

sub_ncc = ncc[d1:h-d1, d2:w-d2]

n = 3

max_positions = Find_N_MaxValues(sub_ncc, n)
coords = []

for pos in max_positions:
    cordx = pos[0] + d1
    cordy = pos[1] + d2
    coords.append((cordx, cordy))

for c in coords:
    print(c)

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_final = cv2.rectangle(img_rgb, (cordy-d2, cordx-d1), (cordy+d2, cordx+d1), (255, 0, 0), 2)

for (cordx, cordy) in coords:
    img_final = cv2.rectangle(img_final, (cordy-d2, cordx-d1), (cordy+d2, cordx+d1), (255, 0, 0), 2)
    
f = plt.figure(figsize=(10, 5))
plt.imshow(img_final, cmap='gray') 
plt.show()