import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def calcula_homografia(src, dst):
    A = []
    b = []

    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b.append(u)
        b.append(v)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    h = np.linalg.solve(A, b)
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)
    return H

img = np.array(Image.open(r"Visao_de_maquina\Figuras_APS2\Fig4_Campo_Persp1a.bmp").convert("L"))
h, w = img.shape

img_ref = np.array(Image.open(r"Visao_de_maquina\Figuras_APS2\fig4_campo_exemplo.png").convert("L"))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Imagem distorcida\nClique nos 4 cantos aqui\n na ordem: superior esquerdo, superior direito, inferior esquerdo, inferior direito")
ax[0].axis("on")

ax[1].imshow(img_ref, cmap='gray')
ax[1].set_title("Imagem de comparação")
ax[1].axis("on")

plt.sca(ax[0]) 
pts = plt.ginput(4, timeout=0)
plt.close()

src_pts = np.array(pts, dtype=np.float64)

# Tamanho da imagem retificada
w_out = 400
h_out = 500

dst_pts = np.array([
    [0, 0],
    [w_out - 1, 0],
    [0, h_out - 1],
    [w_out - 1, h_out - 1]
], dtype=np.float64)

H = calcula_homografia(src_pts, dst_pts)
Hinv = np.linalg.inv(H)

print("H =\n", H)

# Transformação inversa
img_out = np.ones((h_out, w_out), dtype=np.uint8) * 255

for v in range(h_out):
    for u in range(w_out):
        P1 = np.array([u, v, 1], dtype=np.float64)
        P0 = Hinv @ P1

        x = P0[0] / P0[2]
        y = P0[1] / P0[2]

        xi = int(round(x))
        yi = int(round(y))

        if 0 <= xi < w and 0 <= yi < h:
            img_out[v, u] = img[yi, xi]


y1, y2 = 10, 490
x1, x2 = 10, 390

img_crop = img_out[y1:y2, x1:x2]
hc, wc = img_crop.shape

col_255 = []
col_0 = []

for j in range(100,wc):
    tem_255 = False
    tem_0 = False

    for i in range(hc):
        if img_crop[i, j] == 255:
            col_255.append(j)
        if img_crop[i, j] == 0:
            col_0.append(j)

print("Colunas com 255:", col_255)
print("Colunas com 0:", col_0)

if len(col_255) > 0:
    x_linha = min(col_255)
    if x_linha >= wc:
        x_linha = 0
else:
    print("Nao encontrou branco.")
    x_linha = wc // 2


cruza = False
for i in range(hc):
    if img_crop[i, x_linha] == 0:
        cruza = True
        break

if cruza:
    print("Está em impedimento.")
else:
    print("Não está em impedimento.")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Imagem distorcida")

ax[1].imshow(img_out, cmap='gray')
ax[1].set_title("Imagem retificada")

ax[2].imshow(img_crop, cmap='gray')
ax[2].axvline(x=x_linha, color='red', linewidth=2)
ax[2].set_title("Recorte + linha vermelha")

for a in ax:
    a.axis("on")

plt.show()