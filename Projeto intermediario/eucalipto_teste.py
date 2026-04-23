import cv2
import matplotlib.pyplot as plt
import numpy as np
from library.selectBlob import *
from skimage.morphology import skeletonize


img = cv2.imread("Projeto intermediario/Dataset_Projeto1/_Eucalipto_Escolhidos1/Eucalipto5.jpg")
if img is None:
    print("File not found. Bye!")
    exit(0)
h, w, l = img.shape

# Converte imagem para RGB e Mostra
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure('Fig Color')
plt.imshow(img_rgb)
plt.axis('off')

# Separa em 3 canais
(canal_r, canal_g, canal_b) = cv2.split(img_rgb)

# Gera mascara inicial (remove o fundo)
mask_fundo = np.where((canal_b > 140), 0, 255).astype(np.uint8)
#plt.figure('Sem Fundo')
#plt.imshow(mask_fundo, cmap='gray')
#plt.axis('off')


# Separando vaso da planta
# kernel um pouco maior que a espessura do caule
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))

# erosão para quebrar a ligação planta-vaso
mask_erodida = cv2.erode(mask_fundo, kernel, iterations=2)

# detectar componentes conectados na imagem erodida
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_erodida, connectivity=8)

# escolhe o componente mais baixo e com área suficiente
melhor_id = -1
melhor_score = -1

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    cy = centroids[i][1]

    if area < 300:
        continue

    # favorece o objeto mais abaixo na imagem
    score = area + cy * 5

    if score > melhor_score:
        melhor_score = score
        melhor_id = i


# gera máscara só do vaso
mascara_vaso = np.uint8(labels == melhor_id) * 255

# fecha pequenos buracos pretos dentro do vaso
kernel_fill = np.ones((15, 15), np.uint8)
mascara_vaso = cv2.morphologyEx(mascara_vaso, cv2.MORPH_CLOSE, kernel_fill)

# Aumenta um pouco a mascara para não ficar com "bordas" na hora de subtrair (maior e Y para não prejudicar o caule)
kernel = np.ones((5, 40), np.uint8)
mascara_vaso_ajustada = cv2.dilate(mascara_vaso, kernel, iterations=1)


# Subtraindo vaso da imagem vaso+planta
img_planta = cv2.subtract(mask_fundo, mascara_vaso_ajustada)


# Remove ruídos da imagem
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_planta, connectivity=8)
img_limpa = np.zeros_like(img_planta)

for i in range(1, num_labels):  
    area = stats[i, cv2.CC_STAT_AREA]

    if area > 685:  
        img_limpa[labels == i] = 255


# Isolando o caule: 
# Esqueletiza a máscara da planta para obter a linha central
img_planta_bool = img_limpa.astype(bool)
skeleton = skeletonize(img_planta_bool).astype(np.uint8) * 255


# Analisa componentes conectados do esqueleto
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)


#  Seleciona o componente do esqueleto mais vertical e mais longo
melhor_id = -1
melhor_score = -1


for i in range(1, num_labels):
    area  = stats[i, cv2.CC_STAT_AREA]
    w_cc  = stats[i, cv2.CC_STAT_WIDTH]
    h_cc  = stats[i, cv2.CC_STAT_HEIGHT]

    if area < 30:          # ignora ruídos pequenos
        continue

    # quanto mais alto e fino, mais provável de ser o caule
    verticalidade = h_cc / (w_cc + 1)
    score = area * verticalidade

    if score > melhor_score:
        melhor_score = score
        melhor_id = i

# Máscara do esqueleto do caule
skeleton_caule = np.uint8(labels == melhor_id) * 255


# Dilata o esqueleto para reconstruir a espessura real do caule e ajuste o tamanho do kernel conforme a espessura do caule na imagem
espessura_caule = 7   # px — aumente se o caule for mais grosso
kernel_caule = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (espessura_caule, espessura_caule))
mascara_caule = cv2.dilate(skeleton_caule, kernel_caule, iterations=2)

# Intersecta com a máscara original para não vazar para fora da planta
mascara_caule = cv2.bitwise_and(mascara_caule, img_planta)

# Fecha pequenos buracos internos
kernel_close = np.ones((5, 5), np.uint8)
mascara_caule = cv2.morphologyEx(mascara_caule, cv2.MORPH_CLOSE, kernel_close)





plt.figure("Máscara do Caule", figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Planta (sem vaso)")
plt.imshow(img_limpa, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Esqueleto filtrado")
plt.imshow(skeleton, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Máscara final do caule")
plt.imshow(mascara_caule, cmap="gray")
plt.axis("off")

plt.tight_layout()


# Subtrai caule da imagem da planta
mask_folhas = cv2.subtract(img_limpa, mascara_caule)

# plt.figure("Folhas sem caule")
# plt.imshow(mask_folhas, cmap='gray')
# plt.axis('off')

# Fecha as plantas
kernel = np.ones((15, 15), np.uint8)
mask_fechada = cv2.morphologyEx(mask_folhas, cv2.MORPH_CLOSE, kernel)

# plt.figure("Folhas preenchidas")
# plt.imshow(mask_fechada, cmap='gray')
# plt.axis('off')


# Contando as folhas:
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Função: vizinhos 8-conectados
def neighbors8(x, y, skel):
    h, w = skel.shape
    pts = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            xx, yy = x + dx, y + dy

            if 0 <= xx < w and 0 <= yy < h and skel[yy, xx] > 0:
                pts.append((xx, yy))

    return pts


# Encontrar endpoints
def find_endpoints(skel):
    endpoints = []
    h, w = skel.shape

    for y in range(h):
        for x in range(w):
            if skel[y, x] == 0:
                continue

            vizinhos = neighbors8(x, y, skel)

            if len(vizinhos) == 1:
                endpoints.append((x, y))

    return endpoints


# Medir comprimento do ramo
def branch_length(skel, start):
    visited = set()
    current = start
    prev = None
    length = 0

    while True:
        visited.add(current)
        nbs = neighbors8(current[0], current[1], skel)

        if prev is not None:
            nbs = [p for p in nbs if p != prev]

        if len(nbs) == 0:
            break

        # parou numa bifurcação
        if len(nbs) > 1:
            break

        nxt = nbs[0]

        if nxt in visited:
            break

        length += 1
        prev = current
        current = nxt

    return length


# Pipeline completo
def contar_pontas(skeleton, min_length=10):
    
    # encontra endpoints
    endpoints = find_endpoints(skeleton)

    # filtra por tamanho do ramo
    endpoints_filtrados = []

    for (x, y) in endpoints:
        length = branch_length(skeleton, (x, y))

        if length > min_length:
            endpoints_filtrados.append((x, y))

    # resultado
    print("Pontas totais:", len(endpoints))
    print("Pontas filtradas:", len(endpoints_filtrados))

    # visualização
    img_debug = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)

    # vermelho = todos
    for (x, y) in endpoints:
        cv2.circle(img_debug, (x, y), 3, (255, 0, 0), -1)

    # verde = filtrados
    for (x, y) in endpoints_filtrados:
        cv2.circle(img_debug, (x, y), 3, (0, 255, 0), -1)

    plt.figure("Endpoints")
    plt.imshow(img_debug)
    plt.title("Vermelho = todos | Verde = filtrados")
    plt.axis('off')
   

    return endpoints_filtrados

# chama função
endpoints_final = contar_pontas(skeleton, min_length=40)
plt.show()