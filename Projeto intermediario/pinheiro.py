import cv2
import numpy as np
from library.selectBlob import *
from skimage.morphology import skeletonize
from collections import deque
import os
import pandas as pd


pasta = "Visao_de_maquina/Projeto intermediario/Dataset_Projeto1/_Pinheiro_Escolhidos1"

arquivos = [f for f in os.listdir(pasta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

resultados = []

for idx, nome_arquivo in enumerate(arquivos):

    caminho_img = os.path.join(pasta, nome_arquivo)
    img = cv2.imread(caminho_img)

    if img is None:
        print(f"Erro ao abrir {nome_arquivo}")
        continue

    # Converte imagem para RGB e Mostra
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Separa em 3 canais
    (canal_r, canal_g, canal_b) = cv2.split(img_rgb)

    # Gera mascara inicial (remove o fundo)
    mask_fundo = np.where((canal_b > 140), 0, 255).astype(np.uint8)

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

    # Subtrai vaso da imagem vaso+planta
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

    # Função para encontrar as pontas
    def find_endpoints(skel):
        endpoints = []
        h, w = skel.shape

        for y in range(h):
            for x in range(w):
                if skel[y, x] == 0:
                    continue

                if len(neighbors8(x, y, skel)) == 1:
                    endpoints.append((x, y))

        return endpoints


    def bfs_path(skel, start, goal):
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == goal:
                return path

            for nx, ny in neighbors8(x, y, skel):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return []


    def extract_stem(skel):
        endpoints = find_endpoints(skel)

        if len(endpoints) < 2:
            return []

        h, w = skel.shape

        # Base da planta (mais baixo e mais central)
        base = max(endpoints, key=lambda p: (p[1], -abs(p[0] - w//2)))

        def path_score(path):
            ys = [p[1] for p in path]
            xs = [p[0] for p in path]
            altura = max(ys) - min(ys)
            lateral = np.std(xs)
            return altura - 0.5 * lateral

        best_path = []
        best_score = -1

        for ep in endpoints:
            if ep == base:
                continue

            path = bfs_path(skel, base, ep)

            if len(path) < 5:
                continue

            score = path_score(path)

            if score > best_score:
                best_score = score
                best_path = path

        return best_path


    def trim_stem_at_first_branch(path, skel, min_length_ratio=0.75):
        """
        Só corta na primeira ramificação encontrada no TERÇO SUPERIOR do caule.
        Ramificações no meio (folhas laterais) são ignoradas.
        """
        if not path:
            return path

        min_idx = int(len(path) * min_length_ratio)

        for i, (x, y) in enumerate(path):
            if i < min_idx:
                continue  # ignora ramificações nos primeiros 75% do caminho

            if grau_no(x, y, skel) >= 3:
                return path[:i]

        return path

    caminho_caule = extract_stem(skeleton)

    mask_caule = np.zeros_like(img_planta)


    for x, y in caminho_caule:
        mask_caule[y, x] = 255




    # Engrossa até cobrir o caule real
    # Distância até a borda da planta
    dist_bg = cv2.distanceTransform(img_limpa, cv2.DIST_L2, 5)


    def grau_no(x, y, skel):
        return len(neighbors8(x, y, skel))

    def trim_stem_at_first_branch(path, skel, min_length_ratio=0.3):
        if not path:
            return path
        min_idx = int(len(path) * min_length_ratio)
        for i, (x, y) in enumerate(path):
            if i < min_idx:
                continue
            if grau_no(x, y, skel) >= 3:
                return path[:i]
        return path

    caminho_filtrado = trim_stem_at_first_branch(caminho_caule, skeleton, min_length_ratio=0.75)



        

    # Distância até o caminho do caule
    mask_linha = np.zeros_like(img_limpa, dtype=np.uint8)
    for x, y in caminho_filtrado:
        mask_linha[y, x] = 255

    dist_caule = cv2.distanceTransform(255 - mask_linha, cv2.DIST_L2, 5)

    # Mantém só o que está mais perto do caule do que da borda
    mascara_caule = np.zeros_like(img_limpa, dtype=np.uint8)

    mask = dist_caule <= (dist_bg * 8)

    # remove pixels muito longe do eixo (folhas grudadas)
    mask &= dist_caule < 9

    mascara_caule[mask] = 255


    # Subtrai caule da imagem da planta
    mask_folhas = cv2.subtract(img_limpa, mascara_caule)

    # Aumenta para recuperar pixels perdidos
    kernel = np.ones((3,3), np.uint8)
    mask_folhas = cv2.dilate(mask_folhas, kernel, iterations=1)

    # Contando as folhas:
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


        return endpoints_filtrados

    # chama função
    endpoints_final = contar_pontas(skeleton, min_length=40)


    num_brancos = cv2.countNonZero(mask_folhas)


    # Desenhando contornos na imagem original
    # encontra contornos na máscara
    contornos, _ = cv2.findContours(mask_folhas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # copia imagem original
    img_contorno = img_rgb.copy()

    # desenha contornos em vermelho
    cv2.drawContours(img_contorno, contornos, -1, (255, 0, 0), 2)


    # cria uma cópia da imagem original
    img_caule_overlay = img_contorno.copy()

    # onde tem caule pinta de azul
    img_caule_overlay[mascara_caule == 255] = [0, 0, 255]  

    # altura vertical da planta
    ys, xs = np.where(img_limpa > 0)
    altura_planta = ys.max() - ys.min()



    if len(caminho_filtrado) == 0:
        print(f"[ERRO] Caminho do caule vazio: {nome_arquivo}")
        comprimento = altura_planta
        diametro = 0

    else:
        # diâmetro
        # encontra a base
        ys, xs = np.where(mascara_caule > 0)
        y_base = ys.max()

        # sobe 10 pixels
        y_target = y_base - 10

        # mede a largura horizontal
        linha = mascara_caule[y_target]
        xs_linha = np.where(linha > 0)[0]
        diametro = (xs_linha.max() - xs_linha.min())+2


        # comprimento total do caule
        comprimento = 0
        for i in range(1, len(caminho_filtrado)):
            x1, y1 = caminho_filtrado[i-1]
            x2, y2 = caminho_filtrado[i]
            comprimento += np.sqrt((x2-x1)**2 + (y2-y1)**2)

    # Número de folhas
    n_folhas = len(endpoints_final)

    # Área foliar
    area = cv2.countNonZero(mask_folhas)

    # Altura vertical
    altura = altura_planta

    # Comprimento total
    compr = comprimento

    # Diâmetro
    diam = diametro

    # Salva resultado
    resultados.append({
        "Img": nome_arquivo,
        "Altura Vert.": altura,
        "Compr Total": compr,
        "Diâmetro": diam,
        "Área": area,
        "Nro Folhas": n_folhas
    })

df = pd.DataFrame(resultados)
df.to_csv("resultados_pinheiros.csv", index=False)
print("CSV gerado com sucesso!")