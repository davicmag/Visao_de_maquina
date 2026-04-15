import cv2
import numpy as np


arquivos = [
    "aps3/Figuras_APS3/Fig4_Esteira1.png",
    "aps3/Figuras_APS3/Fig4_Esteira2.png",
    "aps3/Figuras_APS3/Fig4_Esteira3.png"
]


AREA_MIN_PECA = 1500
AREA_MAX_PECA = 50000

AREA_MIN_ETIQUETA = 30
AREA_MAX_ETIQUETA = 1500

# parte inferior da imagem
Y_MIN_ESTEIRA_RATIO = 0.25

# etiqueta branca deve estar na lateral direita da peça
X_ETIQUETA_MIN_RATIO = 0.55

# branco em HSV
LOWER_WHITE = np.array([0, 0, 180])
UPPER_WHITE = np.array([180, 60, 255])

# azul em HSV
LOWER_BLUE = np.array([85, 60, 40])
UPPER_BLUE = np.array([130, 255, 255])

# verde em HSV
LOWER_GREEN = np.array([35, 50, 40])
UPPER_GREEN = np.array([85, 255, 255])


def detectar_pecas_por_cor(img_bgr, cor):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if cor == "azul":
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    elif cor == "verde":
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    else:
        raise ValueError("Cor inválida. Use 'azul' ou 'verde'.")

    # limpeza morfológica
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img_bgr.shape[:2]
    y_min_esteira = int(h * Y_MIN_ESTEIRA_RATIO)

    pecas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN_PECA or area > AREA_MAX_PECA:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = int(x + bw / 2)
        cy = int(y + bh / 2)

        # filtra objetos fora da esteira 
        if cy < y_min_esteira:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        pecas.append({
            "cnt": cnt,
            "x": cx,
            "y": cy,
            "bbox": (x, y, bw, bh),
            "area": area,
            "cor": cor
        })

    return mask, pecas


def contar_etiquetas_brancas(img_bgr, peca):
    x, y, w, h = peca["bbox"]

    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return 0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, mask_white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    # lateral direita da peça
    x_ini = int(w * 0.60)
    mask_right = np.zeros_like(mask_white)
    mask_right[:, x_ini:] = mask_white[:, x_ini:]

    # erosão para separar etiquetas próximas
    kernel = np.ones((3, 3), np.uint8)
    mask_right = cv2.erode(mask_right, kernel, iterations=2)

   
    mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    etiquetas = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 <= area <= 250:
            etiquetas += 1

    return etiquetas


for arquivo in arquivos:
    print("\n==================================================")
    print("Processando:", arquivo)

    img = cv2.imread(arquivo)
    if img is None:
        print("File not found. Bye!")
        continue

    h, w = img.shape[:2]

    # detecta peças
    mask_azul, pecas_azuis = detectar_pecas_por_cor(img, "azul")
    mask_verde, pecas_verdes = detectar_pecas_por_cor(img, "verde")

    print("Pecas azuis detectadas:", len(pecas_azuis))
    print("Pecas verdes detectadas:", len(pecas_verdes))

    # junta tudo
    todas = []
    for p in pecas_azuis:
        todas.append(p)
    for p in pecas_verdes:
        todas.append(p)

    if len(todas) == 0:
        print("Nenhuma peça detectada.")
        continue

    # escolhe a peça mais à direita
    peca_mais_direita = max(todas, key=lambda p: p["x"])

    # conta etiquetas brancas nessa peça
    n_etiquetas = contar_etiquetas_brancas(img, peca_mais_direita)

    # status
    if n_etiquetas >= 3:
        status = "aprovado"
    else:
        status = "reprovado"

    cor_peca = peca_mais_direita["cor"]
    x = peca_mais_direita["x"]
    y = peca_mais_direita["y"]

    print("\n=== RESULTADO ===")
    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"Cor = {cor_peca}")
    print(f"Etiquetas brancas = {n_etiquetas}")
    print(f"Status = {status}")


    img_out = img.copy()

    cores_bgr = {
        "azul": (255, 0, 0),
        "verde": (0, 255, 0)
    }

    # desenha todas as peças detectadas
    for p in todas:
        cv2.drawContours(img_out, [p["cnt"]], -1, cores_bgr[p["cor"]], 2)
        cv2.circle(img_out, (p["x"], p["y"]), 4, (0, 0, 0), -1)

    # destaca a peça mais à direita
    cv2.circle(img_out, (x, y), 8, (0, 0, 255), 2)
    cv2.putText(img_out, f"{cor_peca} | {status}", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # escreve o texto final
    cv2.putText(img_out, f"X={x} Y={y}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img_out, f"Cor: {cor_peca}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img_out, f"Status: {status}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img_out, f"Etiquetas: {n_etiquetas}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


    cv2.imshow("Mascara Azul", mask_azul)
    cv2.imshow("Mascara Verde", mask_verde)
    cv2.imshow("Resultado", img_out)

    cv2.waitKey(0)

cv2.destroyAllWindows()