import cv2
import numpy as np


arquivos = [
    "aps3/Figuras_APS3/Fig2_Curling1.bmp",
    "aps3/Figuras_APS3/Fig2_Curling2.bmp",
    "aps3/Figuras_APS3/Fig2_Curling3.bmp"
]

# Função para detectar a casa
def detectar_casa(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=80,
        maxRadius=min(img_bgr.shape[0], img_bgr.shape[1]) // 2
    )

    if circles is None:
        return None, None, None

    circles = np.squeeze(circles)

    if len(circles.shape) == 1:
        circles = np.array([circles])

    idx = np.argmax(circles[:, 2])
    cx, cy, raio = circles[idx]
    return int(cx), int(cy), int(raio)

# Função para detectar as pedras
def detectar_pedras_por_cor(img_bgr, cor, cx, cy):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if cor == "vermelho":
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )

    elif cor == "amarelo":
        lower = np.array([20, 100, 100])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pedras = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        pedras.append({
            "cnt": cnt,
            "x": x,
            "y": y,
            "area": area,
            "dist": dist
        })

    return mask, pedras


# Loop para testar todas as imagens
for arquivo in arquivos:
    print("\n==============================")
    print("Processando:", arquivo)

    img = cv2.imread(arquivo)
    if img is None:
        print("Erro ao abrir imagem.")
        continue

    h, w = img.shape[:2]

    # Detecta casa
    cx, cy, raio_casa = detectar_casa(img)

    if cx is None:
        print("Falha na detecção automática da casa. Usando fallback.")
        cx, cy = w // 2, h // 2
        raio_casa = min(h, w) // 3

    print(f"Centro: ({cx}, {cy}) | Raio: {raio_casa}")

    # Detecta pedras
    mask_red, pedras_red = detectar_pedras_por_cor(img, "vermelho", cx, cy)
    mask_yellow, pedras_yellow = detectar_pedras_por_cor(img, "amarelo", cx, cy)

    print("Vermelhas:", len(pedras_red))
    print("Amarelas:", len(pedras_yellow))

    # Junta
    todas = []
    for p in pedras_red:
        p["time"] = "vermelho"
        todas.append(p)

    for p in pedras_yellow:
        p["time"] = "amarelo"
        todas.append(p)

    # Filtra dentro da casa
    pedras_validas = [p for p in todas if p["dist"] <= raio_casa]
    pedras_validas = sorted(pedras_validas, key=lambda k: k["dist"])

    # Pontuação
    vencedor = None
    pontos = 0

    if len(pedras_validas) > 0:
        time = pedras_validas[0]["time"]

        dist_limite = None
        for p in pedras_validas:
            if p["time"] != time:
                dist_limite = p["dist"]
                break

        if dist_limite is None:
            pontos = sum(1 for p in pedras_validas if p["time"] == time)
        else:
            pontos = sum(1 for p in pedras_validas
                         if p["time"] == time and p["dist"] < dist_limite)

        vencedor = time

    print("Vencedor:", vencedor)
    print("Pontos:", pontos)

    img_out = img.copy()

    cv2.circle(img_out, (cx, cy), raio_casa, (255, 0, 0), 2)
    cv2.circle(img_out, (cx, cy), 5, (0, 0, 0), -1)

    cores = {
        "vermelho": (0, 0, 255),
        "amarelo": (0, 255, 255)
    }

    for p in pedras_validas:
        cv2.drawContours(img_out, [p["cnt"]], -1, cores[p["time"]], 2)
        cv2.circle(img_out, (p["x"], p["y"]), 4, (0, 0, 0), -1)

    cv2.putText(img_out, f"Vencedor: {vencedor}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img_out, f"Pontos: {pontos}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Mostra resultado
    cv2.imshow("Resultado", img_out)
    cv2.waitKey(0)

cv2.destroyAllWindows()