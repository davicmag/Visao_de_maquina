import cv2
import matplotlib.pyplot as plt
import numpy as np

base_original = cv2.imread("Figuras_APS2/Fig3_Base.bmp", cv2.IMREAD_GRAYSCALE)
if base_original is None:
    print("File not found. Bye!")
    exit(0)

arm1_original = cv2.imread("Figuras_APS2/Fig3_Arm1.bmp", cv2.IMREAD_GRAYSCALE)
if arm1_original is None:
    print("File not found. Bye!")
    exit(0)

arm2_original = cv2.imread("Figuras_APS2/Fig3_Arm2.bmp", cv2.IMREAD_GRAYSCALE)
if arm2_original is None:
    print("File not found. Bye!")
    exit(0)

import numpy as np

dados_mvimento = np.loadtxt("Robo_Cinematica.csv", delimiter=";", skiprows=1)

theta1_vals = dados_mvimento[:, 0]
theta2_vals = dados_mvimento[:, 1]

hb, wb = base_original.shape
h1, w1 = arm1_original.shape
h2, w2 = arm2_original.shape


# Matrizes de transação
matriz_translacao_arm1 = np.array([[1, 0, 140], [0, 1, 180], [0, 0, 1]])
matriz_translacao_arm2 = np.array([[1, 0, 285], [0, 1, 75], [0, 0, 1]])

# número de frames do vídeo
num_frames = min(len(theta1_vals), len(theta2_vals))
frame_atual = 0

# salvar o vídeo
saida = cv2.VideoWriter(
    "robo_animado.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    80,
    (wb, hb),
    False
)

while (frame_atual < num_frames):

    # recria as imagens a cada frame
    base = base_original.copy()
    arm1 = arm1_original.copy()
    arm2 = arm2_original.copy()

    # Matrizes de rotação - variáveis
    theta1 = theta1_vals[frame_atual]
    theta2 = theta2_vals[frame_atual]

    matriz_rotacao_arm1 = np.array([[np.cos(np.deg2rad(theta1)), -(np.sin(np.deg2rad(theta1))), 0],[np.sin(np.deg2rad(theta1)),  np.cos(np.deg2rad(theta1)), 0],[0, 0, 1]])
    matriz_rotacao_arm2 = np.array([[np.cos(np.deg2rad(theta2)), -(np.sin(np.deg2rad(theta2))), 0],[np.sin(np.deg2rad(theta2)),  np.cos(np.deg2rad(theta2)), 0],[0, 0, 1]])

    # Posicionando Braço 2
    for x in range(w2):
        for y in range(h2):

            p = np.array([[x], [y], [1]])
            p1 = matriz_translacao_arm2 @ matriz_rotacao_arm2 @ p

            u = int(p1[0][0])
            v = int(p1[1][0])

            if u in range(0, w1) and v in range(0, h1):
                arm1[v, u] = arm2[y, x]

    # Posicionando Braço 1
    for x in range(w1):
        for y in range(h1):

            p = np.array([[x], [y], [1]])
            p1 = matriz_translacao_arm1 @ matriz_rotacao_arm1 @ p

            u = int(p1[0][0])
            v = int(p1[1][0])

            if u in range(0, w1) and v in range(0, h1):
                base[v, u] = arm1[y, x]

    # mostra o frame
    cv2.imshow("Video", base)

    # grava o frame no vídeo
    saida.write(base)

    frame_atual += 1

    # Press "q" on keyboard to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

saida.release()
cv2.destroyAllWindows()