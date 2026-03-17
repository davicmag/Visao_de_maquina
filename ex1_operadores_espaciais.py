import cv2
import matplotlib.pyplot as plt
import numpy as np


seixosP = cv2.imread("Figuras_APS2/Fig1_SeixoP.bmp", cv2.IMREAD_GRAYSCALE)
if seixosP is None:
    print("File 1 not found. Bye!")
    exit(0)

seixosM = cv2.imread("Figuras_APS2/Fig1_SeixoM.bmp", cv2.IMREAD_GRAYSCALE)
if seixosM is None:
    print("File not 2 found. Bye!")
    exit(0)


seixosG = cv2.imread("Figuras_APS2/Fig1_SeixoG.bmp", cv2.IMREAD_GRAYSCALE)
if seixosG is None:
    print("File not 3 found. Bye!")
    exit(0)

# Binariza todas as imagens
fig_binP = np.where(seixosP > 100, 255, 0)
fig_binM = np.where(seixosM > 100, 255, 0)
fig_binG = np.where(seixosG > 100, 255, 0)



# Soma os pixels pretos de cada imagem
ranhurasP = np.sum(fig_binP == 0)
ranhurasM = np.sum(fig_binM == 0)
ranhurasG = np.sum(fig_binG == 0)


# Função para classificar
def classificar(ranhuras):
    if 35000 <= ranhuras <= 40000:
        return "P"
    elif 65000 <= ranhuras <= 73000:
        return "M"
    elif 95000 <= ranhuras <= 103000:
        return "G"
    else:
        return "Desconhecido"


print("Qual imagem você quer verificar?")

opcao = input("Digite 1, 2 ou 3: ")

if opcao == "1":
    ranhuras = np.sum(fig_binP == 0)
    resultado = classificar(ranhuras)
    print(f"Imagem 1 tem {ranhuras} ranhuras → Classificação: {resultado}")

elif opcao == "2":
    ranhuras = np.sum(fig_binM == 0)
    resultado = classificar(ranhuras)
    print(f"Imagem 2 tem {ranhurasM} ranhuras → Classificação: {resultado}")

elif opcao == "3":
    ranhuras = np.sum(fig_binG == 0)
    resultado = classificar(ranhuras)
    print(f"Imagem 3 tem {ranhurasG} ranhuras → Classificação: {resultado}")

else:
    print("Opção inválida!")






