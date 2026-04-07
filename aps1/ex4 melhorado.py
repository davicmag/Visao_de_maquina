import numpy as np
import cv2

# read video
cap = cv2.VideoCapture('aps1\Figuras_APS1\Video_APS1_4.mp4')
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Nro frames = ", num_frames)

SCALE = 0.5

frame_atual = 1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # setar o frame atual para o primeiro frame do video
ret0, primeiro_frame = cap.read()
primeiro_frame_gray = cv2.cvtColor(primeiro_frame, cv2.COLOR_BGR2GRAY)
(h,w) = primeiro_frame_gray.shape
primeiro_frame_gray2 = cv2.resize(primeiro_frame_gray, (int(w*SCALE),int(h*SCALE)) )


while (frame_atual < num_frames):

    # leitura sequencia dos frames do video
	ret, frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# reduzir o tamanho do frame pela metade (a fim de reduzir o tempo de processamento)
	(h,w) = frame_gray.shape
	frame_gray2 = cv2.resize(frame_gray, (int(w*SCALE),int(h*SCALE)) )
	cv2.imshow("Video", frame_gray2)
	(h2,w2) = frame_gray2.shape
	#SEU PROCESSAMENTO DEVE VIR AQUI

	frame_gray2 = np.where(np.abs(frame_gray2.astype(np.int32) - primeiro_frame_gray2.astype(np.int32)) > 32, frame_gray2, 0)

	frame_atual += 1
	cv2.imshow("Video", frame_gray2)
	
	# Press "q" on keyboard to exit
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

