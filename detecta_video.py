from ultralytics import YOLO
import cv2

# Caminho do vídeo e do modelo
video_path = "videos/teste1.mp4"
modelo_path = "modelos/yolov8n.pt"

# Carregar o modelo YOLO
model = YOLO(modelo_path)

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

# Verificar se abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Loop para processar cada frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fazer a predição com YOLO
    results = model(frame)

    # Mostrar os resultados com bounding boxes
    annotated_frame = results[0].plot()  # Adiciona caixas no frame
    #tamahho da tela
    resized_frame = cv2.resize(annotated_frame, (1200, 900))
    cv2.imshow("Deteccao em video", resized_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
