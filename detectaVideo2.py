import cv2
from ultralytics import YOLO

# Carregar modelo médio
model = YOLO("yolov8m.pt")

# Caminho do vídeo
cap = cv2.VideoCapture("videos\CERJ2.mp4")

# Classes de veículos
veiculos_ids = [2, 3, 5, 7]
contador_total = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar frame (tela menor = mais rápido)
    frame = cv2.resize(frame, (1200, 900))

    # Rodar detecção
    results = model(frame, verbose=False)[0]
    veiculos_detectados = [d for d in results.boxes.data if int(d[5]) in veiculos_ids]

    for box in veiculos_detectados:
        x1, y1, x2, y2, conf, cls_id = box
        label = model.names[int(cls_id)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    contador_total += len(veiculos_detectados)

    # Mostrar contador no canto
    cv2.putText(frame, f"Total detectado: {contador_total}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Contador de Veículos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total final: {contador_total}")
