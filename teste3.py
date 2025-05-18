import cv2 ##inclinada 
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8m.pt")
model.trackers = "botsort.yaml"

cap = cv2.VideoCapture("videos\intelbras1.teste.mp4")

# Posição da linha inclinada
ponto1 = (-300, -300)
ponto2 = (600, 000)

ids_contados = set()
contador_total = 0
veiculos_ids = [2, 3, 5, 7]
posicoes_anteriores = {}

def lado_do_ponto(px, py, x1, y1, x2, y2):
    """Determina de que lado da linha está o ponto (px, py)"""
    return np.sign((x2 - x1)*(py - y1) - (y2 - y1)*(px - x1))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model.track(source=frame, persist=True, verbose=False)[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().cpu().tolist()
        boxes = results.boxes.xyxy.cpu().tolist()
        classes = results.boxes.cls.int().cpu().tolist()

        for box, cls, track_id in zip(boxes, classes, ids):
            if cls in veiculos_ids:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Desenhar bounding box e ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                lado_atual = lado_do_ponto(cx, cy, *ponto1, *ponto2)
                lado_anterior = posicoes_anteriores.get(track_id)

                if lado_anterior is not None and lado_atual != lado_anterior:
                    if track_id not in ids_contados:
                        ids_contados.add(track_id)
                        contador_total += 1

                posicoes_anteriores[track_id] = lado_atual

    # Desenhar linha inclinada
    cv2.line(frame, ponto1, ponto2, (255, 0, 0), 2)

    # Mostrar contador
    cv2.putText(frame, f"Veiculos contados: {contador_total}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Contagem com Linha Inclinada", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total final: {contador_total}")
