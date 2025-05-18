
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Carregar modelo YOLOv8 + StrongSORT
model = YOLO("yolov8m.pt")
model.trackers = "botsort.yaml"  # ou 'strongsort.yaml' se disponível no seu setup

# Abrir vídeo
cap = cv2.VideoCapture("videos\intelbras1.teste.mp4")

# Linha virtual (posição y para contagem)
linha_contagem_y = 90
ids_contados = set()
contador_total = 0

# IDs de veículos no COCO
veiculos_ids = [2, 3, 5, 7]


# Obter largura e altura do frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Criar VideoWriter para salvar o vídeo
out = cv2.VideoWriter("saida_com_contagem.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 360))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reduzir resolução se quiser melhorar o desempenho
    frame = cv2.resize(frame, (640, 360))

    # Rastrear com YOLO + StrongSORT
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

                # Verifica se cruzou a linha e se ainda não foi contado
                if cy > linha_contagem_y - 5 and cy < linha_contagem_y + 5:
                    if track_id not in ids_contados:
                        ids_contados.add(track_id)
                        contador_total += 1

    # Mostrar linha de contagem
    cv2.line(frame, (0, linha_contagem_y), (frame.shape[1], linha_contagem_y), (255, 0, 0), 2)

    # Mostrar contador
    cv2.putText(frame, f"Veiculos contados: {contador_total}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Contagem com Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total final: {contador_total}")
cap.release()
out.release()
cv2.destroyAllWindows()

