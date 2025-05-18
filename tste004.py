import os
import cv2
from ultralytics import YOLO

# Criar pasta de saída se não existir
pasta_saida = "resultados"
os.makedirs(pasta_saida, exist_ok=True)

# Caminho para salvar o vídeo
caminho_saida = os.path.join(pasta_saida, "saida_com_contagem.mp4")

# Carregar modelo YOLOv8 + StrongSORT
model = YOLO("yolov8m.pt")
model.trackers = "botsort.yaml"

# Abrir vídeo
cap = cv2.VideoCapture("videos/teste1.mp4")

linha_contagem_y = 140
ids_contados = set()
contador_total = 0
veiculos_ids = [2, 3, 5, 7]

# Configuração do vídeo de saída
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(caminho_saida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1200, 900))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1200, 900))
    results = model.track(source=frame, persist=True, verbose=False)[0]

    if results.boxes.id is not None:
        ids = results.boxes.id.int().cpu().tolist()
        boxes = results.boxes.xyxy.cpu().tolist()
        classes = results.boxes.cls.int().cpu().tolist()

        for box, cls, track_id in zip(boxes, classes, ids):
            if cls in veiculos_ids:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if linha_contagem_y - 5 < cy < linha_contagem_y + 5:
                    if track_id not in ids_contados:
                        ids_contados.add(track_id)
                        contador_total += 1

    # Mostrar linha e contador
    cv2.line(frame, (0, linha_contagem_y), (frame.shape[1], linha_contagem_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Veiculos contados: {contador_total}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Mostrar e salvar o frame
    cv2.imshow("Contagem com Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Total final: {contador_total}")
print(f"Vídeo salvo em: {caminho_saida}")
