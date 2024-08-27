from ultralytics import YOLO

model = YOLO('yolov10n.pt') # Скачиваем модель yolo

results = model.predict('input_video\input.mp4',save = True)
print(results[0])
print('=============================================')
for box in results[0].boxes:
    print(box)
