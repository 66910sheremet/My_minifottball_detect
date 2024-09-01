from ultralytics import YOLO

model = YOLO('bestl.pt') # Скачиваем модель yolo

results = model.predict(r'videos_for_cvat\test.mp4',save = True)
print(results[0])
print('=============================================')
for box in results[0].boxes:
    print(box)
