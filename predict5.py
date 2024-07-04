from ultralytics import YOLO
  
model = YOLO("./runs/detect/train/weights/best.pt")
model = model.to('cpu')
results = model.predict(source='./test/images (3).jpeg', stream=True, imgsz=620) # source already setup
names = model.names

for r in results:
    for c in r.boxes.cls:
        naaam = names[int(c)]
        print(names[int(c)])

print(naaam)