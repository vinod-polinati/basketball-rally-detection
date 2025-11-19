from ultralytics import YOLO

model = YOLO("yolov8x.pt")

def detect(frame):
    result = model(frame, verbose=False)[0]
    players = []
    balls = []

    for obj in result.boxes:
        cls = int(obj.cls)
        x1,y1,x2,y2 = obj.xyxy[0]

        if cls == 0:  # person
            players.append([int(x1),int(y1),int(x2),int(y2)])
        if cls == 32: # sports ball
            balls.append([int(x1),int(y1),int(x2),int(y2)])

    return players, balls
