import cv2
from ultralytics import YOLO

model = YOLO(r"D:\Helmet_Detection\models\bestyolo.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.4)

    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
