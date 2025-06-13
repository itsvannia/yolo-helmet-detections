from ultralytics import YOLO

model = YOLO(r"D:\Helmet_Detection\bestyolo.pt")

#Export sang ONNX
model.export(format="onnx", imgsz=[640,640],opset = 16, simplify = True, dynamic = True)