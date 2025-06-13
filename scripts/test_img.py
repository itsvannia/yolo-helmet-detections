import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import time
from collections import deque

class YOLOHelmetDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Đang sử dụng thiết bị: {self.device}")

        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.model.half()

        raw_names = self.model.names
        self.class_names = {k: self._normalize_class_name(v) for k, v in raw_names.items()}
        print("Tên lớp chuẩn hóa:", self.class_names)

        self.colors = {
            'helmet': (0, 255, 0),
            'no_helmet': (0, 0, 255),
            'unknown': (255, 255, 0)
        }

        self.fps_deque = deque(maxlen=30)
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()

        self.skip_frames = 0
        self.frame_counter = 0

        self.helmet_count = 0
        self.no_helmet_count = 0

        self.display_mode = 0
        self.show_fps = True
        self.show_detection = True

        self.model_input_size = (320, 320)

        self.last_detections = []

    def _normalize_class_name(self, name):
        name = name.lower()
        if "no" in name or "without" in name:
            return "no_helmet"
        elif "helmet" in name or "with" in name:
            return "helmet"
        return "unknown"

    def draw_detection_results(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['conf']
            class_name = det['class']

            box_color = self.colors.get(class_name, (255, 255, 0))
            text_color = (255, 255, 255) if class_name == 'no_helmet' else (0, 0, 0)

            if conf > 0.4:
                label = f"{class_name}: {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                (w, h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_x = x1
                text_y = y1 - 5 if y1 - 5 > 5 else y1 + 15
                cv2.rectangle(frame, (text_x - 2, text_y - h - 2), (text_x + w + 2, text_y + 2), box_color, -1)
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

        return frame

    def create_info_overlay(self, frame, helmet_count, no_helmet_count, fps):
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Helmet: {helmet_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"No Helmet: {no_helmet_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame

    def process_frame(self, frame, conf_threshold):
        try:
            with torch.no_grad():
                results = self.model.predict(
                    frame, 
                    conf=conf_threshold, 
                    device=self.device,
                    imgsz=self.model_input_size,
                    verbose=False
                )

            detection_data = []
            helmet_count_temp = 0
            no_helmet_count_temp = 0

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name_raw = self.model.names[cls_id]
                        class_name = self._normalize_class_name(class_name_raw)

                        detection_data.append({
                            'box': (x1, y1, x2, y2),
                            'conf': conf,
                            'class': class_name
                        })

                        if class_name == 'helmet':
                            helmet_count_temp += 1
                        elif class_name == 'no_helmet':
                            no_helmet_count_temp += 1

            self.helmet_count = helmet_count_temp
            self.no_helmet_count = no_helmet_count_temp

            return detection_data

        except Exception as e:
            print(f"Lỗi trong quá trình phát hiện: {e}")
            return []

    def process_image(self, image_path, conf_threshold=0.3, save_output=False, output_path='output.jpg'):
        if not os.path.exists(image_path):
            print(f"Không tìm thấy ảnh: {image_path}")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi khi đọc ảnh: {image_path}")
            return

        detections = self.process_frame(image, conf_threshold)
        output_image = self.draw_detection_results(image.copy(), detections)
        output_image = self.create_info_overlay(output_image, self.helmet_count, self.no_helmet_count, self.fps)

        cv2.imshow("Helmet Detection - Image", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if save_output:
            cv2.imwrite(output_path, output_image)
            print(f"Đã lưu kết quả vào: {output_path}")

    def process_folder(self, folder_path, conf_threshold=0.3, save_output=False, output_dir='output_images'):
        if not os.path.exists(folder_path):
            print(f"Không tìm thấy thư mục: {folder_path}")
            return

        if save_output and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

        if not image_files:
            print("Không có ảnh nào trong thư mục.")
            return

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            print(f"Đang xử lý ảnh: {img_name}")

            image = cv2.imread(img_path)
            if image is None:
                print(f"Không thể đọc: {img_path}")
                continue

            detections = self.process_frame(image, conf_threshold)
            output_image = self.draw_detection_results(image.copy(), detections)
            output_image = self.create_info_overlay(output_image, self.helmet_count, self.no_helmet_count, self.fps)

            cv2.imshow("Helmet Detection - Folder", output_image)
            cv2.waitKey(1)

            if save_output:
                save_path = os.path.join(output_dir, img_name)
                cv2.imwrite(save_path, output_image)
                print(f"Đã lưu: {save_path}")

        cv2.destroyAllWindows()
        print("Hoàn tất xử lý tất cả ảnh trong thư mục.")

    def process_realtime(self, source=0, conf_threshold=0.3, resolution=(640, 480)):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Không thể mở nguồn video: {source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        print(f"Độ phân giải video: {resolution[0]}x{resolution[1]}")
        print(f"Độ phân giải đầu vào mô hình: {self.model_input_size}")
        print(f"Thiết bị: {self.device}")
        print(f"Ngưỡng tin cậy: {conf_threshold}")
        print(f"Nhấn 'D' để đổi chế độ hiển thị, 'S' để bỏ qua frames, 'Q' để thoát")

        fps_update_interval = 0.5
        last_fps_update = time.time()

        while True:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Kết thúc video hoặc lỗi đọc frame")
                break

            self.frame_counter += 1
            if self.frame_counter % (self.skip_frames + 1) == 0:
                detections = self.process_frame(frame, conf_threshold)
                self.last_detections = detections
                self.frame_counter = 0

            output_frame = frame.copy()

            if self.show_detection and self.last_detections:
                output_frame = self.draw_detection_results(output_frame, self.last_detections)

            output_frame = self.create_info_overlay(output_frame, self.helmet_count, self.no_helmet_count, self.fps)

            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - last_fps_update

            if elapsed_time >= fps_update_interval:
                this_fps = self.frame_count / elapsed_time
                self.fps_deque.append(this_fps)
                self.fps = sum(self.fps_deque) / len(self.fps_deque)
                self.frame_count = 0
                last_fps_update = current_time

            cv2.imshow("Helmet Detection", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Người dùng thoát")
                break
            elif key == ord('d'):
                self.display_mode = (self.display_mode + 1) % 2
                self.show_detection = self.display_mode == 0
                print(f"Chế độ hiển thị: {'Đầy đủ' if self.display_mode == 0 else 'Chỉ thông tin'}")
            elif key == ord('s'):
                self.skip_frames = (self.skip_frames + 1) % 5
                print(f"Bỏ qua {self.skip_frames} frame")

            frame_process_time = time.time() - frame_start_time
            if frame_process_time > 0.1:
                if self.skip_frames < 2:
                    self.skip_frames += 1
                    print(f"Tự động điều chỉnh: Bỏ qua {self.skip_frames} frame")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = r'D:\Helmet_Detection\models\bestyolo.pt'
    detector = YOLOHelmetDetector(model_path)

    print("--- Nhận diện từ ảnh đơn ---")
    image_path = r"D:\Helmet_Detection\data_images\test_images\giaothong.jpg"
    detector.process_image(image_path, conf_threshold=0.3, save_output=True)

    # print("--- Nhận diện từ thư mục ảnh ---")
    # folder_path = r"D:\Documents\DATN_Helmet_Detected\test_set\tf1.jpg"
    # detector.process_folder(folder_path, conf_threshold=0.3, save_output=True, output_dir="output_images")

    # print("--- Nhận diện video thời gian thực ---")
    # detector.process_realtime(
    #     source=r"D:\Helmet_Detection\data_images\test_video\atgt0.mp4",
    #     conf_threshold=0.3,
    #     resolution=(640, 480)
    # )
