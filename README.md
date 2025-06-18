
# Đồ án tốt nghiệp 2025 - Nguyễn Văn Nghĩa - CNTT3-K62 - UTC

Ứng dụng Học sâu trong nhận diện hành vi **không đội mũ bảo hiểm** khi tham gia giao thông bằng YOLOv11 và deploy bằng Streamlit

---

## 🧠 Tính năng chính

✅ Nhận diện hành vi **không đội mũ bảo hiểm**  
🎥 Xử lý **ảnh, video** theo thời gian thực  
📁 Tự động lưu ảnh kết quả và báo cáo  
⚡ Sử dụng **YOLOv11** tốc độ cao và chính xác  
🧩 Giao diện đơn giản, dễ dùng với **Streamlit**

---

## 📁 Cấu trúc dự án

```
HELMET_DETECTION/
│
├── app/                    # Thư mục chứa toàn bộ code xử lý
│   ├── main.py             # Giao diện Streamlit chính
│   ├── load_model.py       # Load mô hình YOLO
│   ├── processing.py       # Xử lý ảnh/video
│   ├── draw_box.py         # Vẽ bounding box
│   └── report.py           # Lưu kết quả và tạo báo cáo
│
├── weights/                # Thư mục chứa mô hình YOLO
│   ├── bestyolo.pt
│   └── bestyolo.onnx
│
├── test_images/            # Ảnh test
├── reports/                # Kết quả nhận diện đầu ra
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Cài đặt & Chạy

### 1. Clone dự án
```bash
git clone https://github.com/nghiands/yolo-helmet-detections.git
cd helmet-detection
```

### 2. Tạo môi trường ảo (tùy chọn nhưng nên dùng)
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate    # macOS/Linux
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng Streamlit
```bash
streamlit run app/main.py
```

👉 Mở trình duyệt và truy cập: `http://localhost:8501`

---

## 📂 Cấu hình & Tài nguyên

- **Model YOLOv11**: đặt trong thư mục `weights/`
- **Thư mục đầu vào**:
  - Ảnh: `test_images/`
  - Video: hỗ trợ định dạng `.mp4`, `.avi`
- **Thư mục đầu ra**:
  - Lưu tại `reports/` gồm ảnh có box, báo cáo .csv

---

## 🧪 Ví dụ sử dụng

### ✔️ Phát hiện từ ảnh:
- Tải ảnh lên giao diện Streamlit
- Hệ thống sẽ trả kết quả đã gắn box

### ✔️ Phát hiện từ video:
- Tải video lên
- Hệ thống xử lý và lưu video + kết quả

### ✔️ Phát hiện từ webcam:
- Nhấn nút kích hoạt webcam trực tiếp

---

## 📄 License & Tác giả

- **Giấy phép**: MIT License
- **Tác giả**: Nguyễn Văn Nghĩa  
- **Email**: vannghiands@gmail.com  
- **GitHub**: https://github.com/nghiands

---

## 🌱 Hướng phát triển tương lai

- 📸 Nhận diện kết hợp với **biển số xe**
- 📊 Thống kê dữ liệu vi phạm theo thời gian
- ☁️ Kết nối lưu dữ liệu lên **SQL hoặc Firebase**
- 📱 Tích hợp ứng dụng điện thoại cảnh báo vi phạm
