# from ultralytics import YOLO

# model = YOLO(r"models\best.pt")

# path = model.export(
#     format="onnx",
#     imgsz=640,       # Cố định kích thước ảnh đầu vào để tối ưu đồ thị tĩnh
#     half=False,      # Đặt True nếu muốn dùng FP16 (Half-precision) để tăng tốc độ hơn nữa trên GPU
#     dynamic=False,   # Đặt False để khóa cứng kích thước batch và ảnh, giúp ONNX Runtime biên dịch nhanh nhất
#     simplify=True    # BẢN CHẤT: Gọi thư viện onnx-simplifier để tự động gộp node (Node Fusion) và loại bỏ các toán tử thừa
# )

# print(f"Quá trình chuyển đổi hoàn tất. File ONNX được lưu tại: {path}")


from ultralytics import YOLO

model = YOLO(r"models\best.pt")
# Lệnh này sẽ sinh ra một thư mục chứa file .xml và .bin của OpenVINO
model.export(format="openvino", imgsz=640, half=False)