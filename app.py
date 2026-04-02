import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. Cấu hình giao diện trang web
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")
st.title("Hệ thống Nhận diện Khẩu trang")
st.write("Dự án áp dụng mô hình YOLO để nhận diện khuôn mặt có đeo khẩu trang hay không.")

# 2. Khởi tạo mô hình với cơ chế Cache
# Bản chất: Hàm @st.cache_resource giúp Streamlit lưu mô hình vào bộ nhớ đệm (RAM). 
# Nếu không có decorator này, mỗi khi người dùng thao tác trên web, ứng dụng sẽ tải lại file weights từ đầu, gây lag.
@st.cache_resource
def load_model():
    # Khuyến nghị trỏ đến file ONNX để chạy mượt nhất trên CPU của Cloud
    return YOLO("models/best.onnx")

model = load_model()

# 3. Luồng xử lý giao diện người dùng
# Tải ảnh từ máy tính (Client) lên Server
uploaded_file = st.file_uploader("Tải ảnh cần kiểm tra lên...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc byte ảnh và chuyển thành định dạng PIL
    image = Image.open(uploaded_file)
    
    # Hiển thị ảnh gốc
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    # Nút bấm kích hoạt luồng suy luận
    if st.button("Bắt đầu Nhận diện"):
        with st.spinner("Đang xử lý dữ liệu..."):
            # Chuyển đổi định dạng: PIL (RGB) -> NumPy Array -> OpenCV (BGR)
            # Bản chất: YOLO và OpenCV xử lý ma trận ảnh ở hệ màu BGR
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Chạy dự đoán (Inference)
            results = model.predict(image_cv, conf=0.25)
            
            # Trích xuất ma trận ảnh đã vẽ bounding box từ kết quả
            res_plotted = results[0].plot()
            
            # Chuyển đổi ngược lại từ BGR sang RGB để Streamlit render đúng màu
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Hiển thị kết quả
            st.image(res_rgb, caption="Kết quả nhận diện", use_container_width=True)
            st.success("Hoàn tất!")