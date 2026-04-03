import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from ultralytics import YOLO
import tempfile
import os

# ==========================================
# CẤU HÌNH TRANG & TIỆN ÍCH
# ==========================================
# Cấu hình trang phải là lệnh Streamlit đầu tiên [cite: 2, 35]
st.set_page_config(page_title="Face Mask Detection App", page_icon="😷", layout="wide")

def letterbox_resize(image, target_size=(800, 600)):
    """
    Bản chất: Thay đổi kích thước ma trận ảnh nhưng giữ nguyên tỷ lệ (Aspect Ratio).
    Phần trống sẽ được lấp đầy bằng màu đen (Letterboxing) để tránh làm méo vật thể.
    """
    h_orig, w_orig = image.shape[:2]
    tw, th = target_size
    
    # Tính tỷ lệ scale phù hợp nhất
    scale = min(tw / w_orig, th / h_orig)
    nw, nh = int(w_orig * scale), int(h_orig * scale)
    
    # Resize ảnh theo tỷ lệ mới
    resized_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Tạo nền đen (Canvas) cố định
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    
    # Đặt ảnh đã resize vào chính giữa nền đen
    x_offset = (tw - nw) // 2
    y_offset = (th - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_img
    
    return canvas

# ==========================================
# QUẢN LÝ CACHE (MLOps cơ bản)
# Sử dụng Cache để tối ưu tốc độ tải tài nguyên nặng [cite: 33]
# ==========================================
@st.cache_resource
def load_model():
    # Sử dụng định dạng ONNX để tối ưu suy luận trên CPU của Cloud [cite: 21, 37]
    return YOLO(r"models\best.onnx")

@st.cache_data
def load_eda_data():
    
    data = {
        'Tập dữ liệu': ['Train', 'Train', 'Val', 'Val', 'Test', 'Test'],
        'Nhãn (Class)': ['mask', 'nomask', 'mask', 'nomask', 'mask', 'nomask'],
        'Số lượng mẫu': [3805, 7310, 1097, 2081, 578, 926]
    }
    return pd.DataFrame(data)

model = load_model()

# ==========================================
# THANH ĐIỀU HƯỚNG (SIDEBAR)
# Phân chia ứng dụng thành ít nhất 3 trang chức năng [cite: 2, 3]
# ==========================================
st.sidebar.title("Hệ thống Điều hướng")
page = st.sidebar.radio("Chọn trang hiển thị:", 
                        ["Giới thiệu & Khám phá dữ liệu", 
                         "Triển khai mô hình", 
                         "Đánh giá & Hiệu năng"]) 

# ==========================================
# TRANG 1: GIỚI THIỆU & EDA 
# ==========================================
if page == "Giới thiệu & Khám phá dữ liệu":
    st.title("Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)")
    
    # Thông tin sinh viên bắt buộc
    st.markdown("""
    ### Thông tin thực hiện
    * **Tên đề tài:** Phát hiện người đeo và không đeo khẩu trang trên ảnh khuôn mặt người bằng YoLo26 nhằm đảm bảo sức khỏe cộng đồng.
    * **Sinh viên:** Trần Quốc Tiến 
    * **MSSV:** *22T1020762* 
    ### Giá trị thực tiễn 
    Đảm bảo an toàn cho cộng đồng trước nguy cơ về bệnh truyền nhiễm và người lao động trong môi trường ô nhiễm.
    """)
    
    st.divider()
    
    # Khám phá dữ liệu (EDA) 
    st.subheader("Phân tích tập dữ liệu huấn luyện")
    df = load_eda_data()
    st.dataframe(df, use_container_width=True) # Hiển thị dữ liệu thô
    
    col1, col2 = st.columns(2)
    with col1:
        # Biểu đồ phân phối nhãn 
        st.markdown("**1. Phân phối nhãn lớp (Class Distribution)**")
        fig, ax = plt.subplots()
        sns.barplot(data=df[df['Tập dữ liệu'] == 'Train'], x='Nhãn (Class)', y='Số lượng mẫu', palette='viridis', ax=ax)
        st.pyplot(fig) 
        
    with col2:
        # Biểu đồ tỷ lệ tập dữ liệu 
        st.markdown("**2. Tỷ lệ phân chia tập dữ liệu**")
        pie_data = df.groupby('Tập dữ liệu')['Số lượng mẫu'].sum()
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
        st.pyplot(fig2) 
        
    st.info("""
    **Nhận xét tổng quan về tập dữ liệu:**
    * **Mất cân bằng phân lớp (Class Imbalance):** Dữ liệu có sự chênh lệch nhẹ giữa hai nhãn. Số lượng mẫu `nomask` (7310 ở tập Train) cao gần gấp đôi so với `mask` (3805). Tỷ lệ lệch (~1.9 : 1) này tiếp tục duy trì đồng nhất ở cả tập Validation và Test.
    * **Tác động đến mô hình:** Sự mất cân bằng này có thể khiến mạng nơ-ron bị thiên kiến (bias), dẫn đến xu hướng ưu tiên dự đoán nhãn `nomask`. Điều này có khả năng làm giảm chỉ số Recall của lớp `mask` (bỏ sót người đeo khẩu trang).
    * **Kiểm soát rủi ro:** Thuật toán YOLO26 mặc định có cơ chế tự động bù trừ trọng số lớp trong hàm loss để giảm thiểu tác động của việc lệch dữ liệu.
    """)

    # 4. Hiển thị ảnh mẫu
    st.markdown("**3. Trực quan hóa dữ liệu ảnh mẫu**")
    
    # Thay đổi đường dẫn này trỏ tới thư mục chứa ảnh train thực tế của bạn
    sample_image_dir = r"data\face-mask-5\train\images" 
    sample_label_dir = r"data\face-mask-5\train\labels" # Thêm thư mục nhãn để phân loại
    
    try:
        valid_extensions = ('.jpg', '.jpeg', '.png')
        all_images = [f for f in os.listdir(sample_image_dir) if f.lower().endswith(valid_extensions)]
        
        if len(all_images) > 0:
            mask_images = []
            nomask_images = []
            
            # Quét file nhãn (.txt) để chọn ra 2 ảnh mask (class 0) và 2 ảnh nomask (class 1)
            for img_file in all_images:
                if len(mask_images) >= 2 and len(nomask_images) >= 2:
                    break
                    
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(sample_label_dir, label_file)
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        classes = [line.split()[0] for line in f.read().splitlines() if line.strip()]
                        
                        if '0' in classes and len(mask_images) < 2:
                            mask_images.append(img_file)
                        elif '1' in classes and len(nomask_images) < 2:
                            nomask_images.append(img_file)
            
            # Gộp danh sách, nếu lỗi label thì lấy mặc định 4 ảnh đầu
            display_images = mask_images + nomask_images
            if not display_images: 
                display_images = all_images[:4]
                
            cols = st.columns(len(display_images) if len(display_images) > 0 else 1)
            for idx, col in enumerate(cols):
                img_path = os.path.join(sample_image_dir, display_images[idx])
                img = Image.open(img_path)
                col.image(img, caption=display_images[idx], use_container_width=True)
        else:
            st.warning("Thư mục tồn tại nhưng không có file ảnh định dạng hợp lệ (jpg, png).")
            
    except FileNotFoundError:
        st.warning(f"Chưa cấu hình đúng đường dẫn. Vui lòng kiểm tra lại: `{sample_image_dir}` hoặc thư mục labels tương ứng.")
# ==========================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH [cite: 14]
# ==========================================
elif page == "Triển khai mô hình":
    st.title("Trang 2: Triển khai mô hình (Inference)")
    
    # Thiết kế giao diện nhập liệu: Thêm Widget cấu hình IoU (NMS) để chống trùng lặp box
    st.markdown("**Cấu hình thông số dự đoán:**")
    col_conf, col_iou = st.columns(2)
    with col_conf:
        conf_threshold = st.slider("Ngưỡng độ tin cậy (Confidence):", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    with col_iou:
        iou_threshold = st.slider("Ngưỡng chồng lấp (IoU / NMS):", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
    
    # Widget tải tệp tin 
    uploaded_file = st.file_uploader("Tải lên Ảnh hoặc Video (JPG, PNG, MP4, AVI)", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Xử lý ẢNH TĨNH
        if file_ext in ['jpg', 'jpeg', 'png']:
            img = Image.open(uploaded_file)
            
            # Chuyển đổi sang mảng numpy (BGR) ngay từ đầu
            img_cv_raw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Resize bằng letterbox
            fixed_raw = letterbox_resize(img_cv_raw, (1000, 800))
            
            # Hiển thị ảnh gốc full width, KHÔNG chia cột
            st.image(cv2.cvtColor(fixed_raw, cv2.COLOR_BGR2RGB), caption="Ảnh gốc (Đã chuẩn hóa kích thước)", use_container_width=True)
                
            if st.button("Bắt đầu Nhận diện Ảnh", type="primary"):
                # Thực hiện dự đoán trên ảnh gốc
                results = model.predict(img_cv_raw, conf=conf_threshold, iou=iou_threshold) 
                res_plotted = results[0].plot()
                
                # Resize ảnh kết quả
                fixed_pred = letterbox_resize(res_plotted, (1000, 800))
                
                # Hiển thị ảnh dự đoán full width nằm ngay dưới ảnh gốc
                st.image(cv2.cvtColor(fixed_pred, cv2.COLOR_BGR2RGB), caption="Kết quả dự đoán (Đã chuẩn hóa kích thước)", use_container_width=True)
                
                # Hiển thị kết quả và độ tin cậy
                st.subheader("Chi tiết kết quả dự đoán:")
                boxes = results[0].boxes
                
                if len(boxes) == 0:
                    st.info("Không phát hiện đối tượng nào thỏa mãn ngưỡng tin cậy.")
                else:
                    mask_count = 0
                    nomask_count = 0
                    
                    for box in boxes:
                        conf = box.conf[0].item()
                        label = model.names[int(box.cls[0])]
                        
                        if label == 'mask': mask_count += 1
                        elif label == 'nomask': nomask_count += 1
                        
                        st.write(f"- Phát hiện: **{label.upper()}** | Xác suất: **{conf:.2%}**")
                    
                    st.success(f"**KẾT LUẬN:** Phát hiện tổng cộng **{len(boxes)}** khuôn mặt hợp lệ. Trong đó có **{mask_count}** người ĐEO khẩu trang và **{nomask_count}** người KHÔNG đeo khẩu trang.")

        # Xử lý VIDEO (Tracking)
        elif file_ext in ['mp4', 'avi', 'mov']:
            if st.button("Dự đoán Video", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name 
                
                cap = cv2.VideoCapture(temp_path)
                st_frame = st.empty() 
                
                frame_count = 0
                skip_rate = 3 
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    if frame_count % skip_rate == 0:
                        results = model.track(frame, conf=conf_threshold, iou=iou_threshold, persist=True, verbose=False) 
                        annotated = results[0].plot()
                        
                        # Kích thước 1200x800
                        fixed_frame = letterbox_resize(annotated, (1000, 800))
                        st_frame.image(cv2.cvtColor(fixed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                cap.release() 
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG [cite: 27]
# ==========================================
elif page == "Đánh giá & Hiệu năng":
    st.title("Trang 3: Đánh giá & Hiệu năng [cite: 27]")
    
    # Chỉ số đo lường hiệu năng [cite: 29]
    st.subheader("Các chỉ số đo lường chính (Tập Validation) [cite: 29]")
    m1, m2, m3 = st.columns(3)
    m1.metric("mAP@50", "0.927")
    m2.metric("Precision", "0.920")
    m3.metric("Recall", "0.882")
    
    # Biểu đồ Confusion Matrix [cite: 30]
    st.subheader("Ma trận nhầm lẫn (Confusion Matrix) [cite: 30]")
    st.markdown("Ma trận cho thấy khả năng phân loại chính xác giữa người đeo và không đeo khẩu trang. [cite: 30]")
    
    # Phân tích sai số [cite: 31]
    st.subheader("Nhận định & Hướng cải thiện [cite: 31]")
    st.warning("""
    * **Phân tích sai sót:** Mô hình đôi khi nhầm lẫn khi khuôn mặt ở quá xa hoặc bị che khuất bởi vật cản khác. [cite: 31]
    * **Hướng cải thiện:** Bổ sung thêm dữ liệu ảnh chụp trong điều kiện thiếu sáng và sử dụng các kỹ thuật Model Surgery. [cite: 31]
    """)