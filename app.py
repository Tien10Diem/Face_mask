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
    return YOLO("models/best.onnx")

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
    sample_image_dir = "data/face-mask-5/train/images" 
    sample_label_dir = "data/face-mask-5/train/labels" # Thêm thư mục nhãn để phân loại
    
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
elif page == "Đánh giá & Hiệu năng":
    st.title("Trang 3: Đánh giá & Hiệu năng")
    
    # 1. Các chỉ số đo lường hiệu năng
    st.subheader("1. Các chỉ số đo lường tổng quan (Tập Validation)")
    st.markdown("Hiệu năng của mô hình YOLO26 trên tập Validation (Tổng số 3178 đối tượng):")
    
    m1, m2, m3, m4 = st.columns(4)
    # Hiển thị chỉ số trung bình (all) của cả 2 lớp
    m1.metric("Precision", "0.923")
    m2.metric("Recall", "0.89")
    m3.metric("mAP@50", "0.929")
    m4.metric("mAP@50-95", "0.695")
    
    # 2. Biểu đồ kỹ thuật
    st.subheader("2. Biểu đồ kỹ thuật")
    tab1, tab2 = st.tabs(["Ma trận nhầm lẫn (Confusion Matrix)", "Đồ thị Huấn luyện (Loss/Metrics)"])
    
    with tab1:
        st.markdown("**Ma trận nhầm lẫn:** Thể hiện chi tiết số lượng dự đoán đúng/sai trên tập Validation.")
        
        # Sửa đường dẫn trỏ về đúng thư mục val
        cm_path = r"confusion_matrix.png" 
        
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix (Tập Validation)", use_container_width=True)
        else:
            st.info(f"Vui lòng lưu ảnh Confusion Matrix chuẩn vào đường dẫn: `{cm_path}` để hiển thị.")
            
    with tab2:
        st.markdown("**Đồ thị Huấn luyện:** Theo dõi sự hội tụ của hàm Loss và sự gia tăng của các chỉ số mAP, Precision, Recall qua các kỷ nguyên (Epochs).")
        
        results_path = r"results.png"
        
        if os.path.exists(results_path):
            st.image(results_path, caption="Đồ thị Loss/Metrics trong quá trình huấn luyện", use_container_width=True)
        else:
            st.info(f"Vui lòng lưu ảnh Đồ thị kết quả vào đường dẫn: `{results_path}` để hiển thị.")
    
    # 3. Phân tích sai số và hướng cải thiện
    st.subheader("3. Nhận định & Phân tích chuyên sâu")
    st.warning("""
    **Phân tích học thuật dựa trên Ma trận nhầm lẫn thực tế:**
    * **Độ chính xác nội lớp (True Positives):** Mô hình phân loại xuất sắc lớp `mask` với 1013/1097 trường hợp đúng (chỉ số mAP@50 đạt 0.983).
    * **Vấn đề bỏ sót đối tượng (False Negatives - Lỗi Background):** Đây là điểm yếu lớn nhất của mô hình hiện tại. Nhìn vào hàng `background`, có tới **337** khuôn mặt `nomask` và **14** khuôn mặt `mask` bị mô hình bỏ qua hoàn toàn (không vẽ bounding box). Con số 337 lỗi này chính là nguyên nhân trực tiếp kéo chỉ số Recall của lớp `nomask` xuống mức thấp (0.814).
    * **Nhầm lẫn giữa các lớp (Class Confusion):** Tỷ lệ phân loại nhầm giữa 2 lớp khá thấp. Tuy nhiên, mô hình có xu hướng nhầm người đeo khẩu trang thành không đeo (70 trường hợp) nhiều hơn là nhầm mặt trần thành có khẩu trang (18 trường hợp).
    * **Báo động giả (False Positives):** Có 110 cảnh vật bị nhận diện nhầm thành `mask` và 123 cảnh vật bị nhầm thành `nomask`.

    **Hướng cải thiện:**
    * **Giải quyết lỗi Background:** Để giảm thiểu con số 337 trường hợp bị bỏ sót, cần bổ sung thêm dữ liệu huấn luyện chứa các khuôn mặt `nomask` có kích thước nhỏ (ở xa), bị che khuất một phần hoặc chụp trong điều kiện ánh sáng yếu, màu sắc tương đồng da người.
    * **Tinh chỉnh kỹ thuật:** Cân nhắc hạ nhẹ ngưỡng Confidence Threshold trong thực tế triển khai để "vớt" lại các khuôn mặt bị bỏ sót. Đồng thời, cấu hình tăng cường các kỹ thuật Data Augmentation (như Mosaic, Zoom out, Random Crop) trong quá trình huấn luyện để ép YOLO26 học cách trích xuất đặc trưng của các vật thể cực nhỏ.
    """)