import cv2
import os
import numpy as np
from ultralytics import YOLO


model_path = r"models\best_openvino_model"
model = YOLO(model_path)

TARGET_WIDTH = 640
TARGET_HEIGHT = 640

print("Đang khởi động mô hình...")
dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
for _ in range(10):
    model.predict(dummy_input, verbose=False)

def process_source(source_path):

    ext = os.path.splitext(source_path)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']


    if ext in image_extensions:
        print(f"Đang xử lý ảnh: {source_path}")
        results = model.predict(source_path, imgsz=640, conf=0.5)
        
        for r in results:
            annotated_frame = r.plot()
            resized_frame = cv2.resize(annotated_frame, (TARGET_WIDTH, TARGET_HEIGHT))
            cv2.imshow("Detection Result", resized_frame)
            print("Nhấn phím bất kỳ để đóng ảnh...")
            cv2.waitKey(0) 

    # TRƯỜNG HỢP 2: XỬ LÝ VIDEO (CẦN TRACKING)
    elif ext in video_extensions:
        print(f"Đang xử lý video: {source_path}")
        # Sử dụng .track để duy trì ID của đối tượng qua các frame
        results = model.track(source_path, 
                               show=False, 
                               tracker="bytetrack.yaml", 
                               imgsz=640, 
                               conf=0.5, 
                               stream=True) 

        for r in results:
            annotated_frame = r.plot()
            resized_frame = cv2.resize(annotated_frame, (TARGET_WIDTH, TARGET_HEIGHT))
            cv2.imshow("Detect face mask", resized_frame)

            # Chờ 1ms để duy trì luồng video. Nhấn 'q' để thoát.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("Định dạng file không hỗ trợ.")

    cv2.destroyAllWindows()


input_path = r"inference_data\img5.jpg" 
process_source(input_path)