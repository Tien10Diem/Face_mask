from ultralytics import YOLO

def evaluate_test_set():

    model_path = r"models\best.pt"
    model = YOLO(model_path)

    data_yaml = r"data\face-mask-5\data.yaml" 


    metrics = model.val(
        data=data_yaml,
        split='val',      
        imgsz=640,         
        device='',         # Để trống là tự động chọn GPU nếu có, không thì chạy CPU. Có thể gán '0' cho GPU số 0.
        plots=True         # Cho phép sinh ra các biểu đồ đánh giá (Confusion Matrix, PR curve,...)
    )

    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ CHÍNH:")
    print(f"Precision (Toàn bộ class): {metrics.box.mp:.4f}")   # Bổ sung Precision
    print(f"Recall    (Toàn bộ class): {metrics.box.mr:.4f}")   # Bổ sung Recall
    print(f"mAP@50    (Toàn bộ class): {metrics.box.map50:.4f}")
    print(f"mAP@50-95 (Toàn bộ class): {metrics.box.map:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_test_set()