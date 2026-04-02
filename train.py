from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data=r"data\face-mask-5\data.yaml",
    epochs=100,          
    patience=10,         # Early Stopping
    imgsz=640,           
    batch=32,            
    device=0,            
    optimizer='auto',   
    lr0=0.01,            
    fliplr=0.5,             #Lật ngang ảnh với xác suất 50%.
    flipud=0.0,             # KHÔNG lật dọc 
    mosaic=1.0,             # Ghép 4 ảnh thành 1, giúp mô hình học các vật thể ở nhiều tỷ lệ khác nhau
    hsv_h=0.015,            # Biến thiên sắc độ.
    hsv_s=0.7,              # Biến thiên độ bão hòa.
    hsv_v=0.4,              # Biến thiên độ sáng
    project = "first_model"
)