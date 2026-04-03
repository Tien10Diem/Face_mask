# ObjectDetection_FaceMask

Project phát hiện đối tượng `mask` / `nomask` bằng YOLO, gồm script huấn luyện - đánh giá - suy luận và ứng dụng Streamlit để demo.

## Mô tả nhanh

- Dataset YOLO nằm tại `data/face-mask-5/`
- Nhãn lớp theo `data.yaml`:
  - `0: mask`
  - `1: nomask`
- Các model đang sử dụng trong project:
  - `models/best.pt`
  - `models/best.onnx`

## Cấu trúc chính

```text
ObjectDetection_FaceMask/
|-- app.py
|-- train.py
|-- test.py
|-- requirements.txt
|-- data/face-mask-5/
|   |-- data.yaml
|   |-- train/
|   |-- valid/
|   `-- test/
|-- models/
|   |-- best.pt
|   `-- best.onnx
|-- confusion_matrix.png
`-- results.png
```

## Cài đặt môi trường

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Huấn luyện mô hình

Chạy:

```bash
python train.py
```

`train.py` hiện đang train với:

- dataset config: `data\face-mask-5\data.yaml`
- epochs: `100`
- patience: `10`
- imgsz: `640`
- batch: `32`
- device: `0`

## Đánh giá trên tập test

Chạy:

```bash
python test.py
```

`test.py` đang dùng:

- model: `models\best.pt`
- split: `test`
- `plots=True` để xuất biểu đồ đánh giá

Chỉ số in ra terminal:

- Precision
- Recall
- mAP@50
- mAP@50-95

## Chạy web app Streamlit

Chạy:

```bash
streamlit run app.py
```

`app.py` hiện load model ONNX tại `models/best.onnx`, có 3 trang:

- Giới thiệu và EDA
- Triển khai nhận diện ảnh/video
- Đánh giá hiệu năng

## Phụ thuộc chính

Xem trong `requirements.txt`, nổi bật:

- `ultralytics==8.4.33`
- `onnxruntime==1.24.4`
- `streamlit`
- `opencv-python-headless`
- `numpy<2.0.0`
