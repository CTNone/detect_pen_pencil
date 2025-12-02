# Nhận diện Bút chì và Bút bi (Pen & Pencil Detection)

README này mô tả ngắn gọn nội dung và cách sử dụng code/data trong bài toán của tôi.

**Mục tiêu**
- Xây dựng mô hình phát hiện đối tượng (YOLOv8) để nhận diện `pen`, `pencil`.
- Hợp nhất dataset từ Roboflow, huấn luyện, đánh giá và chạy inference realtime (dùng điện thoại làm webcam qua iVCam).

**Yêu cầu**
- Hệ điều hành: Windows (hướng dẫn PowerShell bên dưới). Có thể chạy trên Linux/Colab/Kaggle với điều chỉnh đường dẫn.
- Python 3.8+ và các thư viện:
  - `ultralytics` (YOLOv8 API)
  - `opencv-python` (video & hiển thị)
  - `torch` (cài theo CPU/CUDA từ https://pytorch.org)
  - Tùy chọn: `roboflow`, `pyyaml`, `tqdm` (được dùng trong notebook)

Ví dụ cài (PowerShell):
```powershell
python -m pip install --upgrade pip
pip install ultralytics opencv-python
# Nếu cần Roboflow/tiện ích khác:
pip install roboflow pyyaml tqdm
# Cài torch riêng nếu ultralytics không tự cài (chọn wheel phù hợp GPU/CPU):
# xem https://pytorch.org
```

**File & script chính**
- `File gốc.ipynb` — notebook chính: cài thư viện, tải dataset từ Roboflow, merge dataset, tạo `data.yaml`, huấn luyện, đánh giá, hiển thị sample predictions và xử lý video.
- `weights/best.pt` — file model đã train (đường dẫn mẫu trên máy bạn):
  `C:\HỌC TRÊN TRƯỜNG\NĂM 4\kỳ 1\MyProject\ObjDetection\detect_pen_pencil\weights\best.pt`
- `realtime_ivcam.py` — script Python để infer realtime từ webcam do iVCam (phone) cung cấp.

**Cấu trúc thư mục (mô tả ngắn)**
- `weights/` — chứa `best.pt`, `last.pt` nếu có.
- `Dataset/`, `TEST/` — (nếu có) chứa data gốc.
- Notebook: `File gốc.ipynb` chứa toàn bộ pipeline (download, merge, train, eval, visualize).

Hướng dẫn sử dụng (Quick start)

1) Chuẩn bị mô hình/weights
- Nếu bạn đã có `weights/best.pt` (đường dẫn như trên), đặt file đó vào `weights/` như đã nêu hoặc chỉnh lại tham số `--model` khi chạy script.

2) Chạy realtime inference (iVCam)
- Kết nối iVCam trên điện thoại với máy Windows — iVCam sẽ hiện như một webcam DirectShow.
- Chạy script realtime (PowerShell):
```powershell
cd "c:\HỌC TRÊN TRƯỜNG\NĂM 4\kỳ 1\MyProject\ObjDetection\detect_pen_pencil"
python realtime_ivcam.py --device 0
```
- Thử đổi `--device` sang `1`, `2`, ... nếu camera không mở.
- Lưu video kết quả:
```powershell
python realtime_ivcam.py --device 0 --save --output result.mp4 --conf 0.3
```

3) Chạy huấn luyện (notebook / CLI)
- Notebook đã chứa ví dụ dùng API `ultralytics.YOLO` để train. Nếu muốn dùng CLI (yolo command), ví dụ:
```powershell
# Ví dụ dùng yolov8m pretrained (nếu có) và data.yaml đã chuẩn bị
yolo task=detect mode=train model=yolov8m.pt data="path/to/data.yaml" epochs=100 imgsz=640 batch=32
```
- Trong `File gốc.ipynb` có đoạn merge dataset và sinh `data.yaml` (đường dẫn trong notebook đang dùng `/kaggle/working/...` — bạn cần sửa về đường dẫn trên máy local nếu chạy offline).

4) Đánh giá trên test set
- Ví dụ dùng API (đã có trong notebook):
```python
from ultralytics import YOLO
model = YOLO(r"C:\...\weights\best.pt")
metrics = model.val(data=r"path\to\data.yaml", split='test')
print(metrics)
```

5) Hiển thị sample predictions
- Notebook có cell mẫu: chọn vài ảnh test, dùng `model.predict(...).plot()` rồi hiển thị bằng `matplotlib`.

Ghi chú về đường dẫn và môi trường
- Notebook ban đầu dùng đường dẫn Colab/Kaggle như `/kaggle/working/...`. Trên máy Windows bạn cần chỉnh các biến `dataset`/`merged`/`training_dir` sang đường dẫn tương ứng trên Windows.
- Model mặc định trong `realtime_ivcam.py` đã set sẵn đến đường dẫn `weights/best.pt` như trên; nếu bạn để model ở vị trí khác, truyền `--model "đường_dẫn"` khi chạy.

Tối ưu & Troubleshooting
- Nếu inference chậm: giảm `--imgsz` (ví dụ 320), hoặc chạy trên GPU với phiên bản `torch` có CUDA.
- Nếu `ultralytics` báo thiếu `torch`, cài `torch` theo hướng dẫn chính thức (phiên bản phù hợp CUDA).
- Nếu webcam không mở: kiểm tra iVCam đang chạy, thử device index khác, hoặc mở Device Manager/Camera để xem tên thiết bị.
- Để xem log/training outputs: mở thư mục `runs/detect/<run_name>` sau khi train.

