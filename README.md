**Realtime detection with iVCam (phone as webcam)**

- **File**: `realtime_ivcam.py`
- **Model**: default path set to `C:\HỌC TRÊN TRƯỜNG\NĂM 4\kỳ 1\MyProject\ObjDetection\detect_pen_pencil\weights\best.pt`

Quick steps:

1. Install dependencies (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install ultralytics opencv-python
# If you have a GPU, install a matching torch build first (see https://pytorch.org)
```

2. Run iVCam on your phone and connect to the PC. iVCam exposes a webcam device (try device index 0,1,...).

3. Run the realtime script (try device index 0):

```powershell
cd "c:\HỌC TRÊN TRƯỜNG\NĂM 4\kỳ 1\MyProject\ObjDetection\detect_pen_pencil"
python realtime_ivcam.py --device 0
```

Optional: save output video and adjust confidence:

```powershell
python realtime_ivcam.py --device 0 --save --output result.mp4 --conf 0.3
```

Troubleshooting:
- If the camera doesn't open, try different device indexes (0,1,2...).
- If `ultralytics` requires `torch` not installed automatically, install it with the appropriate CUDA/CPU wheel from https://pytorch.org.
- For lower latency, reduce `--imgsz` to 320 or lower.
