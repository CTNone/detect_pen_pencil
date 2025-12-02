import argparse
import time
import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description='Realtime YOLOv8 inference from iVCam (phone as webcam)')
    p.add_argument('--model', type=str, default=r'C:\HỌC TRÊN TRƯỜNG\NĂM 4\kỳ 1\MyProject\ObjDetection\detect_pen_pencil\weights\best.pt',
                   help='Path to the .pt model')
    p.add_argument('--device', type=str, default='0',
                   help='OpenCV video device index (0,1,...) or device string')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--save', action='store_true', help='Save output video')
    p.add_argument('--output', type=str, default='ivcam_output.mp4', help='Output video file when --save')
    p.add_argument('--fps', type=float, default=20.0, help='Output video FPS when saving')
    return p.parse_args()


def open_capture(device):
    # Try integer index first, then fallback to string
    try:
        idx = int(device)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    return cap


def main():
    args = parse_args()

    print(f'Loading model: {args.model}')
    model = YOLO(args.model)

    cap = open_capture(args.device)
    if not cap.isOpened():
        print('ERROR: Cannot open video device. Make sure iVCam is running and try different device index (0,1,2...).')
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video opened: {width}x{height}')

    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
        print(f'Saving output to: {args.output}')

    window_name = 'iVCam YOLO'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Frame read failed, exiting')
                break

            t0 = time.time()
            # Run inference (returns a Results object)
            results = model(frame, imgsz=args.imgsz, conf=args.conf)
            r = results[0]

            # Prepare visualization image (copy original frame)
            vis = frame.copy()

            # Get class names
            names = None
            if hasattr(model, 'names') and model.names is not None:
                names = model.names
            elif hasattr(r, 'names') and r.names is not None:
                names = r.names

            # Draw boxes and labels (class + confidence)
            try:
                boxes = r.boxes.xyxy
                confs = r.boxes.conf
                clss = r.boxes.cls
            except Exception:
                boxes = None

            if boxes is not None:
                # Convert to numpy if tensors
                try:
                    xy = boxes.cpu().numpy()
                    confs_np = confs.cpu().numpy()
                    clss_np = clss.cpu().numpy()
                except Exception:
                    xy = boxes
                    confs_np = confs
                    clss_np = clss

                for (x1, y1, x2, y2), conf, cls in zip(xy, confs_np, clss_np):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # label text
                    cls_name = str(int(cls))
                    if names is not None:
                        try:
                            cls_name = names[int(cls)]
                        except Exception:
                            cls_name = str(int(cls))

                    label = f"{cls_name} {float(conf):.2f}"

                    # text size and background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    # make background rectangle
                    cv2.rectangle(vis, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
                    # put text (black on green background)
                    cv2.putText(vis, label, (x1 + 3, y1 - 4), font, font_scale, (0, 0, 0), thickness)

            fps = 1.0 / max(1e-6, (time.time() - t0))
            cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow(window_name, vis)

            if args.save and out is not None:
                out.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
