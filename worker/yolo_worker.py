import torch
import numpy as np
from ultralytics import YOLO

class YOLOPoseWorker:
    def __init__(self, model_path="models/yolo26n-pose.pt", conf_thresh=0.25, imgsz=640):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WORKER] Loading model {model_path} tren thiet bi: {self.device}...")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Warm-up model để khởi tạo đồ thị tính toán
        dummy_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy_img, imgsz=self.imgsz, device=self.device, verbose=False)
        print("[WORKER] Model da san sang.")

    def process_frame(self, frame):
        """
        Nhan vao: ma tran frame (BGR numpy array)
        Tra ve: Danh sach cac person info:
        [
            {
                "track_id": int,
                "box": [x1, y1, x2, y2],
                "keypoints": ndarray shape (17, 3) -> [x_norm, y_norm, conf]
            }, ...
        ]
        """
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            device=self.device,
            classes=[0]  # Chi nhan dien nguoi (Person)
        )

        detected_persons = []

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            kpts_xyn = results[0].keypoints.xyn.cpu().numpy()  # [N, 17, 2]
            
            # Neu khong co conf cua keypoint thi tao mac dinh 1.0
            if results[0].keypoints.conf is not None:
                kpts_conf = results[0].keypoints.conf.cpu().numpy()
            else:
                kpts_conf = np.ones((len(boxes), 17))

            for box, track_id, xy, conf in zip(boxes, track_ids, kpts_xyn, kpts_conf):
                # Gop toa do [x, y] va do tin cay [conf] thanh ma tran (17, 3)
                keypoints_data = np.hstack([xy, conf[:, None]])
                
                detected_persons.append({
                    "track_id": int(track_id),
                    "box": box.tolist(),
                    "keypoints": keypoints_data
                })

        return detected_persons