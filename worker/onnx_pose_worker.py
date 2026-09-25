import cv2
import numpy as np
import onnxruntime as ort

class ONNXPoseWorker:
    def __init__(self, model_path="models/yolo26n-pose.onnx", conf_thresh=0.35, iou_thresh=0.45, imgsz=416):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        # Cấu hình ONNX Runtime tối ưu cho CPU ARM
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # Tận dụng 4 core hiệu năng cao của điện thoại
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, im, new_shape=(416, 416), color=(114, 114, 114)):
        shape = im.shape[:2]  # shape hiện tại [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2  # Chia đều padding 2 bên

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def process_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        
        # 1. Preprocessing (Letterbox + Chuyển kênh màu + Chuẩn hóa)
        img, ratio, (dw, dh) = self._letterbox(frame, new_shape=(self.imgsz, self.imgsz))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)   # CHW -> 1CHW

        # 2. Suy luận ONNX
        preds = self.session.run(None, {self.input_name: blob})[0]
        preds = np.squeeze(preds, axis=0)     # Shape: [56, N]
        preds = np.transpose(preds, (1, 0))   # Đổi thành [N, 56] (xywh + conf + 17*3 kpts)

        # 3. Lọc Boxes và Điểm tin cậy (Confidence)
        boxes, confidences, keypoints_list = [], [], []
        
        for pred in preds:
            conf = pred[4]
            if conf < self.conf_thresh:
                continue

            # Tọa độ bounding box (cx, cy, w, h) trên ảnh 320x320
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            
            # Khôi phục tọa độ về kích thước ảnh gốc
            x1 = (cx - w / 2 - dw) / ratio
            y1 = (cy - h / 2 - dh) / ratio
            box_w = w / ratio
            box_h = h / ratio

            # Tách 17 Keypoints (x, y, visibility)
            kpts_raw = pred[5:]  # 51 phần tử (17 điểm x 3)
            kpts = []
            for i in range(17):
                kx = (kpts_raw[i * 3] - dw) / ratio
                ky = (kpts_raw[i * 3 + 1] - dh) / ratio
                kconf = kpts_raw[i * 3 + 2]
                # Chuẩn hóa kx, ky về tỉ lệ [0.0 - 1.0] để tương thích với State Machine
                kpts.append((kx / orig_w, ky / orig_h, kconf))

            boxes.append([int(x1), int(y1), int(box_w), int(box_h)])
            confidences.append(float(conf))
            keypoints_list.append(kpts)

        # 4. Áp dụng Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)
        
        detected_persons = []
        if len(indices) > 0:
            for i, idx in enumerate(indices.flatten()):
                x, y, w, h = boxes[idx]
                detected_persons.append({
                    "track_id": i + 1,  # Gán ID tuần tự
                    "box": [x, y, x + w, y + h],
                    "keypoints": keypoints_list[idx],
                    "confidence": confidences[idx]
                })

        return detected_persons


if __name__ == "__main__":
    import os
    import sys
    import time

    # Tắt thông báo warning quét GPU của ONNX Runtime
    os.environ["ORT_LOGGING_LEVEL"] = "3"

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from producer.video_stream import PhoneCameraProducer

    print("==================================================")
    print("  KHOI DONG HE THONG ONNX (HEADLESS MODE)")
    print("  Khong giao dien Web - Toi uu 100% CPU")
    print("==================================================")

    producer = PhoneCameraProducer().start()
    # Kích thước 416x416 chuẩn với model ONNX hiện tại của bạn
    worker = ONNXPoseWorker(model_path="models/yolo26n-pose.onnx", conf_thresh=0.25, imgsz=416)

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            frame = producer.get_latest_frame(timeout=1.0)
            if frame is None:
                time.sleep(0.01)
                continue

            # Chạy suy luận trực tiếp bằng ONNX
            t0 = time.time()
            persons = worker.process_frame(frame)
            infer_time = (time.time() - t0) * 1000

            frame_count += 1
            now = time.time()

            # Mỗi 1 giây in tốc độ FPS và thời gian suy luận một lần
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                print(f"[AI RUNNING] FPS: {fps:.1f} | Latency: {infer_time:.1f}ms | So nguoi: {len(persons)}")
                frame_count = 0
                prev_time = now

    except KeyboardInterrupt:
        print("\nDa dung he thong.")    

    