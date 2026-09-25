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
    import math
    from collections import deque

    os.environ["ORT_LOGGING_LEVEL"] = "3"
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from producer.video_stream import PhoneCameraProducer

    # ==================== CẤU HÌNH LOGIC TÉ NGÃ ====================
    ANGLE_THRESHOLD = 50           # Góc nghiêng thân người (>50 độ)
    SPEED_THRESHOLD = 0.20         # Tốc độ rơi đột ngột
    CONFIRM_DURATION = 2.5         # Bất động 2.5s để xác nhận ngã
    MOVEMENT_TOLERANCE = 0.03      
    ASPECT_RATIO_THRESHOLD = 0.88  # W/H >= 0.88 (nằm ngang)

    STATE_NORMAL = "NORMAL"
    STATE_SUSPECTED = "FALL_SUSPECTED"
    STATE_CONFIRMED = "FALL_CONFIRMED"

    class PersonTracker:
        def __init__(self, track_id):
            self.track_id = track_id
            self.state = STATE_NORMAL
            self.suspected_start_time = None
            self.suspected_pos = None
            self.confirmed_time = None

            self.position_history = deque(maxlen=5)
            self.time_history = deque(maxlen=5)
            self.speed_history = deque(maxlen=15)
            self.angle_history = deque(maxlen=5)
            self.last_seen = time.time()
            self.aspect_ratio = 0.0

        def reset_to_normal(self):
            self.state = STATE_NORMAL
            self.suspected_start_time = None
            self.suspected_pos = None
            self.confirmed_time = None
            self.position_history.clear()
            self.time_history.clear()
            self.speed_history.clear()

        def update(self, keypoints, box, current_time):
            self.last_seen = current_time
            ls, rs = keypoints[5], keypoints[6]
            lh, rh = keypoints[11], keypoints[12]
            la, ra = keypoints[15], keypoints[16]

            if ls[2] < 0.20 or rs[2] < 0.20 or lh[2] < 0.20 or rh[2] < 0.20:
                return None, 0.0, self.aspect_ratio

            bw = abs(box[2] - box[0])
            bh = abs(box[3] - box[1])
            self.aspect_ratio = bw / (bh + 1e-6)

            mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)

            dx = mid_hip[0] - mid_shoulder[0]
            dy = mid_hip[1] - mid_shoulder[1]
            body_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

            has_ankles = (la[2] > 0.20 and ra[2] > 0.20)
            mid_ankle_y = (la[1] + ra[1]) / 2 if has_ankles else None
            hip_ankle_drop = (mid_ankle_y - mid_hip[1]) if mid_ankle_y is not None else 0.0

            centroid = ((ls[0] + rs[0] + lh[0] + rh[0]) / 4,
                        (ls[1] + rs[1] + lh[1] + rh[1]) / 4)

            self.position_history.append(centroid)
            self.time_history.append(current_time)
            self.angle_history.append(body_angle)

            speed = 0.0
            if len(self.position_history) >= 2:
                total_dist = 0.0
                for i in range(1, len(self.position_history)):
                    p1 = self.position_history[i - 1]
                    p2 = self.position_history[i]
                    total_dist += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                total_time = self.time_history[-1] - self.time_history[0]
                if total_time > 0:
                    speed = total_dist / total_time

            self.speed_history.append(speed)
            recent_max_speed = max(self.speed_history) if self.speed_history else 0.0

            is_bounding_box_horizontal = self.aspect_ratio >= ASPECT_RATIO_THRESHOLD
            is_standing_bent = (self.aspect_ratio < 0.75) or (has_ankles and hip_ankle_drop > 0.35)
            is_true_lying = (body_angle > ANGLE_THRESHOLD) and (is_bounding_box_horizontal or body_angle > 75) and (not is_standing_bent)

            if self.state == STATE_NORMAL:
                had_fast_motion = recent_max_speed > SPEED_THRESHOLD
                if is_true_lying and (had_fast_motion or (body_angle > 70 and is_bounding_box_horizontal)):
                    self.state = STATE_SUSPECTED
                    self.suspected_start_time = current_time
                    self.suspected_pos = centroid
                    print(f"--> [NGHI NGO TE NGA] ID {self.track_id} - Goc: {body_angle:.0f} do - W/H: {self.aspect_ratio:.2f}")

            elif self.state == STATE_SUSPECTED:
                elapsed = current_time - self.suspected_start_time
                dx_m = centroid[0] - self.suspected_pos[0]
                dy_m = centroid[1] - self.suspected_pos[1]
                moved_dist = math.sqrt(dx_m**2 + dy_m**2)

                if body_angle < 38 or self.aspect_ratio < 0.70 or moved_dist > MOVEMENT_TOLERANCE * 3:
                    self.reset_to_normal()
                    print(f"--> [HUY NGHI NGO] ID {self.track_id} tro lai binh thuong.")
                elif elapsed >= CONFIRM_DURATION:
                    if is_true_lying:
                        self.state = STATE_CONFIRMED
                        if self.confirmed_time is None:
                            self.confirmed_time = current_time
                            print(f"\n🚨🚨🚨 [CANH BAO NGUY HIEM] ID {self.track_id} XAC NHAN TE NGA! 🚨🚨🚨\n")
                    else:
                        self.reset_to_normal()

            elif self.state == STATE_CONFIRMED:
                if body_angle < 38 or self.aspect_ratio < 0.70:
                    print(f"--> [THONG TIN] ID {self.track_id} da dung day.")
                    self.reset_to_normal()

            return body_angle, speed, self.aspect_ratio

    print("==================================================")
    print("  KHOI DONG ONNX FALL DETECTOR (HEADLESS)")
    print("  He thong dang theo doi te nga...")
    print("==================================================")

    producer = PhoneCameraProducer().start()
    worker = ONNXPoseWorker(model_path="models/yolo26n-pose.onnx", conf_thresh=0.25, imgsz=416)
    trackers = {}

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            frame = producer.get_latest_frame(timeout=1.0)
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            persons = worker.process_frame(frame)
            frame_count += 1

            for p in persons:
                tid = p["track_id"]
                if tid not in trackers:
                    trackers[tid] = PersonTracker(tid)
                tracker = trackers[tid]
                tracker.update(p["keypoints"], p["box"], now)

            # Don dep tracker cu (>10s)
            expired = [t for t, obj in trackers.items() if now - obj.last_seen > 10.0]
            for t in expired:
                del trackers[t]

            # In thong so dinh ky moi giay
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                print(f"[RUNNING] FPS: {fps:.1f} | Tracking: {len(trackers)}")
                frame_count = 0
                prev_time = now

    except KeyboardInterrupt:
        print("\nDa dung he thong.")