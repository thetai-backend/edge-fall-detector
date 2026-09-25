import sys
import os
import cv2
import time
import math
import numpy as np
from collections import deque
from flask import Flask, Response

# Thêm đường dẫn gốc để import module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from producer.video_stream import PhoneCameraProducer
from worker.yolo_worker import YOLOPoseWorker

# ==================== CẤU HÌNH LOGIC TÉ NGÃ ====================
ANGLE_THRESHOLD = 50           # Góc nghiêng thân người (>50 độ)
SPEED_THRESHOLD = 0.20         # Tốc độ rơi đột ngột (tỷ lệ màn hình / giây)
CONFIRM_DURATION = 2.5         # Duy trì bất động trong 2.5s để xác nhận ngã
MOVEMENT_TOLERANCE = 0.03      # Ngưỡng dịch chuyển tối đa khi nằm
ASPECT_RATIO_THRESHOLD = 0.88  # W/H >= 0.88 (nằm ngang)

STATE_NORMAL = "NORMAL"
STATE_SUSPECTED = "FALL_SUSPECTED"
STATE_CONFIRMED = "FALL_CONFIRMED"

KEYPOINT_CONNECTIONS = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


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
        ls, rs = keypoints[5], keypoints[6]    # Vai trái, vai phải
        lh, rh = keypoints[11], keypoints[12]  # Hông trái, hông phải
        la, ra = keypoints[15], keypoints[16]  # Cổ chân trái, phải

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

        # Điều kiện nằm sàn thực sự
        is_bounding_box_horizontal = self.aspect_ratio >= ASPECT_RATIO_THRESHOLD
        is_standing_bent = (self.aspect_ratio < 0.75) or (has_ankles and hip_ankle_drop > 0.35)
        is_true_lying = (body_angle > ANGLE_THRESHOLD) and (is_bounding_box_horizontal or body_angle > 75) and (not is_standing_bent)

        # Cỗ máy trạng thái (State Machine)
        if self.state == STATE_NORMAL:
            had_fast_motion = recent_max_speed > SPEED_THRESHOLD
            if is_true_lying and (had_fast_motion or (body_angle > 70 and is_bounding_box_horizontal)):
                self.state = STATE_SUSPECTED
                self.suspected_start_time = current_time
                self.suspected_pos = centroid

        elif self.state == STATE_SUSPECTED:
            elapsed = current_time - self.suspected_start_time
            dx_m = centroid[0] - self.suspected_pos[0]
            dy_m = centroid[1] - self.suspected_pos[1]
            moved_dist = math.sqrt(dx_m**2 + dy_m**2)

            if body_angle < 38 or self.aspect_ratio < 0.70 or moved_dist > MOVEMENT_TOLERANCE * 3:
                self.reset_to_normal()
            elif elapsed >= CONFIRM_DURATION:
                if is_true_lying:
                    self.state = STATE_CONFIRMED
                    if self.confirmed_time is None:
                        self.confirmed_time = current_time
                        print(f"\n[CANH BAO] ID {self.track_id} TE NGA! Goc: {body_angle:.0f} do, W/H: {self.aspect_ratio:.2f}")
                else:
                    self.reset_to_normal()

        elif self.state == STATE_CONFIRMED:
            if body_angle < 38 or self.aspect_ratio < 0.70:
                print(f"[THONG TIN] ID {self.track_id} da dung day.")
                self.reset_to_normal()

        return body_angle, speed, self.aspect_ratio


# ==================== FLASK APP & RUNTIME ====================
app = Flask(__name__)

producer = PhoneCameraProducer().start()
worker = YOLOPoseWorker(model_path="models/yolo26n-pose.pt", conf_thresh=0.25, imgsz=640)
trackers = {}


def generate_frames():
    global trackers
    prev_time = time.time()
    fps_smooth = 15.0

    while True:
        frame = producer.get_latest_frame(timeout=1.0)
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        dt = now - prev_time
        prev_time = now
        current_fps = (1.0 / dt) if dt > 0 else 15.0
        fps_smooth = 0.85 * fps_smooth + 0.15 * current_fps

        h, w, _ = frame.shape
        persons = worker.process_frame(frame)
        any_confirmed_fall = False

        active_ids = set()
        for p in persons:
            tid = p["track_id"]
            box = p["box"]
            kpts = p["keypoints"]
            active_ids.add(tid)

            if tid not in trackers:
                trackers[tid] = PersonTracker(tid)

            tracker = trackers[tid]
            angle, speed, ar = tracker.update(kpts, box, now)
            state = tracker.state

            # Chọn màu theo trạng thái
            if state == STATE_CONFIRMED:
                box_color = (0, 0, 255)      # Đỏ
                any_confirmed_fall = True
            elif state == STATE_SUSPECTED:
                box_color = (0, 165, 255)    # Cam
            else:
                box_color = (0, 255, 0)      # Xanh lá

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Nhãn trạng thái
            tag = f"ID:{tid} | {state}"
            if state == STATE_SUSPECTED and tracker.suspected_start_time:
                remain = max(0.0, CONFIRM_DURATION - (now - tracker.suspected_start_time))
                tag += f" ({remain:.1f}s)"
            elif angle is not None:
                tag += f" | {angle:.0f}deg | W/H:{ar:.2f}"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            # Vẽ Skeleton
            for p1_idx, p2_idx in KEYPOINT_CONNECTIONS:
                if kpts[p1_idx][2] > 0.20 and kpts[p2_idx][2] > 0.20:
                    pt1 = (int(kpts[p1_idx][0] * w), int(kpts[p1_idx][1] * h))
                    pt2 = (int(kpts[p2_idx][0] * w), int(kpts[p2_idx][1] * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            # Vẽ Khớp xương
            for kpt in kpts:
                if kpt[2] > 0.20:
                    cv2.circle(frame, (int(kpt[0] * w), int(kpt[1] * h)), 3, (0, 255, 255), -1)

        # Xóa tracker cũ không xuất hiện sau 10 giây
        expired_ids = [tid for tid, trk in trackers.items() if now - trk.last_seen > 10.0]
        for tid in expired_ids:
            del trackers[tid]

        # Thanh trạng thái trên cùng
        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"Fall Detection Live | FPS: {fps_smooth:.1f}", (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracking: {len(active_ids)}", (w - 160, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Cảnh báo toàn màn hình khi có té ngã
        if any_confirmed_fall:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            cv2.rectangle(frame, (w // 2 - 220, 50), (w // 2 + 220, 95), (0, 0, 255), -1)
            cv2.putText(frame, "!!! FALL DETECTED !!!", (w // 2 - 190, 83),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Mã hóa JPEG chất lượng 65 để giảm băng thông và tải CPU
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return """
    
    
    
        
        Giám Sát Té Ngã Trực Tiếp
        """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("==================================================")
    print("  HE THONG DA SAN SANG!")
    print("  Mo trinh duyet truy cap: http://127.0.0.1:5000")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)