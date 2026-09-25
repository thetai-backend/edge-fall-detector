import sys
import os
import time

# Them duong dan goc vao sys.path de import producer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from producer.video_stream import PhoneCameraProducer
from worker.yolo_worker import YOLOPoseWorker

def main():
    # 1. Khoi dong Producer
    print("[SYSTEM] Khoi dong Producer...")
    producer = PhoneCameraProducer().start()

    # 2. Khoi tao Worker
    model_file = "models/yolo26n-pose.pt"
    if not os.path.exists(model_file):
        print(f"[LOI] Khong tim thay file {model_file}! Vui long copy file vao thu muc models/.")
        producer.stop()
        return

    worker = YOLOPoseWorker(model_path=model_file, conf_thresh=0.25, imgsz=640)

    print("[SYSTEM] Bat dau pipeline nhan dien...")
    fps_count = 0
    start_time = time.time()

    try:
        while True:
            # Lay frame moi nhat (khong bi lag)
            frame = producer.get_latest_frame(timeout=1.0)
            if frame is None:
                continue

            # Inference bang YOLO Pose
            t0 = time.time()
            persons = worker.process_frame(frame)
            infer_time = (time.time() - t0) * 1000  # mili-giay

            fps_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                print(f"[PERF] FPS Inference: {fps:.1f} | Latency: {infer_time:.1f}ms | Tracked: {len(persons)} nguoi")
                
                # In thong tin track_id neu co nguoi xuat hien
                for p in persons:
                    print(f"  -> Track ID: {p['track_id']} | Box: {[round(x, 1) for x in p['box']]}")

                fps_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n[SYSTEM] Dang dung he thong...")
    finally:
        producer.stop()
        print("[SYSTEM] Da giai phong tai nguyen thanh cong.")

if __name__ == "__main__":
    main()