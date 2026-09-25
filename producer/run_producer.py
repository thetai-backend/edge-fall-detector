import time
import cv2
from video_stream import PhoneCameraProducer

def main():
    print("[INFO] Khoi dong Producer tren Termux PRoot...")
    producer = PhoneCameraProducer().start()

    time.sleep(1.0)  # Cho Producer ket noi den 127.0.0.1
    fps_count = 0
    start_time = time.time()

    try:
        while True:
            # Lay frame moi nhat
            frame = producer.get_latest_frame(timeout=2.0)
            if frame is None:
                print("[CANH BAO] Dang cho frame tu camera...")
                continue

            fps_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                h, w, _ = frame.shape
                fps = fps_count / elapsed
                print(f"[OK] Lay frame thanh cong: {w}x{h} | FPS doc duoc: {fps:.1f}")
                fps_count = 0
                start_time = time.time()

            # Gia lap Consumer ton 0.05s de xu ly
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Dang dung Producer...")
    finally:
        producer.stop()
        print("[INFO] Da dong luong Producer an toan.")

if __name__ == "__main__":
    main()