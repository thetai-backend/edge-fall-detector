import cv2
import threading
import queue
import time
from config import config

class PhoneCameraProducer:
    def __init__(self, source=config.SOURCE_URL):
        self.source = source
        self.frame_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
        self.stopped = False
        self.thread = None
        self.is_connected = False

    def start(self):
        """Khoi chay luong Producer ngam"""
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self):
        while not self.stopped:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_connected = False
                time.sleep(config.RECONNECT_DELAY_SEC)
                continue

            self.is_connected = True

            while not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    self.is_connected = False
                    break

                # Neu queue dang day (Consumer chua doc kip), vut bo frame cu
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                # Day frame moi nhat vao
                self.frame_queue.put(frame)

            cap.release()
            if not self.stopped:
                time.sleep(config.RECONNECT_DELAY_SEC)

    def get_latest_frame(self, timeout=1.0):
        """Ham de Worker AI lay frame moi nhat"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Dung luong an toan"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)