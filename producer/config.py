import os

class StreamConfig:
    # 1. NGUON STREAM NOI BO (LOOPBACK)
    # 127.0.0.1 tro truc tiep vao RAM may, khong qua mang ngoai
    SOURCE_URL = os.getenv("CAMERA_SOURCE", "http://127.0.0.1:8080/video")

    # 2. HANG DOI (QUEUE)
    # Cực kỳ quan trọng tren dien thoai: Queue = 1 de khong bao gio bi tran RAM
    QUEUE_MAX_SIZE = 1

    # 3. KET NOI LAI (SELF-HEALING)
    # De phong truong hop app camera bi Android tam pause khi chuyen tab
    AUTO_RECONNECT = True
    RECONNECT_DELAY_SEC = 1.0

config = StreamConfig()