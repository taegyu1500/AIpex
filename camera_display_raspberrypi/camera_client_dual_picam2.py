#!/usr/bin/env python3
# camera_client_dual_picam2.py
# -------------------------------------------
# B 라즈베리파이:
# - Picamera2(0) : FRONT 카메라
# - Picamera2(1) : REAR  카메라
# - 각 프레임을 A 파이로 전송 → A 파이가 디텍팅 → JSON 결과 회신
# - FRONT HUD : 항상 전체화면 (480x320로 리사이즈)
#   + 우측 상단에 X1200 배터리 퍼센트 표시
# - REAR  HUD : 디텍팅 있을 때만 좌측 상단에 팝업처럼 표시
# -------------------------------------------

import socket
import struct
import json
import cv2
import numpy as np
import threading
import time

from picamera2 import Picamera2
import smbus2  # X1200 I2C 배터리 게이지용

# ====== A 파이 주소/포트 설정 ======
A_HOST = "192.168.0.10"    # A 라즈베리파이 IP로 바꿔줘
FRONT_PORT = 50000
REAR_PORT = 50001

# ====== 공용 유틸 ======
def recvall(sock, n: int):
    """정확히 n바이트를 받을 때까지 반복 수신."""
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# ====== 디텍션 결과 공유 변수 ======
front_result = {"width": 640, "height": 480, "detections": []}
rear_result  = {"width": 640, "height": 480, "detections": []}
front_lock = threading.Lock()
rear_lock  = threading.Lock()

# ====== X1200 배터리 퍼센트 읽기 ======
I2C_BUS_ID = 1       # /dev/i2c-1
FG_ADDR    = 0x36    # X120x fuel gauge I2C 주소 (일반적으로 0x36)

def _read_word_swapped(bus, reg):
    """
    read_word_data 결과가 리틀엔디안이라 바이트 스왑 한 번 해줌.
    (X120x 관련 예제 코드 스타일)
    """
    raw = bus.read_word_data(FG_ADDR, reg)
    # low/high 바이트 스왑
    return ((raw & 0xFF) << 8) | (raw >> 8)

def get_battery_percentage() -> int | None:
    """
    X1200 / X120x 시리즈 배터리 잔량(%) 읽기.

    - I2C 주소: 0x36
    - SoC 레지스터: 0x04
    - 값 스케일: raw / 256.0 (%)
    """
    try:
        bus = smbus2.SMBus(I2C_BUS_ID)

        # state-of-charge(%) 레지스터 0x04
        raw_soc = _read_word_swapped(bus, 0x04)
        bus.close()

        percent = raw_soc / 256.0   # 0.0 ~ 100.x
        # 안전하게 범위 제한
        if percent < 0:
            percent = 0
        if percent > 100:
            percent = 100

        return int(round(percent))
    except Exception as e:
        print("[BAT] Failed to read battery:", e)
        return None

# ====== 카메라 + 소켓 스레드 (Picamera2) ======
def camera_thread_picam(cam_name: str,
                        cam_index: int,
                        port: int,
                        result_ref: dict,
                        lock: threading.Lock):
    """
    cam_name : "FRONT" / "REAR"
    cam_index : 0 또는 1 (rpicam-hello 기준 카메라 인덱스)
    port : A 파이 디텍터 서버 포트 (50000 / 50001)
    result_ref : front_result 또는 rear_result
    lock : 해당 결과용 Lock
    """
    print(f"[B][{cam_name}] Starting Picamera2 index {cam_index} ...")

    # Picamera2 초기화
    picam = Picamera2(cam_index)
    config = picam.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()
    time.sleep(0.5)  # 워밍업

    print(f"[B][{cam_name}] Connecting to A {A_HOST}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((A_HOST, port))
    except Exception as e:
        print(f"[B][{cam_name}] Failed to connect:", e)
        picam.close()
        return

    print(f"[B][{cam_name}] Connected to A.")

    try:
        while True:
            # RGB 배열로 캡처
            frame_rgb = picam.capture_array()
            # OpenCV BGR로 변환
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # JPEG 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_param)
            if not ok:
                print(f"[B][{cam_name}] Failed to encode frame.")
                continue

            frame_bytes = encoded.tobytes()

            # 길이 + 데이터 전송
            sock.sendall(struct.pack(">I", len(frame_bytes)))
            sock.sendall(frame_bytes)

            # JSON 길이 수신
            length_buf = recvall(sock, 4)
            if not length_buf:
                print(f"[B][{cam_name}] Server closed connection.")
                break
            (json_len,) = struct.unpack(">I", length_buf)

            # JSON 데이터 수신
            json_data = recvall(sock, json_len)
            if json_data is None:
                print(f"[B][{cam_name}] Failed to receive JSON.")
                break

            result = json.loads(json_data.decode("utf-8"))

            # 공유 결과 갱신
            with lock:
                result_ref.clear()
                result_ref.update(result)

    except Exception as e:
        print(f"[B][{cam_name}] Exception:", e)
    finally:
        print(f"[B][{cam_name}] Thread exit.")
        sock.close()
        picam.close()

# ====== 바운딩박스 렌더링 ======
def render_black_canvas_from_result(result: dict) -> np.ndarray:
    """
    result = {
      "width":  W,
      "height": H,
      "detections": [
         {"x1":..., "y1":..., "x2":..., "y2":..., "cls_name":"person", "conf":0.87}, ...
      ]
    }
    """
    w = result.get("width", 640)
    h = result.get("height", 480)
    detections = result.get("detections", [])

    black = np.zeros((h, w, 3), dtype=np.uint8)

    for det in detections:
        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        cls_name = det.get("cls_name", "obj")
        conf = float(det.get("conf", 0.0))

        # 박스
        cv2.rectangle(black, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 중앙 정렬 라벨
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        box_w = x2 - x1
        text_x = x1 + (box_w - text_w) // 2
        text_y = y2 + text_h + 5

        text_x = max(0, min(text_x, w - text_w))
        text_y = max(text_h, min(text_y, h - 1))

        cv2.putText(
            black,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    return black

# ====== 메인 (HUD 렌더 루프) ======
def main():
    global front_result, rear_result

    # Waveshare 3.5" LCD 해상도
    SCREEN_W = 480
    SCREEN_H = 320

    # FRONT / REAR 카메라 스레드 시작
    t_front = threading.Thread(
        target=camera_thread_picam,
        args=("FRONT", 0, FRONT_PORT, front_result, front_lock),
        daemon=True,
    )
    t_rear = threading.Thread(
        target=camera_thread_picam,
        args=("REAR", 1, REAR_PORT, rear_result, rear_lock),
        daemon=True,
    )

    t_front.start()
    t_rear.start()

    # FRONT HUD 전체화면
    cv2.namedWindow("Front HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Front HUD",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # REAR HUD 상태 플래그
    rear_window_created = False
    REAR_WIN_W = 320
    REAR_WIN_H = 240
    REAR_WIN_X = 0
    REAR_WIN_Y = 0

    # 배터리 표시 관련 상태 (1초에 한 번만 I2C 읽기)
    last_batt_read_time = 0.0
    batt_percent_cached = None

    print("[B] HUD started. ESC 로 종료.")

    while True:
        now = time.time()

        # ------ 배터리 값 1초에 한 번만 갱신 ------
        if now - last_batt_read_time > 1.0:
            batt_percent_cached = get_battery_percentage()
            last_batt_read_time = now

        # ---------- FRONT ----------
        with front_lock:
            fr = dict(front_result)

        front_canvas = render_black_canvas_from_result(fr)

        # 640x480 → 480x320 으로 리사이즈해서 LCD를 꽉 채우기
        front_canvas_fs = cv2.resize(
            front_canvas,
            (SCREEN_W, SCREEN_H),
            interpolation=cv2.INTER_LINEAR
        )

        # ----- 우측 상단에 배터리 퍼센트 표시 -----
        if batt_percent_cached is not None:
            batt_text = f"BAT {batt_percent_cached:3d}%"
        else:
            batt_text = "BAT --%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (bt_w, bt_h), _ = cv2.getTextSize(batt_text, font, font_scale, thickness)

        margin = 8
        text_x = SCREEN_W - bt_w - margin
        text_y = bt_h + margin

        # 테두리(검은색) 먼저
        cv2.putText(
            front_canvas_fs,
            batt_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        # 실제 초록색 글자
        cv2.putText(
            front_canvas_fs,
            batt_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("Front HUD", front_canvas_fs)

        # ---------- REAR (디텍팅 있을 때만 좌측 상단) ----------
        with rear_lock:
            rr = dict(rear_result)
        rear_dets = rr.get("detections", [])

        if rear_dets:
            rear_canvas = render_black_canvas_from_result(rr)
            if not rear_window_created:
                cv2.namedWindow("Rear HUD", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Rear HUD", REAR_WIN_W, REAR_WIN_H)
                cv2.moveWindow("Rear HUD", REAR_WIN_X, REAR_WIN_Y)
                rear_window_created = True

            cv2.imshow("Rear HUD", rear_canvas)
        else:
            if rear_window_created:
                cv2.destroyWindow("Rear HUD")
                rear_window_created = False

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    if rear_window_created:
        cv2.destroyWindow("Rear HUD")
    cv2.destroyWindow("Front HUD")
    cv2.destroyAllWindows()
    print("[B] Exit.")

if __name__ == "__main__":
    main()
