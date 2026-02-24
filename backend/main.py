import os
import time
import threading
import logging
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# 可选依赖（语音/TTS）
try:
    import speech_recognition as sr
except Exception:
    sr = None
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
try:
    import pyaudio
except Exception:
    pyaudio = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============ 环境配置 ============
TARGET_LABEL_DEFAULT = os.getenv("TARGET_LABEL", "bottle")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
ENABLE_LOCAL_CAM = os.getenv("ENABLE_LOCAL_CAM", "0") == "1"
ENABLE_LOCAL_VOICE = os.getenv("ENABLE_LOCAL_VOICE", "0") == "1"
ENABLE_TTS = os.getenv("ENABLE_TTS", "0") == "1"

# ============ FastAPI 应用初始化 ============
app = FastAPI(title="SmartReach Accessibility", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

# ============ 模型与手部检测 ============
YOLO_MODEL = YOLO(MODEL_PATH)
MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands(max_num_hands=1, min_detection_confidence=0.7)
MP_DRAW = mp.solutions.drawing_utils

# ============ 系统状态 ============
class SystemState:
    def __init__(self):
        self.current_instruction = "System initializing..."
        self.target_detected = False
        self.hand_detected = False
        self.confidence = 0.0
        self.distance_ratio = 0.0
        self.last_infer_ms = 0.0
        self.target_label = TARGET_LABEL_DEFAULT
        self.last_command = None

state = SystemState()

# ============ 语音/TTS 工具 ============
def _has_default_mic() -> bool:
    if pyaudio is None:
        return False
    pa = pyaudio.PyAudio()
    try:
        pa.get_default_input_device_info()
        return True
    except Exception:
        return False
    finally:
        try:
            pa.terminate()
        except Exception:
            pass

TTS_ENGINE = None
if ENABLE_TTS and pyttsx3:
    try:
        TTS_ENGINE = pyttsx3.init()
        TTS_ENGINE.setProperty("rate", 160)
        TTS_ENGINE.setProperty("volume", 1.0)
    except Exception as e:
        logging.warning("TTS init failed: %s", e)
        TTS_ENGINE = None

def speak_async(text: str):
    if not (ENABLE_TTS and TTS_ENGINE):
        return
    def _run():
        try:
            TTS_ENGINE.say(text)
            TTS_ENGINE.runAndWait()
        except Exception as e:
            logging.warning("TTS speak failed: %s", e)
    threading.Thread(target=_run, daemon=True).start()

# ============ 核心指令生成 ============
def generate_accessible_instruction(hand_info, target_info, h, w):
    if not target_info:
        return "Searching for target", 0.0
    if not hand_info:
        return "Hand not detected", 0.0

    hx, hy = hand_info["norm_center"]
    tx, ty = target_info["center"][0] / w, target_info["center"][1] / h
    dx, dy = hx - tx, hy - ty
    distance_ratio = target_info["area"]

    threshold = 0.1
    instr = []

    if distance_ratio < 0.05:
        instr.append("Move forward")
    elif distance_ratio > 0.2:
        instr.append("Stop. Grasp now")
    else:
        if dx > threshold:
            instr.append("Left")
        elif dx < -threshold:
            instr.append("Right")
        if dy > threshold:
            instr.append("Up")
        elif dy < -threshold:
            instr.append("Down")
        if not instr:
            instr.append("Move forward slowly")

    return " ".join(instr) if instr else "Aligned", distance_ratio

# ============ 视觉处理 ============
def process_frame(frame: np.ndarray, target_label: str) -> Dict[str, Any]:
    h, w, _ = frame.shape
    results = YOLO_MODEL(frame, verbose=False, conf=0.5)[0]
    target_info = None
    detections: List[Dict[str, Any]] = []

    for box in results.boxes:
        label = YOLO_MODEL.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        detections.append(
            {"label": label, "confidence": conf, "bbox": [float(x1), float(y1), float(x2), float(y2)]}
        )
        if target_info is None and label == target_label:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1) / (h * w)
            target_info = {"center": (cx, cy), "bbox": (x1, y1, x2, y2), "area": area, "conf": conf}

    hand_center = None
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = HANDS.process(img_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_lms.landmark]
            ys = [lm.y * h for lm in hand_lms.landmark]
            hand_center = {
                "x": int(sum(xs) / len(xs)),
                "y": int(sum(ys) / len(ys))
            }
            break

    instruction, distance_ratio = generate_accessible_instruction(hand_center, target_info, h, w)
    state.current_instruction = instruction
    state.target_detected = target_info is not None
    state.hand_detected = hand_center is not None
    state.confidence = target_info["conf"] if target_info else 0.0
    state.distance_ratio = distance_ratio
    state.target_label = target_label

    return {
        "width": w,
        "height": h,
        "instruction": instruction,
        "distance_ratio": distance_ratio,
        "target_detected": state.target_detected,
        "hand_detected": state.hand_detected,
        "confidence": state.confidence,
        "detections": detections,
        "hand_center": hand_center,  # 新增
    }

# ============ 本地视频流（可选） ============
def video_stream_generator():
    if not ENABLE_LOCAL_CAM:
        # 占位空帧，确保旧前端不崩
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        while True:
            _, buffer = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.warning("Local camera not available, fallback to blank frames.")
        return video_stream_generator()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" + frame_bytes + b"\r\n"
        )
    cap.release()

# ============ 语音指令监听（可选） ============
def listen_for_commands():
    if not (ENABLE_LOCAL_VOICE and sr and _has_default_mic()):
        logging.warning("Voice command disabled or no microphone.")
        return
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    mic = sr.Microphone()
    commands_map = {"bottle": "bottle", "cup": "cup", "phone": "phone"}

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=4)
            text = recognizer.recognize_google(audio, language="en-US").lower()
            logging.info("Voice command: %s", text)
            for k, v in commands_map.items():
                if k in text:
                    state.target_label = v
                    speak_async(f"Target set to {v}")
                    state.last_command = text
                    break
        except Exception as e:
            logging.warning("Voice listen error: %s", e)
            time.sleep(2)

# ============ 路由 ============
@app.get("/")
def index():
    return FileResponse("../frontend/index.html")

@app.get("/api/status")
def api_status():
    return {
        "status": "ready",
        "model": MODEL_PATH,
        "target": state.target_label,
        "instruction": state.current_instruction,
        "target_detected": state.target_detected,
        "hand_detected": state.hand_detected,
        "confidence": state.confidence,
        "distance_ratio": state.distance_ratio,
        "last_infer_ms": state.last_infer_ms,
        "voice_enabled": ENABLE_LOCAL_VOICE,
        "tts_enabled": ENABLE_TTS,
        "local_cam": ENABLE_LOCAL_CAM,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.post("/api/target")
def api_set_target(body: Dict[str, str] = Body(...)):
    target = body.get("target", "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="target is required")
    state.target_label = target
    return {"target": target}

@app.post("/api/command")
def api_command(body: Dict[str, str] = Body(...)):
    cmd = body.get("command", "").strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="command is required")
    state.last_command = cmd
    state.target_label = cmd.lower()
    speak_async(f"Target set to {state.target_label}")
    return {"command": cmd, "target": state.target_label}

@app.post("/api/infer")
async def api_infer(file: UploadFile = File(...), target: Optional[str] = None):
    target_label = target.strip() if target else state.target_label
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    t0 = time.time()
    result = process_frame(frame, target_label)
    state.last_infer_ms = round((time.time() - t0) * 1000, 2)
    return result

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# ============ 启动语音线程（可选） ============
if ENABLE_LOCAL_VOICE and sr and _has_default_mic():
    threading.Thread(target=listen_for_commands, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)