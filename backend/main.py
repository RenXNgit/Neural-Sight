import os
import time
import threading
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
try:
    import torch
except Exception:
    torch = None

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

SEARCH_TIMEOUT_SEC = 20
HAND_TIMEOUT_SEC = 10
COMPLETE_HOLD_SEC = 1.2
COMPLETE_DIST_THRESHOLD = 0.08
TARGET_MOVE_COMPLETE_THRESHOLD = 0.12
GRASP_ALIGN_THRESHOLD = 0.12

TARGET_CN_MAP = {
    "bottle": "瓶子",
    "cup": "杯子",
    "phone": "手机",
}


def target_to_cn(label: str) -> str:
    return TARGET_CN_MAP.get(label, label)

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
@contextmanager
def temporary_torch_load_patch_for_yolo():
    if torch is None:
        yield
        return

    original_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        # Trust local model file in this project and keep legacy behavior for YOLO checkpoint loading.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = original_load


def load_yolo_model(model_path: str):
    # PyTorch 2.6+ changed torch.load defaults and can break older YOLO checkpoints.
    with temporary_torch_load_patch_for_yolo():
        return YOLO(model_path)


YOLO_MODEL = load_yolo_model(MODEL_PATH)
MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands(max_num_hands=1, min_detection_confidence=0.7)
MP_DRAW = mp.solutions.drawing_utils

# ============ 系统状态 ============
class SystemState:
    def __init__(self):
        self.current_instruction = "请问您需要什么？"
        self.target_detected = False
        self.hand_detected = False
        self.confidence = 0.0
        self.distance_ratio = 0.0
        self.last_infer_ms = 0.0
        self.target_label = TARGET_LABEL_DEFAULT
        self.target_selected = False
        self.last_command = None
        self.phase = "idle"
        self.search_started_at = time.time()
        self.target_seen_at = None
        self.grab_candidate_since = None
        self.last_target_center = None
        self.completed_until = 0.0

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
def generate_realtime_guidance(hand_info, target_info, h, w):
    hx, hy = hand_info["x"] / w, hand_info["y"] / h
    tx, ty = target_info["center"][0] / w, target_info["center"][1] / h
    dx, dy = hx - tx, hy - ty
    norm_dist = float(np.hypot(dx, dy))
    distance_ratio = target_info["area"]

    threshold = 0.08
    moves = []

    # 手在目标后方时，中心可能已对齐但视觉上无明显位移，直接进入可抓取提示。
    if abs(dx) <= GRASP_ALIGN_THRESHOLD and abs(dy) <= GRASP_ALIGN_THRESHOLD and distance_ratio >= 0.02:
        return "可以抓取", distance_ratio, norm_dist

    if dx > threshold:
        moves.append("手向左移动")
    elif dx < -threshold:
        moves.append("手向右移动")

    if dy > threshold:
        moves.append("手向上移动")
    elif dy < -threshold:
        moves.append("手向下移动")

    if distance_ratio < 0.05:
        moves.append("向前一点")
    elif distance_ratio > 0.25:
        moves.append("稍微后退一点")

    if not moves:
        moves.append("继续向前一点")

    return "，".join(moves), distance_ratio, norm_dist

# ============ 视觉处理 ============
def process_frame(frame: np.ndarray, target_label: str, target_selected: bool = False) -> Dict[str, Any]:
    now = time.time()
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

    target_cn = target_to_cn(target_label)
    distance_ratio = target_info["area"] if target_info else 0.0

    if not target_selected:
        instruction = "请问您需要什么？"
        state.phase = "idle"
        state.search_started_at = now
        state.target_seen_at = None
        state.grab_candidate_since = None
        state.last_target_center = None
    elif now < state.completed_until:
        instruction = "任务完成，请问您需要什么？"
    elif target_info is None:
        if state.phase != "searching":
            state.search_started_at = now
            state.phase = "searching"
        state.target_seen_at = None
        state.grab_candidate_since = None
        state.last_target_center = None
        if now - state.search_started_at >= SEARCH_TIMEOUT_SEC:
            instruction = f"视野中未发现{target_cn}，请确认物品是否在桌面上。"
        else:
            instruction = f"正在寻找{target_cn}。"
    elif hand_center is None:
        if state.target_seen_at is None:
            state.target_seen_at = now
        state.phase = "locked"
        state.grab_candidate_since = None
        if now - state.target_seen_at >= HAND_TIMEOUT_SEC:
            instruction = "未检测到手，请稍微抬高或向前伸。"
        else:
            instruction = f"已锁定{target_cn}，请在胸前伸出手。"
        state.last_target_center = target_info["center"]
    else:
        state.phase = "guiding"
        state.target_seen_at = now
        instruction, distance_ratio, norm_dist = generate_realtime_guidance(hand_center, target_info, h, w)

        target_moved = False
        if state.last_target_center is not None:
            last_tx, last_ty = state.last_target_center
            cur_tx, cur_ty = target_info["center"]
            move_norm = float(np.hypot((cur_tx - last_tx) / w, (cur_ty - last_ty) / h))
            target_moved = move_norm >= TARGET_MOVE_COMPLETE_THRESHOLD

        ready_to_grasp = (instruction == "可以抓取") or (norm_dist < COMPLETE_DIST_THRESHOLD)

        if ready_to_grasp:
            if state.grab_candidate_since is None:
                state.grab_candidate_since = now
            if target_moved or (now - state.grab_candidate_since >= COMPLETE_HOLD_SEC):
                instruction = "任务完成"
                state.phase = "idle"
                state.target_selected = False
                state.target_label = ""
                state.completed_until = now + 2.5
                state.search_started_at = state.completed_until
                state.target_seen_at = None
                state.grab_candidate_since = None
                state.last_target_center = None
            else:
                state.last_target_center = target_info["center"]
        else:
            state.grab_candidate_since = None
            state.last_target_center = target_info["center"]

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
        "target": state.target_label if state.target_selected else None,
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
    state.target_selected = True
    state.phase = "searching"
    state.search_started_at = time.time()
    state.target_seen_at = None
    state.grab_candidate_since = None
    state.last_target_center = None
    state.completed_until = 0.0
    state.current_instruction = f"正在寻找{target_to_cn(target)}。"
    return {"target": target, "instruction": state.current_instruction}

@app.post("/api/command")
def api_command(body: Dict[str, str] = Body(...)):
    cmd = body.get("command", "").strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="command is required")
    state.last_command = cmd
    state.target_label = cmd.lower()
    state.target_selected = True
    speak_async(f"Target set to {state.target_label}")
    return {"command": cmd, "target": state.target_label}

@app.post("/api/infer")
async def api_infer(file: UploadFile = File(...), target: Optional[str] = None):
    target_label = target.strip() if target else state.target_label
    target_selected = bool(target.strip()) if target else state.target_selected
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    t0 = time.time()
    result = process_frame(frame, target_label, target_selected=target_selected)
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