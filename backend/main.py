from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time
import asyncio
from ultralytics import YOLO
import speech_recognition as sr
from queue import Queue
import io

# ============ FastAPI 应用初始化 ============
app = FastAPI(title="SmartReach Accessibility", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件挂载
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

# ============ 全局配置 ============
TARGET_LABEL = "bottle"
YOLO_MODEL = YOLO("yolov8n.pt")
MP_HANDS = mp.solutions.hands
HANDS = MP_HANDS.Hands(max_num_hands=1, min_detection_confidence=0.7)
MP_DRAW = mp.solutions.drawing_utils

# 语音识别引擎
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000

# 语音合成引擎
TTS_ENGINE = pyttsx3.init()
TTS_ENGINE.setProperty('rate', 160)
TTS_ENGINE.setProperty('volume', 1.0)

# 系统状态
class SystemState:
    def __init__(self):
        self.current_instruction = "System initializing..."
        self.fps = 0.0
        self.target_detected = False
        self.hand_detected = False
        self.confidence = 0.0
        self.distance_ratio = 0.0  # 0.0 到 1.0，表示手与目标的接近程度
        self.is_running = False
        self.lock = threading.Lock()

state = SystemState()

# ============ 无障碍语音引导核心 ============

def generate_accessible_instruction(hand_info, target_info, h, w):
    """
    为盲人用户生成简洁、清晰的语音指令
    不同于视觉版本，这里优先考虑听觉的清晰度和简洁性
    """
    if not target_info:
        return "Searching for target", 0.0
    if not hand_info:
        return "Hand not detected", 0.0
    
    hx, hy = hand_info["norm_center"]
    tx, ty = target_info["center"][0] / w, target_info["center"][1] / h
    
    dx = hx - tx
    dy = hy - ty
    
    # 计算距离比例（用于调整播报频率）
    distance_ratio = target_info["area"]  # 面积越大，距离越近
    
    threshold = 0.1
    instr = []
    
    # 优先级排序：距离 > 水平 > 垂直
    if distance_ratio < 0.05:
        instr.append("Move forward")
    elif distance_ratio > 0.2:
        instr.append("Stop. Grasp now")
    else:
        # 水平方向
        if dx > threshold:
            instr.append("Left")
        elif dx < -threshold:
            instr.append("Right")
        
        # 垂直方向
        if dy > threshold:
            instr.append("Up")
        elif dy < -threshold:
            instr.append("Down")
        
        if not instr:
            instr.append("Move forward slowly")
    
    # 返回简洁指令和距离比例
    return " ".join(instr) if instr else "Aligned", distance_ratio

def speak_async(text, speed_factor=1.0):
    """
    异步播放语音
    speed_factor: 用于根据距离调整语速（接近时加快）
    """
    def _speak():
        try:
            engine = pyttsx3.init()
            # 根据距离动态调整语速（类似倒车雷达）
            base_rate = 160
            adjusted_rate = int(base_rate * speed_factor)
            engine.setProperty('rate', adjusted_rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    threading.Thread(target=_speak, daemon=True).start()

# ============ 视觉处理函数 ============

def process_frame(frame):
    """处理单帧图像"""
    h, w, _ = frame.shape
    
    # 1. 物体检测
    results = YOLO_MODEL(frame, verbose=False, conf=0.5)[0]
    target_info = None
    
    for box in results.boxes:
        label = YOLO_MODEL.names[int(box.cls[0])]
        if label == TARGET_LABEL:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1) / (h * w)
            target_info = {
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "conf": conf
            }
            # 绘制目标框（用于调试和演示）
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break
    
    # 2. 手部追踪
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = HANDS.process(img_rgb)
    hand_info = None
    
    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            lm = hand_lms.landmark[9]
            hand_info = {
                "center": (lm.x * w, lm.y * h),
                "norm_center": (lm.x, lm.y),
                "landmarks": hand_lms
            }
            MP_DRAW.draw_landmarks(frame, hand_lms, MP_HANDS.HAND_CONNECTIONS)
            break
    
    # 3. 生成无障碍指令
    instruction, distance_ratio = generate_accessible_instruction(hand_info, target_info, h, w)
    
    # 4. 更新系统状态
    with state.lock:
        state.current_instruction = instruction
        state.target_detected = target_info is not None
        state.hand_detected = hand_info is not None
        state.confidence = target_info["conf"] if target_info else 0.0
        state.distance_ratio = distance_ratio
    
    # 5. 绘制状态信息（用于调试）
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Target: {TARGET_LABEL}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Instruction: {instruction}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return frame, instruction, distance_ratio

# ============ 视频流生成器 ============

def video_stream_generator():
    """生成 MJPEG 视频流（用于调试和演示）"""
    cap = cv2.VideoCapture(0)
    prev_time = 0
    last_speech_time = 0
    speech_interval = 1.0  # 无障碍版本播报频率更高
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        frame, instruction, distance_ratio = process_frame(frame)
        
        # 计算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        with state.lock:
            state.fps = fps
        
        # 动态语音播报（根据距离调整频率和语速）
        if instruction not in ["Searching for target", "Hand not detected"]:
            # 距离越近，播报越频繁，语速越快
            if distance_ratio > 0.1:
                speech_interval = 0.5 + (1.0 - distance_ratio) * 0.5  # 0.5 到 1.0 秒
            else:
                speech_interval = 1.5
            
            if curr_time - last_speech_time > speech_interval:
                # 根据距离调整语速
                speed_factor = 1.0 + distance_ratio * 0.5  # 1.0 到 1.5 倍速
                speak_async(instruction, speed_factor)
                last_speech_time = curr_time
        
        # 编码为 JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
               + frame_bytes + b'\r\n')

# ============ 语音命令识别 ============

def listen_for_commands():
    """
    后台线程：持续监听用户的语音命令
    支持的命令：
    - "find bottle", "find cup", "find phone" 等
    - "start", "stop", "pause"
    """
    global TARGET_LABEL
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening for voice commands...")
                audio = recognizer.listen(source, timeout=5)
            
            # 使用 Google Speech Recognition
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {command}")
            
            # 命令处理
            if "find" in command:
                for obj in ["bottle", "cup", "cell phone", "mouse", "keyboard"]:
                    if obj in command:
                        TARGET_LABEL = obj
                        speak_async(f"Searching for {obj}")
                        break
            elif "stop" in command:
                speak_async("System stopped")
                with state.lock:
                    state.is_running = False
            elif "start" in command:
                speak_async("System started")
                with state.lock:
                    state.is_running = True
        
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Speech Recognition Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(0.5)

# ============ FastAPI 路由 ============

@app.on_event("startup")
async def startup_event():
    """启动时初始化语音命令监听线程"""
    threading.Thread(target=listen_for_commands, daemon=True).start()
    speak_async("SmartReach Accessibility system started. Say find bottle, find cup, or find phone")
    with state.lock:
        state.is_running = True

@app.get("/")
async def index():
    """返回主页"""
    return FileResponse("../frontend/index.html")

@app.get("/video_feed")
async def video_feed():
    """返回 MJPEG 视频流（用于调试）"""
    return StreamingResponse(
        video_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    with state.lock:
        return {
            "instruction": state.current_instruction,
            "fps": round(state.fps, 1),
            "target_detected": state.target_detected,
            "hand_detected": state.hand_detected,
            "confidence": round(state.confidence, 3),
            "distance_ratio": round(state.distance_ratio, 2),
            "is_running": state.is_running,
            "target": TARGET_LABEL
        }

@app.get("/api/speak/{text}")
async def speak(text: str):
    """手动触发语音播报（用于测试）"""
    speak_async(text)
    return {"status": "speaking", "text": text}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "accessibility_mode": True}

if __name__ == "__main__":
    import uvicorn
    print("Starting SmartReach Accessibility Edition...")
    print("Open http://localhost:8000 in your browser (optional)")
    print("System will respond to voice commands")
    uvicorn.run(app, host="0.0.0.0", port=8000)
