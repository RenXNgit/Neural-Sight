# SmartReach Accessibility Edition

**Vision-Guided Object Grasping Assistant for Blind Users**

一个专为盲人用户设计的实时视觉引导系统。通过计算机视觉和自然语言语音交互，帮助视障用户准确抓取目标物体。

## 核心设计理念

本系统采用**无障碍优先（Accessibility-First）**的设计方法，与传统的"先做功能，再加无障碍"不同：

1.  **语音为主，视觉为辅**：系统的核心交互是语音，而非屏幕。视频流仅用于调试和演示。
2.  **听觉反馈优化**：指令简洁清晰，采用"倒车雷达"式的动态播报频率和语速变化。
3.  **零视觉依赖**：用户无需看屏幕，完全通过耳朵和手的触觉完成任务。

## 功能特性

✅ **语音命令识别**：用户可以说"Find bottle"、"Find cup"等来指定目标物体  
✅ **实时语音指导**：系统播报方向指令（"Left", "Right", "Forward" 等）  
✅ **动态播报频率**：距离越近，播报越频繁，语速越快（类似倒车雷达）  
✅ **高精度检测**：YOLOv8 物体检测 + MediaPipe 手部追踪  
✅ **后台监听**：系统启动后自动进入语音监听状态  
✅ **可选调试界面**：提供 Web 界面用于调试和演示（但不是必需的）  

## 项目结构

```
SmartReach_Accessibility/
├── backend/
│   └── main.py              # FastAPI 后端（语音优先）
├── frontend/
│   ├── index.html           # 调试界面（可选）
│   └── static/
│       └── js/
│           └── app.js       # 前端逻辑
├── requirements.txt         # Python 依赖
├── run.sh                   # 启动脚本
└── README.md               # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：`pyaudio` 在某些系统上需要特殊处理：
- **macOS**: `brew install portaudio && pip install pyaudio`
- **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev && pip install pyaudio`
- **Windows**: 通常自动安装，如有问题可下载预编译的 wheel 文件

### 2. 启动系统

```bash
bash run.sh
```

或直接运行：
```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. 系统启动后

系统会自动：
1.  初始化摄像头和麦克风
2.  播放欢迎语音："SmartReach Accessibility system started. Say find bottle, find cup, or find phone"
3.  进入语音监听状态

### 4. 使用语音命令

系统支持以下语音命令：

| 命令 | 效果 |
|------|------|
| "Find bottle" | 开始检测瓶子 |
| "Find cup" | 开始检测杯子 |
| "Find phone" / "Find cell phone" | 开始检测手机 |
| "Find mouse" | 开始检测鼠标 |
| "Find keyboard" | 开始检测键盘 |
| "Start" | 启动系统 |
| "Stop" | 停止系统 |

### 5. 跟随语音指导

系统会不断播报指令，例如：
- "Left" - 手在物体左侧，向左移动
- "Right" - 手在物体右侧，向右移动
- "Forward" - 物体距离太远，向前移动
- "Stop. Grasp now" - 手已对准物体，可以抓取

**关键特性**：距离越近，播报频率越快，语速越快（类似倒车雷达的"嘟嘟嘟"声）。

## 核心算法

### 1. 视觉处理流程

```
摄像头输入
    ↓
YOLOv8 物体检测（找到目标物体）
    ↓
MediaPipe 手部追踪（找到用户的手）
    ↓
空间关系计算（计算手与物体的相对位置）
    ↓
指令生成（生成简洁的语音指令）
    ↓
TTS 语音播报（播放指令）
```

### 2. 听觉反馈优化

系统采用**距离感知的动态播报**：

- **距离远**（面积 < 5%）：
  - 播报频率：1.5 秒一次
  - 语速：正常（160 wpm）
  - 指令："Move forward"

- **距离中等**（面积 5%-20%）：
  - 播报频率：0.5-1.0 秒一次（动态变化）
  - 语速：加快（160-240 wpm）
  - 指令：方向指令（"Left", "Right" 等）

- **距离近**（面积 > 20%）：
  - 播报频率：立即播报
  - 语速：最快（240 wpm）
  - 指令："Stop. Grasp now"

这种设计模仿了**倒车雷达**的工作原理，用户可以通过播报频率和语速的变化直观地感受到距离。

### 3. 语音命令识别

系统使用 Google Speech Recognition API 进行语音识别。后台线程持续监听麦克风输入，识别用户的命令并做出相应反应。

## 高分扩展建议

1.  **触觉反馈**：集成振动反馈设备（如手机振动马达），当手接近目标时振动加快。
2.  **多语言支持**：支持中文、英文等多种语言的语音指导。
3.  **性能评估**：记录每次抓取的耗时、成功率等数据，生成实验报告。
4.  **离线语音识别**：使用本地语音识别模型（如 Vosk），避免依赖网络。
5.  **深度相机集成**：使用 RealSense 或 Kinect 获取真实的 Z 轴距离，提高精度。

## 常见问题

**Q: 系统无法识别我的语音？**  
A: 确保：
- 麦克风工作正常
- 环境噪音不要过大
- 说话清晰，语速适中
- 网络连接正常（Google Speech Recognition 需要网络）

**Q: 模型加载很慢？**  
A: 首次运行会下载 YOLOv8 模型文件（约 250MB）。请耐心等待。

**Q: 如何在没有网络的情况下使用？**  
A: 可以使用离线语音识别库（如 Vosk 或 PocketSphinx）替代 Google Speech Recognition。

**Q: 能否在移动设备上运行？**  
A: 可以，但需要适配移动设备的摄像头和麦克风 API。建议使用 Kivy 或 React Native 重新开发移动版本。

## 技术栈

- **后端框架**：FastAPI + Uvicorn
- **计算机视觉**：OpenCV, YOLOv8, MediaPipe
- **语音处理**：pyttsx3 (TTS), SpeechRecognition (STT)
- **前端**：HTML5, TailwindCSS, Vanilla JavaScript（可选调试界面）

## 无障碍设计原则

本系统遵循以下无障碍设计原则：

1.  **感知可用性**：所有信息通过语音传达，不依赖视觉。
2.  **可操作性**：用户只需说话，无需复杂的手势或按键操作。
3.  **可理解性**：指令简洁清晰，易于理解。
4.  **鲁棒性**：系统能够处理口音、背景噪音等干扰。

## 许可证

MIT License

## 致谢

感谢 OpenAI、Google、Ultralytics 等开源社区的贡献。
