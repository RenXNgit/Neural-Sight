let videoEl, overlayEl, streamReady = false, sending = false;
let lastInstruction = "";
let recog, recogActive = false;
let lastVoiceTime = Date.now();
let listenTimeout = null;
let handTrail = [];



function setupCamera() {
    videoEl = document.getElementById('video');
    if (!videoEl) {
        videoEl = document.createElement('video');
        videoEl.id = 'video';
        videoEl.autoplay = true;
        videoEl.playsInline = true;
        videoEl.style.width = '640px';
        videoEl.style.height = '480px';
        document.body.appendChild(videoEl);
    }
    overlayEl = document.getElementById('overlay');
    if (!overlayEl) {
        overlayEl = document.createElement('canvas');
        overlayEl.id = 'overlay';
        overlayEl.style.position = 'absolute';
        overlayEl.style.left = '0';
        overlayEl.style.top = '0';
        overlayEl.width = 640;
        overlayEl.height = 480;
        document.body.appendChild(overlayEl);
    }
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(stream => {
            videoEl.srcObject = stream;
            videoEl.onloadedmetadata = () => {
                streamReady = true;
                overlayEl.width = videoEl.videoWidth;
                overlayEl.height = videoEl.videoHeight;
            };
        })
        .catch(e => {
            speak("无法访问摄像头");
        });
}

async function sendFrame() {
    if (!streamReady || sending) return;
    sending = true;
    try {
        const off = document.createElement('canvas');
        off.width = videoEl.videoWidth;
        off.height = videoEl.videoHeight;
        const ctx = off.getContext('2d');
        // === 镜像采集 ===
        ctx.save();
        ctx.translate(off.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoEl, 0, 0, off.width, off.height);
        ctx.restore();
        // === end 镜像采集 ===
        const blob = await new Promise(res => off.toBlob(res, 'image/jpeg', 0.7));
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        const resp = await fetch('/api/infer', { method: 'POST', body: fd });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        drawDetections(data);
        updateInstruction(data.instruction);
    } catch (e) {
        speak("推理失败");
    } finally {
        sending = false;
    }
}

function drawDetections(data) {
    const { detections = [], width, height, hand_center } = data;
    overlayEl.width = width;
    overlayEl.height = height;
    const ctx = overlayEl.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    // 目标检测框
    ctx.lineWidth = 2;
    ctx.font = '14px sans-serif';
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeStyle = det.label === data.target ? '#00ff55' : '#ffcc00';
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = '#000';
        ctx.fillRect(x1, y1 - 18, ctx.measureText(det.label).width + 60, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(`${det.label} ${(det.confidence * 100).toFixed(1)}%`, x1 + 4, y1 - 5);
    });

    // === 轨迹直接用 hand_center 原始坐标 ===
    if (hand_center) {
        handTrail.push(hand_center);
        if (handTrail.length > 50) handTrail.shift();
    }
    if (handTrail.length > 1) {
        for (let i = 1; i < handTrail.length; i++) {
            const alpha = i / handTrail.length;
            ctx.strokeStyle = `rgba(0,255,255,${alpha})`;
            ctx.lineWidth = 6 * alpha;
            ctx.beginPath();
            ctx.moveTo(handTrail[i - 1].x, handTrail[i - 1].y);
            ctx.lineTo(handTrail[i].x, handTrail[i].y);
            ctx.stroke();
        }
    }
    if (hand_center) {
        ctx.beginPath();
        ctx.arc(hand_center.x, hand_center.y, 10, 0, 2 * Math.PI);
        ctx.fillStyle = "#00ffff";
        ctx.fill();
    }
}

function updateInstruction(instr) {
    if (instr && instr !== lastInstruction) {
        speak(instr);
        lastInstruction = instr;
    }
    const el = document.getElementById('instruction');
    if (el) el.textContent = instr;
}

// 语音播报
function speak(text) {
    if (!window.speechSynthesis) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = 'zh-CN';
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
}

// 智能语音输入（持续监听，交互友好）
function setupVoiceInput() {
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recog = new SpeechRecognition();
    recog.lang = 'zh-CN';
    recog.continuous = true;
    recog.interimResults = false;

    recog.onresult = function(event) {
        lastVoiceTime = Date.now();
        clearTimeout(listenTimeout);
        let found = false;
        for (let i = 0; i < event.results.length; i++) {
            const text = event.results[i][0].transcript.trim();
            if (text.includes("瓶")) {
                sendTarget("bottle");
                speak("已收到您的指令，目标切换为瓶子");
                found = true;
            } else if (text.includes("杯")) {
                sendTarget("cup");
                speak("已收到您的指令，目标切换为杯子");
                found = true;
            } else if (text.includes("手机")) {
                sendTarget("phone");
                speak("已收到您的指令，目标切换为手机");
                found = true;
            }
        }
        if (!found) {
            speak("未能识别您的语音，请再说一次");
        }
        restartListenTimeout();
    };

    recog.onerror = function(event) {
        // 只播报一次错误
        if (event.error === "no-speech") {
            speak("未听到您的指令，请再试一次");
        }
        setTimeout(() => recog.start(), 1000);
        restartListenTimeout();
    };

    recog.onend = function() {
        setTimeout(() => recog.start(), 1000);
        restartListenTimeout();
    };

    recog.start();
    speak("语音识别已开启，请说出目标");
    restartListenTimeout();
}

// 如果10秒没听到语音，提醒用户
function restartListenTimeout() {
    clearTimeout(listenTimeout);
    listenTimeout = setTimeout(() => {
        speak("未听到您的指令，请再试一次");
    }, 10000);
}

function sendTarget(target) {
    fetch('/api/target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target })
    });
}

document.addEventListener('DOMContentLoaded', function() {
    setupCamera();
    setInterval(sendFrame, 500); // 0.5秒一帧，不卡
    setupVoiceInput();
});