let videoEl, overlayEl, streamReady = false, sending = false;
let lastInstruction = "";
let recog, recogActive = false;
let lastVoiceTime = Date.now();
let listenTimeout = null;
let handTrail = [];
let currentTarget = "";
let isSpeaking = false;
let lastSpokenText = "";
let lastSpokenAt = 0;
let lastFoundSpokenAt = 0;
let welcomePromptPlayed = false;

const SPEAK_COOLDOWN_MS = 1200;
const FOUND_ANNOUNCE_COOLDOWN_MS = 8000;
const GUIDANCE_REPEAT_MS = 2000;



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
        console.error("具体的推理报错信息:", e); // 把错误打印到浏览器控制台
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
    const text = normalizeText(instr);
    if (!text) {
        const el = document.getElementById('instruction');
        if (el) el.textContent = instr || '';
        return;
    }

    // 一轮任务结束后清空前端目标缓存，允许下一轮继续选择同一目标。
    if (text.includes('任务完成')) {
        currentTarget = "";
    }

    if (shouldSpeakInstruction(text)) {
        speak(text);
        lastInstruction = text;
    }
    const el = document.getElementById('instruction');
    if (el) el.textContent = instr;
}

function normalizeText(text) {
    return (text || '').replace(/\s+/g, ' ').trim();
}

function isFoundInstruction(text) {
    const compact = text.replace(/\s+/g, '').toLowerCase();
    return compact.includes('找到') || compact.includes('已发现') || compact.includes('found');
}

function isGuidanceInstruction(text) {
    return text.includes('手向')
        || text.includes('向前一点')
        || text.includes('继续向前一点')
        || text.includes('稍微后退一点');
}

function shouldSpeakInstruction(text) {
    const now = Date.now();
    if (text === lastInstruction) {
        // 引导类指令允许按固定频率重复播报，避免用户只听到一次。
        if (isGuidanceInstruction(text) && now - lastSpokenAt >= GUIDANCE_REPEAT_MS) {
            return true;
        }
        return false;
    }

    if (isFoundInstruction(text)) {
        if (now - lastFoundSpokenAt < FOUND_ANNOUNCE_COOLDOWN_MS) {
            return false;
        }
        lastFoundSpokenAt = now;
    }

    if (now - lastSpokenAt < SPEAK_COOLDOWN_MS) {
        return false;
    }

    return true;
}

// 语音播报
function speak(text, options = {}) {
    const { force = false, cooldownMs = SPEAK_COOLDOWN_MS } = options;
    if (!window.speechSynthesis) return;
    const normalized = normalizeText(text);
    if (!normalized) return;

    const now = Date.now();
    if (!force) {
        if (normalized === lastSpokenText && now - lastSpokenAt < cooldownMs) return;
        if (now - lastSpokenAt < cooldownMs) return;
    }

    const utter = new SpeechSynthesisUtterance(normalized);
    utter.lang = 'zh-CN';
    utter.onstart = () => {
        isSpeaking = true;
    };
    utter.onend = () => {
        isSpeaking = false;
    };
    utter.onerror = () => {
        isSpeaking = false;
    };

    // 队列里只保留最新一句，避免连续积压造成“循环播报”错觉。
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
    lastSpokenText = normalized;
    lastSpokenAt = now;
}

function playWelcomePrompt(force = false) {
    const welcome = "请问您需要什么？";
    if (!force && welcomePromptPlayed) return;
    speak(welcome, { force: true, cooldownMs: 300 });
    welcomePromptPlayed = true;
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
        if (isSpeaking) return;
        lastVoiceTime = Date.now();
        clearTimeout(listenTimeout);
        let found = false;
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (!event.results[i].isFinal) continue;
            const text = event.results[i][0].transcript.trim();
            if (text.includes("瓶")) {
                setTargetWithFeedback("bottle", "正在寻找瓶子。");
                found = true;
            } else if (text.includes("杯")) {
                setTargetWithFeedback("cup", "正在寻找杯子。");
                found = true;
            } else if (text.includes("手机")) {
                setTargetWithFeedback("phone", "正在寻找手机。");
                found = true;
            }
        }
        if (!found) {
            speak("未能识别您的语音，请再说一次", { cooldownMs: 5000 });
        }
        restartListenTimeout();
    };

    recog.onerror = function(event) {
        // 只播报一次错误
        if (event.error === "no-speech") {
            speak("未听到您的指令，请再试一次", { cooldownMs: 10000 });
        }
        setTimeout(() => recog.start(), 1000);
        restartListenTimeout();
    };

    recog.onend = function() {
        setTimeout(() => recog.start(), 1000);
        restartListenTimeout();
    };

    recog.start();
    restartListenTimeout();
}

// 如果10秒没听到语音，提醒用户
function restartListenTimeout() {
    clearTimeout(listenTimeout);
    listenTimeout = setTimeout(() => {
        if (!isSpeaking) {
            speak("未听到您的指令，请再试一次", { cooldownMs: 10000 });
        }
    }, 10000);
}

function setTargetWithFeedback(target, feedback) {
    // 同一目标不重复下发，防止识别到回声后不断触发。
    if (target === currentTarget) return;
    currentTarget = target;
    sendTarget(target);
    speak(feedback, { force: true });
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
    setTimeout(() => playWelcomePrompt(), 400);

    // 某些浏览器会拦截首次自动播报，用户首次交互后补播一次。
    const replayOnFirstGesture = () => {
        const now = Date.now();
        if (lastSpokenText !== "请问您需要什么？" || now - lastSpokenAt > 4000) {
            playWelcomePrompt(true);
        }
    };
    document.addEventListener('click', replayOnFirstGesture, { once: true });

    setupVoiceInput();
});