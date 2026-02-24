// 定期更新系统状态
setInterval(updateStatus, 500);

function updateStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            // 更新指令显示
            document.getElementById('instruction').textContent = data.instruction;
            document.getElementById('debug_instruction').textContent = 
                `Instruction: ${data.instruction}`;
            
            // 更新 FPS
            document.getElementById('fps').textContent = data.fps;
            document.getElementById('debug_fps').textContent = `FPS: ${data.fps}`;
            
            // 更新目标标签
            document.getElementById('target_label').textContent = data.target;
            document.getElementById('debug_target').textContent = `Target: ${data.target}`;
            
            // 更新手部检测状态
            const handStatus = data.hand_detected ? 'Yes' : 'No';
            document.getElementById('hand_status').textContent = handStatus;
            document.getElementById('debug_hand').textContent = `Hand: ${handStatus}`;
            
            // 更新目标检测状态
            const targetStatus = data.target_detected ? 'Yes' : 'No';
            document.getElementById('target_status').textContent = targetStatus;
            
            // 更新距离比例
            const distancePercent = (data.distance_ratio * 100).toFixed(1);
            document.getElementById('distance_ratio').textContent = distancePercent + '%';
            
            // 更新状态指示灯
            const indicator = document.getElementById('status_indicator');
            if (data.is_running) {
                indicator.className = 'status-indicator active';
            } else {
                indicator.className = 'status-indicator inactive';
            }
        })
        .catch(error => console.error('Error fetching status:', error));
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('SmartReach Accessibility Interface Loaded');
    console.log('System is listening for voice commands...');
    updateStatus();
});
