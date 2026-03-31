#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Input - 极速本地语音输入工具 (V2 稳定加速版)
修复: FunASR 1.3.1 chunk_size 类型冲突
优化: CPU线程绑定 / 向量化RMS / 剪贴板延迟压缩
开源协议: MIT
注意: Windows 全局热键需以管理员身份运行终端/IDE
"""

import sys
import os
import time
import queue
import logging
import warnings
import ctypes
import traceback
from ctypes import wintypes
from pathlib import Path

# ✅ 提前绑定 CPU 线程，避免推理时动态分配开销
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() or 4))
os.environ["MKL_NUM_THREADS"] = str(max(1, os.cpu_count() or 4))

import numpy as np
import pyaudio
import keyboard
import pyperclip
import yaml
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget, QFrame,
    QSizePolicy, QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPainter, QIcon, QPixmap, QImage

from funasr import AutoModel

# 抑制无关底层警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# ==========================================
# 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("VoiceInput")

# ==========================================
# 热键信号桥接
# ==========================================
class HotKeyBridge(QObject):
    pressed = pyqtSignal()
    released = pyqtSignal()

# ==========================================
# Windows 系统交互工具
# ==========================================
class Win32Utils:
    @staticmethod
    def is_admin():
        try:
            return os.getuid() == 0
        except AttributeError:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0

    @staticmethod
    def set_ime_status(open_status: bool):
        try:
            imm32 = ctypes.WinDLL("imm32", use_last_error=True)
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            h_wnd = user32.GetForegroundWindow()
            if not h_wnd: return
            h_imc = imm32.ImmGetContext(h_wnd)
            if h_imc:
                imm32.ImmSetOpenStatus(h_imc, wintypes.BOOL(open_status))
                imm32.ImmReleaseContext(h_wnd, h_imc)
        except Exception as e:
            logger.debug(f"IME switch failed: {e}")

# ==========================================
# 音频录制器
# ==========================================
class AudioRecorder:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024

    def __init__(self, device_index=None, sample_rate=16000, max_seconds=30):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.max_frames = int((sample_rate / self.CHUNK) * max_seconds)
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self._rms = 0.0
        
    def start_recording(self):
        self.frames = []
        self.is_recording = True
        try:
            self.stream = self.p.open(
                format=self.FORMAT, channels=self.CHANNELS, rate=self.sample_rate,
                input=True, input_device_index=self.device_index,
                frames_per_buffer=self.CHUNK, stream_callback=self._audio_callback
            )
            self.stream.start_stream()
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self.is_recording = False
            
    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return None
        audio_bytes = b''.join(self.frames)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        max_amp = np.max(np.abs(audio_float))
        if max_amp > 0.005:
            audio_float = audio_float * min(1.0 / max_amp, 10.0)
        return audio_float
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
            if len(self.frames) >= self.max_frames:
                self.is_recording = False
                return (in_data, pyaudio.paComplete)
            try:
                # 向量化计算 RMS，降低音频线程 CPU 占用
                samples = np.frombuffer(in_data, dtype=np.int16)
                self._rms = np.sqrt(np.mean(samples.astype(np.float32)**2)) / 32768.0
            except Exception:
                self._rms = 0.0
        return (in_data, pyaudio.paContinue)
    
    def get_rms(self):
        return self._rms
    
    def close(self):
        if self.p and self.p.is_active():
            self.p.terminate()

# ==========================================
# FunASR 推理线程
# ==========================================
class ASRWorker(QThread):
    status_changed = pyqtSignal(str)
    final_text_ready = pyqtSignal(str)
    
    def __init__(self, audio_queue, config):
        super().__init__()
        self.audio_queue = audio_queue
        self.config = config
        self.model = None
        self._is_running = True
        
    def run(self):
        # 绑定 PyTorch CPU 线程
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        torch.set_num_interop_threads(1)
        
        self.status_changed.emit("加载模型...")
        try:
            Path(self.config['model_dir']).mkdir(parents=True, exist_ok=True)
            self.model = AutoModel(
                model=self.config['asr_model'],
                vad_model=self.config['vad_model'],
                punc_model=self.config['punc_model'],
                model_dir=self.config['model_dir'],
                device=self.config['device'],
                disable_update=True  # 跳过版本检查，加快启动
            )
            logger.info("FunASR model loaded successfully")
            
            # [新增优化]：给模型喂一段全零的假数据进行预热，消除第一次按键的卡顿
            logger.info("正在预热模型...")
            dummy_audio = np.zeros(16000, dtype=np.float32) 
            self.model.generate(input=dummy_audio, fs=self.config['sample_rate'])
            
            self.status_changed.emit("就绪")
        except Exception as e:
            logger.error(f"Failed to load FunASR: {e}")
            self.status_changed.emit("模型加载失败")
            return
            
        while self._is_running:
            try:
                task = self.audio_queue.get(timeout=0.5)
                if task['type'] == 'transcribe':
                    self._transcribe(task['data'])
            except queue.Empty:
                continue
                
    def _transcribe(self, audio_data):
        self.status_changed.emit("识别中...")
        start_time = time.time()
        try:
            # ✅ 核心修复：移除引发类型错误的 chunk_size，使用官方离线默认策略
            # 离线全量推理对 <10s 语音更快，且准确率 100% 保持
            result = self.model.generate(
                input=audio_data,
                fs=self.config['sample_rate']
            )
            
            text = ""
            if result and isinstance(result, list) and len(result) > 0:
                res_item = result[0]
                if isinstance(res_item, dict):
                    text = res_item.get('text', '').strip()
                elif isinstance(res_item, str):
                    text = res_item.strip()
                else:
                    text = str(res_item).strip()
                    
            elapsed = time.time() - start_time
            if text:
                logger.info(f"识别结果 ({elapsed:.2f}s): {text}")
                self.final_text_ready.emit(text)
                self.status_changed.emit("完成")
            else:
                logger.info("VAD 过滤后无有效语音")
                self.final_text_ready.emit("")
                self.status_changed.emit("无语音")
        except Exception as e:
            logger.error(f"ASR error: {e}")
            logger.debug(traceback.format_exc())
            self.final_text_ready.emit("")
            self.status_changed.emit("识别失败")
            
    def stop(self):
        self._is_running = False
        self.wait()

# ==========================================
# 悬浮窗 HUD
# ==========================================
class VoiceHUD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bar_heights = [2, 2, 2, 2, 2]
        self.bar_weights = [0.5, 0.8, 1.0, 0.75, 0.55]
        self.base_bar_height = 14
        self.rms_history = []
        self._rms_source = None
        self._setup_ui()
        self._setup_timer()
        
    def _setup_ui(self):
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("QMainWindow { background: transparent; }")
        
        self.capsule = QFrame(self)
        self.capsule.setObjectName("Capsule")
        self.capsule.setStyleSheet("""
            QFrame#Capsule { background-color: rgba(10, 10, 10, 240); border-radius: 16px; border: 1px solid rgba(80, 80, 80, 180); }
        """)
        self.setCentralWidget(self.capsule)
        
        layout = QHBoxLayout(self.capsule)
        layout.setContentsMargins(12, 0, 16, 0)
        layout.setSpacing(8)
        
        self.waveform_widget = QWidget()
        self.waveform_widget.setFixedSize(32, 32)
        self.waveform_widget.paintEvent = self._paint_waveform
        layout.addWidget(self.waveform_widget)
        
        self.text_label = QLabel("就绪")
        self.text_label.setStyleSheet("color: #e0e0e0; font-size: 12px; font-weight: 500; font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;")
        self.text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.text_label)
        
        self.setMinimumSize(110, 32)
        self.setMaximumSize(500, 60)
        self.hide()
        
    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_animation)
        self.timer.start(33)
        
    def set_rms_source(self, func): self._rms_source = func
        
    def _update_animation(self):
        if not self.isVisible(): return
        rms = self._rms_source() if self._rms_source else 0.0
        target = min(1.0, rms * 15.0)
        self.rms_history.append(target)
        if len(self.rms_history) > 5: self.rms_history.pop(0)
        smoothed = 0.7 * target + 0.3 * (self.rms_history[-2] if len(self.rms_history) >= 2 else target)
        for i in range(5):
            target_h = max(2, smoothed * self.base_bar_height * self.bar_weights[i])
            cur = self.bar_heights[i]
            self.bar_heights[i] = cur + (target_h - cur) * 0.4 if target_h > cur else cur - (cur - target_h) * 0.15
        self.waveform_widget.update()
        
    def _paint_waveform(self, event):
        painter = QPainter(self.waveform_widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 220))
        start_x = (self.waveform_widget.width() - 18) // 2
        for i, h in enumerate(self.bar_heights):
            shake = np.random.uniform(-0.8, 0.8) if h > 3 else 0
            final_h = max(2.0, float(h + shake))
            y_pos = int(16 - final_h / 2)
            painter.drawRoundedRect(start_x + i * 4, y_pos, 2, int(final_h), 1, 1)
        painter.end()
            
    def show_hud(self):
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, screen.height() - self.height() - 80)
        self.text_label.setText("聆听中")
        self.text_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        self.show()
        self.setWindowOpacity(0.0)
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(150)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_anim.start()
        
    def hide_hud(self):
        if not self.isVisible(): return
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(100)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._fade_anim.finished.connect(self.hide)
        self._fade_anim.start()
        
    def set_processing(self):
        self.text_label.setText("解码中")
        self.text_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")

# ==========================================
# 主控制器
# ==========================================
class VoiceInputApp:
    def __init__(self, config_path="config.yaml"):
        if not Win32Utils.is_admin():
            logger.warning("⚠️ 未检测到管理员权限，全局热键可能无法注册。请以管理员身份运行终端/IDE。")
            
        self.config = self._load_config(config_path)
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        
        self.audio_recorder = AudioRecorder(device_index=self.config.get('mic_device_index'), sample_rate=self.config['sample_rate'])
        self.audio_queue = queue.Queue()
        self.asr_worker = ASRWorker(self.audio_queue, self.config)
        self.hud = VoiceHUD()
        self.hud.set_rms_source(self.audio_recorder.get_rms)
        
        self.asr_worker.final_text_ready.connect(self._on_transcript)
        
        self._hotkey_bridge = HotKeyBridge()
        self._hotkey_bridge.pressed.connect(self._on_hotkey_pressed)
        self._hotkey_bridge.released.connect(self._on_hotkey_released)
        
        self._key_pressed = False
        self._setup_tray()
        self._setup_hotkey()
        
        self.asr_worker.start()
        logger.info("✅ 应用已启动，按住 %s 键说话，松开自动识别粘贴", self.config['hotkey'].upper())
        
    def _load_config(self, path):
        default = {'hotkey': 'f8', 'mic_device_index': None, 'sample_rate': 16000,
                   'asr_model': 'paraformer-zh', 'vad_model': 'fsmn-vad', 'punc_model': 'ct-punc',
                   'device': 'cpu', 'model_dir': './models/funasr'}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                if cfg: default.update(cfg)
        else:
            with open(path, 'w', encoding='utf-8') as f: yaml.dump(default, f, allow_unicode=True)
        default['model_dir'] = str(Path(default['model_dir']).resolve())
        return default
        
    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self._create_icon(), self.app)
        menu = QMenu()
        self.status_action = menu.addAction("状态: 初始化")
        self.status_action.setEnabled(False)
        self.asr_worker.status_changed.connect(lambda s: self.status_action.setText(f"状态: {s}"))
        menu.addSeparator()
        quit_action = menu.addAction("退出")
        quit_action.triggered.connect(self._quit)
        self.tray.setContextMenu(menu)
        self.tray.show()
        
    def _create_icon(self):
        img = QImage(32, 32, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(180, 180, 180))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(10, 5, 12, 18, 5, 5)
        p.drawRect(14, 20, 4, 10)
        p.drawRect(8, 28, 16, 2)
        p.end()
        return QIcon(QPixmap.fromImage(img))
        
    def _setup_hotkey(self):
        keyboard.on_press_key(self.config['hotkey'], self._on_keyboard_press)
        keyboard.on_release_key(self.config['hotkey'], self._on_keyboard_release)
        
    def _on_keyboard_press(self, e):
        if not self._key_pressed:
            self._key_pressed = True
            self._hotkey_bridge.pressed.emit()
            
    def _on_keyboard_release(self, e):
        if self._key_pressed:
            self._key_pressed = False
            self._hotkey_bridge.released.emit()
            
    def _on_hotkey_pressed(self):
        if self.audio_recorder.is_recording or self.hud.isVisible(): return
        self.hud.show_hud()
        self.audio_recorder.start_recording()
        
    def _on_hotkey_released(self):
        if not self.audio_recorder.is_recording: return
        self.hud.set_processing()
        audio_data = self.audio_recorder.stop_recording()
        if audio_data is None or len(audio_data) == 0:
            self.hud.hide_hud()
            return
        self.audio_queue.put({'type': 'transcribe', 'data': audio_data})
        
    def _on_transcript(self, text):
        if text and text.strip(): self._paste_text(text)
        self.hud.hide_hud()
        
    def _paste_text(self, text):
        try: old_clip = pyperclip.paste()
        except: old_clip = ""
        pyperclip.copy(text)
        Win32Utils.set_ime_status(False)
        time.sleep(0.02)  # 剪贴板同步极短等待
        keyboard.press_and_release('ctrl+v')
        time.sleep(0.08)  # 按键释放与IME恢复短等待
        Win32Utils.set_ime_status(True)
        try: pyperclip.copy(old_clip)
        except: pass
        
    def _quit(self):
        logger.info("正在退出...")
        keyboard.unhook_all()
        self.audio_recorder.stop_recording()
        self.asr_worker.stop()
        self.audio_recorder.close()
        self.app.quit()
        
    def run(self):
        sys.exit(self.app.exec())

if __name__ == "__main__":
    try:
        app = VoiceInputApp("config.yaml")
        app.run()
    except KeyboardInterrupt: pass
    except Exception as e:
        logger.exception("💥 致命错误")
        input("按 Enter 退出...")