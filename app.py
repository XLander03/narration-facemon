import sys
import cv2
import face_recognition
import pickle
import pygame
import numpy as np
import os
import tempfile
import sounddevice as sd
from pathlib import Path
from cryptography.fernet import Fernet
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QStackedWidget, QFrame, QFileDialog, QTabWidget,
                             QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

# --- PREVENT CRASHES IN PACKAGED APP ---
if getattr(sys, 'frozen', False):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# --- CONSTANTS ---
TOLERANCE = 0.5
# Define the standard location for Data
DOCS_DIR = Path.home() / "Documents" / "AudioGuard"

# Ensure the directory exists
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --- HELPER: ASSET DISCOVERY ---
def get_asset_path(filename):
    """Looks for assets ONLY in ~/Documents/AudioGuard"""
    target = DOCS_DIR / filename
    if target.exists():
        return str(target)
    return None

def get_audio_device_name():
    try:
        sd._terminate()
        sd._initialize()
        idx = sd.default.device[1]
        info = sd.query_devices(idx)
        return info.get('name', 'Unknown').lower()
    except: return "speaker (error)"

# ==========================================
#   WORKER: PLAYER (Client Side)
# ==========================================
class SecurityPlayerWorker(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, str) 
    unlock_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None
        self.mode = "LOCKED"
        self.authorized_encodings = []
        self.decryption_key = None
        self.audio_path = None
        pygame.mixer.init()
        self.audio_paused = True
        self.audio_loaded = False
        self.temp_file = None

    def load_assets(self):
        # Look in Documents/AudioGuard
        lock_path = get_asset_path("access.lock")
        if lock_path:
            try:
                with open(lock_path, "rb") as f:
                    data = pickle.load(f)
                    self.authorized_encodings = data if isinstance(data, list) else [data]
            except: pass
        
        key_path = get_asset_path("master.key")
        if key_path:
            try: 
                with open(key_path, "rb") as f: self.decryption_key = f.read()
            except: pass

        self.audio_path = get_asset_path("secure_audio.enc")

    def decrypt_audio(self):
        if not self.decryption_key or not self.audio_path: return False
        try:
            fernet = Fernet(self.decryption_key)
            with open(self.audio_path, 'rb') as f: encrypted = f.read()
            decrypted = fernet.decrypt(encrypted)
            
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            self.temp_file.write(decrypted)
            self.temp_file.close()
            
            pygame.mixer.music.load(self.temp_file.name)
            pygame.mixer.music.play(-1)
            pygame.mixer.music.pause()
            self.audio_loaded = True
            return True
        except Exception as e:
            print(f"Decryption Error: {e}")
            return False

    def run(self):
        self.load_assets()
        self.cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize for speed
            small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = np.ascontiguousarray(small, dtype=np.uint8)
            boxes = face_recognition.face_locations(rgb_small)
            
            msg = "Initializing..."
            color = "yellow"

            if not self.authorized_encodings or not self.decryption_key or not self.audio_path:
                msg = "NO PACKAGE FOUND in Documents/AudioGuard"; color = "gray"
            else:
                if len(boxes) == 0:
                    msg = "Scanning for Authorized User..."; color = "yellow"
                    if not self.audio_paused: pygame.mixer.music.pause(); self.audio_paused = True
                elif len(boxes) > 1:
                    msg = "Multiple Faces Detected!"; color = "red"
                    if not self.audio_paused: pygame.mixer.music.pause(); self.audio_paused = True
                else:
                    live_enc = face_recognition.face_encodings(rgb_small, boxes)[0]
                    matches = face_recognition.compare_faces(self.authorized_encodings, live_enc, tolerance=TOLERANCE)
                    
                    if True in matches:
                        dev_name = get_audio_device_name()
                        if 'speaker' in dev_name:
                            msg = "Connect Headphones to Play"; color = "red"
                            if not self.audio_paused: pygame.mixer.music.pause(); self.audio_paused = True
                        else:
                            msg = f"SECURE PLAYBACK | {dev_name[:10]}"; color = "green"
                            if self.mode == "LOCKED":
                                self.mode = "MONITORING"
                                self.unlock_signal.emit()
                                self.decrypt_audio()
                            
                            if self.audio_loaded and self.audio_paused:
                                pygame.mixer.music.unpause()
                                self.audio_paused = False
                    else:
                        msg = "ACCESS DENIED"; color = "red"
                        if not self.audio_paused: pygame.mixer.music.pause(); self.audio_paused = True

            self.status_signal.emit(msg, color)

            h, w, ch = rgb_frame.shape
            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            p = qt_img.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

        self.cap.release()
        pygame.mixer.music.stop()
        if self.temp_file:
            try: os.remove(self.temp_file.name)
            except: pass

    def stop(self):
        self._run_flag = False
        self.wait()

# ==========================================
#   MAIN APP WINDOW
# ==========================================
class UnifiedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioGuard: Generator & Player")
        self.resize(900, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: white; font-family: Arial;")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 10px 20px; }
            QTabBar::tab:selected { background: #007acc; color: white; }
        """)
        self.layout.addWidget(self.tabs)

        self.player_tab = QWidget()
        self.setup_player_tab()
        self.tabs.addTab(self.player_tab, "Secure Player")

        self.generator_tab = QWidget()
        self.setup_generator_tab()
        self.tabs.addTab(self.generator_tab, "Package Generator (Admin)")
        
        self.player_thread = None
        self.tabs.currentChanged.connect(self.handle_tab_change)
        
        # Default: Check if files exist in Documents. If yes, Player. Else, Generator.
        if get_asset_path("access.lock"): 
            self.tabs.setCurrentIndex(0)
            self.handle_tab_change(0)
        else:
            self.tabs.setCurrentIndex(1)
            self.handle_tab_change(1)

    def setup_player_tab(self):
        layout = QVBoxLayout(self.player_tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title = QLabel("SECURE ACCESS LOCKED")
        self.lbl_title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet("color: #ff5555")
        layout.addWidget(self.lbl_title)
        
        self.lbl_info = QLabel(f"Looking for keys in:\n{DOCS_DIR}")
        self.lbl_info.setStyleSheet("color: #888; margin-bottom: 10px;")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.lbl_status = QLabel("System Idle")
        self.lbl_status.setFont(QFont("Arial", 14))
        self.lbl_status.setStyleSheet("background: #333; padding: 8px; border-radius: 4px;")
        layout.addWidget(self.lbl_status)
        self.lbl_video = QLabel()
        self.lbl_video.setFixedSize(640, 480)
        self.lbl_video.setStyleSheet("background: black; border: 2px solid #444;")
        layout.addWidget(self.lbl_video)

    def handle_tab_change(self, index):
        if self.player_thread:
            self.player_thread.stop()
            self.player_thread = None
            self.lbl_video.clear()
            self.lbl_title.setText("SECURE ACCESS LOCKED")
            self.lbl_title.setStyleSheet("color: #ff5555")

        if index == 0: # Player Tab
            self.player_thread = SecurityPlayerWorker()
            self.player_thread.change_pixmap_signal.connect(lambda x: self.lbl_video.setPixmap(QPixmap.fromImage(x)))
            self.player_thread.status_signal.connect(self.update_player_ui)
            self.player_thread.unlock_signal.connect(lambda: self.lbl_title.setText("DECRYPTING & PLAYING"))
            self.player_thread.start()

    def update_player_ui(self, msg, color):
        self.lbl_status.setText(msg)
        border = "#00ff00" if color == "green" else "#ff0000"
        if color == "yellow": border = "#ffff00"
        elif color == "gray": border = "#444444"
        self.lbl_video.setStyleSheet(f"background: black; border: 3px solid {border};")
        if color == "green": self.lbl_title.setStyleSheet("color: #00ff00")

    def setup_generator_tab(self):
        layout = QVBoxLayout(self.generator_tab)
        title = QLabel("Create Secure Package")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        layout.addWidget(title)
        self.inputs = {}
        form_layout = QVBoxLayout()
        for label in ["Photo 1 (Front)", "Photo 2 (Left)", "Photo 3 (Right)", "Audio File (.mp3)"]:
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setFixedWidth(120)
            entry = QLineEdit(); entry.setReadOnly(True); entry.setStyleSheet("padding: 5px; background: #333;")
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda checked, e=entry, t=label: self.browse_file(e, t))
            row.addWidget(lbl); row.addWidget(entry); row.addWidget(btn)
            form_layout.addLayout(row)
            self.inputs[label] = entry
        layout.addLayout(form_layout)
        self.btn_gen = QPushButton("GENERATE SECURE PACKAGE")
        self.btn_gen.setStyleSheet("background: #007acc; padding: 15px; font-weight: bold; font-size: 14px;")
        self.btn_gen.clicked.connect(self.run_generation_safe)
        layout.addWidget(self.btn_gen)
        layout.addStretch()

    def browse_file(self, entry_widget, file_type):
        filter_str = "Images (*.png *.jpg *.jpeg *.bmp)" if "Photo" in file_type else "Audio (*.mp3 *.wav)"
        fname, _ = QFileDialog.getOpenFileName(self, f"Select {file_type}", "", filter_str)
        if fname: entry_widget.setText(fname)

    def run_generation_safe(self):
        p1 = self.inputs["Photo 1 (Front)"].text()
        p2 = self.inputs["Photo 2 (Left)"].text()
        p3 = self.inputs["Photo 3 (Right)"].text()
        audio = self.inputs["Audio File (.mp3)"].text()

        if not all([p1, p2, p3, audio]):
            QMessageBox.warning(self, "Error", "Please select all files.")
            return

        self.btn_gen.setText("Generating... Please Wait...")
        self.btn_gen.setEnabled(False)
        QApplication.processEvents()

        try:
            # Output to a folder named "New_Package" in the project directory
            out_dir = "New_Package"
            if not os.path.exists(out_dir): os.makedirs(out_dir)

            encodings = []
            for i, p in enumerate([p1, p2, p3]):
                self.btn_gen.setText(f"Processing Photo {i+1}/3...")
                QApplication.processEvents()
                img = face_recognition.load_image_file(p)
                if img.shape[0] > 800:
                    scale = 800 / img.shape[0]
                    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
                encs = face_recognition.face_encodings(img)
                if not encs: raise Exception(f"No face in Photo {i+1}")
                encodings.append(encs[0])

            with open(os.path.join(out_dir, "access.lock"), "wb") as f: pickle.dump(encodings, f)

            self.btn_gen.setText("Encrypting Audio...")
            QApplication.processEvents()
            key = Fernet.generate_key()
            with open(os.path.join(out_dir, "master.key"), "wb") as f: f.write(key)
            with open(audio, "rb") as f: raw_audio = f.read()
            fernet = Fernet(key)
            enc_audio = fernet.encrypt(raw_audio)
            with open(os.path.join(out_dir, "secure_audio.enc"), "wb") as f: f.write(enc_audio)

            QMessageBox.information(self, "Success", f"Files created in folder: '{out_dir}'\n\nZip this folder (contents only) and send to the client.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_gen.setText("GENERATE SECURE PACKAGE")
            self.btn_gen.setEnabled(True)

    def closeEvent(self, event):
        if self.player_thread: self.player_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UnifiedApp()
    window.show()
    sys.exit(app.exec())