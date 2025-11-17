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
                             QStackedWidget, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont

# --- FIX FOR WINDOWED APP CRASHES ---
# Redirects stdout/stderr to devnull if packaged as an app to prevent crashes
if getattr(sys, 'frozen', False):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

# --- CONFIGURATION & PATHS ---
USER_DATA_DIR = Path.home() / "Documents" / "AudioGuard"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = USER_DATA_DIR / "authorized_users.pkl"
TOLERANCE = 0.5 # Stricter tolerance for the Lock Screen

# Function to find bundled resources (like the encrypted audio)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# PATHS
ENCRYPTED_AUDIO_FILE = resource_path(os.path.join("src", "secure_narration.enc"))

# Helper to find external assets (Identity Photo & Decryption Key) next to the App
def get_external_asset_path(filename):
    # 1. Check next to the executable (Dist mode)
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        # Go up 3 levels from Contents/MacOS/AudioGuard to the folder containing .app
        bundle_dir = os.path.abspath(os.path.join(base_dir, "../../..")) 
        path = os.path.join(bundle_dir, filename)
        if os.path.exists(path): return path
    
    # 2. Check next to script (Dev mode)
    if os.path.exists(filename):
        return filename
    
    return None

# --- GLOBAL HELPERS ---
def load_database():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        except: return {}
    return {}

def save_database(db):
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)

def get_audio_device_name():
    try:
        sd._terminate()
        sd._initialize()
        idx = sd.default.device[1]
        info = sd.query_devices(idx)
        return info.get('name', 'Unknown').lower()
    except:
        return "speaker (error)"

# --- WORKER THREAD ---
class VideoWorker(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str, str) 
    face_detected_signal = pyqtSignal(object)
    unlock_signal = pyqtSignal(bool)

    def __init__(self, mode="monitor"):
        super().__init__()
        self.mode = mode
        self._run_flag = True
        self.cap = None
        self.db = load_database()
        self.reference_encoding = None
        self.last_status_print = ""
        
        # Decryption State
        self.decrypted_temp_file = None
        self.current_key_loaded = False

        # --- SETUP MODES ---
        if self.mode == "lock":
            # Load identity.jpg for verification
            path = get_external_asset_path("identity.jpg")
            if path:
                try:
                    print(f"[Lock] Loading identity from: {path}")
                    img = face_recognition.load_image_file(path)
                    encs = face_recognition.face_encodings(img)
                    if len(encs) > 0:
                        self.reference_encoding = encs[0]
                    else:
                        print("[Lock] Error: No face found in identity.jpg")
                except Exception as e:
                    print(f"[Lock] Error loading identity: {e}")
            else:
                print("[Lock] Warning: identity.jpg not found next to app.")

        elif self.mode == "monitor":
            # Pre-load DB encodings
            self.all_encodings = []
            for name, data in self.db.items():
                # Handle potential DB structure differences
                if isinstance(data, dict): encs = data.get('encodings', [])
                else: encs = data 
                self.all_encodings.extend(encs)
            
            pygame.mixer.init()
            self.audio_paused = True

    def decrypt_and_play(self):
        """Decrypts audio using master.key if face is valid"""
        if self.current_key_loaded: return True

        key_path = get_external_asset_path("master.key")
        if not key_path:
            self.status_signal.emit("Error: master.key missing!", "red")
            return False

        try:
            with open(key_path, 'rb') as f:
                key = f.read()
            
            fernet = Fernet(key)
            with open(ENCRYPTED_AUDIO_FILE, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Write to temp file
            self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            self.temp_file.write(decrypted_data)
            self.temp_file.close()
            
            self.decrypted_temp_file = self.temp_file.name
            pygame.mixer.music.load(self.decrypted_temp_file)
            pygame.mixer.music.play(-1) # Loop
            pygame.mixer.music.pause()
            
            self.current_key_loaded = True
            print("[Security] Audio decrypted and loaded.")
            return True
        except Exception as e:
            print(f"[Security] Decryption failed: {e}")
            self.status_signal.emit("Decryption Failed", "red")
            return False

    def run(self):
        self.cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret: break

            # Preprocessing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = np.ascontiguousarray(small_frame, dtype=np.uint8)
            face_locations = face_recognition.face_locations(small_frame)

            # --- LOCK MODE ---
            if self.mode == "lock":
                if self.reference_encoding is None:
                    self.status_signal.emit("Error: 'identity.jpg' missing", "red")
                elif len(face_locations) == 0:
                    self.status_signal.emit("Scanning for Recipient...", "yellow")
                else:
                    live_encs = face_recognition.face_encodings(small_frame, face_locations)
                    match = face_recognition.compare_faces([self.reference_encoding], live_encs[0], tolerance=TOLERANCE)
                    
                    if match[0]:
                        self.status_signal.emit("IDENTITY CONFIRMED. UNLOCKING...", "green")
                        self.unlock_signal.emit(True)
                        # Green Box
                        for (t, r, b, l) in face_locations:
                            t*=4; r*=4; b*=4; l*=4
                            cv2.rectangle(rgb_frame, (l, t), (r, b), (0, 255, 0), 2)
                    else:
                        self.status_signal.emit("ACCESS DENIED: Face Mismatch", "red")
                        # Red Box
                        for (t, r, b, l) in face_locations:
                            t*=4; r*=4; b*=4; l*=4
                            cv2.rectangle(rgb_frame, (l, t), (r, b), (255, 0, 0), 2)

            # --- ENROLL MODE ---
            elif self.mode == "enroll":
                if len(face_locations) == 1:
                    self.status_signal.emit("Face Detected - Ready", "green")
                    self.face_detected_signal.emit((rgb_frame, face_locations))
                elif len(face_locations) == 0:
                    self.status_signal.emit("No Face Detected", "red")
                else:
                    self.status_signal.emit("Multiple Faces", "red")

            # --- MONITOR MODE ---
            elif self.mode == "monitor":
                dev_name = get_audio_device_name()
                is_speaker = 'speaker' in dev_name
                face_valid = False
                status_msg = "Locked"
                color = "red"

                if len(face_locations) == 0:
                    status_msg = "No Face"
                elif len(face_locations) > 1:
                    status_msg = "Multiple Faces"
                elif is_speaker:
                    status_msg = "⚠️ SPEAKERS DETECTED"
                else:
                    # Authenticate
                    encs = face_recognition.face_encodings(small_frame, face_locations)
                    # Check against DB OR against Identity.jpg (Dual fallback)
                    valid_user = False
                    
                    # 1. Check DB
                    if self.all_encodings:
                        matches = face_recognition.compare_faces(self.all_encodings, encs[0], tolerance=TOLERANCE)
                        if True in matches: valid_user = True
                    
                    # 2. If DB empty (first run), check Identity Match again
                    if not valid_user and self.reference_encoding is not None:
                        match = face_recognition.compare_faces([self.reference_encoding], encs[0], tolerance=TOLERANCE)
                        if match[0]: valid_user = True

                    if valid_user:
                        face_valid = True
                        status_msg = f"Secure Playback | {dev_name[:10]}"
                        color = "green"
                    else:
                        status_msg = "⛔ Unauthorized"

                # Update UI
                self.status_signal.emit(status_msg, color)
                
                # Audio Logic
                if face_valid and not is_speaker:
                    # Attempt Decryption
                    if not self.current_key_loaded:
                        success = self.decrypt_and_play()
                        if not success: face_valid = False
                    
                    if face_valid and self.audio_paused:
                        pygame.mixer.music.unpause()
                        self.audio_paused = False
                else:
                    if not self.audio_paused:
                        pygame.mixer.music.pause()
                        self.audio_paused = True
                
                # Draw Boxes
                for (t, r, b, l) in face_locations:
                    t*=4; r*=4; b*=4; l*=4
                    c = (0, 255, 0) if color == "green" else (255, 0, 0)
                    cv2.rectangle(rgb_frame, (l, t), (r, b), c, 2)

            # Render
            h, w, ch = rgb_frame.shape
            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            p = qt_img.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

        self.cap.release()
        if self.mode == "monitor": pygame.mixer.music.stop()
        self.cleanup()

    def cleanup(self):
        if self.decrypted_temp_file and os.path.exists(self.decrypted_temp_file):
            try:
                os.remove(self.decrypted_temp_file)
                print("[Security] Temp file wiped.")
            except: pass

    def stop(self):
        self._run_flag = False
        self.wait()

# --- MAIN GUI ---
class AudioGuardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioGuard - Secure Courier")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.create_sidebar()
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        self.page_lock = self.create_lock_page()
        self.page_home = self.create_home_page()
        self.page_enroll = self.create_enroll_page()
        self.page_monitor = self.create_monitor_page()

        self.stack.addWidget(self.page_lock)
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_enroll)
        self.stack.addWidget(self.page_monitor)
        
        # Start in Lock Mode
        self.sidebar.hide()
        self.start_lock_mode()
        
        self.enroll_step = 0
        self.temp_encodings = []
        self.enroll_name = ""

    def create_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setStyleSheet("background-color: #2d2d2d; border-right: 1px solid #3d3d3d;")
        self.sidebar.setFixedWidth(220)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setSpacing(20)
        sb_layout.setContentsMargins(20, 40, 20, 20)
        
        title = QLabel("AUDIO\nGUARD")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.addWidget(title)
        
        btn_style = "QPushButton { background-color: #3d3d3d; padding: 15px; border-radius: 5px; text-align: left; } QPushButton:hover { background-color: #4d4d4d; }"
        
        btn_home = QPushButton("  🏠  Home")
        btn_home.setStyleSheet(btn_style)
        btn_home.clicked.connect(lambda: self.switch_page(1))
        
        btn_enroll = QPushButton("  👤  Registration")
        btn_enroll.setStyleSheet(btn_style)
        btn_enroll.clicked.connect(lambda: self.switch_page(2))
        
        btn_monitor = QPushButton("  🔐  Decrypt & Play")
        btn_monitor.setStyleSheet(btn_style)
        btn_monitor.clicked.connect(lambda: self.switch_page(3))
        
        sb_layout.addWidget(btn_home)
        sb_layout.addWidget(btn_enroll)
        sb_layout.addWidget(btn_monitor)
        sb_layout.addStretch()
        self.layout.addWidget(self.sidebar)

    # --- PAGE CREATORS ---
    def create_lock_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("SECURE COURIER LOCKED")
        lbl.setFont(QFont("Arial", 26, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #ff5555")
        layout.addWidget(lbl)
        self.lock_status = QLabel("Verifying Recipient Identity...")
        self.lock_status.setFont(QFont("Arial", 16))
        layout.addWidget(self.lock_status)
        self.lock_video = QLabel()
        self.lock_video.setFixedSize(640, 480)
        self.lock_video.setStyleSheet("background: black; border: 2px solid #555;")
        layout.addWidget(self.lock_video, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("Identity Verified")
        lbl.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        layout.addWidget(lbl)
        desc = QLabel("1. Register your face angles for better accuracy.\n2. Proceed to Decrypt & Play.")
        desc.setStyleSheet("color: #aaa; font-size: 14px;")
        layout.addWidget(desc)
        return page

    def create_enroll_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("User Registration", font=QFont("Arial", 22)))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter Name")
        self.name_input.setStyleSheet("padding: 10px; background: #333; color: white;")
        layout.addWidget(self.name_input)
        self.enroll_status = QLabel("Enter name and start")
        self.enroll_status.setStyleSheet("color: yellow; font-size: 16px;")
        layout.addWidget(self.enroll_status)
        self.enroll_video = QLabel()
        self.enroll_video.setFixedSize(640, 480)
        self.enroll_video.setStyleSheet("background: black;")
        layout.addWidget(self.enroll_video, alignment=Qt.AlignmentFlag.AlignCenter)
        self.enroll_btn = QPushButton("Start Registration")
        self.enroll_btn.setStyleSheet("background: #007acc; padding: 15px;")
        self.enroll_btn.clicked.connect(self.handle_enroll_click)
        layout.addWidget(self.enroll_btn)
        return page

    def create_monitor_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Secure Decryption Player", font=QFont("Arial", 22)))
        self.monitor_status = QLabel("Initializing...")
        self.monitor_status.setStyleSheet("background: #333; padding: 10px;")
        layout.addWidget(self.monitor_status)
        self.monitor_video = QLabel()
        self.monitor_video.setFixedSize(640, 480)
        self.monitor_video.setStyleSheet("background: black;")
        layout.addWidget(self.monitor_video, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

    # --- NAVIGATION & LOGIC ---
    def switch_page(self, index):
        if self.thread: self.thread.stop()
        self.stack.setCurrentIndex(index)
        if index == 2: self.start_enroll_mode()
        elif index == 3: self.start_monitor_mode()

    def start_lock_mode(self):
        self.thread = VideoWorker(mode="lock")
        self.thread.change_pixmap_signal.connect(lambda x: self.lock_video.setPixmap(QPixmap.fromImage(x)))
        self.thread.status_signal.connect(lambda m, c: self.lock_status.setText(m))
        self.thread.unlock_signal.connect(self.handle_unlock)
        self.thread.start()

    def handle_unlock(self, success):
        if success:
            self.thread.stop()
            self.sidebar.show()
            self.switch_page(1) # Go Home
            QMessageBox.information(self, "Access Granted", "Identity Verified.\nEnvironment Unlocked.")

    def start_enroll_mode(self):
        self.enroll_step = -1
        self.name_input.setEnabled(True); self.name_input.clear()
        self.enroll_btn.setText("Start Registration")
        self.enroll_status.setText("Enter Name")
        self.thread = VideoWorker(mode="enroll")
        self.thread.change_pixmap_signal.connect(lambda x: self.enroll_video.setPixmap(QPixmap.fromImage(x)))
        self.thread.status_signal.connect(self.update_enroll_status)
        self.thread.face_detected_signal.connect(self.store_frame)
        self.thread.start()

    def update_enroll_status(self, msg, color):
        if "Capture" in self.enroll_btn.text():
            self.enroll_status.setText(msg)
            self.enroll_status.setStyleSheet(f"color: {color}; font-size: 16px;")

    def store_frame(self, data): self.current_frame = data

    def handle_enroll_click(self):
        if self.enroll_step == -1:
            name = self.name_input.text().strip()
            if not name: return
            self.enroll_name = name
            self.name_input.setEnabled(False)
            self.enroll_step = 0
            self.enroll_btn.setText("Capture FRONT")
            self.enroll_status.setText("Look at camera")
        elif self.enroll_step < 3:
            if not hasattr(self, 'current_frame'): return
            rgb, boxes = self.current_frame
            if len(boxes) != 1: return
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            enc = face_recognition.face_encodings(rgb, boxes)[0]
            self.temp_encodings.append(enc)
            self.enroll_step += 1
            if self.enroll_step == 1: self.enroll_btn.setText("Capture LEFT")
            elif self.enroll_step == 2: self.enroll_btn.setText("Capture RIGHT")
            elif self.enroll_step == 3:
                db = load_database()
                db[self.enroll_name] = {'encodings': self.temp_encodings}
                save_database(db)
                self.enroll_status.setText("Saved!")
                self.enroll_btn.setText("Done")
                self.enroll_btn.clicked.disconnect()
                self.enroll_btn.clicked.connect(lambda: self.switch_page(1))

    def start_monitor_mode(self):
        self.thread = VideoWorker(mode="monitor")
        self.thread.change_pixmap_signal.connect(lambda x: self.monitor_video.setPixmap(QPixmap.fromImage(x)))
        self.thread.status_signal.connect(self.update_monitor_status)
        # Identity image is pre-loaded by init if found
        # If user hasn't enrolled, we can try to set the reference encoding from the Lock mode
        if self.page_lock.findChild(QLabel): # Quick hack to re-use identity if needed
             pass 
        self.thread.start()

    def update_monitor_status(self, msg, color):
        bg = "#1a4d1a" if color == "green" else "#4d1a1a"
        self.monitor_status.setText(msg)
        self.monitor_status.setStyleSheet(f"background: {bg}; padding: 10px;")

    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioGuardApp()
    window.show()
    sys.exit(app.exec())