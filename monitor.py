import cv2
import face_recognition
import pickle
import pygame
import numpy as np
import os
import sounddevice as sd

# --- Settings ---
DB_FILE = "data/people/authorized_users.pkl"
AUDIO_FILE = "src/your_narration.mp3" # <-- Ensure this file exists
TOLERANCE = 0.6
# --- End Settings ---

# --- Audio Device Check ---
def get_current_audio_device():
    """Returns the name of the current default output device."""
    try:
        # Refresh device list (helps detect plug/unplug events)
        sd._terminate() 
        sd._initialize()
        
        default_output_index = sd.default.device[1]
        device_info = sd.query_devices(default_output_index)
        return device_info.get('name', 'Unknown').lower()
    except Exception as e:
        # If we can't detect, assume it's a speaker for safety
        return "speaker (error detection)"

# 1. Initialize Audio
pygame.mixer.init()
if not os.path.exists(AUDIO_FILE):
    print(f"Error: Audio file '{AUDIO_FILE}' not found.")
    exit()

try:
    pygame.mixer.music.load(AUDIO_FILE)
except Exception as e:
    print(f"Error loading audio: {e}")
    exit()

# 2. Load Database
all_known_encodings = []
all_known_names = []

if os.path.exists(DB_FILE):
    with open(DB_FILE, 'rb') as f:
        database = pickle.load(f)
        for name, encodings in database.items():
            for encoding in encodings:
                all_known_encodings.append(encoding)
                all_known_names.append(name)
    print(f"Loaded {len(database)} users.")
else:
    print("Error: No database found. Please run enroll.py first.")
    exit()

# 3. Initialize Video
cap = cv2.VideoCapture(0)

print("Starting monitoring... Press 'q' to quit.")
pygame.mixer.music.play()
audio_is_paused = False

# To prevent spamming the terminal, we only print device changes
last_device_name = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and Process
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Force contiguous array
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    status_text = "UNKNOWN"
    status_color = (0, 0, 255) # Red

    # --- CHECK AUDIO DEVICE ---
    current_device = get_current_audio_device()
    
    # Debug print (only prints if device changes)
    if current_device != last_device_name:
        print(f"[System] Audio Output Switched to: {current_device}")
        last_device_name = current_device

    # Check if "speaker" is in the name (e.g. "macbook pro speakers")
    is_on_speaker = 'speaker' in current_device

    # --- LOGIC GATES ---
    if len(face_locations) == 0:
        status_text = "WARNING: No face detected"
        should_play = False
            
    elif len(face_locations) > 1:
        status_text = "WARNING: Multiple faces detected"
        should_play = False

    elif is_on_speaker:
        status_text = "WARNING: Headphones not detected"
        # Ensure we explicitly mention the device on screen for debugging
        cv2.putText(frame, f"Device: {current_device[:20]}...", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        should_play = False
            
    else:
        # Valid Face + Headphones
        matches = face_recognition.compare_faces(all_known_encodings, face_encodings[0], tolerance=TOLERANCE)
        
        if True in matches:
            should_play = True
            status_text = "Authorized User Detected"
            status_color = (0, 255, 0) # Green
        else:
            should_play = False
            status_text = "WARNING: Unauthorized Person"

    # --- AUDIO CONTROL ---
    if should_play:
        if audio_is_paused:
            pygame.mixer.music.unpause()
            audio_is_paused = False
    else:
        if not audio_is_paused:
            pygame.mixer.music.pause()
            audio_is_paused = True

    # Display Status
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    for (top, right, bottom, left) in face_locations:
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), status_color, 2)

    cv2.imshow('Narration Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Stopping...")
pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()