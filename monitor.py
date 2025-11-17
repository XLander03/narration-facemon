import cv2
import face_recognition
import pickle
import pygame
import numpy as np

# --- Settings ---
ENCODING_FILE = "authorized_user.pkl"
AUDIO_FILE = "src/your_narration.mp3" # <-- PUT YOUR AUDIO FILE HERE
CAM_INDEX = 0 # Use the same working camera index
# --- End Settings ---

# 1. Initialize Audio
pygame.mixer.init()
try:
    pygame.mixer.music.load(AUDIO_FILE)
    print(f"Successfully loaded audio file: {AUDIO_FILE}")
except pygame.error as e:
    print(f"Error: Could not load audio file: {e}")
    print("Please make sure the file is in the correct path and is a valid format (e.g., mp3, wav).")
    exit()

# 2. Load the Authorized User's "Faceprint"
try:
    with open(ENCODING_FILE, 'rb') as f:
        authorized_encoding = pickle.load(f)
    print(f"Successfully loaded authorized user from {ENCODING_FILE}")
except FileNotFoundError:
    print(f"Error: Could not find encoding file: {ENCODING_FILE}")
    print("Please run the enroll.py script first to save your face.")
    exit()
except Exception as e:
    print(f"Error loading encoding file: {e}")
    exit()

# 3. Initialize Video
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"Error: Cannot open camera at index {CAM_INDEX}.")
    exit()

print("Starting monitoring... Press 'q' to quit.")
pygame.mixer.music.play()
audio_is_paused = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find all faces in the current frame
    # We resize for speed, but process the RGB frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    status = "UNKNOWN"
    color = (0, 0, 255) # Red for warnings

    if len(face_locations) == 0:
        status = "WARNING: No face detected."
        if not audio_is_paused:
            pygame.mixer.music.pause()
            audio_is_paused = True
            
    elif len(face_locations) > 1:
        status = "WARNING: Multiple faces detected."
        if not audio_is_paused:
            pygame.mixer.music.pause()
            audio_is_paused = True
            
    else:
        # One face found, check if it's the authorized one
        matches = face_recognition.compare_faces([authorized_encoding], face_encodings[0])
        
        if matches[0]:
            status = "Authorized User Detected. Audio Playing."
            color = (0, 255, 0) # Green for OK
            if audio_is_paused:
                pygame.mixer.music.unpause()
                audio_is_paused = False
        else:
            status = "WARNING: Unauthorized person."
            if not audio_is_paused:
                pygame.mixer.music.pause()
                audio_is_paused = True
                
    # Display the results
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow('Narration Monitoring (Press q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Stopping...")
pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()