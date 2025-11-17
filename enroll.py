import cv2
import face_recognition
import pickle
import os

# --- Settings ---
ENCODING_FILE = "authorized_user.pkl"
# --- End Settings ---

print("Starting enrollment...")
print("A window will open. Please look at the camera.")
print("Press 'c' to capture your image. Press 'q' to quit.")

# Use the working camera index you found from test_cam.py
CAM_INDEX = 0 
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"Error: Cannot open camera at index {CAM_INDEX}.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't get frame.")
        break
        
    cv2.imshow('Enrollment - Press C to capture', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        print("Capturing image...")
        
        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find the face
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            print("No face found. Please try again.")
        elif len(face_locations) > 1:
            print("Multiple faces found. Please ensure only you are in the frame.")
        else:
            # Get the encoding (the "faceprint")
            user_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            
            # Save this encoding to a file
            with open(ENCODING_FILE, 'wb') as f:
                pickle.dump(user_encoding, f)
            
            print(f"Success! Your 'faceprint' has been saved to {ENCODING_FILE}")
            break # Exit after successful capture

cap.release()
cv2.destroyAllWindows()