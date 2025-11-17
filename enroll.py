import cv2
import face_recognition
import pickle
import os
import numpy as np

# --- Settings ---
DB_FILE = "data/people/authorized_users.pkl"
# --- End Settings ---

def load_database():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        except EOFError:
            return {}
    return {}

def save_database(db):
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)

def enroll_user():
    database = load_database()
    
    name = input("Enter the name of the user to enroll: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    if name in database:
        print(f"Warning: '{name}' already exists. Overwriting...")
    
    print(f"\n--- Enrolling {name} ---")
    print("We need to capture 3 angles: Front, Left, and Right.")
    print("Press 'c' to capture each angle. Press 'q' to quit at any time.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    angles = ["Front View", "Turn Head Left", "Turn Head Right"]
    user_encodings = []

    for angle in angles:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break

            # Display instructions on screen
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capture: {angle}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'c' to capture", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Enrollment', display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Enrollment cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                # Process image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # CRITICAL FIX: Force contiguous array
                rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

                face_locations = face_recognition.face_locations(rgb_frame)

                if len(face_locations) == 0:
                    print(f"❌ No face detected. Please look at the camera for {angle}.")
                elif len(face_locations) > 1:
                    print("❌ Multiple faces detected. Ensure only you are in frame.")
                else:
                    # Get encoding
                    encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    user_encodings.append(encoding)
                    print(f"✅ Captured {angle}!")
                    break # Move to next angle

    # Save to database
    database[name] = user_encodings
    save_database(database)
    print(f"\nSuccess! User '{name}' enrolled with {len(user_encodings)} angles.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll_user()