import cv2
import numpy as np
import os
import pickle
import csv
from datetime import datetime
import time

class SimpleFaceAttendance:
    def __init__(self):
        # Load trained model
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load trained data if exists
        self.load_trained_data()
        
        # Attendance file
        self.attendance_file = "attendance.csv"
        self.init_attendance_file()
        
        # Track marked attendance today
        self.marked_today = set()
        self.load_today_attendance()
    
    def load_trained_data(self):
        """Load trained face recognizer and labels"""
        try:
            if os.path.exists("face_model.yml"):
                self.face_recognizer.read("face_model.yml")
                print("✓ Loaded trained face model")
            
            if os.path.exists("labels.pickle"):
                with open("labels.pickle", 'rb') as f:
                    self.labels = pickle.load(f)
                print(f"✓ Loaded {len(self.labels)} known faces")
                
                # Invert labels dictionary
                self.label_to_name = {v: k for k, v in self.labels.items()}
            else:
                print("⚠️ No trained faces found. Please train the model first.")
                print("   Run: python train_model.py")
                
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def init_attendance_file(self):
        """Initialize attendance CSV file"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time', 'Status'])
            print(f"✓ Created attendance file: {self.attendance_file}")
    
    def load_today_attendance(self):
        """Load today's attendance to avoid duplicates"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[1] == today:
                        self.marked_today.add(row[0])
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        if name in self.marked_today:
            print(f"Already marked today: {name}")
            return False
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str, "Present"])
        
        self.marked_today.add(name)
        print(f"✓ Attendance marked: {name} at {time_str}")
        return True
    
    def recognize_face(self, face_img):
        """Recognize face using LBPH recognizer"""
        if not hasattr(self, 'label_to_name') or not self.label_to_name:
            return "Unknown", 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (200, 200))
        
        # Predict
        label, confidence = self.face_recognizer.predict(gray)
        
        # Convert confidence to percentage (lower is better in LBPH)
        confidence_percent = max(0, 100 - confidence)
        
        if confidence_percent > 50:  # Confidence threshold
            name = self.label_to_name.get(label, "Unknown")
            return name, confidence_percent
        
        return "Unknown", confidence_percent
    
    def simple_face_match(self, face_img):
        """Simple face matching using template matching"""
        if not hasattr(self, 'known_faces') or not self.known_faces:
            return "Unknown", 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        
        best_match = None
        best_score = float('inf')
        
        for name, known_face in self.known_faces.items():
            # Calculate MSE (Mean Squared Error)
            mse = np.mean((gray - known_face) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_match = name
        
        # Threshold for recognition
        if best_score < 2000:
            confidence = max(0, 100 - (best_score / 40))
            return best_match, confidence
        
        return "Unknown", 0
    
    def run(self):
        """Run the face recognition attendance system"""
        print("\n" + "="*50)
        print("SIMPLE FACE RECOGNITION ATTENDANCE")
        print("="*50)
        print(f"Known faces: {len(self.labels)}")
        print(f"Already marked today: {len(self.marked_today)}")
        print("\nCommands:")
        print("  Press 'Q' to quit")
        print("  Press 'T' to take a photo and add new person")
        print("  Press 'M' to manually mark attendance")
        print("="*50 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam!")
            return
        
        # For FPS calculation
        prev_time = time.time()
        recognized_names = set()
        
        print("Starting face recognition...")
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame!")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each face
            current_names = []
            for (x, y, w, h) in faces:
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Recognize face
                name, confidence = self.recognize_face(face_roi)
                
                # If unknown, try simple matching
                if name == "Unknown":
                    name, confidence = self.simple_face_match(face_roi)
                
                # Display name and confidence
                if name != "Unknown":
                    display_text = f"{name} ({confidence:.1f}%)"
                    color = (0, 255, 0)
                    current_names.append(name)
                    
                    # Mark attendance if confidence is high
                    if confidence > 70 and name not in recognized_names:
                        self.mark_attendance(name)
                        recognized_names.add(name)
                else:
                    display_text = f"Unknown ({confidence:.1f}%)"
                    color = (0, 0, 255)
                
                # Put text
                cv2.putText(frame, display_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Display info
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Marked today: {len(self.marked_today)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Face Recognition Attendance', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nStopping system...")
                break
            elif key == ord('t') or key == ord('T'):
                # Take photo for new person
                self.take_photo_for_training(frame, faces)
            elif key == ord('m') or key == ord('M'):
                # Manual attendance marking
                self.manual_attendance()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        self.show_summary()
    
    def take_photo_for_training(self, frame, faces):
        """Take photo of a face for training"""
        if len(faces) == 0:
            print("No face detected for photo!")
            return
        
        # Take the first face
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Ask for name
        name = input("\nEnter name for this person: ").strip()
        if not name:
            print("No name entered. Cancelled.")
            return
        
        # Create faces directory if not exists
        if not os.path.exists("faces"):
            os.makedirs("faces")
        
        # Save face image
        face_filename = f"faces/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(face_filename, face_roi)
        print(f"✓ Photo saved: {face_filename}")
        
        # Ask to retrain
        retrain = input("Retrain model now? (y/n): ").lower()
        if retrain == 'y':
            print("Please run: python train_model.py")
    
    def manual_attendance(self):
        """Manually mark attendance"""
        print("\n" + "-"*40)
        print("MANUAL ATTENDANCE MARKING")
        print("-"*40)
        
        if self.labels:
            print("Known people:")
            for name in self.labels.keys():
                print(f"  - {name}")
        
        name = input("\nEnter name to mark attendance: ").strip()
        if name:
            self.mark_attendance(name)
        else:
            print("No name entered.")
    
    def show_summary(self):
        """Show attendance summary"""
        print("\n" + "="*50)
        print("ATTENDANCE SUMMARY")
        print("="*50)
        print(f"Total marked today: {len(self.marked_today)}")
        
        if self.marked_today:
            print("\nAttendees today:")
            for name in sorted(self.marked_today):
                print(f"  ✓ {name}")
        
        # Show recent entries
        if os.path.exists(self.attendance_file):
            print(f"\nAttendance saved to: {self.attendance_file}")
            print("\nLatest entries:")
            with open(self.attendance_file, 'r') as f:
                lines = f.readlines()
                # Show last 10 entries
                for line in lines[-10:]:
                    print(f"  {line.strip()}")
        
        print("\nSession ended.")

def check_dependencies():
    """Check if OpenCV is installed"""
    try:
        import cv2
        print("✓ OpenCV is installed")
        return True
    except ImportError:
        print("❌ OpenCV not installed!")
        print("   Run: pip install opencv-python numpy")
        return False

def main():
    """Main function"""
    print("="*50)
    print("SIMPLE FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*50)
    print("No dlib required! Only OpenCV needed.")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if model exists
    if not os.path.exists("face_model.yml"):
        print("\n⚠️  No trained model found!")
        print("Please train the model first:")
        print("  1. Add photos of people to 'faces' folder")
        print("  2. Run: python train_model.py")
        print("\nOr press Enter to continue with basic face detection...")
        input()
    
    # Create necessary directories
    os.makedirs("faces", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    
    # Run the system
    system = SimpleFaceAttendance()
    system.run()

if __name__ == "__main__":
    main()