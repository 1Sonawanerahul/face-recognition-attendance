import cv2
import os
import numpy as np
import pickle

def train_face_model():
    """Train face recognition model from images in 'faces' folder"""
    print("="*50)
    print("FACE MODEL TRAINING")
    print("="*50)
    
    # Check if faces directory exists
    if not os.path.exists("faces"):
        print("❌ 'faces' folder not found!")
        print("Please create 'faces' folder and add photos.")
        print("Name files like: John_1.jpg, John_2.jpg, Sarah_1.jpg")
        return
    
    # Get all image files
    image_paths = []
    for root, dirs, files in os.walk("faces"):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("❌ No images found in 'faces' folder!")
        print("Add photos of people first.")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Prepare training data
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Process each image
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        print(f"Processing: {filename}")
        
        # Get name from filename (format: Name_1.jpg)
        # Remove extension first
        name_without_ext = os.path.splitext(filename)[0]
        
        # Extract name (part before first underscore or dot)
        if '_' in name_without_ext:
            name = name_without_ext.split('_')[0]
        else:
            name = name_without_ext
        
        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1
        
        label_id = label_ids[name]
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ⚠️  Could not read: {filename}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(detected_faces) == 0:
            print(f"  ⚠️  No face detected in: {filename}")
            continue
        
        # Use the first face found
        (x, y, w, h) = detected_faces[0]
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to standard size
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Add to training data
        faces.append(face_roi)
        labels.append(label_id)
        
        print(f"  ✓ Added: {name} (ID: {label_id})")
    
    if not faces:
        print("❌ No valid faces found for training!")
        return
    
    print(f"\nTraining with {len(faces)} face samples...")
    
    # Train the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    # Save the model
    recognizer.save("face_model.yml")
    
    # Save labels
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    
    print("\n✅ Training completed!")
    print(f"✅ Model saved: face_model.yml")
    print(f"✅ Labels saved: labels.pickle")
    print(f"✅ Trained {len(label_ids)} people:")
    for name, label_id in label_ids.items():
        count = labels.count(label_id)
        print(f"   - {name}: {count} image(s)")
    
    # Create simple face templates for backup matching
    create_simple_templates(faces, labels, label_ids)

def create_simple_templates(faces, labels, label_ids):
    """Create simple average templates for each person"""
    print("\nCreating simple templates...")
    
    templates = {}
    
    # Group faces by label
    for face, label in zip(faces, labels):
        # Find name for this label
        for name, lid in label_ids.items():
            if lid == label:
                if name not in templates:
                    templates[name] = []
                templates[name].append(face)
                break
    
    # Calculate average face for each person
    avg_faces = {}
    for name, face_list in templates.items():
        if face_list:
            # Average all faces
            avg_face = np.mean(face_list, axis=0).astype(np.uint8)
            avg_faces[name] = avg_face
    
    # Save templates
    with open("face_templates.pickle", 'wb') as f:
        pickle.dump(avg_faces, f)
    
    print(f"✅ Created {len(avg_faces)} face templates")

def collect_photos():
    """Collect photos for training via webcam"""
    print("\n" + "="*50)
    print("COLLECT TRAINING PHOTOS")
    print("="*50)
    
    name = input("Enter person's name: ").strip()
    if not name:
        print("No name entered. Cancelled.")
        return
    
    num_photos = input("How many photos to take? (default: 20): ").strip()
    try:
        num_photos = int(num_photos) if num_photos else 20
    except:
        num_photos = 20
    
    print(f"\nWill take {num_photos} photos of {name}")
    print("Press 'C' to capture, 'Q' to quit")
    
    # Create directory for this person
    person_dir = f"dataset/{name}"
    os.makedirs(person_dir, exist_ok=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    count = 0
    
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display count
            cv2.putText(frame, f"Photos: {count}/{num_photos}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Person: {name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'C' to capture, 'Q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Collect Photos', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nStopping collection...")
            break
        elif key == ord('c') or key == ord('C'):
            if len(faces) > 0:
                # Take the first face
                x, y, w, h = faces[0]
                
                # Save face
                face_roi = frame[y:y+h, x:x+w]
                filename = f"{person_dir}/{name}_{count+1}.jpg"
                cv2.imwrite(filename, face_roi)
                count += 1
                print(f"  ✓ Captured photo {count}/{num_photos}")
                
                # Also save to faces folder for immediate training
                faces_dir = "faces"
                os.makedirs(faces_dir, exist_ok=True)
                cv2.imwrite(f"{faces_dir}/{name}_{count}.jpg", face_roi)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Collected {count} photos of {name}")
    print(f"✅ Saved to: {person_dir}/")
    
    # Ask to train now
    train_now = input("\nTrain model now? (y/n): ").lower()
    if train_now == 'y':
        train_face_model()

def main():
    """Main training function"""
    print("SIMPLE FACE RECOGNITION TRAINING")
    print("="*50)
    print("1. Train from existing photos")
    print("2. Collect new photos via webcam")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        train_face_model()
    elif choice == '2':
        collect_photos()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()