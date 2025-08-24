import cv2
import numpy as np
import time
from collections import deque

class FaceDetector:
    def __init__(self):
        # Load multiple cascades for better accuracy
        self.face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        self.profile_cascade = self._load_cascade("haarcascade_profileface.xml")
        self.eye_cascade = self._load_cascade("haarcascade_eye.xml")
        
        # Detection parameters
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_face_size = (30, 30)
        self.max_face_size = (300, 300)
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.detection_history = deque(maxlen=10)
        
        # Face tracking for stability
        self.face_tracker = []
        
    def _load_cascade(self, filename):
        """Load Haar cascade with error handling"""
        cascade_path = cv2.data.haarcascades + filename
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            print(f"⚠️ Warning: Could not load {filename}")
            return None
        return cascade
    
    def _get_camera(self, indexes=(0, 1, 2)):
        """Try multiple camera indexes and backends"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
        
        for idx in indexes:
            for backend in backends:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Test if we can actually read frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Using camera {idx} with backend {backend}")
                        # Optimize camera settings
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                        return cap
                    cap.release()
        
        raise RuntimeError("❌ No working camera found")
    
    def _preprocess_frame(self, frame):
        """Optimize frame for better detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Optional: Apply Gaussian blur to reduce noise
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        return gray
    
    def _detect_faces(self, gray_frame):
        """Enhanced face detection using multiple methods"""
        all_faces = []
        
        # Primary frontal face detection
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                maxSize=self.max_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces)
        
        # Profile face detection (less frequent for performance)
        if self.profile_cascade is not None and self.frame_count % 3 == 0:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=self.min_face_size,
                maxSize=self.max_face_size
            )
            all_faces.extend(profile_faces)
        
        return self._merge_overlapping_faces(all_faces)
    
    def _merge_overlapping_faces(self, faces):
        """Merge overlapping face detections"""
        if len(faces) == 0:
            return []
        
        faces = list(faces)
        merged = []
        
        for face in faces:
            x, y, w, h = face
            merged_with_existing = False
            
            for i, existing in enumerate(merged):
                ex, ey, ew, eh = existing
                
                # Calculate overlap
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                face_area = w * h
                existing_area = ew * eh
                
                # If significant overlap, merge faces
                if overlap_area > 0.3 * min(face_area, existing_area):
                    # Keep the larger face
                    if face_area > existing_area:
                        merged[i] = face
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(face)
        
        return merged
    
    def _detect_eyes_in_face(self, gray_frame, face_roi):
        """Detect eyes within a face region for validation"""
        if self.eye_cascade is None:
            return []
        
        x, y, w, h = face_roi
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(15, 15),
            maxSize=(50, 50)
        )
        
        # Convert eye coordinates to global frame coordinates
        global_eyes = []
        for (ex, ey, ew, eh) in eyes:
            global_eyes.append((x + ex, y + ey, ew, eh))
        
        return global_eyes
    
    def _calculate_fps(self, start_time):
        """Calculate and track FPS"""
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        return np.mean(self.fps_history)
    
    def _draw_detections(self, frame, faces, eyes, fps):
        """Draw face and eye detections with enhanced visualization"""
        output = frame.copy()
        
        # Draw faces
        for i, (x, y, w, h) in enumerate(faces):
            # Face rectangle with rounded corners effect
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Face label with background
            label = f"Face {i + 1}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Confidence indicator (based on face size)
            confidence = min(100, (w * h) / 100)  # Simple confidence metric
            cv2.putText(output, f"{confidence:.0f}%", (x + w - 50, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(output, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
            cv2.circle(output, (ex + ew // 2, ey + eh // 2), 2, (255, 0, 0), -1)
        
        # Performance info overlay
        self._draw_info_overlay(output, len(faces), len(eyes), fps)
        
        return output
    
    def _draw_info_overlay(self, frame, face_count, eye_count, fps):
        """Draw performance and detection info overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Faces: {face_count}",
            f"Eyes: {eye_count}",
            f"Frame: {self.frame_count}",
            "Press 'q' to quit"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main detection loop"""
        cap = self._get_camera()
        
        print("🎥 Face detection started")
        print("📊 Features: Multi-cascade detection, eye validation, FPS tracking")
        print("⌨️  Controls: 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("⚠️ Failed to grab frame")
                    continue
                
                self.frame_count += 1
                
                # Preprocess frame
                gray = self._preprocess_frame(frame)
                
                # Detect faces
                faces = self._detect_faces(gray)
                
                # Detect eyes within faces for validation
                all_eyes = []
                validated_faces = []
                
                for face in faces:
                    eyes = self._detect_eyes_in_face(gray, face)
                    all_eyes.extend(eyes)
                    
                    # Consider face valid if it has at least 1 eye or is large enough
                    if len(eyes) >= 1 or (face[2] * face[3]) > 5000:
                        validated_faces.append(face)
                
                # Calculate FPS
                fps = self._calculate_fps(start_time)
                
                # Draw detections
                output = self._draw_detections(frame, validated_faces, all_eyes, fps)
                
                # Display result
                cv2.imshow("Enhanced Face Detection", output)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"face_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, output)
                    print(f"📸 Screenshot saved: {filename}")
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            print(f"✅ Detection completed")
            print(f"📈 Average FPS: {avg_fps:.1f}")
            print(f"🎯 Total frames processed: {self.frame_count}")

def main():
    """Main function with error handling"""
    try:
        detector = FaceDetector()
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try checking camera permissions or updating OpenCV")

if __name__ == "__main__":
    main()