import cv2
import numpy as np
import time
import os
from typing import Optional, List, Tuple, Dict

class BlueDetector:
    def __init__(self):
        # Default blue color range (HSV)
        self.color_ranges = {
            'light_blue': ((90, 50, 50), (130, 255, 255)),
            'dark_blue': ((100, 150, 0), (140, 255, 255)),
            'navy_blue': ((110, 100, 0), (130, 255, 100))
        }
        
        self.current_range = 'light_blue'
        self.min_area = 500
        self.frame_count = 0
        self.total_detections = 0
        self.detection_history = []
        self.save_output = False
        self.output_dir = "detections"
        os.makedirs(self.output_dir, exist_ok=True)

    def set_blue_range(self, range_name: str):
        """Switch to a different predefined blue color range"""
        if range_name in self.color_ranges:
            self.current_range = range_name
            print(f"🎨 Switched to {range_name} detection range")
        else:
            print(f"❌ Invalid range: {range_name}")

    def detect_blue_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Detect blue objects in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.color_ranges[self.current_range]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.medianBlur(mask, 5)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
        
        return mask, frame, detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            cx, cy = det['center']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Blue {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def add_info_panel(self, frame: np.ndarray, fps: float, detections_count: int) -> np.ndarray:
        """Add FPS and detection info overlay"""
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detections_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Range: {self.current_range}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame

    def update_stats(self, detections_count: int):
        """Update detection statistics"""
        self.total_detections += detections_count
        self.detection_history.append(detections_count)
        if len(self.detection_history) > 1000:
            self.detection_history.pop(0)

    def save_frame(self, frame: np.ndarray, prefix: str):
        """Save frame if enabled"""
        if self.save_output:
            timestamp = int(time.time() * 1000)
            cv2.imwrite(os.path.join(self.output_dir, f"{prefix}_{timestamp}.jpg"), frame)

    def print_controls(self):
        print("\n🎮 Controls:")
        print("1: Light Blue detection")
        print("2: Dark Blue detection")
        print("3: Navy Blue detection")
        print("s: Show statistics")
        print("c: Capture frame")
        print("p: Pause/Resume")
        print("q/ESC: Quit\n")

    def print_statistics(self):
        avg_detections = (sum(self.detection_history) / len(self.detection_history)
                          if self.detection_history else 0)
        print("\n📊 Current Statistics:")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        print(f"Average detections/frame: {avg_detections:.2f}")
        print(f"Current detection range: {self.current_range}\n")

    def print_final_statistics(self):
        print("\n📊 Final Statistics:")
        print(f"Total frames: {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        if self.frame_count > 0:
            print(f"Average detections/frame: {self.total_detections/self.frame_count:.2f}")
        print("✅ Detection complete\n")

    def run_detection(self, source: Optional[int] = None, input_file: Optional[str] = None):
        """Run detection on webcam or video"""
        if input_file:
            cap = cv2.VideoCapture(input_file)
            print(f"📁 Processing video file: {input_file}")
        else:
            cap = cv2.VideoCapture(source if source is not None else 0)
            print(f"📹 Starting camera detection (Camera {source if source is not None else 0})")
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        self.print_controls()
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0.0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("📺 End of video stream")
                        break
                    
                    self.frame_count += 1
                    mask, _, detections = self.detect_blue_objects(frame)
                    frame = self.draw_detections(frame, detections)
                    frame = self.add_info_panel(frame, fps, len(detections))
                    self.update_stats(len(detections))
                    
                    fps_counter += 1
                    if fps_counter >= 30:
                        fps = fps_counter / (time.time() - fps_start_time)
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    self.save_frame(frame, "detection")
                    
                    # ✅ Show only one window
                    cv2.imshow("🔵 Blue Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('1'):
                    self.set_blue_range('light_blue')
                elif key == ord('2'):
                    self.set_blue_range('dark_blue')
                elif key == ord('3'):
                    self.set_blue_range('navy_blue')
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('c'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"capture_{timestamp}.jpg", frame)
                    print(f"📸 Frame captured: capture_{timestamp}.jpg")
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️ {'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_statistics()


if __name__ == "__main__":
    detector = BlueDetector()
    detector.save_output = False  # Change to True if you want to save detections
    detector.run_detection(0)  # 0 = default webcam
