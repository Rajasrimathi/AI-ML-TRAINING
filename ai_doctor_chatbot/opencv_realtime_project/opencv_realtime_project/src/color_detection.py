import cv2
import numpy as np
import csv
import time
from collections import Counter

def get_camera(indexes=(0, 1, 2, 3)):
    """Try multiple backends and camera indexes until one works"""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    for idx in indexes:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    print(f"✅ Using camera index {idx} with backend {backend}")
                    return cap
                cap.release()
    raise RuntimeError("❌ No working camera found. Try checking drivers/permissions.")

# Enhanced HSV ranges with better accuracy
COLORS_HSV = {
    "Red": [[(0, 50, 50), (10, 255, 255)], [(160, 50, 50), (180, 255, 255)]],  # Red wraps around
    "Green": [[(40, 50, 50), (80, 255, 255)]],
    "Blue": [[(100, 50, 50), (130, 255, 255)]],
    "Yellow": [[(20, 50, 50), (30, 255, 255)]],
    "Orange": [[(10, 50, 50), (20, 255, 255)]],
    "Purple": [[(130, 50, 50), (160, 255, 255)]],
    "Cyan": [[(80, 50, 50), (100, 255, 255)]],
    "Pink": [[(140, 30, 50), (170, 255, 255)]],
    "Brown": [[(8, 50, 20), (20, 255, 200)]],
    "White": [[(0, 0, 200), (180, 30, 255)]],
    "Black": [[(0, 0, 0), (180, 255, 50)]],
    "Gray": [[(0, 0, 50), (180, 30, 200)]]
}

def preprocess_frame(frame):
    """Apply preprocessing to improve color detection"""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Apply bilateral filter to smooth colors while preserving edges
    filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
    
    return filtered

def create_color_mask(hsv, color_ranges):
    """Create mask for colors that might have multiple HSV ranges"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in color_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        temp_mask = cv2.inRange(hsv, lower_np, upper_np)
        mask = cv2.bitwise_or(mask, temp_mask)
    
    return mask

def post_process_mask(mask):
    """Apply morphological operations to clean up the mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def get_dominant_colors(frame, hsv, num_colors=3):
    """Get the most dominant colors in the frame using contour analysis"""
    detected_colors = []
    color_areas = {}
    
    for color_name, ranges in COLORS_HSV.items():
        mask = create_color_mask(hsv, ranges)
        mask = post_process_mask(mask)
        
        # Find contours to get actual color regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small noise
                total_area += area
        
        if total_area > 1000:  # Significant presence threshold
            color_areas[color_name] = total_area
            detected_colors.append((color_name, mask, total_area))
    
    # Sort by area and return top colors
    detected_colors.sort(key=lambda x: x[2], reverse=True)
    return detected_colors[:num_colors], color_areas

def draw_color_info(frame, detected_colors, color_areas):
    """Draw color information on the frame"""
    output = frame.copy()
    
    # Draw semi-transparent overlay for text background
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (400, 60 + len(detected_colors) * 35), (0, 0, 0), -1)
    output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
    
    # Title
    cv2.putText(output, "Detected Colors (by dominance):", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw detected colors with area percentages
    total_area = sum(color_areas.values()) if color_areas else 1
    
    for i, (color_name, mask, area) in enumerate(detected_colors):
        percentage = (area / total_area) * 100 if total_area > 0 else 0
        text = f"{i+1}. {color_name}: {percentage:.1f}%"
        
        # Color indicator
        color_bgr = get_color_bgr(color_name)
        cv2.rectangle(output, (20, 50 + i * 35), (40, 70 + i * 35), color_bgr, -1)
        cv2.rectangle(output, (20, 50 + i * 35), (40, 70 + i * 35), (255, 255, 255), 1)
        
        # Text
        cv2.putText(output, text, (50, 68 + i * 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output

def get_color_bgr(color_name):
    """Get BGR color for visualization"""
    color_map = {
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "Orange": (0, 165, 255),
        "Purple": (128, 0, 128),
        "Cyan": (255, 255, 0),
        "Pink": (203, 192, 255),
        "Brown": (42, 42, 165),
        "White": (255, 255, 255),
        "Black": (0, 0, 0),
        "Gray": (128, 128, 128)
    }
    return color_map.get(color_name, (128, 128, 128))

def main():
    cap = get_camera()
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create CSV file for logging
    with open("detected_colors.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Rank", "Color", "Percentage", "Area"])
        
        print("🎥 Color detection started. Press 'q' to quit, 'c' to calibrate.")
        print("📝 Results are being logged to 'detected_colors.csv'")
        
        frame_count = 0
        detection_history = []
        
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                frame_count += 1
                
                # Preprocess frame
                processed_frame = preprocess_frame(frame)
                hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                
                # Get dominant colors
                detected_colors, color_areas = get_dominant_colors(frame, hsv)
                
                # Stabilize detection using history (every 5 frames)
                if frame_count % 5 == 0 and detected_colors:
                    detection_history.append([color[0] for color in detected_colors])
                    if len(detection_history) > 5:
                        detection_history.pop(0)
                
                # Draw visualization
                output = draw_color_info(frame, detected_colors, color_areas)
                
                # Add frame count and FPS info
                cv2.putText(output, f"Frame: {frame_count}", (output.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Log to CSV (every 30 frames to avoid spam)
                if frame_count % 30 == 0 and detected_colors:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    total_area = sum(color_areas.values()) if color_areas else 1
                    
                    for rank, (color_name, _, area) in enumerate(detected_colors, 1):
                        percentage = (area / total_area) * 100 if total_area > 0 else 0
                        writer.writerow([timestamp, rank, color_name, f"{percentage:.1f}", area])
                    
                    file.flush()
                
                # Show result
                cv2.imshow("Enhanced Color Detection", output)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("📸 Calibration frame captured - current detection:")
                    for i, (color, _, area) in enumerate(detected_colors, 1):
                        print(f"  {i}. {color}: {area} pixels")
                        
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Color detection completed. Check 'detected_colors.csv' for logs.")

if __name__ == "__main__":
    main()