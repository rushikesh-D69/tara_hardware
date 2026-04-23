import cv2
import os
import time
import argparse

"""
TARA ADAS — Data Collection Tool
This script helps you quickly capture and label images for your traffic sign classifier.
Run this on your RPi or Laptop with your USB webcam.

Usage:
  python3 capture_data.py --id 14   # To capture 'Stop' signs
  python3 capture_data.py --id 1    # To capture '30 Speed Limit' signs
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True, help="Class ID (e.g. 14 for Stop, 1 for 30km/h)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()

    # Create directories
    class_dir = os.path.join(args.output, str(args.id))
    os.makedirs(class_dir, exist_ok=True)

    # Initialize camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.cam}")
        return

    print(f"\n--- TARA Data Collection: Class {args.id} ---")
    print("Instructions:")
    print("1. Place your printed sign inside the GREEN box.")
    print("2. Press 'S' to SAVE a photo.")
    print("3. Press 'Q' to QUIT when done.")
    print("-------------------------------------------\n")

    count = len(os.listdir(class_dir))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # --- ROI Settings (matching traffic_sign.py) ---
        # We look at the center 40% of the frame.
        roi_y1, roi_y2 = int(h * 0.3), int(h * 0.7)
        roi_x1, roi_x2 = int(w * 0.3), int(w * 0.7)
        
        # Draw guide UI
        display = frame.copy()
        cv2.rectangle(display, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(display, f"Class: {args.id} | Saved: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press 'S' to Save | 'Q' to Quit", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("TARA Data Collector", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Crop the ROI
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Resize to the standard 96x96 for MobileNet
            roi_sized = cv2.resize(roi, (96, 96))
            
            # Save the file
            filename = os.path.join(class_dir, f"img_{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, roi_sized)
            
            count += 1
            print(f"Saved image {count} to {class_dir}")
            
            # Flash the screen text briefly
            cv2.rectangle(display, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), -1)
            cv2.imshow("TARA Data Collector", display)
            cv2.waitKey(50)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! Collected {count} images for class {args.id}.")

if __name__ == "__main__":
    main()
