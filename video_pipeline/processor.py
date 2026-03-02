import cv2
import os
from pathlib import Path

class VideoProcessor:
    def __init__(self, detector, recognizer, verifier):
        self.detector = detector
        self.recognizer = recognizer
        self.verifier = verifier
    
    def extract_frames(self, video_path, fps=5, output_dir="outputs/frames"):
        """Extract frames from video"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def process_video(self, video_path, output_file="outputs/video_results.txt"):
        """Process entire video pipeline"""
        print("Extracting frames...")
        frames = self.extract_frames(video_path)
        
        results = []
        seen_plates = set()
        
        print(f"Processing {len(frames)} frames...")
        for frame_path in frames:
            # Detect
            detections = self.detector.detect(frame_path)
            
            for det in detections:
                # Crop plate
                plate_img = self.detector.crop_plate(frame_path, det['plate_bbox'])
                
                # Recognize
                recognition = self.recognizer.predict(plate_img)
                
                plate_text = recognition.get('plate', '')
                
                # Skip duplicates
                if plate_text in seen_plates:
                    continue
                seen_plates.add(plate_text)
                
                # Verify
                verification = self.verifier.verify(
                    plate_text,
                    recognition.get('color', ''),
                    recognition.get('type', '')
                )
                
                result = {
                    "frame": frame_path,
                    "detection": det,
                    "recognition": recognition,
                    "verification": verification
                }
                
                results.append(result)
                
                # Print suspicious cases
                if verification['status'] == 'suspicious':
                    print(f"⚠️ SUSPICIOUS: {plate_text} - {verification['flags']}")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"{r}\n")
        
        return results
