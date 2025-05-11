from ultralytics import YOLO
import cv2
import torch
import gc
import numpy as np
import os
import threading
import base64
from threading import Lock
import time
import requests
from datetime import datetime, timedelta
import io
import tempfile
from config import API_ENDPOINTS

class ShopliftDetector:
    def __init__(self):
        # Memory optimization 
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Initialize CUDA settings once
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        # Display settings
        self.target_width = 640
        self.target_height = 480
        self.display_width = 1024
        self.display_height = 768

        # Thread safety
        self.lock = Lock()
        self.frame_data = None
        self.is_running = False
        self.thread = None
        
        # Analysis tracking
        self.camera_id = None
        self.auth_token = None
        self.last_analysis_time = datetime.now()
        self.total_events = 0
        self.suspicious_events = 0

        # Alert mechanism - improved settings for pre-alert video
        self.min_alert_seconds = 3     # Minimum alert video length
        self.max_buffer_seconds = 20
        self.frame_buffer = []         # All frames including pre-alert frames
        self.fps = 30  # Default FPS, will be updated from video
        self.buffer_size = 0           # Will be calculated based on FPS
        self.cooldown_active = False
        self.cooldown_duration = timedelta(seconds=30)  # 30 second cooldown between alerts
        self.cooldown_start_time = None
        self.last_buffer_adjustment_time = datetime.now()
        
        # Continuous buffer for proper video recording
        self.continuous_frame_buffer = []  # Keep a separate continuous buffer just for video creation
        self.continuous_buffer_size = 300  # Store 10 seconds @ 30fps (adjust based on actual FPS later)
        self.continuous_buffer_lock = Lock()  # Separate lock for the continuous buffer
        
        # Detection thresholds - adjusted for single-person shoplifting
        # Path 1: Moderate ratio but normal confidence
        self.suspicious_ratio_threshold_normal = 0.50  # Lowered from 0.65 to 0.50
        self.confidence_threshold_normal = 0.70       # Maintained at 0.70
        
        # Path 2: Low ratio but high confidence (for single-person shoplifting)
        self.suspicious_ratio_threshold_early = 0.30  # Lowered from 0.75 to 0.30
        self.confidence_threshold_early = 0.85       # Increased from 0.80 to 0.85
        
        # Test mode flag
        self.test_mode = False
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "bestm.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model.half()

        # Add verbose setting
        self.verbose = False
        
        print("ShopliftDetector initialized with adjusted detection settings for single-person shoplifting")
        
    def update_continuous_buffer(self, new_frame):
        """Add a new frame to the continuous buffer for proper video recording"""
        with self.continuous_buffer_lock:
            # Save the frame with timestamp
            frame_entry = {
                "frame": new_frame.copy(),
                "timestamp": datetime.now()
            }
            self.continuous_frame_buffer.append(frame_entry)
            
            # Maintain the buffer size
            while len(self.continuous_frame_buffer) > self.continuous_buffer_size:
                self.continuous_frame_buffer.pop(0)
                
    def get_continuous_buffer_video(self, n_seconds):
        """Get the last n_seconds of footage from the continuous buffer"""
        with self.continuous_buffer_lock:
            # Calculate how many frames we need based on FPS
            n_frames = int(self.fps * n_seconds)
            
            # If we don't have enough frames, use what we have
            if len(self.continuous_frame_buffer) == 0:
                return [], []
                
            n_frames = min(n_frames, len(self.continuous_frame_buffer))
            
            # Check if we have at least 1 second of footage (based on FPS)
            min_frames = int(self.fps)
            if n_frames < min_frames:
                print(f"Warning: Only have {n_frames} frames, which is less than 1 second at {self.fps} FPS")
                if len(self.continuous_frame_buffer) >= min_frames:
                    n_frames = min_frames
                    
            # Get the frames, most recent last
            frames = self.continuous_frame_buffer[-n_frames:]
            
            # Return both the raw frame entries and just the frames
            return frames, [f["frame"] for f in frames]

    def trigger_alert(self):
        """Trigger shoplifting alert and save evidence clip"""
        if not self.camera_id or not self.auth_token:
            print(f"Cannot trigger alert: camera_id={self.camera_id}, auth_token={bool(self.auth_token)}")
            return False
            
        try:
            # First check if the camera is still active in the database
            try:
                import requests
                # Use a status check API endpoint
                status_url = f"{API_ENDPOINTS['check_camera_status']}{self.camera_id}/"
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                status_response = requests.get(status_url, headers=headers, timeout=5)
                
                if status_response.status_code != 200:
                    # Camera might be inactive or deleted
                    print(f"Camera status check failed: {status_response.status_code} - {status_response.text}")
                    # Check if we got a specific "camera inactive" response
                    response_data = status_response.json()
                    if response_data.get('status') == 'inactive':
                        print(f"Camera {self.camera_id} is inactive, stopping detection")
                        self.is_running = False
                        # Signal to the detector that it should stop
                        self.is_stopping = True
                        return False
            except Exception as status_err:
                # If we can't check status, continue with alert creation
                # This is a fallback in case the status endpoint isn't available
                print(f"Error checking camera status: {status_err}")
            
            # Check if we're stopping - don't send alerts when shutting down
            if hasattr(self, 'is_stopping') and self.is_stopping:
                print("Detector is shutting down, skipping alert creation")
                return False
                
            # Mark the alert start time
            alert_start_time = datetime.now()
            print(f"🚨 Creating alert at {alert_start_time.strftime('%H:%M:%S')}")
            
            # Store the current buffer for processing - so we're not affected by ongoing updates
            with self.continuous_buffer_lock:
                # Make a deep copy to ensure we're not affected by updates during processing
                alert_buffer = []
                for entry in self.continuous_frame_buffer:
                    alert_buffer.append({
                        "timestamp": entry["timestamp"],
                        "frame": entry["frame"].copy()  # Make a copy of each frame
                    })
            
            # Create a video using the copied buffer - get 9 seconds of footage
            # We'll use a fixed length of 9 seconds to ensure consistent videos
            seconds_of_footage = 9
            
            # Calculate how many frames we need based on FPS
            n_frames = int(self.fps * seconds_of_footage)
            
            # If we don't have enough frames, use what we have
            if len(alert_buffer) == 0:
                print("No frames in buffer for video")
                return False
                
            n_frames = min(n_frames, len(alert_buffer))
            
            # Check if we have at least 3 seconds of footage (based on FPS)
            min_frames = int(self.fps * 3)
            if n_frames < min_frames:
                print(f"Warning: Only have {n_frames} frames, which is less than 3 seconds at {self.fps} FPS")
                if len(alert_buffer) < min_frames:
                    print("Not enough frames for a meaningful video")
                    # Try to get more frames - wait a bit for buffer to fill
                    time.sleep(2)
                    
                    # Check buffer again
                    with self.continuous_buffer_lock:
                        if len(self.continuous_frame_buffer) < min_frames:
                            print(f"Still not enough frames after waiting, have {len(self.continuous_frame_buffer)}")
                            return False
                        
                        # Try again with the updated buffer
                        alert_buffer = []
                        for entry in self.continuous_frame_buffer:
                            alert_buffer.append({
                                "timestamp": entry["timestamp"],
                                "frame": entry["frame"].copy()
                            })
                            
                        n_frames = min(len(alert_buffer), int(self.fps * seconds_of_footage))
            
            # Get the most recent n_frames, properly sorted by timestamp
            alert_buffer.sort(key=lambda x: x["timestamp"])
            frames = alert_buffer[-n_frames:]
            video_frames = [f["frame"] for f in frames]
            
            print(f"Creating alert video with {len(video_frames)} frames ({len(video_frames)/self.fps:.1f} seconds @ {self.fps} FPS)")
            
            # Find suspicious frames for thumbnail selection
            # Use the standard buffer for this since it has the labels
            suspicious_frames = []
            with self.lock:
                recent_frames = self.frame_buffer[-30:]  # Get last 30 frames
                for f in recent_frames:
                    if f.get("label") == "suspicious" and f.get("frame") is not None:
                        suspicious_frames.append(f.get("frame"))
            
            # If no suspicious frames found, use the most recent frames from the continuous buffer
            if not suspicious_frames and video_frames:
                # Use the last 1/3 of the video frames
                suspicious_frames = video_frames[len(video_frames)*2//3:]
            
            # Select a thumbnail - preferably from suspicious frames
            if suspicious_frames:
                thumbnail_idx = len(suspicious_frames) // 2  # Middle of suspicious activity
                thumbnail_frame = suspicious_frames[thumbnail_idx]
            elif video_frames:  # Fall back to middle of video_frames
                thumbnail_idx = len(video_frames) // 2
                thumbnail_frame = video_frames[thumbnail_idx]
            else:
                print("No frames available for thumbnail")
                return False
                
            _, thumbnail_buffer = cv2.imencode('.jpg', thumbnail_frame)
            thumbnail_data = thumbnail_buffer.tobytes()
            
            # Create video from frames
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            temp_video_path = os.path.join(temp_dir, f"shoplifting_alert_{self.camera_id}_{timestamp}.mp4")
            
            # Ensure all frames have the same dimensions
            if not video_frames:
                print("No frames to create video")
                return False
                
            height, width = video_frames[0].shape[:2]
            
            # Create video writer with appropriate codec
            # For Windows compatibility - try avc1 first, then mp4v
            video_created = False
            
            try:
                # Define a list of codecs to try
                codecs = [
                    ('avc1', 'mp4'),  # H.264 in MP4
                    ('mp4v', 'mp4'),  # MPEG-4 in MP4
                    ('XVID', 'avi'),  # XVID in AVI
                    ('MJPG', 'avi')   # Motion JPEG in AVI - fallback
                ]
                
                for codec, extension in codecs:
                    try:
                        temp_path = os.path.join(temp_dir, f"shoplifting_alert_{self.camera_id}_{timestamp}.{extension}")
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        writer = cv2.VideoWriter(temp_path, fourcc, self.fps, (width, height))
                        
                        if writer.isOpened():
                            # Write frames with progress
                            print(f"Created video writer with codec {codec}")
                            total_frames = len(video_frames)
                            for i, frame in enumerate(video_frames):
                                writer.write(frame)
                                if i % max(1, total_frames//10) == 0:  # Report at 10% intervals
                                    print(f"Writing frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
                            
                            writer.release()
                            print(f"Video saved to {temp_path}")
                            
                            # Check if file was created and has content
                            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                                temp_video_path = temp_path
                                video_created = True
                                break
                            else:
                                print(f"Video file is too small with codec {codec}, trying next codec")
                    except Exception as e:
                        print(f"Error with codec {codec}: {e}")
                        if writer:
                            writer.release()
                
                if not video_created:
                    raise Exception("Failed to create video with any codec")
                
            except Exception as e:
                print(f"All video writing attempts failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Create a temporary thumbnail file
            temp_thumb_path = os.path.join(temp_dir, f"thumbnail_{self.camera_id}_{timestamp}.jpg")
            with open(temp_thumb_path, "wb") as f:
                f.write(thumbnail_data)
                
            print(f"Thumbnail saved to {temp_thumb_path}")
            
            # Final verification
            if not os.path.exists(temp_video_path) or not os.path.exists(temp_thumb_path):
                print("Error: Video or thumbnail file doesn't exist")
                return False
                
            # Check file sizes
            video_size = os.path.getsize(temp_video_path)
            thumb_size = os.path.getsize(temp_thumb_path)
            
            if video_size < 1000 or thumb_size < 100:
                print(f"Error: Files too small - video: {video_size} bytes, thumbnail: {thumb_size} bytes")
                return False
                
            print(f"Final video size: {video_size/1024:.1f} KB, thumbnail: {thumb_size/1024:.1f} KB")
            
            # Prepare files for upload - using try/finally to ensure proper cleanup
            files = {}
            try:
                video_file = open(temp_video_path, 'rb')
                thumb_file = open(temp_thumb_path, 'rb')
                
                files = {
                    'video_clip': (os.path.basename(temp_video_path), video_file, 'video/mp4'),
                    'video_thumbnail': (os.path.basename(temp_thumb_path), thumb_file, 'image/jpeg')
                }
                
                # Send alert to backend
                alert_url = API_ENDPOINTS['create_shoplifting_alert']
                headers = {'Authorization': f'Bearer {self.auth_token}'}
                data = {'camera_id': self.camera_id}
                
                # Add more debug output
                print(f"Sending alert to {alert_url} with camera_id={self.camera_id}")
                print(f"Auth token present: {bool(self.auth_token)}")
                
                # Try multiple times with increasing timeout
                max_attempts = 3
                response = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        timeout = 30 * attempt  # Increase timeout with each attempt
                        print(f"Alert request attempt {attempt}/{max_attempts} with timeout={timeout}s")
                        response = requests.post(alert_url, headers=headers, data=data, files=files, timeout=timeout)
                        
                        if response.status_code == 201:
                            print(f"Alert created successfully!")
                            
                            # CRITICAL: Clear the continuous buffer after a successful alert
                            # This prevents footage from this alert appearing in the next one
                            with self.continuous_buffer_lock:
                                # Keep only the very latest frames (last 1.5 seconds) for continuity
                                now = datetime.now()
                                recent_frames = []
                                for entry in self.continuous_frame_buffer:
                                    if (now - entry["timestamp"]).total_seconds() < 1.5:
                                        recent_frames.append(entry)
                                
                                # Reset the buffer
                                self.continuous_frame_buffer = recent_frames
                                print(f"🗑️ Cleared continuous buffer - kept only {len(recent_frames)} recent frames")
                            
                            # Set cooldown to prevent too many alerts
                            self.cooldown_active = True
                            self.cooldown_start_time = datetime.now()
                            return True
                        else:
                            print(f"Failed to create alert: {response.status_code} - {response.text}")
                            # Continue trying if not the last attempt
                            if attempt < max_attempts:
                                print(f"Retrying in 1 second...")
                                time.sleep(1)
                            else:
                                return False
                    except requests.exceptions.RequestException as e:
                        print(f"Network error on attempt {attempt}: {e}")
                        if attempt < max_attempts:
                            print(f"Retrying in 1 second...")
                            time.sleep(1)
                        else:
                            return False
            except Exception as e:
                print(f"Error sending alert data: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                # Clean up files regardless of success/failure
                try:
                    if 'video_clip' in files and files['video_clip'][1]:
                        files['video_clip'][1].close()
                    if 'video_thumbnail' in files and files['video_thumbnail'][1]:
                        files['video_thumbnail'][1].close()
                        
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    if os.path.exists(temp_thumb_path):
                        os.remove(temp_thumb_path)
                except Exception as e:
                    print(f"Error cleaning up temp files: {e}")
                
        except Exception as e:
            print(f"Error triggering alert: {e}")
            import traceback
            traceback.print_exc()
            return False

    def adjust_buffer_size(self):
        """Dynamically adjust buffer size based on activity level"""
        now = datetime.now()
        # Adjust every 3 seconds instead of 5
        if (now - self.last_buffer_adjustment_time).total_seconds() < 3:
            return
            
        if not self.frame_buffer:
            return
            
        # Look at last N frames
        frames_to_analyze = min(int(self.fps * 3), len(self.frame_buffer))  # 3 seconds of frames, ensure integer
        if frames_to_analyze <= 0:
            frames_to_analyze = 1  # Ensure at least 1 frame is analyzed
            
        recent_frames = self.frame_buffer[-frames_to_analyze:]
        
        # Calculate suspicious ratio
        suspicious_frames = sum(1 for frame in recent_frames if frame["label"] == "suspicious")
        activity_score = suspicious_frames / frames_to_analyze if frames_to_analyze > 0 else 0
        
        # Adjust main buffer size - now more responsive
        if activity_score > 0.6:
            new_buffer_size = int(self.fps * 8)   # 8 seconds for high activity
        elif activity_score > 0.4:
            new_buffer_size = int(self.fps * 6)   # 6 seconds for medium activity
        elif activity_score > 0.2:
            new_buffer_size = int(self.fps * 5)   # 5 seconds for low activity 
        else:
            new_buffer_size = int(self.fps * 4)   # 4 seconds for minimal activity
            
        self.buffer_size = max(int(new_buffer_size), 1)  # Ensure buffer is at least 1 frame
        
        self.last_buffer_adjustment_time = now
        
    def calculate_alert_metrics(self):
        """Calculate voting metrics for alert triggering"""
        if not self.frame_buffer:
            return 0, 0
            
        suspicious_frames = [f for f in self.frame_buffer if f["label"] == "suspicious"]
        suspicious_count = len(suspicious_frames)
        total_count = len(self.frame_buffer)
        
        suspicious_ratio = suspicious_count / total_count if total_count > 0 else 0
        
        # Calculate weighted average confidence
        if suspicious_count == 0:
            average_confidence = 0
        else:
            confidence_sum = sum(f["confidence"] for f in suspicious_frames)
            average_confidence = confidence_sum / suspicious_count
            
        return suspicious_ratio, average_confidence
        
    def check_cooldown(self):
        """Check if cooldown is active and update its state"""
        if not self.cooldown_active:
            return False
            
        now = datetime.now()
        elapsed = (now - self.cooldown_start_time).total_seconds()
        
        if elapsed >= self.cooldown_duration.total_seconds():
            self.cooldown_active = False
            self.cooldown_start_time = None
            return False
            
        return True
        
    def send_analysis_data(self):
        """Send detection data to analysis endpoint"""
        if not self.camera_id or not self.auth_token:
            return
            
        current_time = datetime.now()
        if (current_time - self.last_analysis_time).total_seconds() >= 10:
            try:
                headers = {
                    'Authorization': f'Bearer {self.auth_token}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'camera_id': self.camera_id,
                    'total_events': self.total_events,
                    'suspicious_events': self.suspicious_events
                }
                
                response = requests.post(
                    API_ENDPOINTS['update_detection_data'],
                    json=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    # Reset counters after successful update
                    self.total_events = 0
                    self.suspicious_events = 0
                    self.last_analysis_time = current_time
                    
            except Exception as e:
                print(f"Error sending analysis data: {e}")

    def check_for_alerts(self):
        """Check for shoplifting alerts based on detection results"""
        # Don't check if in cooldown period
        if self.cooldown_active:
            now = datetime.now()
            elapsed = (now - self.cooldown_start_time).total_seconds()
            
            if elapsed >= self.cooldown_duration.total_seconds():
                self.cooldown_active = False
                self.cooldown_start_time = None
                print("Alert cooldown period ended")
            else:
                return False

        # Check if we have enough frames to analyze
        if len(self.frame_buffer) < 15:  # Need at least 15 frames
            return False

        # Count frames with valid image data
        valid_frames = len([f for f in self.frame_buffer if f.get("frame") is not None])
        if valid_frames < 15:  # Require at least 15 frames for analysis
            print(f"Not enough valid frames: {valid_frames}/15")
            return False

        # Check for suspicious activity
        recent_frame_count = min(30, len(self.frame_buffer))
        if recent_frame_count > 0:
            recent_frames = self.frame_buffer[-recent_frame_count:]
            recent_suspicious = sum(1 for f in recent_frames if f.get("label") == "suspicious")
            recent_confidence_sum = sum(f.get("confidence", 0) for f in recent_frames if f.get("label") == "suspicious")
            
            suspicious_ratio = recent_suspicious / recent_frame_count
            avg_confidence = recent_confidence_sum / recent_suspicious if recent_suspicious > 0 else 0
            
            # Check for alert conditions:
            # 1. Regular trigger: Moderate suspicious ratio with normal confidence
            # 2. High confidence trigger: Lower ratio but higher confidence (for single-person)
            alert_triggered = (
                (suspicious_ratio >= self.suspicious_ratio_threshold_normal and avg_confidence >= self.confidence_threshold_normal) or
                (suspicious_ratio >= self.suspicious_ratio_threshold_early and avg_confidence >= self.confidence_threshold_early)
            )
            
            if alert_triggered:
                print(f"⚠️ Alert triggered! {recent_suspicious}/{recent_frame_count} frames suspicious")
                print(f"Ratio: {suspicious_ratio:.2f} (Threshold: {self.suspicious_ratio_threshold_normal:.2f}/{self.suspicious_ratio_threshold_early:.2f})")
                print(f"Confidence: {avg_confidence:.2f} (Threshold: {self.confidence_threshold_normal:.2f}/{self.confidence_threshold_early:.2f})")
                
                # Trigger the alert directly
                result = self.trigger_alert()
                
                # Set cooldown regardless of result to prevent immediate re-triggering
                self.cooldown_active = True
                self.cooldown_start_time = datetime.now()
                
                return result
                
        return False

    def set_test_mode(self, enabled=True, lower_thresholds=True):
        """Enable test mode for easier triggering of alerts during testing"""
        self.test_mode = enabled
        
        # Lower thresholds in test mode for easier triggering
        if enabled and lower_thresholds:
            # Lower thresholds for testing
            self.suspicious_ratio_threshold_normal = 0.3  # Much lower ratio for testing
            self.confidence_threshold_normal = 0.3  # Lower standard confidence for testing
            self.suspicious_ratio_threshold_early = 0.2  # Lower early threshold
            self.confidence_threshold_early = 0.5  # Lower early confidence
            self.cooldown_duration = timedelta(seconds=5)  # Shorter cooldown in test mode
            print("Test mode enabled with significantly lowered thresholds")
        else:
            # Reset to our updated values for single-person shoplifting
            self.suspicious_ratio_threshold_normal = 0.50  # Regular threshold
            self.confidence_threshold_normal = 0.70  # Regular confidence
            self.suspicious_ratio_threshold_early = 0.30  # Lower ratio for single-person
            self.confidence_threshold_early = 0.85  # High confidence threshold
            self.cooldown_duration = timedelta(seconds=30)  # Standard cooldown between alerts
            print("Standard detection thresholds restored for single-person shoplifting")
        
        return True

    def force_trigger_alert(self):
        """Force trigger an alert for testing purposes"""
        if not self.frame_buffer:
            print("Cannot force trigger: No frames in buffer")
            return False
        
        # Make sure we have at least some frames with frames (not just metadata)
        frames_with_image = [f for f in self.frame_buffer if f.get("frame") is not None]
        
        print(f"Pre-force trigger: {len(frames_with_image)} main buffer frames")
        
        if len(frames_with_image) < 3:
            print(f"Not enough frames with images in buffer, found {len(frames_with_image)}")
            # If we don't have enough frames, wait and retry
            for i in range(10):
                time.sleep(0.5)  # Wait half a second
                frames_with_image = [f for f in self.frame_buffer if f.get("frame") is not None]
                if len(frames_with_image) >= 3:
                    print(f"Now have {len(frames_with_image)} frames with images after waiting")
                    break
        
        # Add some suspicious frames to the buffer with very high confidence
        # Mark fewer frames but with higher confidence to match single-person shoplifting scenario
        marked_frames = 0
        for i in range(min(8, len(self.frame_buffer))):
            if self.frame_buffer[i].get("frame") is not None:
                self.frame_buffer[i]["label"] = "suspicious"
                self.frame_buffer[i]["confidence"] = 0.95  # Very high confidence
                marked_frames += 1
        
        print(f"Marked {marked_frames} frames as suspicious with high confidence")
        
        # Trigger the alert
        result = self.trigger_alert()
        print(f"Force trigger alert result: {result}")
        return result

    def _detection_loop(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
                
            # Get actual FPS from video
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # Default to 30 if can't get valid FPS
                
            # Initialize buffer sizes based on FPS
            self.buffer_size = int(self.max_buffer_seconds * self.fps)
            
            # Set continuous buffer size (10 seconds of footage)
            self.continuous_buffer_size = int(10 * self.fps)
            
            print(f"Detection initialized: FPS={self.fps}, main_buffer_size={self.buffer_size}, continuous_buffer_size={self.continuous_buffer_size}")

            frame_count = 0
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video
                    continue

                frame_count += 1
                if frame_count % 2 != 0:  # Process every other frame instead of every third
                    continue

                # Process frame
                frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                # Add frame to continuous buffer first, before any detection processing
                # This ensures we have clean frames for video recording
                self.update_continuous_buffer(frame)
                
                with torch.no_grad():
                    results = self.model.predict(
                        frame,
                        conf=0.25,
                        device=self.device,
                        half=True if self.device == 'cuda' else False,
                        iou=0.45,
                        max_det=10,
                        retina_masks=False,
                        show_boxes=True,
                        verbose=self.verbose
                    )

                frame_result = {"label": "normal", "confidence": 0.0, "timestamp": datetime.now(), "frame": frame.copy()}
                
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    boxes_np = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    labels = boxes.cls.cpu().numpy()
                    masks = results[0].masks

                    # Apply NMS
                    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
                    idxs = cv2.dnn.NMSBoxes(
                        boxes_np.tolist(),
                        scores.tolist(),
                        0.25,  # confidence threshold
                        0.45   # NMS IoU threshold
                    )

                    if len(idxs) > 0:
                        if isinstance(idxs, np.ndarray):
                            idxs = idxs.flatten()

                        # Update detections
                        boxes_np = boxes_np[idxs]
                        scores = scores[idxs]
                        labels = labels[idxs]
                        if masks is not None:
                            masks = masks[idxs]

                        # Create modified results with filtered detections
                        modified_results = results[0].new()
                        
                        # Update analysis counters
                        self.total_events += len(boxes_np)
                        
                        # Modify labels based on confidence threshold
                        # In test mode, lower the detection threshold
                        confidence_threshold = 0.3 if self.test_mode else 0.4
                        
                        mask = (labels == 1) & (scores < confidence_threshold)
                        modified_labels = labels.copy()
                        modified_labels[mask] = 0

                        # Track highest suspicious confidence for frame
                        max_suspicious_confidence = 0.0
                        is_frame_suspicious = False
                        
                        # Draw boxes with proper classification
                        for i in range(len(boxes_np)):
                            box = boxes_np[i].astype(int)
                            score = scores[i]
                            # Use adjusted threshold for test mode
                            is_suspicious = labels[i] == 1 and score >= confidence_threshold
                            label = "Suspicious" if is_suspicious else "Normal"
                            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                            
                            if is_suspicious:
                                self.suspicious_events += 1
                                is_frame_suspicious = True
                                max_suspicious_confidence = max(max_suspicious_confidence, score)
                            
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(frame, 
                                      f"{label} {score:.2f}", 
                                      (box[0], box[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, color, 2)
                                      
                        # Update frame result
                        if is_frame_suspicious:
                            frame_result["label"] = "suspicious"
                            frame_result["confidence"] = max_suspicious_confidence

                # Add frame to main buffer
                self.frame_buffer.append(frame_result)
                
                # Maintain sliding buffer
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer = self.frame_buffer[-self.buffer_size:]
                
                # Adjust buffer size based on activity
                self.adjust_buffer_size()
                
                # Check for alerts
                self.check_for_alerts()

                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                with self.lock:
                    self.frame_data = frame_base64
                
                # Send analysis data if needed
                self.send_analysis_data()

                # Small delay to prevent CPU overload
                time.sleep(0.01)

        except Exception as e:
            print(f"Error in detection loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()

    def start_detection(self, video_path, camera_id=None, auth_token=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        with self.lock:
            if not self.is_running:
                print(f"Starting detection for camera ID: {camera_id}, auth token present: {bool(auth_token)}")
                self.camera_id = camera_id
                self.auth_token = auth_token
                self.is_running = True
                self.thread = threading.Thread(target=self._detection_loop, args=(video_path,))
                self.thread.daemon = True
                self.thread.start()
                return True
        return False

    def stop_detection(self):
        with self.lock:
            if self.is_running:
                self.is_running = False
                if self.thread:
                    self.thread.join(timeout=1.0)
                    self.thread = None
                return True
        return False

    def get_frame(self):
        with self.lock:
            return self.frame_data

    def finalize_alert_recording(self):
        """Complete the alert recording process and send the full video evidence"""
        if not self.camera_id or not self.auth_token:
            print("Cannot finalize alert: missing camera_id or auth_token")
            return False
            
        try:
            # Get alert ID if we're updating an existing alert
            alert_id = getattr(self, 'current_alert_id', None)
            
            if not alert_id:
                print("No current alert ID found, cannot finalize alert")
                return False
                
            print(f"Finalizing alert recording for alert ID: {alert_id}")
            
            # Arrange frames chronologically
            timeordered_frames = []
            suspicious_frames = []
            
            # Process all frames
            for frame_data in self.frame_buffer:
                if frame_data.get("frame") is not None:
                    timeordered_frames.append({
                        "timestamp": frame_data.get("timestamp", datetime.now()),
                        "frame": frame_data.get("frame"),
                        "label": frame_data.get("label", "normal")
                    })
                    
                    # Track suspicious frames for thumbnail selection
                    if frame_data.get("label") == "suspicious":
                        suspicious_frames.append(frame_data.get("frame"))
            
            # Sort all frames by timestamp
            timeordered_frames.sort(key=lambda x: x["timestamp"])
            
            # Get all frames for the video
            frames = [f["frame"] for f in timeordered_frames]
            
            print(f"Collected {len(frames)} frames for final alert video")
            
            # Select thumbnail from middle of suspicious frames
            if suspicious_frames:
                thumbnail_idx = len(suspicious_frames) // 2  # Middle of suspicious activity
                thumbnail_frame = suspicious_frames[thumbnail_idx]
            else:
                # Fall back to middle frame if no suspicious frames
                thumbnail_idx = len(frames) // 2
                thumbnail_frame = frames[thumbnail_idx]
                
            # Create temporary files with unique timestamps
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            temp_video_path = os.path.join(temp_dir, f"final_alert_{self.camera_id}_{timestamp}.mp4")
            temp_thumb_path = os.path.join(temp_dir, f"final_thumb_{self.camera_id}_{timestamp}.jpg")
            
            # Save thumbnail
            _, thumbnail_buffer = cv2.imencode('.jpg', thumbnail_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(temp_thumb_path, "wb") as f:
                f.write(thumbnail_buffer.tobytes())
            
            # Generate high-quality video from frames
            height, width = frames[0].shape[:2]
            
            # Try using H264 codec for better compatibility
            try:
                # For Windows, try different codecs in order of preference
                for codec_name in ['avc1', 'mp4v', 'XVID', 'DIVX']:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
                        
                        if video_writer.isOpened():
                            print(f"Successfully opened video writer with codec {codec_name}")
                            break
                    except Exception as codec_err:
                        print(f"Failed with codec {codec_name}: {codec_err}")
                        continue
                
                if not video_writer.isOpened():
                    print("Failed with all preferred codecs, trying MJPG")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    video_writer = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
            except Exception as e:
                print(f"Error creating video writer: {e}")
                video_writer = None
            
            if video_writer is None or not video_writer.isOpened():
                print("Failed to open any video writer, creating fallback AVI")
                fallback_path = os.path.join(temp_dir, f"fallback_{timestamp}.avi")
                fallback_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(fallback_path, fallback_fourcc, self.fps, (width, height))
                
                if not video_writer.isOpened():
                    print("Failed to create any kind of video, cannot proceed")
                    return False
                
                temp_video_path = fallback_path
                
            # Write frames with progress reporting
            total_frames = len(frames)
            for i, frame in enumerate(frames):
                video_writer.write(frame)
                if i % 30 == 0:
                    print(f"Writing frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
            
            video_writer.release()
            print(f"Video written to {temp_video_path}")
            
            # Send completed alert to backend
            try:
                url = API_ENDPOINTS['update_alert_evidence']
                
                print(f"Sending complete alert evidence to backend for alert ID: {alert_id}")
                
                with open(temp_video_path, 'rb') as video_file, open(temp_thumb_path, 'rb') as thumb_file:
                    files = {
                        'video_clip': (f'final_evidence_{timestamp}.mp4', video_file, 'video/mp4'),
                        'video_thumbnail': (f'final_thumb_{timestamp}.jpg', thumb_file, 'image/jpeg')
                    }
                    
                    # Add alert_id to form data
                    data = {
                        'alert_id': alert_id,
                        'status': 'completed'
                    }
                    
                    headers = {'Authorization': f'Bearer {self.auth_token}'}
                    
                    # Make sure the request can handle larger files by increasing timeout
                    response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
                    
                    if response.status_code == 200:
                        print("☑️ Alert evidence sent successfully!")
                        return True
                    else:
                        print(f"Failed to send alert evidence: {response.status_code} - {response.text}")
                        return False
                        
            except Exception as e:
                print(f"Error sending alert evidence: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Error finalizing alert recording: {e}")
            import traceback
            traceback.print_exc()
            return False