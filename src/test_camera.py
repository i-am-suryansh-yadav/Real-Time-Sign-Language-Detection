"""
test_camera.py - Comprehensive Camera Testing Tool
Tests camera functionality with detailed diagnostics and FPS monitoring
"""
import cv2
import time
import sys

class CameraTest:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.fps_history = []
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        print("=" * 60)
        print("Camera Test - Press 'q' to quit, 'i' for info")
        print("=" * 60)
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ ERROR: Cannot open camera {self.camera_index}")
            return False
        
        # Set optimal resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera initialized successfully")
        print(f"   Resolution: {int(actual_width)}x{int(actual_height)}")
        print(f"   Target FPS: {int(actual_fps)}")
        print()
        
        return True
    
    def get_camera_info(self):
        """Get detailed camera information"""
        if not self.cap or not self.cap.isOpened():
            return
        
        info = {
            'Width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'Height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'FPS': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'Brightness': int(self.cap.get(cv2.CAP_PROP_BRIGHTNESS)),
            'Contrast': int(self.cap.get(cv2.CAP_PROP_CONTRAST)),
            'Saturation': int(self.cap.get(cv2.CAP_PROP_SATURATION)),
        }
        
        print("\n" + "=" * 40)
        print("CAMERA INFORMATION")
        print("=" * 40)
        for key, value in info.items():
            print(f"{key:.<20} {value}")
        print("=" * 40 + "\n")
    
    def calculate_fps(self, prev_time):
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1 / (current_time - prev_time + 1e-6)
        self.fps_history.append(fps)
        
        # Keep only last 30 frames for average
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        return fps, avg_fps, current_time
    
    def draw_overlay(self, frame, fps, avg_fps, frame_count):
        """Draw informative overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        
        # Bottom info bar
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        
        # Blend
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS color coding
        fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255) if fps > 15 else (0, 0, 255)
        
        # Top text
        cv2.putText(frame, f"Camera Test - Frame: {frame_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Bottom text
        cv2.putText(frame, f"FPS: {int(fps)} | Avg: {int(avg_fps)}", 
                    (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        cv2.putText(frame, f"Resolution: {w}x{h}", 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Controls
        controls_text = "Controls: Q=Quit | I=Info | S=Screenshot"
        text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, controls_text, 
                    (w - text_size[0] - 10, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main test loop"""
        if not self.initialize_camera():
            return
        
        prev_time = time.time()
        frame_count = 0
        screenshot_count = 0
        
        print("🎥 Camera feed started. Testing in progress...")
        print("   Press 'q' to quit")
        print("   Press 'i' for camera info")
        print("   Press 's' to take screenshot\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  Warning: Failed to read frame")
                    break
                
                frame_count += 1
                
                # Calculate FPS
                fps, avg_fps, prev_time = self.calculate_fps(prev_time)
                
                # Draw overlay
                frame = self.draw_overlay(frame, fps, avg_fps, frame_count)
                
                # Display
                cv2.imshow('Camera Test', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✅ Test completed successfully")
                    break
                elif key == ord('i'):
                    self.get_camera_info()
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"camera_test_{screenshot_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            print(f"\n📊 Performance Summary:")
            print(f"   Average FPS: {sum(self.fps_history) / len(self.fps_history):.2f}")
            print(f"   Min FPS: {min(self.fps_history):.2f}")
            print(f"   Max FPS: {max(self.fps_history):.2f}")

def main():
    """Main entry point"""
    camera_index = 0
    
    # Allow command line argument for camera index
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("Usage: python test_camera.py [camera_index]")
            sys.exit(1)
    
    tester = CameraTest(camera_index)
    tester.run()

if __name__ == "__main__":
    main()