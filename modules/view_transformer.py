
import cv2
import numpy as np
from ultralytics import YOLO

class ViewTransformer:
    """
    Transforms pixel coordinates to real field coordinates
    Uses pitch keypoint detection model to map positions
    """
    
    def __init__(self, pitch_model_path=None):
        self.pitch_model = None
        if pitch_model_path:
            try:
                self.pitch_model = YOLO(pitch_model_path)
                print("✅ Loaded pitch keypoint model")
            except Exception as e:
                print(f"⚠️  Could not load pitch model: {e}")
        
        # Standard football pitch dimensions (meters)
        self.PITCH_LENGTH = 105.0  # meters
        self.PITCH_WIDTH = 68.0    # meters
        
        # Field zones for position classification
        self.zones = self._define_zones()
        
        # Transformation matrix (will be calculated from keypoints)
        self.transform_matrix = None
        self.has_valid_transform = False
    
    def _define_zones(self):
        """Define field zones for position classification"""
        return {
            'goalkeeper': {
                'defensive': (0, 16.5),    # In penalty box
                'attacking': (88.5, 105)    # Rare but possible
            },
            'defender': {
                'zone': (0, 35)  # Defensive third
            },
            'midfielder': {
                'zone': (35, 70)  # Middle third
            },
            'forward': {
                'zone': (70, 105)  # Attacking third
            }
        }
    
    def detect_keypoints(self, frame):
        """Detect pitch keypoints using the model"""
        if not self.pitch_model:
            return None
        
        try:
            # Use YOLO predict method (not detect)
            results = self.pitch_model.predict(frame, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            
            # Check for keypoints in different possible attributes
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                try:
                    keypoints = result.keypoints.xy[0].cpu().numpy()
                    if len(keypoints) > 0:
                        return keypoints
                except:
                    pass
            
            # Try alternative: boxes with keypoints
            if hasattr(result, 'boxes') and result.boxes is not None:
                try:
                    if hasattr(result.boxes, 'keypoints'):
                        return result.boxes.keypoints
                except:
                    pass
            
            return None
            
        except Exception as e:
            # Silent fail - will use simple scaling
            return None
    
    def calculate_transform_matrix(self, frame):
        """Calculate transformation matrix from detected keypoints"""
        keypoints = self.detect_keypoints(frame)
        
        if keypoints is None or len(keypoints) < 4:
            # Fallback: use simple scaling based on frame dimensions
            self.has_valid_transform = False
            return False
        
        try:
            # Define real-world pitch corners (in meters)
            pitch_points = np.array([
                [0, 0],                           # Bottom-left corner
                [self.PITCH_LENGTH, 0],           # Bottom-right
                [self.PITCH_LENGTH, self.PITCH_WIDTH],  # Top-right
                [0, self.PITCH_WIDTH]             # Top-left
            ], dtype=np.float32)
            
            # Use detected keypoints (select corner points)
            # This assumes your model detects corners - adjust indices as needed
            image_points = keypoints[:4].astype(np.float32)
            
            # Calculate perspective transform
            self.transform_matrix = cv2.getPerspectiveTransform(
                image_points, 
                pitch_points
            )
            self.has_valid_transform = True
            return True
            
        except Exception as e:
            print(f"⚠️  Transform calculation failed: {e}")
            self.has_valid_transform = False
            return False
    
    def transform_point(self, x, y, frame_width=None, frame_height=None):
        """
        Transform pixel coordinates to field coordinates (meters)
        
        Returns: (x_meters, y_meters) on the pitch
        """
        if self.has_valid_transform and self.transform_matrix is not None:
            # Use perspective transform
            point = np.array([[[x, y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.transform_matrix)
            return transformed[0][0]
        
        else:
            # Fallback: simple scaling
            if frame_width and frame_height:
                x_meters = (x / frame_width) * self.PITCH_LENGTH
                y_meters = (y / frame_height) * self.PITCH_WIDTH
                return np.array([x_meters, y_meters])
            
            return np.array([x, y])
    
    def get_position_zone(self, x_meters, y_meters=None):
        """
        Classify player position based on field location
        
        Args:
            x_meters: X position in meters (0 = defensive goal, 105 = attacking goal)
            y_meters: Y position in meters (optional, for more detailed classification)
        
        Returns: Position string (e.g., "Defender", "Central Midfielder")
        """
        # Ensure x is in valid range
        x = max(0, min(x_meters, self.PITCH_LENGTH))
        
        # Basic position classification based on thirds
        if x < 35:  # Defensive third
            if x < 16.5:  # In penalty box
                return "Goalkeeper/Deep Defender"
            else:
                return "Defender"
        
        elif x < 70:  # Middle third
            if y_meters is not None:
                if y_meters < self.PITCH_WIDTH * 0.33:
                    return "Right Midfielder"
                elif y_meters > self.PITCH_WIDTH * 0.67:
                    return "Left Midfielder"
                else:
                    return "Central Midfielder"
            return "Midfielder"
        
        else:  # Attacking third
            if x > 88.5:  # In attacking penalty box
                return "Forward/Striker"
            else:
                return "Attacking Midfielder"
    
    def get_zone_percentages(self, positions_meters):
        """
        Calculate time spent in each third of the pitch
        
        Args:
            positions_meters: List of [x, y] positions in meters
        
        Returns: Dict with percentages for defensive/middle/attacking thirds
        """
        # FIX: Check if positions_meters is empty properly for numpy arrays
        if positions_meters is None or (hasattr(positions_meters, '__len__') and len(positions_meters) == 0):
            return {
                'defensive': 0.0,
                'middle': 0.0,
                'attacking': 0.0
            }
        
        positions = np.array(positions_meters)
        x_positions = positions[:, 0]
        
        total = len(x_positions)
        
        defensive = np.sum(x_positions < 35) / total * 100
        middle = np.sum((x_positions >= 35) & (x_positions < 70)) / total * 100
        attacking = np.sum(x_positions >= 70) / total * 100
        
        return {
            'defensive': round(defensive, 1),
            'middle': round(middle, 1),
            'attacking': round(attacking, 1)
        }
    
    def calculate_distance_meters(self, pos1, pos2):
        """
        Calculate real distance between two positions in meters
        
        Args:
            pos1, pos2: Pixel coordinates (x, y) or meter coordinates
        
        Returns: Distance in meters
        """
        if self.has_valid_transform:
            # Transform both points if needed
            if isinstance(pos1, (list, tuple)) and len(pos1) == 2:
                p1 = self.transform_point(pos1[0], pos1[1])
                p2 = self.transform_point(pos2[0], pos2[1])
            else:
                p1, p2 = pos1, pos2
            
            return np.linalg.norm(p2 - p1)
        else:
            # Fallback: pixel distance * calibration factor
            pixel_dist = np.linalg.norm(np.array(pos2) - np.array(pos1))
            # Rough calibration: assuming ~10 pixels = 1 meter
            return pixel_dist / 10.0