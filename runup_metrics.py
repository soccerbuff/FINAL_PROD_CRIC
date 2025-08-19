"""
runup_metrics.py

This module analyzes runup biomechanics from bowling videos, including:
- Pose estimation from runup video
- Phase detection (runup, bound, FFC, release)
- Runup speed calculation every 30 frames
- Forward-backward arm movement analysis (wrist relative to torso)
- Vertical arm movement analysis
"""

import json
import math
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunupAnalyzer:
    """
    Analyzes runup biomechanics from bowling videos.
    """
    
    def __init__(self, model_complexity: int = 2, min_tracking_confidence: float = 0.6):
        """
        Initialize the runup analyzer.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define keypoint mappings for MediaPipe
        self.keypoint_mappings = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
    
    def extract_pose_keypoints(self, video_path: str) -> tuple[List[Dict], float, int, int]:
        """
        Extract pose keypoints from runup video.
        
        Args:
            video_path: Path to the runup video file
            
        Returns:
            Tuple of (frames_data, fps, width, height)
        """
        logger.info(f"Extracting pose keypoints from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            frame_data = {
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'keypoints': {},
                'phase': 'unknown'
            }
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Extract keypoints
                for key, landmark_idx in self.keypoint_mappings.items():
                    landmark = landmarks[landmark_idx]
                    frame_data['keypoints'][key] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
            
            frames_data.append(frame_data)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        logger.info(f"Completed pose extraction. Total frames: {len(frames_data)}")
        
        return frames_data, fps, width, height
    
    def detect_phases(self, frames_data: List[Dict]) -> List[Dict]:
        """
        Detect bowling phases from pose data.
        
        Args:
            frames_data: List of frames with keypoints
            
        Returns:
            Updated frames_data with phase information
        """
        logger.info("Detecting bowling phases...")
        
        # Detect bound phase (lowest point in torso trajectory)
        bound_start, bound_end = self._detect_bound_phase(frames_data)
        
        # Detect BFC (Back Foot Contact)
        bfc_idx = self._detect_bfc(frames_data, bound_end)
        
        # Detect FFC (Front Foot Contact)
        ffc_idx = self._detect_ffc(frames_data, bfc_idx)
        
        # Detect release
        release_idx = self._detect_release(frames_data, ffc_idx)
        
        # Label phases
        for i, frame in enumerate(frames_data):
            if bound_start <= i <= bound_end:
                frame['phase'] = 'bound'
            elif i == bfc_idx:
                frame['phase'] = 'BFC'
            elif i == ffc_idx:
                frame['phase'] = 'FFC'
            elif i == release_idx:
                frame['phase'] = 'release'
            elif i < bound_start:
                frame['phase'] = 'run_up'
            else:
                frame['phase'] = 'delivery_stride'
        
        logger.info(f"Phase detection complete. BFC: {bfc_idx}, FFC: {ffc_idx}, Release: {release_idx}")
        
        return frames_data
    
    def _detect_bound_phase(self, frames_data: List[Dict]) -> Tuple[int, int]:
        """Detect the bound phase (lowest point in torso trajectory)."""
        mid_ys = []
        for frame in frames_data:
            kps = frame.get('keypoints', {})
            if 'left_hip' in kps and 'right_hip' in kps and 'left_shoulder' in kps and 'right_shoulder' in kps:
                mid_y = (kps['left_hip']['y'] + kps['right_hip']['y'] + 
                        kps['left_shoulder']['y'] + kps['right_shoulder']['y']) / 4
                mid_ys.append(mid_y)
            else:
                mid_ys.append(None)
        
        # Smooth the trajectory
        mid_ys_smoothed = self._moving_average(mid_ys, window=5)
        
        # Find the lowest point (bound)
        valid_ys = [(i, y) for i, y in enumerate(mid_ys_smoothed) if y is not None]
        if not valid_ys:
            return 0, len(frames_data) - 1
        
        min_idx, _ = min(valid_ys, key=lambda x: x[1])
        
        # Expand bound phase around the lowest point
        start_idx = max(0, min_idx - 10)
        end_idx = min(len(frames_data) - 1, min_idx + 10)
        
        return start_idx, end_idx
    
    def _detect_bfc(self, frames_data: List[Dict], bound_end: int) -> Optional[int]:
        """Detect Back Foot Contact."""
        if bound_end >= len(frames_data) - 1:
            return None
        
        # Look for the frame after bound where one foot is clearly behind the other
        for i in range(bound_end + 1, min(bound_end + 30, len(frames_data))):
            kps = frames_data[i].get('keypoints', {})
            if 'left_ankle' in kps and 'right_ankle' in kps:
                left_y = kps['left_ankle']['y']
                right_y = kps['right_ankle']['y']
                
                # Check if one foot is significantly behind the other
                if abs(left_y - right_y) > 0.05:  # 5% of frame height
                    return i
        
        return None
    
    def _detect_ffc(self, frames_data: List[Dict], bfc_idx: Optional[int]) -> Optional[int]:
        """Detect Front Foot Contact."""
        if bfc_idx is None:
            return None
        
        # Look for the frame where front foot makes contact
        for i in range(bfc_idx + 1, min(bfc_idx + 60, len(frames_data))):
            kps = frames_data[i].get('keypoints', {})
            if 'left_ankle' in kps and 'right_ankle' in kps:
                left_y = kps['left_ankle']['y']
                right_y = kps['right_ankle']['y']
                
                # Check if both feet are at similar height (contact)
                if abs(left_y - right_y) < 0.02:  # 2% of frame height
                    return i
        
        return None
    
    def _detect_release(self, frames_data: List[Dict], ffc_idx: Optional[int]) -> Optional[int]:
        """Detect ball release."""
        if ffc_idx is None:
            return None
        
        # Look for the frame where arms are extended forward
        for i in range(ffc_idx + 1, min(ffc_idx + 90, len(frames_data))):
            kps = frames_data[i].get('keypoints', {})
            if 'left_wrist' in kps and 'right_wrist' in kps:
                left_wrist = kps['left_wrist']
                right_wrist = kps['right_wrist']
                
                # Check if wrists are extended forward (lower y values)
                if left_wrist['y'] < 0.3 and right_wrist['y'] < 0.3:
                    return i
        
        return None
    
    def _moving_average(self, values: List[float], window: int = 3) -> List[float]:
        """Apply moving average smoothing to a list of values."""
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            window_vals = [v for v in values[start:end] if v is not None]
            if window_vals:
                smoothed.append(sum(window_vals) / len(window_vals))
            else:
                smoothed.append(None)
        return smoothed
    
    def calculate_runup_speed(self, frames_data: List[Dict], interval: int = 30) -> List[Dict]:
        """
        DEPRECATED: Use analyze_runup() instead for complete analysis with real units.
        
        Calculate runup speed every N frames (in normalized units).
        
        Args:
            frames_data: List of frames with keypoints
            interval: Frame interval for speed calculation (default: 30)
            
        Returns:
            List of speed measurements with frame ranges and speeds (in normalized units)
        """
        logger.info(f"Calculating runup speed every {interval} frames...")
        
        speeds = []
        
        for i in range(0, len(frames_data) - interval, interval):
            start_frame = frames_data[i]
            end_frame = frames_data[i + interval]
            
            # Get pelvis center positions
            start_pos = self._get_pelvis_center(start_frame)
            end_pos = self._get_pelvis_center(end_frame)
            
            if start_pos and end_pos:
                # Calculate displacement
                dx = end_pos['x'] - start_pos['x']
                dy = end_pos['y'] - start_pos['y']
                displacement = math.sqrt(dx**2 + dy**2)
                
                # Calculate time difference
                dt = end_frame['timestamp'] - start_frame['timestamp']
                
                if dt > 0:
                    speed = displacement / dt  # pixels per second
                    
                    speeds.append({
                        'frame_range': (i, i + interval),
                        'start_frame': i,
                        'end_frame': i + interval,
                        'start_time': start_frame['timestamp'],
                        'end_time': end_frame['timestamp'],
                        'displacement': displacement,
                        'time_delta': dt,
                        'speed': speed,
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    })
        
        logger.info(f"Calculated {len(speeds)} speed measurements")
        return speeds
    
    def calculate_forward_backward_arm_movement(self, frames_data: List[Dict]) -> Dict:
        """
        DEPRECATED: Use analyze_runup() instead for complete analysis with real units.
        
        Calculate forward-backward arm movement during runup from side view.
        
        Uses wrist positions relative to torso center for measuring arm swing amplitude.
        Returns values in normalized coordinates (0-1 range).
        
        Args:
            frames_data: List of frames with keypoints
            
        Returns:
            Dictionary with forward-backward arm movement metrics (in normalized units)
        """
        logger.info("Calculating forward-backward arm movement metrics using elbow-torso comparison...")
        
        # Get runup frames
        runup_frames = [f for f in frames_data if f.get('phase') == 'run_up']
        
        if len(runup_frames) < 2:
            logger.warning("Insufficient runup frames for forward-backward arm movement analysis")
            return {}
        
        # Extract arm and torso positions over time (side view perspective)
        left_wrist_positions = []
        right_wrist_positions = []
        torso_positions = []
        
        for frame in runup_frames:
            kps = frame.get('keypoints', {})
            
            # Check if we have all required keypoints
            if all(key in kps for key in ['left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
                # Left wrist
                left_wrist_positions.append((kps['left_wrist']['x'], kps['left_wrist']['y']))
                
                # Right wrist
                right_wrist_positions.append((kps['right_wrist']['x'], kps['right_wrist']['y']))
                
                # Torso center (midpoint between shoulders and hips)
                torso_x = (kps['left_shoulder']['x'] + kps['right_shoulder']['x'] + 
                          kps['left_hip']['x'] + kps['right_hip']['x']) / 4
                torso_y = (kps['left_shoulder']['y'] + kps['right_shoulder']['y'] + 
                          kps['left_hip']['y'] + kps['right_hip']['y']) / 4
                torso_positions.append((torso_x, torso_y))
        
        metrics = {}
        
        # Calculate forward-backward arm movement relative to torso
        if len(left_wrist_positions) > 1 and len(right_wrist_positions) > 1 and len(torso_positions) > 1:
            left_forward_range = self._calculate_wrist_torso_forward_range(left_wrist_positions, torso_positions)
            right_forward_range = self._calculate_wrist_torso_forward_range(right_wrist_positions, torso_positions)
            
            # Calculate additional detailed metrics
            left_forward_stats = self._calculate_wrist_torso_detailed_stats(left_wrist_positions, torso_positions)
            right_forward_stats = self._calculate_wrist_torso_detailed_stats(right_wrist_positions, torso_positions)
            
            metrics['forward_backward_arm_movement'] = {
                'left_arm_range': left_forward_range,
                'right_arm_range': right_forward_range,
                'average_forward_range': (left_forward_range + right_forward_range) / 2,
                'measurement_type': 'wrist_relative_to_torso',
                'units': 'normalized_units_forward_from_torso',
                'left_arm_detailed': left_forward_stats,
                'right_arm_detailed': right_forward_stats
            }
        
        logger.info(f"Calculated forward-backward arm movement metrics using elbow-torso comparison")
        return metrics

    def calculate_vertical_arm_movement(self, frames_data: List[Dict]) -> Dict:
        """
        DEPRECATED: Use analyze_runup() instead for complete analysis with real units.
        
        Calculate vertical arm movement during runup from side view.
        
        Uses wrist positions relative to torso center for measuring arm elevation amplitude.
        Returns values in normalized coordinates (0-1 range).
        
        Args:
            frames_data: List of frames with keypoints
            
        Returns:
            Dictionary with vertical arm movement metrics (in normalized units)
        """
        logger.info("Calculating vertical arm movement metrics using wrist-torso comparison...")
        
        # Get runup frames
        runup_frames = [f for f in frames_data if f.get('phase') == 'run_up']
        
        if len(runup_frames) < 2:
            logger.warning("Insufficient runup frames for vertical arm movement analysis")
            return {}
        
        # Extract arm and torso positions over time (side view perspective)
        left_wrist_positions = []
        right_wrist_positions = []
        torso_positions = []
        
        for frame in runup_frames:
            kps = frame.get('keypoints', {})
            
            # Check if we have all required keypoints
            if all(key in kps for key in ['left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
                # Left wrist
                left_wrist_positions.append((kps['left_wrist']['x'], kps['left_wrist']['y']))
                
                # Right wrist
                right_wrist_positions.append((kps['right_wrist']['x'], kps['right_wrist']['y']))
                
                # Torso center (midpoint between shoulders and hips)
                torso_x = (kps['left_shoulder']['x'] + kps['right_shoulder']['x'] + 
                          kps['left_hip']['x'] + kps['right_hip']['x']) / 4
                torso_y = (kps['left_shoulder']['y'] + kps['right_shoulder']['y'] + 
                          kps['left_hip']['y'] + kps['right_hip']['y']) / 4
                torso_positions.append((torso_x, torso_y))
        
        metrics = {}
        
        # Calculate vertical arm movement relative to torso
        if len(left_wrist_positions) > 1 and len(right_wrist_positions) > 1 and len(torso_positions) > 1:
            left_vertical_range = self._calculate_wrist_torso_vertical_range(left_wrist_positions, torso_positions)
            right_vertical_range = self._calculate_wrist_torso_vertical_range(right_wrist_positions, torso_positions)
            
            # Calculate additional detailed metrics
            left_vertical_stats = self._calculate_wrist_torso_vertical_detailed_stats(left_wrist_positions, torso_positions)
            right_vertical_stats = self._calculate_wrist_torso_vertical_detailed_stats(right_wrist_positions, torso_positions)
            
            metrics['vertical_arm_movement'] = {
                'left_arm_range': left_vertical_range,
                'right_arm_range': right_vertical_range,
                'average_vertical_range': (left_vertical_range + right_vertical_range) / 2,
                'measurement_type': 'wrist_relative_to_torso',
                'units': 'normalized_units_above_below_torso',
                'left_arm_detailed': left_vertical_stats,
                'right_arm_detailed': right_vertical_stats
            }
        
        logger.info(f"Calculated vertical arm movement metrics using wrist-torso comparison")
        return metrics


    
    def _get_pelvis_center(self, frame: Dict) -> Optional[Dict[str, float]]:
        """Get the center position of the pelvis from a frame."""
        kps = frame.get('keypoints', {})
        if 'left_hip' in kps and 'right_hip' in kps:
            left_hip = kps['left_hip']
            right_hip = kps['right_hip']
            return {
                'x': (left_hip['x'] + right_hip['x']) / 2,
                'y': (left_hip['y'] + right_hip['y']) / 2
            }
        return None
    
    def _calculate_vertical_range(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate the vertical (y-axis) range of movement."""
        if len(positions) < 2:
            return 0.0
        
        y_coords = [pos[1] for pos in positions]
        return max(y_coords) - min(y_coords)
    
    def _calculate_wrist_torso_forward_range(self, wrist_positions: List[Tuple[float, float]], 
                                           torso_positions: List[Tuple[float, float]]) -> float:
        """
        Calculate forward-backward arm swing amplitude relative to torso position.
        
        Measures how far forward/backward the wrist swings relative to the torso center.
        This captures the full arm swing amplitude during runup.
        Positive values indicate forward swing, negative values indicate backward swing.
        
        Args:
            wrist_positions: List of (x, y) wrist coordinates
            torso_positions: List of (x, y) torso center coordinates
            
        Returns:
            Amplitude of forward-backward arm swing in pixels
        """
        if len(wrist_positions) < 2 or len(torso_positions) < 2:
            return 0.0
        
        # Calculate wrist position relative to torso for each frame
        relative_positions = []
        for i in range(min(len(wrist_positions), len(torso_positions))):
            wrist_x = wrist_positions[i][0]
            torso_x = torso_positions[i][0]
            
            # Relative position: positive = forward of torso, negative = behind torso
            relative_x = wrist_x - torso_x
            relative_positions.append(relative_x)
        
        if len(relative_positions) < 2:
            return 0.0
        
        # Calculate range of relative movement
        max_forward = max(relative_positions)  # Most forward position
        max_backward = min(relative_positions)  # Most backward position
        
        # Range is the total distance covered
        movement_range = max_forward - max_backward
        
        return movement_range
    
    def _calculate_wrist_torso_vertical_range(self, wrist_positions: List[Tuple[float, float]], 
                                            torso_positions: List[Tuple[float, float]]) -> float:
        """
        Calculate vertical arm swing amplitude relative to torso position.
        
        Measures how far above/below the wrist swings relative to the torso center.
        This captures the arm elevation amplitude during runup.
        Positive values indicate above torso, negative values indicate below torso.
        
        Args:
            wrist_positions: List of (x, y) wrist coordinates
            torso_positions: List of (x, y) torso center coordinates
            
        Returns:
            Amplitude of vertical arm swing in pixels
        """
        if len(wrist_positions) < 2 or len(torso_positions) < 2:
            return 0.0
        
        # Calculate wrist position relative to torso for each frame
        relative_positions = []
        for i in range(min(len(wrist_positions), len(torso_positions))):
            wrist_y = wrist_positions[i][1]
            torso_y = torso_positions[i][1]
            
            # Relative position: positive = above torso, negative = below torso
            # Note: In image coordinates, lower Y values = higher position
            relative_y = torso_y - wrist_y  # Inverted because lower Y = higher
            relative_positions.append(relative_y)
        
        if len(relative_positions) < 2:
            return 0.0
        
        # Calculate range of relative movement
        max_above = max(relative_positions)  # Most elevated position
        max_below = min(relative_positions)  # Most lowered position
        
        # Range is the total distance covered
        movement_range = max_above - max_below
        
        return movement_range
    
    def _calculate_wrist_torso_vertical_detailed_stats(self, wrist_positions: List[Tuple[float, float]], 
                                                     torso_positions: List[Tuple[float, float]]) -> Dict:
        """
        Calculate detailed statistics for wrist vertical swing amplitude relative to torso.
        
        Args:
            wrist_positions: List of (x, y) wrist coordinates
            torso_positions: List of (x, y) torso center coordinates
            
        Returns:
            Dictionary with detailed vertical swing amplitude statistics
        """
        if len(wrist_positions) < 2 or len(torso_positions) < 2:
            return {}
        
        # Calculate relative positions
        relative_positions = []
        for i in range(min(len(wrist_positions), len(torso_positions))):
            wrist_y = wrist_positions[i][1]
            torso_y = torso_positions[i][1]
            relative_y = torso_y - wrist_y  # Inverted because lower Y = higher
            relative_positions.append(relative_y)
        
        if len(relative_positions) < 2:
            return {}
        
        # Calculate statistics
        max_above = max(relative_positions)
        max_below = min(relative_positions)
        avg_position = sum(relative_positions) / len(relative_positions)
        
        # Calculate above vs below dominance
        above_frames = sum(1 for pos in relative_positions if pos > 0)
        below_frames = sum(1 for pos in relative_positions if pos < 0)
        neutral_frames = sum(1 for pos in relative_positions if pos == 0)
        
        total_frames = len(relative_positions)
        above_percentage = (above_frames / total_frames) * 100
        below_percentage = (below_frames / total_frames) * 100
        neutral_percentage = (neutral_frames / total_frames) * 100
        
        return {
            'max_above_position': max_above,
            'max_below_position': max_below,
            'average_position': avg_position,
            'above_frames': above_frames,
            'below_frames': below_frames,
            'neutral_frames': neutral_frames,
            'above_percentage': above_percentage,
            'below_percentage': below_percentage,
            'neutral_percentage': neutral_percentage,
            'elevation_balance': 'elevated_dominant' if above_percentage > 60 else 
                                'lowered_dominant' if below_percentage > 60 else 'balanced'
        }
    
    def _calculate_wrist_torso_detailed_stats(self, wrist_positions: List[Tuple[float, float]], 
                                            torso_positions: List[Tuple[float, float]]) -> Dict:
        """
        Calculate detailed statistics for wrist swing amplitude relative to torso.
        
        Args:
            wrist_positions: List of (x, y) wrist coordinates
            torso_positions: List of (x, y) torso center coordinates
            
        Returns:
            Dictionary with detailed swing amplitude statistics
        """
        if len(wrist_positions) < 2 or len(torso_positions) < 2:
            return {}
        
        # Calculate relative positions
        relative_positions = []
        for i in range(min(len(wrist_positions), len(torso_positions))):
            wrist_x = wrist_positions[i][0]
            torso_x = torso_positions[i][0]
            relative_x = wrist_x - torso_x
            relative_positions.append(relative_x)
        
        if len(relative_positions) < 2:
            return {}
        
        # Calculate statistics
        max_forward = max(relative_positions)
        max_backward = min(relative_positions)
        avg_position = sum(relative_positions) / len(relative_positions)
        
        # Calculate forward vs backward dominance
        forward_frames = sum(1 for pos in relative_positions if pos > 0)
        backward_frames = sum(1 for pos in relative_positions if pos < 0)
        neutral_frames = sum(1 for pos in relative_positions if pos == 0)
        
        total_frames = len(relative_positions)
        forward_percentage = (forward_frames / total_frames) * 100
        backward_percentage = (backward_frames / total_frames) * 100
        neutral_percentage = (neutral_frames / total_frames) * 100
        
        return {
            'max_forward_position': max_forward,
            'max_backward_position': max_backward,
            'average_position': avg_position,
            'forward_frames': forward_frames,
            'backward_frames': backward_frames,
            'neutral_frames': neutral_frames,
            'forward_percentage': forward_percentage,
            'backward_percentage': backward_percentage,
            'neutral_percentage': neutral_percentage,
            'movement_balance': 'forward_dominant' if forward_percentage > 60 else 
                               'backward_dominant' if backward_percentage > 60 else 'balanced'
        }
    
    def _calculate_forward_range(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate the forward-backward (x-axis) range of movement."""
        if len(positions) < 2:
            return 0.0
        
        x_coords = [pos[0] for pos in positions]
        return max(x_coords) - min(x_coords)
    
    def _median_pixels_per_cm(self, frames_data: List[Dict], width: int, height: int) -> Optional[float]:
        """Estimate median pixels-per-cm using right elbow→wrist distance (assumed 30 cm)."""
        reference_cm = 30.0
        px_per_cm_values = []
        for frame in frames_data:
            kps = frame.get('keypoints', {})
            re = kps.get('right_elbow')
            rw = kps.get('right_wrist')
            if re and rw:
                dx = (rw['x'] - re['x']) * width
                dy = (rw['y'] - re['y']) * height
                dist_px = math.sqrt(dx*dx + dy*dy)
                if dist_px > 0:
                    px_per_cm_values.append(dist_px / reference_cm)
        if not px_per_cm_values:
            return None
        # median for robustness
        px_per_cm_values.sort()
        mid = len(px_per_cm_values) // 2
        if len(px_per_cm_values) % 2 == 1:
            return px_per_cm_values[mid]
        return (px_per_cm_values[mid - 1] + px_per_cm_values[mid]) / 2.0

    def _pxps_to_kmh(self, px_per_s: float, px_per_cm: float) -> float:
        """Convert pixels/second to km/h given pixels-per-cm scale."""
        if px_per_cm <= 0:
            return 0.0
        cm_per_s = px_per_s / px_per_cm
        return cm_per_s * 0.036

    def _compute_runup_speeds_kmh(self, frames_data: List[Dict], width: int, height: int, px_per_cm: float, interval: int = 30) -> List[Dict]:
        """Compute run-up speeds in km/h every N frames using pelvis center and real scaling."""
        speeds = []
        for i in range(0, len(frames_data) - interval, interval):
            f0 = frames_data[i]
            f1 = frames_data[i + interval]
            p0 = self._get_pelvis_center(f0)
            p1 = self._get_pelvis_center(f1)
            if not p0 or not p1:
                continue
            dx_px = (p1['x'] - p0['x']) * width
            dy_px = (p1['y'] - p0['y']) * height
            disp_px = math.sqrt(dx_px*dx_px + dy_px*dy_px)
            dt = f1['timestamp'] - f0['timestamp']
            if dt <= 0:
                continue
            speed_pxps = disp_px / dt
            kmh = self._pxps_to_kmh(speed_pxps, px_per_cm)
            speeds.append({
                'segment_start_s': f0['timestamp'],
                'segment_end_s': f1['timestamp'],
                'speed_kmh': kmh
            })
        return speeds

    def _forward_backward_range_cm(self, frames_data: List[Dict], width: int, height: int, px_per_cm: float) -> Optional[Dict]:
        """Compute wrist–torso forward/backward range (cm) for left and right arms over all run-up frames."""
        # Get run-up frames but exclude last 5 frames before bound
        bound_start = next((i for i, f in enumerate(frames_data) if f.get('phase') == 'bound'), None)
        if bound_start is not None:
            runup_frames = [f for i, f in enumerate(frames_data) if f.get('phase') == 'run_up' and i < bound_start - 5]
        else:
            runup_frames = [f for f in frames_data if f.get('phase') == 'run_up']
        if len(runup_frames) < 2:
            return None
        left_rel_px = []
        right_rel_px = []
        for f in runup_frames:
            k = f.get('keypoints', {})
            if all(key in k for key in ['left_wrist','right_wrist','left_hip','right_hip','left_shoulder','right_shoulder']):
                torso_x = (k['left_shoulder']['x'] + k['right_shoulder']['x'] + k['left_hip']['x'] + k['right_hip']['x']) / 4.0
                lwx = k['left_wrist']['x'] * width
                rwx = k['right_wrist']['x'] * width
                tx = torso_x * width
                left_rel_px.append(lwx - tx)
                right_rel_px.append(rwx - tx)
        if len(left_rel_px) < 2 or len(right_rel_px) < 2:
            return None
        left_range_cm = (max(left_rel_px) - min(left_rel_px)) / px_per_cm
        right_range_cm = (max(right_rel_px) - min(right_rel_px)) / px_per_cm
        return {
            'left_arm_range_cm': left_range_cm,
            'right_arm_range_cm': right_range_cm,
            'average_range_cm': (left_range_cm + right_range_cm) / 2.0
        }

    def _vertical_range_cm(self, frames_data: List[Dict], width: int, height: int, px_per_cm: float) -> Optional[Dict]:
        """Compute wrist–torso vertical range (cm) for left and right arms over all run-up frames."""
        # Get run-up frames but exclude last 5 frames before bound
        bound_start = next((i for i, f in enumerate(frames_data) if f.get('phase') == 'bound'), None)
        if bound_start is not None:
            runup_frames = [f for i, f in enumerate(frames_data) if f.get('phase') == 'run_up' and i < bound_start - 5]
        else:
            runup_frames = [f for f in frames_data if f.get('phase') == 'run_up']
        if len(runup_frames) < 2:
            return None
        left_rel_px = []
        right_rel_px = []
        for f in runup_frames:
            k = f.get('keypoints', {})
            if all(key in k for key in ['left_wrist','right_wrist','left_hip','right_hip','left_shoulder','right_shoulder']):
                torso_y = (k['left_shoulder']['y'] + k['right_shoulder']['y'] + k['left_hip']['y'] + k['right_hip']['y']) / 4.0
                lwy = k['left_wrist']['y'] * height
                rwy = k['right_wrist']['y'] * height
                ty = torso_y * height
                # positive up: use torso_y - wrist_y (image coords inverted)
                left_rel_px.append(ty - lwy)
                right_rel_px.append(ty - rwy)
        if len(left_rel_px) < 2 or len(right_rel_px) < 2:
            return None
        left_range_cm = (max(left_rel_px) - min(left_rel_px)) / px_per_cm
        right_range_cm = (max(right_rel_px) - min(right_rel_px)) / px_per_cm
        return {
            'left_arm_range_cm': left_range_cm,
            'right_arm_range_cm': right_range_cm,
            'average_range_cm': (left_range_cm + right_range_cm) / 2.0
        }
    
    def analyze_runup(self, video_path: str, output_dir: str = "runup_analysis") -> Dict:
        """
        Complete runup analysis pipeline.
        
        Args:
            video_path: Path to the runup video
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"Starting complete runup analysis of {video_path}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Step 1: Extract pose keypoints
        frames_data, actual_fps, width, height = self.extract_pose_keypoints(video_path)
        
        # Step 2: Detect phases
        frames_data = self.detect_phases(frames_data)
        
        # Real-unit scaling (pixels per cm) using elbow→wrist reference
        px_per_cm = self._median_pixels_per_cm(frames_data, width, height)
        if not px_per_cm:
            logger.warning("Could not compute scale; output will be empty.")
            return {
                'runup_speeds_kmh': [],
                'forward_backward_arm_movement_cm': {},
                'vertical_arm_movement_cm': {}
            }
        
        # Step 3: Calculate runup speed in km/h (simple list of segments)
        runup_speeds_kmh = self._compute_runup_speeds_kmh(frames_data, width, height, px_per_cm, interval=30)
        
        # Step 4: Arm movement ranges in cm over run-up
        fb_range_cm = self._forward_backward_range_cm(frames_data, width, height, px_per_cm) or {}
        vert_range_cm = self._vertical_range_cm(frames_data, width, height, px_per_cm) or {}
        
        # Compile results
        results = {
            'runup_speeds_kmh': runup_speeds_kmh,
            'forward_backward_arm_movement_cm': fb_range_cm,
            'vertical_arm_movement_cm': vert_range_cm,
            'total_frames': len(frames_data)
        }
        
        # Save results
        output_file = Path(output_dir) / f"{Path(video_path).stem}_runup_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Analysis complete. Results saved to {output_file}")
        
        return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze runup biomechanics from bowling video")
    parser.add_argument("video_path", help="Path to the runup video file")
    parser.add_argument("--output-dir", default="runup_analysis", help="Output directory for results")
    parser.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2], 
                       help="MediaPipe model complexity")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RunupAnalyzer(model_complexity=args.model_complexity)
    
    # Run analysis
    try:
        results = analyzer.analyze_runup(args.video_path, args.output_dir)
        print(f"\nAnalysis complete!")
        print(f"Total frames analyzed: {results['total_frames']}")
        print(f"Runup speed segments: {len(results['runup_speeds_kmh'])}")
        print(f"Forward-backward arm movement (cm): {results.get('forward_backward_arm_movement_cm',{})}")
        print(f"Vertical arm movement (cm): {results.get('vertical_arm_movement_cm',{})}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
