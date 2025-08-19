import json
import mediapipe as mp
import cv2
import sys
import os

def extract_pose_keypoints(video_path, view=None):
    mp_pose = mp.solutions.pose
    
    # Use lower thresholds for better pose detection, especially for front view
    if view == 'front':
        # Front view often has more challenging angles, use lower thresholds
        min_tracking_confidence = 0.3
        min_detection_confidence = 0.3
        model_complexity = 1  # Use medium complexity for better speed/accuracy balance
    else:
        # Keep higher thresholds for side and back views
        min_tracking_confidence = 0.5
        min_detection_confidence = 0.4
        model_complexity = 2
    
    pose = mp_pose.Pose(
        model_complexity=model_complexity, 
        min_tracking_confidence=min_tracking_confidence,
        min_detection_confidence=min_detection_confidence
    )
    cap = cv2.VideoCapture(video_path)
    keypoints_data = []
    frame_idx = 0
    def get_midpoint(p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            # Count how many keypoints are detected
            detected_count = len([lm for lm in results.pose_landmarks.landmark if lm.visibility > 0.1])
            print(f"Frame {frame_idx}: Pose detected with {detected_count} keypoints")
        else:
            print(f"Frame {frame_idx}: No pose detected")
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if results.pose_landmarks:
            h, w, _ = image.shape
            lm = results.pose_landmarks.landmark
            kp = {
                'left_shoulder': lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
                'right_shoulder': lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                'left_elbow': lm[mp_pose.PoseLandmark.LEFT_ELBOW],
                'right_elbow': lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
                'left_wrist': lm[mp_pose.PoseLandmark.LEFT_WRIST],
                'right_wrist': lm[mp_pose.PoseLandmark.RIGHT_WRIST],
                'left_hip': lm[mp_pose.PoseLandmark.LEFT_HIP],
                'right_hip': lm[mp_pose.PoseLandmark.RIGHT_HIP],
                'left_knee': lm[mp_pose.PoseLandmark.LEFT_KNEE],
                'right_knee': lm[mp_pose.PoseLandmark.RIGHT_KNEE],
                'left_ankle': lm[mp_pose.PoseLandmark.LEFT_ANKLE],
                'right_ankle': lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
                'left_heel': lm[mp_pose.PoseLandmark.LEFT_HEEL],
                'right_heel': lm[mp_pose.PoseLandmark.RIGHT_HEEL],
                'left_foot_index': lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
                'right_foot_index': lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
                'nose': lm[mp_pose.PoseLandmark.NOSE],
                'left_eye': lm[mp_pose.PoseLandmark.LEFT_EYE],
                'right_eye': lm[mp_pose.PoseLandmark.RIGHT_EYE],
                'left_ear': lm[mp_pose.PoseLandmark.LEFT_EAR],
                'right_ear': lm[mp_pose.PoseLandmark.RIGHT_EAR],
                'left_index': lm[mp_pose.PoseLandmark.LEFT_INDEX],
                'right_index': lm[mp_pose.PoseLandmark.RIGHT_INDEX],
                'left_thumb': lm[mp_pose.PoseLandmark.LEFT_THUMB],
                'right_thumb': lm[mp_pose.PoseLandmark.RIGHT_THUMB],
                'left_pinky': lm[mp_pose.PoseLandmark.LEFT_PINKY],
                'right_pinky': lm[mp_pose.PoseLandmark.RIGHT_PINKY]
            }
            # Include visibility and z-coordinate for better analysis
            frame_keypoints = {}
            for k, v in kp.items():
                # For front view, be more lenient with visibility thresholds
                if view == 'front':
                    # Include keypoints even with lower visibility for front view
                    if v.visibility > 0.1:  # Very low threshold for front view
                        frame_keypoints[k] = {
                            'x': int(v.x * w), 
                            'y': int(v.y * h), 
                            'z': v.z,
                            'visibility': v.visibility
                        }
                else:
                    # Standard visibility threshold for other views
                    if v.visibility > 0.3:
                        frame_keypoints[k] = {
                            'x': int(v.x * w), 
                            'y': int(v.y * h), 
                            'z': v.z,
                            'visibility': v.visibility
                        }
            keypoints_data.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'keypoints': frame_keypoints,
                'pose_detected': True
            })
        else:
            keypoints_data.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'keypoints': {},
                'pose_detected': False
            })
        frame_idx += 1
    cap.release()
    # Output JSON file path
    base = os.path.splitext(os.path.basename(video_path))[0]
    if view:
        keypoints_json = f"{base}_{view}_keypoints.json"
    else:
        keypoints_json = f"{base}_keypoints.json"
    with open(keypoints_json, 'w') as f:
        json.dump(keypoints_data, f, indent=2)
    return keypoints_json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_est.py <video_path> [view]")
        sys.exit(1)
    video_path = sys.argv[1]
    view = sys.argv[2] if len(sys.argv) > 2 else None
    output_json = extract_pose_keypoints(video_path, view)
    print(f"Keypoints JSON saved to: {output_json}")