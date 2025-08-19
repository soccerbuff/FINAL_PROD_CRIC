"""
back_metrics.py
This module contains functions to compute back-view biomechanical metrics.
"""
import json
import math

def safe_get_keypoint(kps, key):
    """Safely get a keypoint dictionary with 'x' and 'y' or return None if missing or incomplete."""
    if key not in kps:
        return None
    pt = kps[key]
    if pt is None or 'x' not in pt or 'y' not in pt:
        return None
    return pt

# --- Utility Functions ---
def get_midpoint(p1, p2):
    return {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}

def get_angle_with_vertical(p1, p2):
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    return math.degrees(math.atan2(dx, dy))

# --- Back-View Metric Calculations ---

def calculate_ffc_to_release_time(data):
    """Calculate the time from FFC (First Frame Contact) to ball release."""
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    
    if ffc_idx is None or release_idx is None:
        return None, None, None, None
    
    ffc_time = data[ffc_idx]['timestamp']
    release_time = data[release_idx]['timestamp']
    total_time = release_time - ffc_time
    
    # Create frame timings for all frames between FFC and release
    frame_timings = []
    for i in range(ffc_idx, release_idx + 1):
        current_time = data[i]['timestamp'] - ffc_time
        frame_timings.append({
            'frame': i,
            'elapsed_time': current_time,
            'total_time': total_time
        })
    
    return total_time, ffc_idx, release_idx, frame_timings

def calculate_pelvic_drop(frame):
    if not frame: return None
    kps = frame['keypoints']
    left_hip = safe_get_keypoint(kps, 'left_hip')
    right_hip = safe_get_keypoint(kps, 'right_hip')
    if not left_hip or not right_hip:
        return None
    # Angle of the hip line with horizontal
    return get_angle_with_vertical(left_hip, right_hip) - 90

def calculate_knee_valgus_varus(frame, leg):
    if not frame or not leg: return None
    kps = frame['keypoints']
    hip = safe_get_keypoint(kps, f'{leg}_hip')
    knee = safe_get_keypoint(kps, f'{leg}_knee')
    ankle = safe_get_keypoint(kps, f'{leg}_ankle')
    if not hip or not knee or not ankle:
        return None
    # Deviation of the knee from the hip-ankle line
    line_x = hip['x'] + (ankle['x'] - hip['x']) * ((knee['y'] - hip['y']) / (ankle['y'] - hip['y']))
    return knee['x'] - line_x

def get_bfc_ffc_release_indices(data):
    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    return bfc_idx, ffc_idx, release_idx

def get_bfc_ffc_arms(data):
    bfc_idx, ffc_idx, _ = get_bfc_ffc_release_indices(data)
    if bfc_idx is None or ffc_idx is None:
        return None, None, None, None
    kps_bfc = data[bfc_idx]['keypoints']
    left_ankle_y_bfc = kps_bfc['left_ankle']['y']
    right_ankle_y_bfc = kps_bfc['right_ankle']['y']
    bfc_leg = 'left' if left_ankle_y_bfc > right_ankle_y_bfc else 'right'
    ffc_leg = 'right' if bfc_leg == 'left' else 'left'
    bowling_arm = bfc_leg  # Bowling arm is the same side as back foot
    lead_arm = ffc_leg
    return bfc_leg, ffc_leg, bowling_arm, lead_arm

def calculate_hip_shoulder_separation_release(data, release_idx):
    """
    Calculate hip-shoulder separation at release.
    Measures the difference in tilt between shoulder line and hip line.
    """
    kps = data[release_idx]['keypoints']
    # Check if all required keypoints are present
    if 'left_shoulder' in kps and 'right_shoulder' in kps and 'left_hip' in kps and 'right_hip' in kps:
        left_shoulder = kps['left_shoulder']
        right_shoulder = kps['right_shoulder']
        left_hip = kps['left_hip']
        right_hip = kps['right_hip']
        
        # Calculate shoulder line angle (horizontal tilt)
        shoulder_angle = math.degrees(math.atan2(right_shoulder['y'] - left_shoulder['y'], 
                                               right_shoulder['x'] - left_shoulder['x']))
        
        # Calculate hip line angle (horizontal tilt)
        hip_angle = math.degrees(math.atan2(right_hip['y'] - left_hip['y'], 
                                           right_hip['x'] - left_hip['x']))
        
        # Calculate the difference in tilt between shoulder and hip lines
        tilt_difference = shoulder_angle - hip_angle
        
        # Normalize to [-180, 180]
        if tilt_difference > 180:
            tilt_difference -= 360
        elif tilt_difference < -180:
            tilt_difference += 360
            
        return tilt_difference, release_idx, (left_shoulder, right_shoulder, left_hip, right_hip)
    else:
        # Write 'data missing' if keypoints are missing
        return 'data missing', release_idx, 'data missing'

def calculate_pelvic_drop_ffc(data, ffc_idx):
    kps = data[ffc_idx]['keypoints']
    y_lhip = kps['left_hip']['y']
    y_rhip = kps['right_hip']['y']
    delta_y = abs(y_lhip - y_rhip)
    return delta_y, ffc_idx, (y_lhip, y_rhip)

def calculate_bowling_arm_abduction_release(data, release_idx, bowling_arm='right'):
    """
    Calculate true shoulder abduction angle at release.
    Measures the angle between the upper arm (shoulder to elbow) and vertical axis.
    """
    kps = data[release_idx]['keypoints']
    S_key = f'{bowling_arm}_shoulder'
    E_key = f'{bowling_arm}_elbow'
    
    if S_key in kps and E_key in kps:
        S = kps[S_key]
        E = kps[E_key]
        
        # Calculate vector from shoulder to elbow
        arm_vector = (E['x'] - S['x'], E['y'] - S['y'])
        
        # Calculate angle with vertical (y-axis)
        # Positive angle means arm is abducted (raised away from body)
        angle = math.degrees(math.atan2(arm_vector[0], arm_vector[1]))
        
        # Normalize to [0, 180] degrees for abduction
        if angle < 0:
            angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
            
        return angle, release_idx, (S, E)
    else:
        # Handle missing keypoints gracefully
        return None, None, None

def calculate_peak_velocity(angles, timestamps):
    if len(angles) < 2 or len(timestamps) < 2:
        return None
    max_velocity = 0
    for i in range(1, len(angles)):
        dt = timestamps[i] - timestamps[i-1]
        if dt == 0:
            continue
        velocity = abs(angles[i] - angles[i-1]) / dt
        if velocity > max_velocity:
            max_velocity = velocity
    return max_velocity

def calculate_rotation_angles(data, start_idx, end_idx, point1_key1, point1_key2, point2_key1, point2_key2):
    angles = []
    timestamps = []
    for i in range(start_idx, end_idx + 1):
        kps = data[i]['keypoints']
        if all(k in kps for k in [point1_key1, point1_key2, point2_key1, point2_key2]):
            mid_point1 = get_midpoint(kps[point1_key1], kps[point1_key2])
            mid_point2 = get_midpoint(kps[point2_key1], kps[point2_key2])
            angle = get_angle_with_vertical(mid_point1, mid_point2)
            angles.append(angle)
            timestamps.append(data[i]['timestamp'])
        else:
            angles.append(None)
            timestamps.append(data[i]['timestamp'])
    return angles, timestamps

def calculate_x_factor(shoulder_angles, pelvis_angles, timestamps, br_idx):
    x_factors = []
    for s, p in zip(shoulder_angles, pelvis_angles):
        if s is None or p is None:
            x_factors.append(None)
        else:
            x_factors.append(s - p)
    # Find peak X-factor and timing relative to BR
    peak_x_factor = None
    peak_idx = None
    for i, val in enumerate(x_factors):
        if val is not None and (peak_x_factor is None or abs(val) > abs(peak_x_factor)):
            peak_x_factor = val
            peak_idx = i
    if peak_idx is None or br_idx >= len(timestamps):
        return None, None
    time_diff_ms = (timestamps[peak_idx] - timestamps[br_idx]) * 1000  # convert to ms
    return peak_x_factor, time_diff_ms

def calculate_runup_path(data):
    """
    Calculate the runup path as a list of pelvis center points (x, y) with frame indices.
    """
    path = []
    for i, frame in enumerate(data):
        kps = frame.get('keypoints', {})
        left_hip = kps.get('left_hip')
        right_hip = kps.get('right_hip')
        if left_hip and right_hip:
            pelvis_center = get_midpoint(left_hip, right_hip)
            path.append({'frame': i, 'x': pelvis_center['x'], 'y': pelvis_center['y'], 'timestamp': frame.get('timestamp')})
    return path

def calculate_all_back_metrics(phased_json_path):
    with open(phased_json_path) as f:
        data = json.load(f)
    metrics = {}
    bfc_idx, ffc_idx, release_idx = get_bfc_ffc_release_indices(data)

    # 1. FFC to Ball Release Time
    if ffc_idx is not None and release_idx is not None:
        ffc_to_release_result = calculate_ffc_to_release_time(data)
        if ffc_to_release_result:
            metrics['ffc_to_release_time'] = {
                'value': ffc_to_release_result[0],
                'ffc_frame': ffc_to_release_result[1],
                'release_frame': ffc_to_release_result[2],
                'frame_timings': ffc_to_release_result[3]
            }

    

    # 5. Hip-Shoulder Separation at Release
    if release_idx is not None:
        angle, idx, coords = calculate_hip_shoulder_separation_release(data, release_idx)
        if angle == 'data missing':
            metrics['hip_shoulder_separation_release'] = {'value': 'data missing', 'frame': idx, 'coordinates': coords}
        else:
            metrics['hip_shoulder_separation_release'] = {'value': angle, 'frame': idx, 'coordinates': coords}

    # 6. Pelvic Drop at FFC (existing)
    if ffc_idx is not None:
        delta_y, idx, y_coords = calculate_pelvic_drop_ffc(data, ffc_idx)
        metrics['pelvic_drop_ffc'] = {'value': delta_y, 'frame': idx, 'y_coords': y_coords}

    # 7. Bowling Arm Abduction Angle at Release (existing)
    if release_idx is not None:
        # Infer bowling arm from legs
        _, ffc_leg, bowling_arm, _ = get_bfc_ffc_arms(data)
        angle, idx, coords = calculate_bowling_arm_abduction_release(data, release_idx, bowling_arm=bowling_arm or 'right')
        if angle is None:
            metrics['bowling_arm_abduction_release'] = {'value': 'data missing', 'frame': idx, 'coordinates': coords}
        else:
            metrics['bowling_arm_abduction_release'] = {'value': angle, 'frame': idx, 'coordinates': coords}

    return metrics
