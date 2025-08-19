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

def get_midpoint(p1, p2):
    """Calculate the midpoint between two points."""
    return {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}

def calculate_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    norm_ab = (ab[0]**2 + ab[1]**2) ** 0.5
    norm_cb = (cb[0]**2 + cb[1]**2) ** 0.5
    if norm_ab == 0 or norm_cb == 0:
        return 0.0
    cos_angle = dot / (norm_ab * norm_cb)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

def get_bfc_ffc_arms(data):
    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
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

def calculate_front_knee_angle_at_ffc(data):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    if ffc_idx is None or ffc_leg is None:
        return None, None, None, None
    front_leg = ffc_leg
    kps = data[ffc_idx]['keypoints']
    hip = (kps[f'{front_leg}_hip']['x'], kps[f'{front_leg}_hip']['y'])
    knee = (kps[f'{front_leg}_knee']['x'], kps[f'{front_leg}_knee']['y'])
    ankle = (kps[f'{front_leg}_ankle']['x'], kps[f'{front_leg}_ankle']['y'])
    angle = calculate_angle(hip, knee, ankle)
    return angle, ffc_idx, knee, front_leg

def calculate_min_front_knee_angle_post_ffc(data, time_window_ms=120):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if ffc_idx is None or ffc_leg is None:
        return None, None, None, None
    start_time = data[ffc_idx]['timestamp']
    end_time = start_time + time_window_ms / 1000.0
    min_angle = None
    min_idx = None
    min_knee = None
    for i in range(ffc_idx, len(data)):
        if data[i]['timestamp'] > end_time or (release_idx is not None and i > release_idx):
            break
        kps = data[i]['keypoints']
        hip = (kps[f'{ffc_leg}_hip']['x'], kps[f'{ffc_leg}_hip']['y'])
        knee = (kps[f'{ffc_leg}_knee']['x'], kps[f'{ffc_leg}_knee']['y'])
        ankle = (kps[f'{ffc_leg}_ankle']['x'], kps[f'{ffc_leg}_ankle']['y'])
        angle = calculate_angle(hip, knee, ankle)
        if min_angle is None or angle < min_angle:
            min_angle = angle
            min_idx = i
            min_knee = knee
    return min_angle, min_idx, min_knee, ffc_leg

def calculate_front_knee_angle_at_release(data):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if release_idx is None or ffc_leg is None:
        return None, None, None, None
    front_leg = ffc_leg
    kps = data[release_idx]['keypoints']
    hip = (kps[f'{front_leg}_hip']['x'], kps[f'{front_leg}_hip']['y'])
    knee = (kps[f'{front_leg}_knee']['x'], kps[f'{front_leg}_knee']['y'])
    ankle = (kps[f'{front_leg}_ankle']['x'], kps[f'{front_leg}_ankle']['y'])
    angle = calculate_angle(hip, knee, ankle)
    return angle, release_idx, knee, front_leg

def calculate_front_knee_extension_velocity(data):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if ffc_idx is None or release_idx is None or ffc_leg is None:
        return None
    front_leg = ffc_leg
    kps_ffc = data[ffc_idx]['keypoints']
    kps_release = data[release_idx]['keypoints']
    hip_ffc = (kps_ffc[f'{front_leg}_hip']['x'], kps_ffc[f'{front_leg}_hip']['y'])
    knee_ffc = (kps_ffc[f'{front_leg}_knee']['x'], kps_ffc[f'{front_leg}_knee']['y'])
    ankle_ffc = (kps_ffc[f'{front_leg}_ankle']['x'], kps_ffc[f'{front_leg}_ankle']['y'])
    angle_ffc = calculate_angle(hip_ffc, knee_ffc, ankle_ffc)
    hip_release = (kps_release[f'{front_leg}_hip']['x'], kps_release[f'{front_leg}_hip']['y'])
    knee_release = (kps_release[f'{front_leg}_knee']['x'], kps_release[f'{front_leg}_knee']['y'])
    ankle_release = (kps_release[f'{front_leg}_ankle']['x'], kps_release[f'{front_leg}_ankle']['y'])
    angle_release = calculate_angle(hip_release, knee_release, ankle_release)
    time_diff = data[release_idx]['timestamp'] - data[ffc_idx]['timestamp']
    if time_diff == 0:
        return None
    velocity = (angle_release - angle_ffc) / time_diff
    return {'value': velocity, 'frame': release_idx}

def calculate_trunk_forward_flexion_at_release(data):
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if release_idx is None:
        return None
    kps = data[release_idx]['keypoints']
    mid_shoulder = ((kps['left_shoulder']['x'] + kps['right_shoulder']['x']) / 2,
                    (kps['left_shoulder']['y'] + kps['right_shoulder']['y']) / 2)
    mid_hip = ((kps['left_hip']['x'] + kps['right_hip']['x']) / 2,
               (kps['left_hip']['y'] + kps['right_hip']['y']) / 2)
    
    # Calculate trunk line vector (hip to shoulder)
    trunk_dx = mid_shoulder[0] - mid_hip[0]
    trunk_dy = mid_shoulder[1] - mid_hip[1]
    
    # Vertical line passes through hip midpoint
    # Angle between vertical line and trunk line
    # Vertical line has direction (0, -1) pointing upward
    # Trunk line has direction (trunk_dx, trunk_dy) from hip to shoulder
    
    # Calculate angle between vertical line and trunk line
    # Vertical vector: (0, -1) pointing upward
    # Trunk vector: (trunk_dx, trunk_dy) from hip to shoulder
    vertical_vector = (0, -1)  # Pointing upward
    trunk_vector = (trunk_dx, trunk_dy)
    
    # Normalize vectors
    trunk_magnitude = (trunk_dx**2 + trunk_dy**2)**0.5
    if trunk_magnitude == 0:
        return None
    
    trunk_unit = (trunk_dx/trunk_magnitude, trunk_dy/trunk_magnitude)
    vertical_unit = (0, -1)
    
    # Calculate dot product
    dot_product = trunk_unit[0]*vertical_unit[0] + trunk_unit[1]*vertical_unit[1]
    dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
    
    # Calculate angle in degrees
    angle = math.degrees(math.acos(dot_product))
    
    return angle, release_idx, (mid_shoulder, mid_hip)

def calculate_peak_forward_flexion_velocity(data):
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if release_idx is None or release_idx < 2:
        return None
    
    # Use a larger time window (3 frames) to get more stable velocity calculation
    kps_release = data[release_idx]['keypoints']
    kps_prev = data[release_idx - 2]['keypoints']  # Use 2 frames back instead of 1
    
    # Calculate trunk angles using the same method as trunk_forward_flexion_at_release
    def calculate_trunk_angle(kps):
        mid_shoulder = ((kps['left_shoulder']['x'] + kps['right_shoulder']['x']) / 2,
                        (kps['left_shoulder']['y'] + kps['right_shoulder']['y']) / 2)
        mid_hip = ((kps['left_hip']['x'] + kps['right_hip']['x']) / 2,
                   (kps['left_hip']['y'] + kps['right_hip']['y']) / 2)
        
        # Calculate trunk line vector (hip to shoulder)
        trunk_dx = mid_shoulder[0] - mid_hip[0]
        trunk_dy = mid_shoulder[1] - mid_hip[1]
        
        # Calculate angle between trunk and vertical line
        vertical_vector = (0, -1)  # Pointing upward
        trunk_vector = (trunk_dx, trunk_dy)
        
        # Normalize vectors
        trunk_magnitude = (trunk_dx**2 + trunk_dy**2)**0.5
        if trunk_magnitude == 0:
            return None
        
        trunk_unit = (trunk_dx/trunk_magnitude, trunk_dy/trunk_magnitude)
        vertical_unit = (0, -1)
        
        # Calculate dot product
        dot_product = trunk_unit[0]*vertical_unit[0] + trunk_unit[1]*vertical_unit[1]
        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
        
        # Calculate angle in degrees
        angle = math.degrees(math.acos(dot_product))
        return angle
    
    angle_release = calculate_trunk_angle(kps_release)
    angle_prev = calculate_trunk_angle(kps_prev)
    
    if angle_release is None or angle_prev is None:
        return None
    
    time_diff = data[release_idx]['timestamp'] - data[release_idx - 2]['timestamp']
    if time_diff == 0:
        return None
    
    velocity = (angle_release - angle_prev) / time_diff
    return {'value': velocity, 'frame': release_idx}

def calculate_release_height(data):
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if release_idx is None:
        return None
    kps = data[release_idx]['keypoints']
    
    # Get wrist Y position (prefer bowling arm wrist if determinable)
    bfc_leg, ffc_leg, bowling_arm, _ = get_bfc_ffc_arms(data)
    wrist_y = None
    if bowling_arm:
        wrist_y = kps.get(f'{bowling_arm}_wrist', {}).get('y')
    if wrist_y is None:
        wrist_y = kps.get('right_wrist', {}).get('y') or kps.get('left_wrist', {}).get('y')
    
    # Get heel Y position (use the higher heel as ground reference)
    left_heel_y = kps.get('left_heel', {}).get('y')
    right_heel_y = kps.get('right_heel', {}).get('y')
    
    if wrist_y is None:
        return None
    
    # Use the higher heel as ground reference (lower Y value = higher position)
    heel_y = None
    if left_heel_y is not None and right_heel_y is not None:
        heel_y = min(left_heel_y, right_heel_y)  # Lower Y = higher position
    elif left_heel_y is not None:
        heel_y = left_heel_y
    elif right_heel_y is not None:
        heel_y = right_heel_y
    else:
        # Fallback to ankle if heel not available
        left_ankle_y = kps.get('left_ankle', {}).get('y')
        right_ankle_y = kps.get('right_ankle', {}).get('y')
        if left_ankle_y is not None and right_ankle_y is not None:
            heel_y = min(left_ankle_y, right_ankle_y)
        elif left_ankle_y is not None:
            heel_y = left_ankle_y
        elif right_ankle_y is not None:
            heel_y = right_ankle_y
        else:
            return None
    
    # Calculate absolute height of wrist position at release
    # We want the height of the wrist from the ground
    # In image coordinates, Y increases downward, so we need to measure from bottom to wrist
    
    # Get the ground level (use the lower heel as ground reference)
    ground_y = None
    if left_heel_y is not None and right_heel_y is not None:
        ground_y = max(left_heel_y, right_heel_y)  # Higher Y = lower position = ground
    elif left_heel_y is not None:
        ground_y = left_heel_y
    elif right_heel_y is not None:
        ground_y = right_heel_y
    else:
        # Fallback to ankle if heel not available
        left_ankle_y = kps.get('left_ankle', {}).get('y')
        right_ankle_y = kps.get('right_ankle', {}).get('y')
        if left_ankle_y is not None and right_ankle_y is not None:
            ground_y = max(left_ankle_y, right_ankle_y)
        elif left_ankle_y is not None:
            ground_y = left_ankle_y
        elif right_ankle_y is not None:
            ground_y = right_ankle_y
        else:
            return None
    
    # Calculate height from ground to wrist
    # Height = ground_y - wrist_y (since lower Y = higher position)
    wrist_height_from_ground = ground_y - wrist_y
    
    return wrist_height_from_ground, release_idx

def calculate_bound_height(data):
    bound_indices = [i for i, f in enumerate(data) if f.get('phase') == 'bound']
    if not bound_indices:
        return None, None
    
    # Get feet position at bound start
    bound_start_idx = bound_indices[0]
    kps_start = data[bound_start_idx]['keypoints']
    left_ankle_start = kps_start.get('left_ankle')
    right_ankle_start = kps_start.get('right_ankle')
    if not left_ankle_start or not right_ankle_start:
        return None, None
    
    feet_y_start = max(left_ankle_start['y'], right_ankle_start['y'])
    
    # Find frame with highest hip position (lowest Y value) during bound
    min_mid_hip_y = None
    max_mid_hip_frame = None
    for i in bound_indices:
        kps = data[i]['keypoints']
        left_hip = kps.get('left_hip')
        right_hip = kps.get('right_hip')
        if left_hip and right_hip:
            mid_hip_y = (left_hip['y'] + right_hip['y']) / 2
            if min_mid_hip_y is None or mid_hip_y < min_mid_hip_y:
                min_mid_hip_y = mid_hip_y
                max_mid_hip_frame = i
    
    if max_mid_hip_frame is None:
        return None, None
    
    # Get feet position at highest point
    kps_peak = data[max_mid_hip_frame]['keypoints']
    left_ankle_peak = kps_peak.get('left_ankle')
    right_ankle_peak = kps_peak.get('right_ankle')
    if not left_ankle_peak or not right_ankle_peak:
        return None, None
    
    feet_y_peak = max(left_ankle_peak['y'], right_ankle_peak['y'])
    
    # Calculate bound height: feet_y_start - feet_y_peak
    bound_height = feet_y_start - feet_y_peak
    
    return bound_height, max_mid_hip_frame


def calculate_bound_flight_time(data):
    bound_frames = [(i, f['timestamp']) for i, f in enumerate(data) if f.get('phase') == 'bound']
    if not bound_frames:
        return None
    first_frame_idx, first_time = bound_frames[0]
    last_frame_idx, last_time = bound_frames[-1]
    flight_time = last_time - first_time
    frame_timings = [t for i, t in bound_frames]
    return flight_time, first_frame_idx, last_frame_idx, frame_timings

def calculate_stride_length_at_ffc(data):
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    if ffc_idx is None:
        return None
    kps = data[ffc_idx]['keypoints']
    # Use only ankle x-coordinates for stride length
    left_ankle = kps.get('left_ankle')
    right_ankle = kps.get('right_ankle')
    if not left_ankle or not right_ankle:
        return None
    stride_length_x = abs(left_ankle['x'] - right_ankle['x'])
    left_pt = (left_ankle['x'], left_ankle['y'])
    right_pt = (right_ankle['x'], right_ankle['y'])
    return stride_length_x, ffc_idx, (left_pt, right_pt)

def calculate_bowling_arm_hyperextension(data):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    if bowling_arm is None:
        return None

    # Find release frame index
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    if release_idx is None:
        return None

    # Get keypoints at release frame
    kps = data[release_idx]['keypoints']
    shoulder_pt = safe_get_keypoint(kps, f'{bowling_arm}_shoulder')
    elbow_pt = safe_get_keypoint(kps, f'{bowling_arm}_elbow')
    wrist_pt = safe_get_keypoint(kps, f'{bowling_arm}_wrist')
    
    if not shoulder_pt or not elbow_pt or not wrist_pt:
        return None
    
    shoulder = (shoulder_pt['x'], shoulder_pt['y'])
    elbow = (elbow_pt['x'], elbow_pt['y'])
    wrist = (wrist_pt['x'], wrist_pt['y'])
    
    # Calculate elbow angle at release
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    
    # Check if angle indicates hyperextension (more than 180 degrees or negative)
    if elbow_angle > 180 or elbow_angle < 0:
        return elbow_angle, release_idx, (shoulder, elbow, wrist), bowling_arm
    else:
        return None

def calculate_lead_arm_drop_speed(data):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)

    if ffc_idx is None or release_idx is None or lead_arm is None:
        return None

    max_speed = 0
    max_speed_idx = -1
    max_speed_coords = None

    for i in range(ffc_idx, release_idx):
        if i+1 >= len(data):
            break
        kps1 = data[i]['keypoints']
        kps2 = data[i+1]['keypoints']
        time1 = data[i]['timestamp']
        time2 = data[i+1]['timestamp']

        wrist1_y = kps1[f'{lead_arm}_wrist']['y']
        wrist2_y = kps2[f'{lead_arm}_wrist']['y']
        
        time_diff = time2 - time1
        if time_diff == 0:
            continue
            
        speed = (wrist2_y - wrist1_y) / time_diff
        
        if speed > max_speed:
            max_speed = speed
            max_speed_idx = i + 1
            max_speed_coords = (
                (kps1[f'{lead_arm}_wrist']['x'], wrist1_y),
                (kps2[f'{lead_arm}_wrist']['x'], wrist2_y)
            )

    if max_speed_idx == -1:
        return None

    return max_speed, max_speed_idx, lead_arm, max_speed_coords

def calculate_back_knee_angle_at_bfc(data):
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    if bfc_idx is None or bfc_leg is None:
        return None, None, None, None
    
    back_leg = bfc_leg
    kps = data[bfc_idx]['keypoints']
    hip = (kps[f'{back_leg}_hip']['x'], kps[f'{back_leg}_hip']['y'])
    knee = (kps[f'{back_leg}_knee']['x'], kps[f'{back_leg}_knee']['y'])
    ankle = (kps[f'{back_leg}_ankle']['x'], kps[f'{back_leg}_ankle']['y'])
    angle = calculate_angle(hip, knee, ankle)
    return angle, bfc_idx, knee, back_leg

def calculate_approach_speed(data):
    """
    Calculate approach speed in px/s using only the last two run-up frames before bound start, based on pelvis center movement.
    Returns (speed, frame_idx) where frame_idx is the second-to-last run-up frame.
    """
    runup_start_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'run_up'), None)
    bound_start_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'bound'), None)
    if runup_start_idx is None or bound_start_idx is None:
        return None
    # If runup_start_idx is after bound_start_idx, it means our runup fixing logic converted
    # some bound frames to run_up. In this case, we need to find the original bound start.
    if bound_start_idx <= runup_start_idx:
        # Find the first frame that was originally bound (before our runup fixing)
        # This would be the first frame after the runup frames that is still bound
        original_bound_start = next((i for i, f in enumerate(data) if f.get('phase') == 'bound' and i > runup_start_idx), None)
        if original_bound_start is None:
            return None
        bound_start_idx = original_bound_start

    # Find the last two run-up frames before bound_start_idx
    runup_indices = [i for i in range(runup_start_idx, bound_start_idx)
                     if data[i].get('phase') == 'run_up']
    if len(runup_indices) < 2:
        return "Not enough data"
    idx1, idx2 = runup_indices[-2], runup_indices[-1]
    kps1 = data[idx1].get('keypoints', {})
    kps2 = data[idx2].get('keypoints', {})
    left_hip1 = kps1.get('left_hip')
    right_hip1 = kps1.get('right_hip')
    left_hip2 = kps2.get('left_hip')
    right_hip2 = kps2.get('right_hip')
    if not (left_hip1 and right_hip1 and left_hip2 and right_hip2):
        return "Not enough data"
    pelvis1 = ((left_hip1['x'] + right_hip1['x']) / 2, (left_hip1['y'] + right_hip1['y']) / 2)
    pelvis2 = ((left_hip2['x'] + right_hip2['x']) / 2, (left_hip2['y'] + right_hip2['y']) / 2)
    t1 = data[idx1].get('timestamp', 0)
    t2 = data[idx2].get('timestamp', 0)
    dt = t2 - t1
    if dt == 0:
        return "Not enough data"
    dx = pelvis2[0] - pelvis1[0]
    dy = pelvis2[1] - pelvis1[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    speed = dist / dt
    return speed, idx1  # Return speed and the frame index (second-to-last run-up frame)

def calculate_all_side_metrics(keypoints_json):
    with open(keypoints_json) as f:
        data = json.load(f)
    metrics = {}

    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)

    # 1. Front-knee angle @ FFC (deg)
    if ffc_idx is not None:
        front_knee_ffc = calculate_front_knee_angle_at_ffc(data)
        if front_knee_ffc:
            metrics['front_knee_angle_at_ffc'] = {
                'value': front_knee_ffc[0],
                'frame': front_knee_ffc[1],
                'coordinates': front_knee_ffc[2],
                'leg': front_knee_ffc[3]
            }

    # Approach speed removed for side view


    # 2. Minimum front-knee angle in first 0–120 ms after FFC (deg) before release
    min_front_knee_post_ffc = calculate_min_front_knee_angle_post_ffc(data)
    if min_front_knee_post_ffc:
        metrics['min_front_knee_angle_post_ffc'] = {
            'value': min_front_knee_post_ffc[0],
            'frame': min_front_knee_post_ffc[1],
            'coordinates': min_front_knee_post_ffc[2],
            'leg': min_front_knee_post_ffc[3]
        }

    # 3. Front-knee angle @ release (deg)
    if release_idx is not None:
        front_knee_release = calculate_front_knee_angle_at_release(data)
        if front_knee_release:
            metrics['front_knee_angle_at_release'] = {
                'value': front_knee_release[0],
                'frame': front_knee_release[1],
                'coordinates': front_knee_release[2],
                'leg': front_knee_release[3]
            }

    # 4. Front-knee extension velocity (deg/s) in FFC → release
    if ffc_idx is not None and release_idx is not None:
        front_knee_ext_velocity = calculate_front_knee_extension_velocity(data)
        if front_knee_ext_velocity is not None:
            metrics['front_knee_extension_velocity'] = front_knee_ext_velocity

    # 5. Trunk forward flexion @ release (deg) and peak forward-flexion velocity (deg/s)
    if release_idx is not None:
        trunk_flexion = calculate_trunk_forward_flexion_at_release(data)
        if trunk_flexion:
            metrics['trunk_forward_flexion_at_release'] = {
                'value': trunk_flexion[0],
                'frame': trunk_flexion[1],
                'coordinates': trunk_flexion[2]
            }
        peak_flex_velocity = calculate_peak_forward_flexion_velocity(data)
        if peak_flex_velocity is not None:
            metrics['peak_forward_flexion_velocity'] = peak_flex_velocity

    # 6. Release height (wrist y at release)
    if release_idx is not None:
        release_height = calculate_release_height(data)
        if release_height:
            metrics['release_height'] = {
                'value': release_height[0],
                'frame': release_height[1]
            }

    # 7. Bound height
    bound_height = calculate_bound_height(data)
    if bound_height is not None:
        metrics['bound_height'] = {
            'value': bound_height[0],
            'frame': bound_height[1]
        }

    # 8. Bound flight Time
    bound_flight_time = calculate_bound_flight_time(data)
    if bound_flight_time:
        metrics['bound_flight_time'] = {
            'value': bound_flight_time[0],
            'first_frame': bound_flight_time[1],
            'last_frame': bound_flight_time[2],
            'frame_timings': bound_flight_time[3]
        }

    # 9. Stride length (same like before)
    if ffc_idx is not None:
        stride_length = calculate_stride_length_at_ffc(data)
        if stride_length:
            metrics['stride_length_at_ffc'] = {
                'value': stride_length[0],
                'frame': stride_length[1],
                'coordinates': stride_length[2]
            }

    # 10. Hyperextension (like before)
    if release_idx is not None:
        hyperextension = calculate_bowling_arm_hyperextension(data)
        if hyperextension:
            metrics['bowling_arm_hyperextension_angle'] = {
                'value': hyperextension[0],
                'frame': hyperextension[1],
                'coordinates': hyperextension[2],
                'arm': hyperextension[3]
            }

    # 11. Lead arm drop speed
    if ffc_idx is not None and release_idx is not None:
        lead_arm_drop = calculate_lead_arm_drop_speed(data)
        if lead_arm_drop:
            metrics['lead_arm_drop_speed'] = {
                'value': lead_arm_drop[0],
                'frame': lead_arm_drop[1],
                'arm': lead_arm_drop[2],
                'coordinates': lead_arm_drop[3]
            }

    # 12. BFC knee angle
    if bfc_idx is not None:
        bfc_knee_angle = calculate_back_knee_angle_at_bfc(data)
        if bfc_knee_angle:
            metrics['back_knee_angle_at_bfc'] = {
                'value': bfc_knee_angle[0],
                'frame': bfc_knee_angle[1],
                'coordinates': bfc_knee_angle[2],
                'leg': bfc_knee_angle[3]
            }

    # 13. Back knee angle minimum after bfc till 100ms before ffc
    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    if bfc_idx is not None and ffc_idx is not None:
        bfc_leg, _, _, _ = get_bfc_ffc_arms(data)
        if bfc_leg is None:
            pass
        else:
            time_100ms_before_ffc = data[ffc_idx]['timestamp'] - 0.1
            min_angle = None
            min_idx = None
            min_knee = None
            for i in range(bfc_idx, len(data)):
                if data[i]['timestamp'] > time_100ms_before_ffc:
                    break
                kps = data[i]['keypoints']
                hip = (kps[f'{bfc_leg}_hip']['x'], kps[f'{bfc_leg}_hip']['y'])
                knee = (kps[f'{bfc_leg}_knee']['x'], kps[f'{bfc_leg}_knee']['y'])
                ankle = (kps[f'{bfc_leg}_ankle']['x'], kps[f'{bfc_leg}_ankle']['y'])
                angle = calculate_angle(hip, knee, ankle)
                if min_angle is None or angle < min_angle:
                    min_angle = angle
                    min_idx = i
                    min_knee = knee
            if min_angle is not None:
                metrics['min_back_knee_angle_post_bfc'] = {
                    'value': min_angle,
                    'frame': min_idx,
                    'coordinates': min_knee,
                    'leg': bfc_leg
                }

    # 14. Back-knee collapse (deg): knee_angle_at_BFC − min_knee_angle in [BFC, BFC + win_ms]
    back_knee_collapse = calculate_back_knee_collapse_with_trail(data)
    if back_knee_collapse and back_knee_collapse['value'] is not None:
        metrics['back_knee_collapse'] = {
            'value': back_knee_collapse['value'],
            'frame': back_knee_collapse['bfc_idx'],
            'trail': back_knee_collapse['trail'],
            'leg': back_knee_collapse['leg'],
            'knee_angle_at_bfc': back_knee_collapse['knee_angle_at_bfc'],
            'min_knee_angle_in_window': back_knee_collapse['min_knee_angle_in_window'],
            'min_angle_frame_idx': back_knee_collapse['min_idx']
        }

    # 15. Front-knee flexion (deg): knee_angle_at_FFC − min_knee_angle in [FFC, FFC + win_ms]
    front_knee_flexion = calculate_front_knee_flexion_with_trail(data)
    if front_knee_flexion and front_knee_flexion['value'] is not None:
        metrics['front_knee_flexion'] = {
            'value': front_knee_flexion['value'],
            'frame': front_knee_flexion['ffc_idx'],
            'trail': front_knee_flexion['trail'],
            'leg': front_knee_flexion['leg'],
            'knee_angle_at_ffc': front_knee_flexion['knee_angle_at_ffc'],
            'min_knee_angle_in_window': front_knee_flexion['min_knee_angle_in_window'],
            'min_angle_frame_idx': front_knee_flexion['min_idx']
        }

    # 16. Front leg kinematics classification (FFC to release)
    front_leg_kinematics = calculate_front_leg_kinematics(data)
    if front_leg_kinematics:
        # Get the leg information for front leg kinematics
        _, ffc_leg, _, _ = get_bfc_ffc_arms(data)
        metrics['front_leg_kinematics'] = {
            'value': front_leg_kinematics,
            'frame': ffc_idx if ffc_idx is not None else None,
            'leg': ffc_leg if ffc_leg is not None else None
        }

    # 17. ARM extension at FFC (below/above shoulder)
    if ffc_idx is not None:
        arm_extension = calculate_arm_extension_at_ffc(data)
        if arm_extension:
            metrics['arm_extension_at_ffc'] = {
                'value': arm_extension[0],
                'frame': arm_extension[1],
                'coordinates': arm_extension[2]
            }

    # 18. Directional Efficiency Score (Composite Metric)
    directional_result = calculate_directional_efficiency_score(data)
    if directional_result[0] is not None:
        directional_efficiency_score, hand_forward_start_score, arm_early_elevation_score, trunk_lean_back_score, arm_torso_vector_alignment_score = directional_result
        
        # Find bound frame for annotation timing
        bound_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'bound'), None)
        
        metrics['directional_efficiency_score'] = {
            'value': directional_efficiency_score,
            'hand_forward_start_score': hand_forward_start_score,
            'arm_early_elevation_score': arm_early_elevation_score,
            'trunk_lean_back_score': trunk_lean_back_score,
            'arm_torso_vector_alignment_score': arm_torso_vector_alignment_score,
            'frame': bound_idx  # Annotate at bound frame
        }

    # 19. Sequencing Lag Analysis (Kinetic Chain Timing)
    sequencing_lags = calculate_sequencing_lags(data)
    if sequencing_lags is not None:
        metrics['sequencing_lags'] = sequencing_lags

    return metrics

def calculate_arm_extension_at_ffc(data):
    """
    Calculate ARM extension at FFC - check whether wrist is below or above shoulder at the FFC frame.
    Returns: 'below' if wrist is below shoulder, 'above' if wrist is above shoulder, None if data missing
    """
    bfc_leg, ffc_leg, bowling_arm, lead_arm = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    
    if ffc_idx is None or bowling_arm is None:
        return None, None, None
    
    kps = data[ffc_idx]['keypoints']
    shoulder = kps.get(f'{bowling_arm}_shoulder')
    wrist = kps.get(f'{bowling_arm}_wrist')
    
    if not shoulder or not wrist:
        return None, None, None
    
    # Check if wrist is below or above shoulder (in image coordinates, y increases downward)
    if wrist['y'] > shoulder['y']:
        extension_status = 'below'
    else:
        extension_status = 'above'
    
    return extension_status, ffc_idx, (shoulder, wrist)

def calculate_front_leg_kinematics(data):
    """
    Calculate front leg kinematics classification from FFC to release.
    Returns one of: 'flexor', 'extender', 'flexor-extender', 'constant_brace'
    """
    bfc_leg, ffc_leg, _, _ = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    
    if ffc_idx is None or release_idx is None or ffc_leg is None:
        return None
    
    # Get knee angles throughout the FFC to release window
    knee_angles = []
    for i in range(ffc_idx, release_idx + 1):
        kps = data[i]['keypoints']
        hip = (kps[f'{ffc_leg}_hip']['x'], kps[f'{ffc_leg}_hip']['y'])
        knee = (kps[f'{ffc_leg}_knee']['x'], kps[f'{ffc_leg}_knee']['y'])
        ankle = (kps[f'{ffc_leg}_ankle']['x'], kps[f'{ffc_leg}_ankle']['y'])
        angle = calculate_angle(hip, knee, ankle)
        knee_angles.append(angle)
    
    if len(knee_angles) < 2:
        return None
    
    # Find maximum flexion and extension
    initial_angle = knee_angles[0]  # Angle at FFC
    max_flexion = min(knee_angles)  # Minimum angle = maximum flexion
    max_extension = max(knee_angles)  # Maximum angle = maximum extension
    
    # Calculate changes
    flexion_change = initial_angle - max_flexion  # How much it flexed
    extension_change = max_extension - initial_angle  # How much it extended
    
    # Classify based on the criteria
    if flexion_change >= 10 and extension_change < 10:
        return 'flexor'
    elif flexion_change < 10 and extension_change >= 10:
        return 'extender'
    elif flexion_change >= 10 and extension_change >= 10:
        return 'flexor-extender'
    else:  # flexion_change < 10 and extension_change < 10
        return 'constant_brace'

def calculate_back_knee_collapse_with_trail(data, win_ms=120):
    bfc_leg, _, _, _ = get_bfc_ffc_arms(data)
    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    if bfc_idx is None or bfc_leg is None:
        return None
    start_time = data[bfc_idx]['timestamp']
    end_time = start_time + win_ms / 1000.0
    min_angle = None
    min_idx = None
    trail = []
    for i in range(bfc_idx, len(data)):
        if data[i]['timestamp'] > end_time:
            break
        kps = data[i]['keypoints']
        hip = (kps[f'{bfc_leg}_hip']['x'], kps[f'{bfc_leg}_hip']['y'])
        knee = (kps[f'{bfc_leg}_knee']['x'], kps[f'{bfc_leg}_knee']['y'])
        ankle = (kps[f'{bfc_leg}_ankle']['x'], kps[f'{bfc_leg}_ankle']['y'])
        angle = calculate_angle(hip, knee, ankle)
        trail.append({'frame': i, 'hip': hip, 'knee': knee, 'ankle': ankle, 'angle': angle})
        if min_angle is None or angle < min_angle:
            min_angle = angle
            min_idx = i
    # Angle at BFC
    kps_bfc = data[bfc_idx]['keypoints']
    hip_bfc = (kps_bfc[f'{bfc_leg}_hip']['x'], kps_bfc[f'{bfc_leg}_hip']['y'])
    knee_bfc = (kps_bfc[f'{bfc_leg}_knee']['x'], kps_bfc[f'{bfc_leg}_knee']['y'])
    ankle_bfc = (kps_bfc[f'{bfc_leg}_ankle']['x'], kps_bfc[f'{bfc_leg}_ankle']['y'])
    angle_bfc = calculate_angle(hip_bfc, knee_bfc, ankle_bfc)
    collapse = angle_bfc - min_angle if min_angle is not None else None
    return {
        'value': collapse, 
        'bfc_idx': bfc_idx, 
        'min_idx': min_idx, 
        'trail': trail, 
        'leg': bfc_leg,
        'knee_angle_at_bfc': angle_bfc,
        'min_knee_angle_in_window': min_angle
    }

def calculate_front_knee_flexion_with_trail(data, win_ms=120):
    _, ffc_leg, _, _ = get_bfc_ffc_arms(data)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    if ffc_idx is None or ffc_leg is None:
        return None
    start_time = data[ffc_idx]['timestamp']
    end_time = start_time + win_ms / 1000.0
    min_angle = None
    min_idx = None
    trail = []
    for i in range(ffc_idx, len(data)):
        if data[i]['timestamp'] > end_time:
            break
        kps = data[i]['keypoints']
        hip = (kps[f'{ffc_leg}_hip']['x'], kps[f'{ffc_leg}_hip']['y'])
        knee = (kps[f'{ffc_leg}_knee']['x'], kps[f'{ffc_leg}_knee']['y'])
        ankle = (kps[f'{ffc_leg}_ankle']['x'], kps[f'{ffc_leg}_ankle']['y'])
        angle = calculate_angle(hip, knee, ankle)
        trail.append({'frame': i, 'hip': hip, 'knee': knee, 'ankle': ankle, 'angle': angle})
        if min_angle is None or angle < min_angle:
            min_angle = angle
            min_idx = i
    # Angle at FFC
    kps_ffc = data[ffc_idx]['keypoints']
    hip_ffc = (kps_ffc[f'{ffc_leg}_hip']['x'], kps_ffc[f'{ffc_leg}_hip']['y'])
    knee_ffc = (kps_ffc[f'{ffc_leg}_knee']['x'], kps_ffc[f'{ffc_leg}_knee']['y'])
    ankle_ffc = (kps_ffc[f'{ffc_leg}_ankle']['x'], kps_ffc[f'{ffc_leg}_ankle']['y'])
    angle_ffc = calculate_angle(hip_ffc, knee_ffc, ankle_ffc)
    flexion = angle_ffc - min_angle if min_angle is not None else None
    return {
        'value': flexion, 
        'ffc_idx': ffc_idx, 
        'min_idx': min_idx, 
        'trail': trail, 
        'leg': ffc_leg,
        'knee_angle_at_ffc': angle_ffc,
        'min_knee_angle_in_window': min_angle
    }

def calculate_directional_efficiency_score(data):
    """
    Composite metric measuring directional efficiency during pre-bound and post-bound frames.
    Evaluates arm-torso synchronization and body posture to preserve forward momentum.
    """
    # Find bound frame
    bound_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'bound'), None)
    if bound_idx is None or bound_idx < 5 or bound_idx + 5 >= len(data):
        return None, None, None, None, None
    
    t_bound = bound_idx
    t_minus_5 = bound_idx - 5
    t_plus_5 = bound_idx + 5
    
    # Determine bowling arm for hand/arm-dependent subscores
    bfc_leg, ffc_leg, bowling_arm, _ = get_bfc_ffc_arms(data)
    
    # 1. hand_forward_start_score (t-5 to t_bound, 6 frames)
    hand_forward_scores = []
    for frame_idx in range(t_minus_5, t_bound + 1):
        kps = data[frame_idx]['keypoints']
        wrist = kps.get(f'{bowling_arm}_wrist') if bowling_arm else kps.get('right_wrist')
        left_shoulder = kps.get('left_shoulder')
        right_shoulder = kps.get('right_shoulder')
        left_hip = kps.get('left_hip')
        right_hip = kps.get('right_hip')
        
        if not all([wrist, left_shoulder, right_shoulder, left_hip, right_hip]):
            continue
            
        # Compute trunk center
        trunk_center_x = (left_shoulder['x'] + right_shoulder['x'] + left_hip['x'] + right_hip['x']) / 4
        
        # Compute dx = wrist.x - trunk_center.x
        dx = wrist['x'] - trunk_center_x
        
        # Score logic
        if dx <= 25:
            score = 1.0
        elif dx >= 50:
            score = 0.0
        else:
            score = 1.0 - ((dx - 25) / 25)
        
        hand_forward_scores.append(score)
    
    hand_forward_start_score = sum(hand_forward_scores) / len(hand_forward_scores) if hand_forward_scores else 0.0
    
    # 2. arm_early_elevation_score (t-5 to t_bound, 6 frames)
    arm_elevation_scores = []
    for frame_idx in range(t_minus_5, t_bound + 1):
        kps = data[frame_idx]['keypoints']
        wrist = kps.get(f'{bowling_arm}_wrist') if bowling_arm else kps.get('right_wrist')
        right_shoulder = kps.get(f'{bowling_arm}_shoulder') if bowling_arm else kps.get('right_shoulder')
        
        if not all([wrist, right_shoulder]):
            continue
            
        # Compute dy = wrist.y - shoulder.y
        dy = wrist['y'] - right_shoulder['y']
        
        # Score logic
        if dy >= 0:
            score = 1.0
        elif dy <= -30:
            score = 0.0
        else:
            score = 1.0 - (abs(dy) / 30)
        
        arm_elevation_scores.append(score)
    
    arm_early_elevation_score = sum(arm_elevation_scores) / len(arm_elevation_scores) if arm_elevation_scores else 0.0
    
    # 3. trunk_lean_back_score (t-5 only, 1 frame)
    kps_t_minus_5 = data[t_minus_5]['keypoints']
    left_hip = kps_t_minus_5.get('left_hip')
    right_hip = kps_t_minus_5.get('right_hip')
    left_shoulder = kps_t_minus_5.get('left_shoulder')
    right_shoulder = kps_t_minus_5.get('right_shoulder')
    
    if all([left_hip, right_hip, left_shoulder, right_shoulder]):
        # Compute hip center and shoulder center
        hip_center = get_midpoint(left_hip, right_hip)
        shoulder_center = get_midpoint(left_shoulder, right_shoulder)
        
        # Vector: trunk_vec = shoulder_center - hip_center
        trunk_vec = (shoulder_center['x'] - hip_center['x'], shoulder_center['y'] - hip_center['y'])
        
        # Angle to vertical
        angle_to_vertical = math.degrees(math.atan2(trunk_vec[0], trunk_vec[1]))
        
        # Score logic
        if angle_to_vertical <= 15:
            score = 1.0
        elif angle_to_vertical >= 45:
            score = 0.0
        else:
            score = 1.0 - ((angle_to_vertical - 15) / 30)
        
        trunk_lean_back_score = score
    else:
        trunk_lean_back_score = 0.0
    
    # 4. arm_torso_vector_alignment_score (t_bound to t+5, 6 frames)
    alignment_scores = []
    for frame_idx in range(t_bound, t_plus_5 + 1):
        kps = data[frame_idx]['keypoints']
        wrist = kps.get(f'{bowling_arm}_wrist') if bowling_arm else kps.get('right_wrist')
        right_elbow = kps.get(f'{bowling_arm}_elbow') if bowling_arm else kps.get('right_elbow')
        left_hip = kps.get('left_hip')
        right_hip = kps.get('right_hip')
        
        if not all([wrist, right_elbow, left_hip, right_hip]):
            continue
            
        # Arm vector (from elbow to wrist)
        arm_vec = (wrist['x'] - right_elbow['x'], wrist['y'] - right_elbow['y'])
        
        # Hip movement direction (compare with previous frame)
        if frame_idx > 0:
            prev_kps = data[frame_idx - 1]['keypoints']
            prev_left_hip = prev_kps.get('left_hip')
            prev_right_hip = prev_kps.get('right_hip')
            
            if prev_left_hip and prev_right_hip:
                # Current hip center
                hip_center = get_midpoint(left_hip, right_hip)
                # Previous hip center
                prev_hip_center = get_midpoint(prev_left_hip, prev_right_hip)
                
                # Hip movement vector (direction of body movement)
                torso_movement_vec = (hip_center['x'] - prev_hip_center['x'], 
                                    hip_center['y'] - prev_hip_center['y'])
                
                # Cosine similarity between arm direction and torso movement direction
                dot_product = arm_vec[0] * torso_movement_vec[0] + arm_vec[1] * torso_movement_vec[1]
                arm_magnitude = math.sqrt(arm_vec[0]**2 + arm_vec[1]**2)
                torso_movement_magnitude = math.sqrt(torso_movement_vec[0]**2 + torso_movement_vec[1]**2)
                
                if arm_magnitude == 0 or torso_movement_magnitude == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = dot_product / (arm_magnitude * torso_movement_magnitude)
                    cos_sim = max(-1.0, min(1.0, cos_sim))  # Clamp to [-1, 1]
                
                score = (cos_sim + 1) / 2  # Normalize to [0, 1]
                alignment_scores.append(score)
            else:
                alignment_scores.append(0.0)  # Default score if missing data
        else:
            alignment_scores.append(0.0)  # Default score for first frame
    
    arm_torso_vector_alignment_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    
    # Final composite score
    directional_efficiency_score = (
        hand_forward_start_score +
        arm_early_elevation_score +
        trunk_lean_back_score +
        arm_torso_vector_alignment_score
    ) / 4.0
    
    return directional_efficiency_score, hand_forward_start_score, arm_early_elevation_score, trunk_lean_back_score, arm_torso_vector_alignment_score

# ===== SEQUENCING LAG ANALYSIS FUNCTIONS =====

def calculate_hip_rotation_distance(kps):
    """Calculate hip rotation using left_hip to right_hip X-distance from side view"""
    left_hip = kps.get('left_hip')
    right_hip = kps.get('right_hip')
    
    if not left_hip or not right_hip:
        return None
    
    # Line distance directly represents rotation from side view
    # Bigger distance = more rotation, smaller distance = less rotation
    line_distance = abs(right_hip['x'] - left_hip['x'])
    return line_distance

def calculate_trunk_rotation_inclination(kps):
    """Calculate trunk rotation using trunk inclination angle from vertical"""
    left_hip = kps.get('left_hip')
    right_hip = kps.get('right_hip')
    left_shoulder = kps.get('left_shoulder')
    right_shoulder = kps.get('right_shoulder')
    
    if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
        return None
    
    # Calculate trunk center line
    mid_hip = get_midpoint(left_hip, right_hip)
    mid_shoulder = get_midpoint(left_shoulder, right_shoulder)
    
    # Trunk inclination from vertical
    dx = mid_shoulder['x'] - mid_hip['x']
    dy = mid_shoulder['y'] - mid_hip['y']
    
    # Angle from vertical (0° = upright, positive = forward lean)
    inclination = math.degrees(math.atan2(dx, -dy))  # -dy because Y increases downward
    
    return inclination

def calculate_shoulder_rotation_distance(kps):
    """Calculate shoulder rotation using left_shoulder to right_shoulder X-distance"""
    left_shoulder = kps.get('left_shoulder')
    right_shoulder = kps.get('right_shoulder')
    
    if not left_shoulder or not right_shoulder:
        return None
    
    # Line distance represents shoulder girdle rotation
    # Similar principle to hip rotation
    line_distance = abs(right_shoulder['x'] - left_shoulder['x'])
    return line_distance

def calculate_bowling_arm_rotation(kps, bowling_arm):
    """Calculate bowling arm rotation using elbow position relative to shoulder (360° range)"""
    if not bowling_arm:
        return None
    
    shoulder = kps.get(f'{bowling_arm}_shoulder')
    elbow = kps.get(f'{bowling_arm}_elbow')
    
    if not shoulder or not elbow:
        return None
    
    # Calculate elbow position relative to shoulder
    dx = elbow['x'] - shoulder['x']  # Horizontal offset
    dy = elbow['y'] - shoulder['y']  # Vertical offset
    
    # Calculate angle - 90° = elbow above shoulder (release point for cricket bowling)
    angle = math.degrees(math.atan2(dy, dx))
    
    # Normalize to 0-360° range (atan2 gives -180° to +180°)
    angle = (angle + 360) % 360
    
    return angle

def calculate_angular_velocity(angles, timestamps, window_size=3):
    """Calculate angular velocity using central difference method with smoothing"""
    if len(angles) < window_size or len(timestamps) < window_size:
        return []
    
    velocities = []
    for i in range(window_size//2, len(angles) - window_size//2):
        # Use multiple frames for stable velocity calculation
        angle_diff = angles[i + window_size//2] - angles[i - window_size//2]
        time_diff = timestamps[i + window_size//2] - timestamps[i - window_size//2]
        velocity = angle_diff / time_diff if time_diff > 0 else 0
        velocities.append(velocity)
    
    return velocities

def find_peak_velocity(velocities, timestamps, start_idx, end_idx):
    """Find peak velocity within a specific window"""
    if not velocities or start_idx >= len(velocities) or end_idx >= len(velocities):
        return None, None
    
    # Find maximum absolute velocity in the window
    max_velocity = 0
    peak_idx = start_idx
    
    for i in range(start_idx, min(end_idx + 1, len(velocities))):
        abs_velocity = abs(velocities[i])
        if abs_velocity > max_velocity:
            max_velocity = abs_velocity
            peak_idx = i
    
    # Convert back to original frame index and timestamp
    if peak_idx < len(timestamps):
        peak_time = timestamps[peak_idx]
        return peak_time, max_velocity
    
    return None, None

def calculate_sequencing_lags(data):
    """Calculate sequencing lags from FFC to Release for kinetic chain analysis"""
    
    # 1. Get the delivery window
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    
    if ffc_idx is None or release_idx is None:
        return None
    
    # 2. Extract frames from FFC to Release
    delivery_frames = data[ffc_idx:release_idx + 1]
    
    if len(delivery_frames) < 3:  # Need at least 3 frames for velocity calculation
        return None
    
    # 3. Calculate rotation values for each frame in this window
    rotations = {
        'hip': [],
        'trunk': [],
        'shoulder': [],
        'arm': []
    }
    timestamps = []
    
    bfc_leg, _, bowling_arm, _ = get_bfc_ffc_arms(data)
    
    for frame in delivery_frames:
        kps = frame['keypoints']
        
        # Calculate all rotation metrics
        hip_rot = calculate_hip_rotation_distance(kps)
        trunk_rot = calculate_trunk_rotation_inclination(kps)
        shoulder_rot = calculate_shoulder_rotation_distance(kps)
        arm_rot = calculate_bowling_arm_rotation(kps, bowling_arm)
        
        # Store rotations (only if all values are available)
        if all(v is not None for v in [hip_rot, trunk_rot, shoulder_rot, arm_rot]):
            rotations['hip'].append(hip_rot)
            rotations['trunk'].append(trunk_rot)
            rotations['shoulder'].append(shoulder_rot)
            rotations['arm'].append(arm_rot)
            timestamps.append(frame['timestamp'])
    
    # Check if we have enough data
    if len(timestamps) < 3:
        return None
    
    # 4. Calculate velocities for each segment
    velocities = {}
    for segment in ['hip', 'trunk', 'shoulder', 'arm']:
        if len(rotations[segment]) >= 3:
            vel = calculate_angular_velocity(rotations[segment], timestamps)
            if vel:
                velocities[segment] = vel
    
    if len(velocities) < 4:  # Need all 4 segments
        return None
    
    # 5. Find peak velocities within the delivery window
    peak_times = {}
    peak_velocities = {}
    
    for segment in velocities:
        peak_time, peak_vel = find_peak_velocity(
            velocities[segment], 
            timestamps[1:],  # Skip first timestamp since velocity calculation reduces by 1
            0, 
            len(velocities[segment]) - 1
        )
        if peak_time is not None:
            peak_times[segment] = peak_time
            peak_velocities[segment] = peak_vel
    
    if len(peak_times) < 4:  # Need all 4 peaks
        return None
    
    # 6. Calculate sequencing lags
    ffc_time = data[ffc_idx]['timestamp']
    
    # Convert peak times to relative times from FFC
    relative_peak_times = {}
    for segment in peak_times:
        relative_peak_times[segment] = peak_times[segment] - ffc_time
    
    # Calculate individual lags
    lag_hip_to_trunk = relative_peak_times['trunk'] - relative_peak_times['hip']
    lag_trunk_to_shoulder = relative_peak_times['shoulder'] - relative_peak_times['trunk']
    lag_shoulder_to_arm = relative_peak_times['arm'] - relative_peak_times['shoulder']
    lag_hip_to_arm = relative_peak_times['arm'] - relative_peak_times['hip']
    
    # 7. Assess sequencing quality
    sequencing_quality = "good"
    flags = []
    
    # Check for negative lags (out of sequence)
    if lag_hip_to_trunk < 0:
        flags.append("Hip→Trunk out of sequence")
        sequencing_quality = "poor"
    if lag_trunk_to_shoulder < 0:
        flags.append("Trunk→Shoulder out of sequence")
        sequencing_quality = "poor"
    if lag_shoulder_to_arm < 0:
        flags.append("Shoulder→Arm out of sequence")
        sequencing_quality = "poor"
    
    # Check for very small lags (poor separation)
    if 0 <= lag_hip_to_trunk < 0.02:  # Less than 20ms
        flags.append("Hip→Trunk too close")
    if 0 <= lag_trunk_to_shoulder < 0.01:  # Less than 10ms
        flags.append("Trunk→Shoulder too close")
    if 0 <= lag_shoulder_to_arm < 0.005:  # Less than 5ms
        flags.append("Shoulder→Arm too close")
    
    # Check total sequence time
    if lag_hip_to_arm > 0.15:  # More than 150ms
        flags.append("Total sequence too slow")
    elif lag_hip_to_arm < 0.04:  # Less than 40ms
        flags.append("Total sequence too fast")
    
    return {
        'sequencing_quality': sequencing_quality,
        'flags': flags,
        'lags': {
            'hip_to_trunk_ms': lag_hip_to_trunk * 1000,
            'trunk_to_shoulder_ms': lag_trunk_to_shoulder * 1000,
            'shoulder_to_arm_ms': lag_shoulder_to_arm * 1000,
            'hip_to_arm_total_ms': lag_hip_to_arm * 1000
        },
        'peak_times': {
            'hip_ms': relative_peak_times['hip'] * 1000,
            'trunk_ms': relative_peak_times['trunk'] * 1000,
            'shoulder_ms': relative_peak_times['shoulder'] * 1000,
            'arm_ms': relative_peak_times['arm'] * 1000
        },
        'peak_velocities': peak_velocities,
        'delivery_window': {
            'ffc_frame': ffc_idx,
            'release_frame': release_idx,
            'ffc_time': ffc_time,
            'release_time': data[release_idx]['timestamp']
        }
    }