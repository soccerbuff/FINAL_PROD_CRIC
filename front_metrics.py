"""
front_metrics.py

Refactored module to compute front-view biomechanical metrics as requested:
- Shoulder–hip separation @ FFC (deg, signed) and separation velocity (deg/s)
- Trunk lateral flexion @ BR (deg) and peak lateral-flexion rate (deg/s)
- Release point (x,y) of wrist-finger midpoint
- Foot alignment: front foot angle at release with vertical passing from the knee
- Step width @ FFC (ankle-ankle top-down)
- Front-foot lateral offset @ FFC
- Arm slot @ BR (deg) — shoulder joint plane (low/¾/high + degrees)
- Release lateral offset (x) @ BR (wrist relative to shoulder/pelvis center)
- Pelvic drop @ FFC (|L hip y − R hip y|)

Assumes input data is a list of frames with keypoints and phase labels.
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

def get_midpoint(p1, p2):
    return {'x': (p1['x'] + p2['x']) / 2, 'y': (p1['y'] + p2['y']) / 2}

def get_distance(p1, p2):
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def get_angle_with_vertical(p1, p2):
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    # atan2(dx, dy) gives angle relative to vertical axis
    return math.degrees(math.atan2(dx, dy))

def get_angle_with_horizontal(p1, p2):
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    return math.degrees(math.atan2(dy, dx))

def get_phase_indices(data):
    bfc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'BFC'), None)
    ffc_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'FFC'), None)
    release_idx = next((i for i, f in enumerate(data) if f.get('phase') == 'release'), None)
    return bfc_idx, ffc_idx, release_idx

def get_bfc_ffc_legs(data):
    bfc_idx, ffc_idx, _ = get_phase_indices(data)
    if bfc_idx is None or ffc_idx is None:
        return None, None
    kps_bfc = data[bfc_idx]['keypoints']
    
    # Check if ankle keypoints exist
    left_ankle = safe_get_keypoint(kps_bfc, 'left_ankle')
    right_ankle = safe_get_keypoint(kps_bfc, 'right_ankle')
    
    if left_ankle is None or right_ankle is None:
        return None, None
    
    left_ankle_y_bfc = left_ankle['y']
    right_ankle_y_bfc = right_ankle['y']
    bfc_leg = 'left' if left_ankle_y_bfc > right_ankle_y_bfc else 'right'
    ffc_leg = 'right' if bfc_leg == 'left' else 'left'
    return bfc_leg, ffc_leg

def calculate_shoulder_hip_separation(data, idx):
    kps = data[idx]['keypoints']
    left_shoulder = safe_get_keypoint(kps, 'left_shoulder')
    right_shoulder = safe_get_keypoint(kps, 'right_shoulder')
    left_hip = safe_get_keypoint(kps, 'left_hip')
    right_hip = safe_get_keypoint(kps, 'right_hip')
    if not left_shoulder or not right_shoulder or not left_hip or not right_hip:
        return None, None, None

    # Calculate shoulder line angle
    shoulder_angle = get_angle_with_horizontal(left_shoulder, right_shoulder)
    # Calculate hip line angle
    hip_angle = get_angle_with_horizontal(left_hip, right_hip)

    # Signed difference (shoulder - hip)
    separation = shoulder_angle - hip_angle
    # Normalize to [-180, 180]
    if separation > 180:
        separation -= 360
    elif separation < -180:
        separation += 360

    return separation, shoulder_angle, hip_angle

def calculate_separation_velocity(data, ffc_idx):
    # Calculate angular velocity of shoulder-hip separation around FFC frame
    # Use frames before and after FFC to estimate velocity (deg/s)
    if ffc_idx is None or ffc_idx == 0 or ffc_idx >= len(data) - 1:
        return None
    sep_prev, _, _ = calculate_shoulder_hip_separation(data, ffc_idx - 1)
    sep_next, _, _ = calculate_shoulder_hip_separation(data, ffc_idx + 1)
    # Check if both separation values are valid
    if sep_prev is None or sep_next is None:
        return None
    # Assuming frame rate is known or 1 frame = 1 unit time, velocity = (next - prev)/2
    velocity = (sep_next - sep_prev) / 2
    return velocity

def calculate_trunk_lateral_flexion(data, idx):
    kps = data[idx]['keypoints']
    left_shoulder = safe_get_keypoint(kps, 'left_shoulder')
    right_shoulder = safe_get_keypoint(kps, 'right_shoulder')
    left_hip = safe_get_keypoint(kps, 'left_hip')
    right_hip = safe_get_keypoint(kps, 'right_hip')
    if not left_shoulder or not right_shoulder or not left_hip or not right_hip:
        return None, None
    mid_shoulder = get_midpoint(left_shoulder, right_shoulder)
    mid_hip = get_midpoint(left_hip, right_hip)
    
    # Calculate trunk lateral flexion: angle between trunk midline and vertical
    # The trunk midline is the line from mid_hip to mid_shoulder
    # We need the angle between this line and a vertical reference
    
    # Vector from hip to shoulder (trunk line)
    dx = mid_shoulder['x'] - mid_hip['x']
    dy = mid_shoulder['y'] - mid_hip['y']
    
    # Vertical reference vector (0, -1) pointing upward
    vertical_vector = (0, -1)
    trunk_vector = (dx, dy)
    
    # Normalize trunk vector
    trunk_magnitude = (dx**2 + dy**2)**0.5
    if trunk_magnitude == 0:
        return None, None
    
    trunk_unit = (dx/trunk_magnitude, dy/trunk_magnitude)
    
    # Calculate dot product
    dot_product = trunk_unit[0]*vertical_vector[0] + trunk_unit[1]*vertical_vector[1]
    dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
    
    # Calculate angle in degrees
    angle = math.degrees(math.acos(dot_product))
    
    # Determine if it's left or right lateral flexion
    # If dx > 0, it's right lateral flexion (positive)
    # If dx < 0, it's left lateral flexion (negative)
    if dx < 0:
        angle = -angle
    
    return angle, (mid_shoulder, mid_hip)

def calculate_trunk_lateral_flexion_rate(data, release_idx):
    # Calculate peak lateral flexion rate around release phase
    if release_idx is None or release_idx < 1 or release_idx >= len(data) - 1:
        return None
    rates = []
    for i in range(max(0, release_idx - 5), min(len(data) - 1, release_idx + 5)):
        angle_prev_result = calculate_trunk_lateral_flexion(data, i - 1) if i - 1 >= 0 else (None, None)
        angle_next_result = calculate_trunk_lateral_flexion(data, i + 1) if i + 1 < len(data) else (None, None)
        angle_prev = angle_prev_result[0] if angle_prev_result[0] is not None else None
        angle_next = angle_next_result[0] if angle_next_result[0] is not None else None
        if angle_prev is not None and angle_next is not None:
            rate = (angle_next - angle_prev) / 2
            rates.append(abs(rate))
    if not rates:
        return None
    return max(rates)

def calculate_wrist_finger_midpoint(data, idx, arm):
    kps = data[idx]['keypoints']
    wrist = kps.get(f'{arm}_wrist')
    finger = kps.get(f'{arm}_index') or kps.get(f'{arm}_thumb') or kps.get(f'{arm}_pinky')
    if wrist is None or finger is None:
        return None
    midpoint = get_midpoint(wrist, finger)
    return midpoint

def calculate_foot_alignment_angle(data, idx, ffc_leg):
    kps = data[idx]['keypoints']
    ankle = kps.get(f'{ffc_leg}_ankle')
    knee = kps.get(f'{ffc_leg}_knee')
    toe = kps.get(f'{ffc_leg}_foot_index') or kps.get(f'{ffc_leg}_foot_middle') or kps.get(f'{ffc_leg}_foot_tip')
    if not ankle or not knee or not toe:
        return None
    
    # Calculate foot line vector (ankle to toe)
    foot_dx = toe['x'] - ankle['x']
    foot_dy = toe['y'] - ankle['y']
    
    # Vertical reference vector (0, -1) pointing upward
    vertical_vector = (0, -1)
    foot_vector = (foot_dx, foot_dy)
    
    # Normalize foot vector
    foot_magnitude = (foot_dx**2 + foot_dy**2)**0.5
    if foot_magnitude == 0:
        return None
    
    foot_unit = (foot_dx/foot_magnitude, foot_dy/foot_magnitude)
    
    # Calculate dot product
    dot_product = foot_unit[0]*vertical_vector[0] + foot_unit[1]*vertical_vector[1]
    dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
    
    # Calculate angle in degrees
    angle = math.degrees(math.acos(dot_product))
    
    # Determine if it's left or right foot alignment
    # If foot_dx > 0, it's right foot alignment (positive)
    # If foot_dx < 0, it's left foot alignment (negative)
    if foot_dx < 0:
        angle = -angle
    
    return angle

def calculate_step_width(data, ffc_idx):
    if ffc_idx is None:
        return None
    kps = data[ffc_idx]['keypoints']
    left_ankle = kps.get('left_ankle')
    right_ankle = kps.get('right_ankle')
    if not left_ankle or not right_ankle:
        return None
    # Step width is horizontal distance (x) between ankles (top-down view)
    width = abs(left_ankle['x'] - right_ankle['x'])
    return width

def calculate_front_foot_lateral_offset(data, ffc_idx, bfc_leg, ffc_leg):
    if ffc_idx is None or bfc_leg is None or ffc_leg is None:
        return None
    kps = data[ffc_idx]['keypoints']
    back_ankle = kps.get(f'{bfc_leg}_ankle')
    front_ankle = kps.get(f'{ffc_leg}_ankle')
    if not back_ankle or not front_ankle:
        return None
    offset = front_ankle['x'] - back_ankle['x']
    return offset

def calculate_arm_slot_angle(data, release_idx, bowling_arm):
    if release_idx is None or bowling_arm is None:
        return None
    kps = data[release_idx]['keypoints']
    shoulder = kps.get(f'{bowling_arm}_shoulder')
    elbow = kps.get(f'{bowling_arm}_elbow')
    wrist = kps.get(f'{bowling_arm}_wrist')
    if not shoulder or not elbow or not wrist:
        return None
    # Vector shoulder to elbow
    se = (elbow['x'] - shoulder['x'], elbow['y'] - shoulder['y'])
    # Vector elbow to wrist
    ew = (wrist['x'] - elbow['x'], wrist['y'] - elbow['y'])
    # Calculate angle between vectors
    dot = se[0]*ew[0] + se[1]*ew[1]
    norm_se = math.sqrt(se[0]**2 + se[1]**2)
    norm_ew = math.sqrt(ew[0]**2 + ew[1]**2)
    if norm_se == 0 or norm_ew == 0:
        return None
    cos_angle = dot / (norm_se * norm_ew)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def calculate_release_lateral_offset(data, release_idx, bowling_arm, pelvis_center):
    if release_idx is None or bowling_arm is None or pelvis_center is None:
        return None
    kps = data[release_idx]['keypoints']
    wrist = kps.get(f'{bowling_arm}_wrist')
    if not wrist:
        return None
    # Lateral offset is wrist x relative to pelvis center x
    offset = wrist['x'] - pelvis_center['x']
    return offset

def calculate_pelvic_drop(data, ffc_idx):
    if ffc_idx is None:
        return None
    kps = data[ffc_idx]['keypoints']
    left_hip = kps.get('left_hip')
    right_hip = kps.get('right_hip')
    if not left_hip or not right_hip:
        return None
    drop = abs(left_hip['y'] - right_hip['y'])
    return drop

def get_pelvis_center(kps):
    left_hip = kps.get('left_hip')
    right_hip = kps.get('right_hip')
    if not left_hip or not right_hip:
        return None
    return get_midpoint(left_hip, right_hip)

def calculate_all_front_metrics(phased_json_path):
    with open(phased_json_path) as f:
        data = json.load(f)

    metrics = {}

    bfc_idx, ffc_idx, release_idx = get_phase_indices(data)
    bfc_leg, ffc_leg = get_bfc_ffc_legs(data)
    bowling_arm = bfc_leg  # Bowling arm is the same side as back foot

    # 1. Shoulder–hip separation @ FFC (deg, signed)
    if ffc_idx is not None:
        sep, shoulder_angle, hip_angle = calculate_shoulder_hip_separation(data, ffc_idx)
        metrics['shoulder_hip_separation_ffc'] = {'value': sep, 'frame': ffc_idx, 'shoulder_angle': shoulder_angle, 'hip_angle': hip_angle}
        # Separation velocity (deg/s)
        sep_vel = calculate_separation_velocity(data, ffc_idx)
        metrics['shoulder_hip_separation_velocity_ffc'] = {'value': sep_vel, 'frame': ffc_idx}

    # 2. Trunk lateral flexion @ release (deg)
    if release_idx is not None:
        trunk_flexion_result = calculate_trunk_lateral_flexion(data, release_idx)
        if trunk_flexion_result[0] is not None:
            trunk_flexion, coordinates = trunk_flexion_result
            metrics['trunk_lateral_flexion_release'] = {'value': trunk_flexion, 'frame': release_idx, 'coordinates': coordinates}
        else:
            metrics['trunk_lateral_flexion_release'] = {'value': None, 'frame': release_idx, 'coordinates': None}
        # Peak lateral-flexion rate (deg/s)
        peak_rate = calculate_trunk_lateral_flexion_rate(data, release_idx)
        metrics['peak_lateral_flexion_rate_release'] = {'value': peak_rate, 'frame': release_idx}

    # 3. Release point (x,y) of wrist-finger midpoint
    if release_idx is not None:
        midpoint = calculate_wrist_finger_midpoint(data, release_idx, bowling_arm)
        metrics['release_wrist_finger_midpoint'] = {'value': midpoint, 'frame': release_idx}

    # 4. Foot alignment angle at release
    if release_idx is not None and ffc_leg is not None:
        foot_align_angle = calculate_foot_alignment_angle(data, release_idx, ffc_leg)
        if foot_align_angle is not None:
            # Get coordinates for visualization
            kps = data[release_idx]['keypoints']
            ankle = kps.get(f'{ffc_leg}_ankle')
            knee = kps.get(f'{ffc_leg}_knee')
            toe = kps.get(f'{ffc_leg}_foot_index') or kps.get(f'{ffc_leg}_foot_middle') or kps.get(f'{ffc_leg}_foot_tip')
            if ankle and knee and toe:
                coordinates = (ankle, toe, knee)
            else:
                coordinates = None
            metrics['foot_alignment_release'] = {'value': foot_align_angle, 'frame': release_idx, 'coordinates': coordinates}
        else:
            metrics['foot_alignment_release'] = {'value': None, 'frame': release_idx, 'coordinates': None}

    # 5. Step width @ FFC (ankle-ankle top-down)
    if ffc_idx is not None:
        step_width = calculate_step_width(data, ffc_idx)
        metrics['step_width_ffc'] = {'value': step_width, 'frame': ffc_idx}

    # 6. Front-foot lateral offset @ FFC
    if ffc_idx is not None and bfc_leg is not None and ffc_leg is not None:
        front_foot_offset = calculate_front_foot_lateral_offset(data, ffc_idx, bfc_leg, ffc_leg)
        metrics['front_foot_lateral_offset_ffc'] = {'value': front_foot_offset, 'frame': ffc_idx}

    # 7. Arm slot @ release (deg)
    if release_idx is not None:
        arm_slot_angle = calculate_arm_slot_angle(data, release_idx, bowling_arm)
        metrics['arm_slot_release'] = {'value': arm_slot_angle, 'frame': release_idx}

    # 8. Release lateral offset (x) @ release (wrist relative to pelvis center)
    if release_idx is not None:
        kps_release = data[release_idx]['keypoints']
        pelvis_center = get_pelvis_center(kps_release)
        release_lat_offset = calculate_release_lateral_offset(data, release_idx, bowling_arm, pelvis_center)
        metrics['release_lateral_offset_release'] = {'value': release_lat_offset, 'frame': release_idx}

    # 9. Pelvic drop @ FFC (|L hip y − R hip y|)
    if ffc_idx is not None:
        pelvic_drop = calculate_pelvic_drop(data, ffc_idx)
        metrics['pelvic_drop_ffc'] = {'value': pelvic_drop, 'frame': ffc_idx}

    return metrics
