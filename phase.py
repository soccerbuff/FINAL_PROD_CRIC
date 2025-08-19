
"""
phase.py

This module detects phases from a master keypoints file (e.g., side-view)
and synchronizes these phases across multiple keypoint files from different views.
"""
import json
import sys
import math
import os

# --- Helper functions from the original file (unchanged) ---

def get_torso_mid_y(frame):
    # (Implementation from original file)
    kp = frame.get('keypoints', {})
    try:
        y = (
            kp['left_hip']['y'] + kp['right_hip']['y'] +
            kp['left_shoulder']['y'] + kp['right_shoulder']['y']
        ) / 4
        return y
    except KeyError:
        return None

def check_parabolic_path(mid_ys, min_idx, window=3):
    # (Implementation from original file)
    left = max(0, min_idx - window)
    right = min(len(mid_ys) - 1, min_idx + window)
    for i in range(left, min_idx):
        if mid_ys[i] is None or mid_ys[i] < mid_ys[min_idx]:
            return False
    for i in range(min_idx + 1, right + 1):
        if mid_ys[i] is None or mid_ys[i] < mid_ys[min_idx]:
            return False
    return True

def moving_average(values, window=3):
    # (Implementation from original file)
    smoothed = []
    for i in range(len(values)):
        window_vals = [v for v in values[max(0, i-window//2):min(len(values), i+window//2+1)] if v is not None]
        if window_vals:
            smoothed.append(sum(window_vals) / len(window_vals))
        else:
            smoothed.append(None)
    return smoothed

def detect_bound_phase(data):
    # (Implementation from original file)
    mid_ys = [get_torso_mid_y(frame) for frame in data]
    mid_ys_smoothed = moving_average(mid_ys, window=5)
    valid_ys = [(i, y) for i, y in enumerate(mid_ys_smoothed) if y is not None]
    if not valid_ys:
        return None, None
    min_idx, _ = min(valid_ys, key=lambda x: x[1])
    if not check_parabolic_path(mid_ys_smoothed, min_idx):
        return None, None
    start_idx = min_idx
    for i in range(min_idx - 1, -1, -1):
        y_curr = mid_ys_smoothed[i]
        y_next = mid_ys_smoothed[i + 1]
        if y_curr is None or y_next is None: break
        if y_curr < y_next: break
        start_idx = i
    end_idx = min_idx
    for i in range(min_idx + 1, len(mid_ys_smoothed)):
        y_curr = mid_ys_smoothed[i]
        y_prev = mid_ys_smoothed[i - 1]
        y_start = mid_ys_smoothed[start_idx]
        if y_curr is None or y_prev is None or y_start is None: continue
        if y_curr > y_prev: end_idx = i
        if y_curr >= y_start:
            end_idx = i
            break
    return start_idx, end_idx

# --- Main Refactored Function ---

def detect_and_synchronize_phases(master_keypoints_path, all_keypoint_paths):
    """
    Detects phases from the master file and applies them to all files.
    
    Args:
        master_keypoints_path (str): Path to the keypoints JSON for phase detection (e.g., side view).
        all_keypoint_paths (list): A list of paths to all keypoint JSON files to be updated.
        
    Returns:
        list: A list of paths to the newly created phased JSON files.
    """
    with open(master_keypoints_path, 'r') as f:
        data = json.load(f)

    # This list will store the phase information for each frame
    phase_map = [{'phase': 'unknown'} for _ in range(len(data))]

    # --- Phase Detection Logic (from original file) ---
    start_idx, end_idx = detect_bound_phase(data)
    back_foot, front_foot, bfc_idx, ffc_idx, release_idx = None, None, None, None, None

    if start_idx is not None and end_idx is not None:
        for i in range(start_idx, end_idx):
            phase_map[i]['phase'] = 'bound'
        bfc_idx = end_idx
        kp = data[bfc_idx].get('keypoints', {})
        if kp.get('left_ankle') and kp.get('right_ankle'):
            back_foot = 'left' if kp['left_ankle']['y'] > kp['right_ankle']['y'] else 'right'
            front_foot = 'right' if back_foot == 'left' else 'left'
            phase_map[bfc_idx]['phase'] = 'BFC'
            phase_map[bfc_idx]['back_foot'] = back_foot

    if bfc_idx is not None and front_foot is not None:
        bfc_ankle_y = data[bfc_idx]['keypoints'][f'{back_foot}_ankle']['y']

        # Search for FFC frame where front foot makes contact with ground
        # Contact is detected when front foot ankle y >= back foot ankle y at BFC
        # AND either heel OR toe is touching the ground (whichever touches first)
        min_frames_after_bfc = 2  # Start from the frame after BFC
        max_frames_after_bfc = int(0.4 * (1.0 / (data[1]['timestamp'] - data[0]['timestamp'])))
        search_start = bfc_idx + min_frames_after_bfc
        search_end = min(bfc_idx + max_frames_after_bfc, len(data))

        for i in range(search_start, search_end):
            frame_kp = data[i].get('keypoints', {})
            front_ankle_y = frame_kp.get(f'{front_foot}_ankle', {}).get('y')
            if front_ankle_y is None:
                continue
            # Check for ground contact (primary condition)
            # Ankle position is secondary and more flexible
            front_heel_y = frame_kp.get(f'{front_foot}_heel', {}).get('y')
            front_foot_index_y = frame_kp.get(f'{front_foot}_foot_index', {}).get('y')
            back_foot_index_y = frame_kp.get(f'{back_foot}_foot_index', {}).get('y')
                
            if front_heel_y is not None and front_foot_index_y is not None:
                    # Contact is detected when either heel OR toe touches the ground
                    # Ground level is anywhere between back foot's toe and ankle
                    back_ankle_y = frame_kp.get(f'{back_foot}_ankle', {}).get('y')
                    if back_foot_index_y is not None and back_ankle_y is not None:
                        # Ground level range: from back foot toe (lowest) to back foot ankle (highest)
                        ground_level_min = back_foot_index_y  # Toe position (lowest)
                        ground_level_max = back_ankle_y       # Ankle position (highest)
                    else:
                        # Fallback to BFC ankle position
                        ground_level_min = bfc_ankle_y
                        ground_level_max = bfc_ankle_y
                    
                    # Check if either heel or toe is touching/near the ground within the range
                    heel_touching = front_heel_y >= ground_level_min - 5  # 5px tolerance
                    toe_touching = front_foot_index_y >= ground_level_min - 5  # 5px tolerance
                    
                    # Additional check: ankle should be reasonably close to ground level
                    # This prevents detecting FFC when foot is still in the air
                    ankle_near_ground = front_ankle_y >= ground_level_min - 25  # 25px tolerance
                    
                    # FFC occurs when either heel OR toe makes contact AND ankle is near ground
                    if (heel_touching or toe_touching) and ankle_near_ground:
                        ffc_idx = i
                        phase_map[ffc_idx]['phase'] = 'FFC'
                        phase_map[ffc_idx]['front_foot'] = front_foot
                        # Store which part of foot touched first for analysis
                        if heel_touching and toe_touching:
                            phase_map[ffc_idx]['contact_type'] = 'flat'
                        elif heel_touching:
                            phase_map[ffc_idx]['contact_type'] = 'heel_first'
                        else:
                            phase_map[ffc_idx]['contact_type'] = 'toe_first'
                        break
    
    if ffc_idx is not None and back_foot is not None:
        # New logic: detect release when the wrist is inside a ±30° cone around
        # the vertical from the shoulder (overhead), combined with wrist velocity.
        # Fallback to the previous velocity-only approach if the cone condition
        # is never satisfied.

        # Helper to compute "inside cone" condition for a given frame index
        def inside_overhead_cone(frame_index: int) -> bool:
            frame_kp = data[frame_index].get('keypoints', {})
            shoulder = frame_kp.get(f'{back_foot}_shoulder')
            wrist = frame_kp.get(f'{back_foot}_wrist')
            if not shoulder or not wrist:
                return False
            dx = wrist['x'] - shoulder['x']
            dy = wrist['y'] - shoulder['y']
            # Overhead requires wrist to be above shoulder (smaller y in image coords)
            if wrist['y'] >= shoulder['y']:
                return False
            length = math.hypot(dx, dy)
            if length == 0:
                return False
            # Angle from vertical up (0° means perfectly vertical upward)
            # Vertical up vector is (0, -1) in image coordinates
            cos_theta = max(-1.0, min(1.0, (-dy) / length))
            angle_deg = math.degrees(math.acos(cos_theta))
            return angle_deg <= 30.0

        # Search window: up to 1.5 seconds after FFC
        cone_indices = []
        max_vel_in_cone = -1.0
        max_vel_in_cone_idx_prev = None  # we will mark the previous frame as release

        i = ffc_idx + 1
        while i < len(data) and (data[i]['timestamp'] - data[ffc_idx]['timestamp']) <= 1.5:
            if inside_overhead_cone(i):
                cone_indices.append(i)
                # If consecutive frames are inside the cone, evaluate velocity
                if i - 1 >= 0 and inside_overhead_cone(i - 1):
                    p1 = data[i-1]['keypoints'].get(f'{back_foot}_wrist')
                    p2 = data[i]['keypoints'].get(f'{back_foot}_wrist')
                    if p1 and p2:
                        vel = math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])
                        if vel > max_vel_in_cone:
                            max_vel_in_cone = vel
                            max_vel_in_cone_idx_prev = i - 1
            i += 1

        if max_vel_in_cone_idx_prev is not None:
            # Use the frame just before peak wrist velocity, while still inside the cone
            release_idx = max_vel_in_cone_idx_prev
            phase_map[release_idx]['phase'] = 'release'
        elif cone_indices:
            # Fallback: first entry into the cone
            release_idx = cone_indices[0]
            phase_map[release_idx]['phase'] = 'release'
        else:
            # Final fallback: original velocity-only approach within 1.5s window
            max_vel, max_vel_idx = -1.0, -1
            for j in range(ffc_idx + 1, len(data)):
                if data[j]['timestamp'] - data[ffc_idx]['timestamp'] > 1.5:
                    break
                p1 = data[j-1]['keypoints'].get(f'{back_foot}_wrist')
                p2 = data[j]['keypoints'].get(f'{back_foot}_wrist')
                if p1 and p2:
                    vel = math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])
                    if vel > max_vel:
                        max_vel, max_vel_idx = vel, j
            if max_vel_idx != -1:
                release_idx = max_vel_idx - 1
                phase_map[release_idx]['phase'] = 'release'
            else:
                release_idx = None  # No release phase assigned if not within 1.5 seconds

    if bfc_idx is not None and release_idx is not None:
        for i in range(bfc_idx + 1, release_idx):
            if phase_map[i]['phase'] == 'unknown':
                phase_map[i]['phase'] = 'delivery_stride'
    
    if release_idx is not None:
        for i in range(release_idx + 1, len(data)):
            if data[i].get('keypoints'):
                phase_map[i]['phase'] = 'follow_through'

    if start_idx is not None:
        # Label continuous frames with keypoints before start_idx as run_up
        i = start_idx - 1
        while i >= 0 and data[i].get('keypoints'):
            phase_map[i]['phase'] = 'run_up'
            i -= 1
        # Also label frames from 0 to i (inclusive) if they have keypoints
        for j in range(i + 1):
            if data[j].get('keypoints'):
                phase_map[j]['phase'] = 'run_up'
    
    # --- Fix runup phases for side view if needed ---
    # Check if frames before bound are labeled as unknown or if run_up already exists
    frames_before_bound_are_unknown = True
    has_runup_before_bound = False
    if start_idx is not None:
        for i in range(start_idx):
            if phase_map[i].get('phase') == 'run_up':
                has_runup_before_bound = True
                frames_before_bound_are_unknown = False
                break
            elif phase_map[i].get('phase') != 'unknown':
                frames_before_bound_are_unknown = False
                break
    
    # If frames before bound are unknown, label first 4 bound frames with keypoints as run_up
    if frames_before_bound_are_unknown and start_idx is not None:
        bound_count = 0
        for i in range(start_idx, len(phase_map)):
            if phase_map[i].get('phase') == 'bound' and bound_count < 4:
                # Check if keypoints are not empty
                if data[i].get('keypoints') and len(data[i].get('keypoints', {})) > 0:
                    phase_map[i]['phase'] = 'run_up'
                    bound_count += 1
                    print(f"Fixed: Labeled frame {i} as run_up")
            elif phase_map[i].get('phase') == 'bound':
                break
        
        # Label all frames before start_idx as unknown
        for i in range(start_idx):
            phase_map[i]['phase'] = 'unknown'
        
        print(f"Runup fixing completed: {bound_count} frames labeled as run_up")
    elif has_runup_before_bound:
        print("Run_up phases already exist before bound - no fixing needed")
    
    # --- Ensure all frames before first run_up are unknown ---
    first_runup_idx = None
    for i, frame in enumerate(phase_map):
        if frame.get('phase') == 'run_up':
            first_runup_idx = i
            break
    
    if first_runup_idx is not None:
        # Label all frames before the first run_up as unknown
        for i in range(first_runup_idx):
            phase_map[i]['phase'] = 'unknown'
        print(f"Labeled frames 0 to {first_runup_idx-1} as unknown")
    
    print(f"Phase detection complete. BFC: {bfc_idx}, FFC: {ffc_idx}, Release: {release_idx}")

    # --- Synchronization Step ---
    output_paths = []
    for file_path in all_keypoint_paths:
        with open(file_path, 'r') as f:
            view_data = json.load(f)
        
        # Add the synchronized phase information to each frame
        for i, frame in enumerate(view_data):
            if i < len(phase_map):
                frame.update(phase_map[i])

        base, ext = os.path.splitext(file_path)
        output_path = f"{base.replace('_keypoints', '')}_phased.json"
        
        with open(output_path, 'w') as f:
            json.dump(view_data, f, indent=2)
        output_paths.append(output_path)
        
    return output_paths

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python phase.py <master_keypoints_json> <other_keypoints_json_1> [other_keypoints_json_2]...")
        sys.exit(1)
    master_path = sys.argv[1]
    all_paths = sys.argv[1:]
    detect_and_synchronize_phases(master_path, all_paths)