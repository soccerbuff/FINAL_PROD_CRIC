import cv2
import json
import os
import math

def draw_knee_trail(frame, trail_data, current_frame_idx, color=(0, 255, 0)):
    """
    Draw fading trail of hip-knee-ankle lines for knee movement analysis.
    trail_data: list of dicts with 'frame', 'hip', 'knee', 'ankle' coordinates
    current_frame_idx: current frame being processed
    color: base color for the trail
    """
    if not trail_data:
        return
    
    # Find frames in trail that are <= current_frame_idx
    valid_trail = [t for t in trail_data if t['frame'] <= current_frame_idx]
    if not valid_trail:
        return
    
    # Draw trail with fading effect (oldest faint, newest solid)
    for i, trail_point in enumerate(valid_trail):
        # Calculate alpha based on position in trail (0 = oldest, 1 = newest)
        alpha = i / len(valid_trail) if len(valid_trail) > 1 else 1.0
        alpha = max(0.1, alpha)  # Minimum visibility
        
        # Calculate color with alpha
        r = int(color[0] * alpha)
        g = int(color[1] * alpha)
        b = int(color[2] * alpha)
        trail_color = (b, g, r)  # OpenCV uses BGR
        
        # Draw hip-knee-ankle line
        hip = (int(trail_point['hip'][0]), int(trail_point['hip'][1]))
        knee = (int(trail_point['knee'][0]), int(trail_point['knee'][1]))
        ankle = (int(trail_point['ankle'][0]), int(trail_point['ankle'][1]))
        
        # Draw lines with thickness based on alpha
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, hip, knee, trail_color, thickness)
        cv2.line(frame, knee, ankle, trail_color, thickness)
        
        # Draw keypoints with size based on alpha
        radius = max(2, int(4 * alpha))
        cv2.circle(frame, hip, radius, trail_color, -1)
        cv2.circle(frame, knee, radius, trail_color, -1)
        cv2.circle(frame, ankle, radius, trail_color, -1)

def annotate_video(video_path, keypoints_json_path, output_path=None, metrics_json_path=None, view='side'):
    # Define skeleton connections (common for all views)
    skeleton = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('right_shoulder', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('left_hip', 'right_hip')
    ]

    # Define face keypoints to exclude
    face_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']

    # Load keypoints
    with open(keypoints_json_path, 'r') as f:
        keypoints_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Read the first frame to get the correct dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        return None
    height, width, _ = frame.shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset frame position to the beginning

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base}_annotated_{view}.mp4"

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load metrics if provided
    metrics_data = None
    if metrics_json_path and os.path.exists(metrics_json_path):
        with open(metrics_json_path, 'r') as f:
            metrics_data = json.load(f)
        print(f"Loaded metrics from: {metrics_json_path}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints_data):
            break
        keypoints = keypoints_data[frame_idx]['keypoints']
        # Draw skeleton lines (blue with black borders)
        for pt1, pt2 in skeleton:
            if pt1 in keypoints and pt2 in keypoints:
                x1, y1 = keypoints[pt1]['x'], keypoints[pt1]['y']
                x2, y2 = keypoints[pt2]['x'], keypoints[pt2]['y']
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)  # Black border
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue line
        # Draw shoulder-hip midpoint connection (orange with black border)
        if all(k in keypoints for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            sx1, sy1 = keypoints['left_shoulder']['x'], keypoints['left_shoulder']['y']
            sx2, sy2 = keypoints['right_shoulder']['x'], keypoints['right_shoulder']['y']
            hx1, hy1 = keypoints['left_hip']['x'], keypoints['left_hip']['y']
            hx2, hy2 = keypoints['right_hip']['x'], keypoints['right_hip']['y']
            shoulder_mid = (int((sx1 + sx2) / 2), int((sy1 + sy2) / 2))
            hip_mid = (int((hx1 + hx2) / 2), int((hy1 + hy2) / 2))
            # Draw orange line with black border
            cv2.line(frame, shoulder_mid, hip_mid, (0, 0, 0), 5)  # Black border
            cv2.line(frame, shoulder_mid, hip_mid, (0, 165, 255), 3)  # Orange line
            # Draw midpoints (keep original colors)
            cv2.circle(frame, shoulder_mid, 6, (255, 0, 255),-1)  # Magenta dot
            cv2.circle(frame, hip_mid, 6, (255, 0, 255), -1)      # Magenta dot
        # Draw keypoints (grey with black borders) - exclude face keypoints
        for name, pt in keypoints.items():
            # Skip face keypoints
            if name in face_keypoints:
                continue
            x, y = pt['x'], pt['y']
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)  # Black border
            cv2.circle(frame, (x, y), 2, (128, 128, 128), -1)  # Grey fill
        # Overlay phase text at the top in a box
        phase = keypoints_data[frame_idx].get('phase', 'Unknown')
        text = f"Phase: {phase}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        box_x, box_y = 20, 10
        box_w, box_h = text_w + 20, text_h + 20
        # Draw filled rectangle (box) for text background
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
        # Draw the text over the box
        cv2.putText(
            frame,
            text,
            (box_x + 10, box_y + box_h - 10),
            font,
            font_scale,
            (0, 255, 255),  # Yellow
            thickness,
            cv2.LINE_AA
        )
        
        # Check for knee trail metrics and draw them
        trail_metrics = []
        if metrics_data:
            # Check for back-knee collapse trail
            if 'back_knee_collapse' in metrics_data:
                collapse_data = metrics_data['back_knee_collapse']
                if 'trail' in collapse_data:
                    trail_frames = [t['frame'] for t in collapse_data['trail']]
                    if frame_idx in trail_frames:
                        draw_knee_trail(frame, collapse_data['trail'], frame_idx, color=(255, 0, 0))  # Red for back knee
                        trail_metrics.append('back_knee_collapse')
            
            # Check for front-knee flexion trail
            if 'front_knee_flexion' in metrics_data:
                flexion_data = metrics_data['front_knee_flexion']
                if 'trail' in flexion_data:
                    trail_frames = [t['frame'] for t in flexion_data['trail']]
                    if frame_idx in trail_frames:
                        draw_knee_trail(frame, flexion_data['trail'], frame_idx, color=(0, 255, 0))  # Green for front knee
                        trail_metrics.append('front_knee_flexion')
        
        # Check if this frame has metrics to annotate
        metrics_to_show = []
        if metrics_data:
            for metric_name, metric_info in metrics_data.items():
                # Skip if metric_info is not a dictionary
                if not isinstance(metric_info, dict):
                    continue
                # For head sway (front), allow per-frame
                if view == 'front' and metric_name.startswith('head_lateral_sway_') and metric_info.get('frame') == frame_idx:
                    metrics_to_show.append((metric_name, metric_info))
                elif metric_info.get('frame') == frame_idx:
                    metrics_to_show.append((metric_name, metric_info))
                # Special handling for bound flight time - show on all bound frames
                elif metric_name == 'bound_flight_time' and keypoints_data[frame_idx].get('phase') == 'bound':
                    metrics_to_show.append((metric_name, metric_info))
                # Special handling for FFC to release time - show on all frames between FFC and release
                elif metric_name == 'ffc_to_release_time' and any(t['frame'] == frame_idx for t in metric_info.get('frame_timings', [])):
                    metrics_to_show.append((metric_name, metric_info))
        # Apply slow motion for metrics annotation
        if metrics_to_show or trail_metrics:
            for i, (metric_name, metric_info) in enumerate(metrics_to_show):
                metric_frame = frame.copy()
                annotate_metric_on_frame(metric_frame, metric_name, metric_info, i, frame_idx, keypoints_data, view)
                # Slomo logic per metric
                if metric_name == 'bound_flight_time':
                    slomo_factor = 15
                elif metric_name == 'ffc_to_release_time':
                    slomo_factor = 20
                elif metric_name in ['trunk_lateral_flexion_ffc', 'trunk_lateral_flexion_release', 'hip_shoulder_separation_release', 'pelvic_drop_ffc', 'bowling_arm_abduction_release', 'bowling_arm_slot_angle_release']:
                    slomo_factor = 25
                else:
                    slomo_factor = 25
                for _ in range(slomo_factor):
                    out.write(metric_frame)
        elif trail_metrics:
            # Extra slow motion for trail frames
            slomo_factor = 40  # More slow motion for trail visualization
            for _ in range(slomo_factor):
                out.write(frame)
        else:
            slomo_factor = 30 if phase in ["BFC", "FFC", "release"] else 1
            for _ in range(slomo_factor):
                out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")
    return output_path

def annotate_metric_on_frame(frame, metric_name, metric_info, metric_index, frame_idx, keypoints_data, view):
    value = metric_info.get('value')
    coordinates = metric_info.get('coordinates')
    # Define colors for different metrics (extend for front/back)
    colors = {
        # Side
        'stride_length_at_ffc': (255, 255, 0),
        'back_knee_angle_at_bfc': (0, 255, 255),
        'back_leg_collapse_angle': (255, 0, 255),
        'max_front_knee_extension_angle': (0, 255, 0),
        'front_knee_angle_at_ffc': (255, 165, 0),
        'front_knee_flex_angle': (128, 0, 128),
        'front_knee_angle_at_release': (0, 128, 255),
        'back_alignment_at_release': (255, 0, 0),
        'hip_shoulder_separation_at_ffc': (0, 128, 0),
        'lead_arm_drop_speed': (255, 192, 203),
        'bowling_arm_hyperextension_angle': (255, 215, 0),
        'bound_flight_time': (0, 255, 255),
        'back_knee_collapse': (255, 0, 0),  # Red for back knee collapse
        'front_knee_flexion': (0, 255, 0),  # Green for front knee flexion
        'front_leg_kinematics': (255, 165, 0),  # Orange for front leg kinematics
        'directional_efficiency_score': (0, 255, 255),  # Cyan for directional efficiency
        'arm_extension_at_ffc': (255, 20, 147),  # Deep pink for arm extension at FFC
        # Front
        'trunk_lateral_flexion_ffc': (0, 255, 0),  # Changed to green/neon
        'trunk_lateral_flexion_release': (0, 255, 0),  # Changed to green/neon
        'shoulder_alignment_bfc': (255, 0, 255),
        'shoulder_alignment_ffc': (255, 0, 128),
        'hip_alignment_bfc': (0, 255, 128),
        'hip_alignment_ffc': (0, 128, 255),
        'shoulder_hip_separation_ffc': (255, 255, 0),
        'front_foot_lateral_offset': (255, 128, 0),
        'front_foot_landing_angle': (128, 0, 255),
        'bowling_arm_slot_angle_release': (0, 255, 0),
        'head_lateral_sway_release': (255, 0, 0),
        # Additional metrics
        'release_height': (255, 255, 0),  # Yellow for release height
        'bound_height': (0, 255, 255),  # Cyan for bound height
        'front_knee_extension_velocity': (255, 0, 255),  # Magenta for knee extension velocity
        'peak_forward_flexion_velocity': (255, 165, 0),  # Orange for peak flexion velocity
        # Back
        'hip_shoulder_separation_release': (0, 255, 255),
        'pelvic_drop_ffc': (255, 0, 0),
        'bowling_arm_abduction_release': (0, 255, 0),
        'ffc_to_release_time': (255, 128, 0),  # Orange color for FFC to release time
    }
    color = colors.get(metric_name, (255, 255, 255))
    display_name = metric_name.replace('_', ' ').title()
    
    # Special handling for knee collapse and flexion metrics
    if metric_name == 'back_knee_collapse' and isinstance(metric_info, dict):
        min_angle_frame_idx = metric_info.get('min_angle_frame_idx')
        if frame_idx == min_angle_frame_idx:
            # Show all three values at the minimum angle frame
            display_name = "Back Knee Collapse"
            display_value = f"{value:.1f} degrees"
            # Add additional info
            knee_angle_at_bfc = metric_info.get('knee_angle_at_bfc', 0)
            min_knee_angle = metric_info.get('min_knee_angle_in_window', 0)
            additional_info = f"@BFC: {knee_angle_at_bfc:.1f} degrees | Min: {min_knee_angle:.1f} degrees"
        else:
            display_value = f"{value:.1f} degrees"
            additional_info = None
    elif metric_name == 'front_knee_flexion' and isinstance(metric_info, dict):
        # Check if we should show front knee flexion based on kinematics type
        # Note: In extra file, we don't have access to metrics_data, so we'll show it by default
        # The main annotate_video.py file handles the conditional logic
        min_angle_frame_idx = metric_info.get('min_angle_frame_idx')
        if frame_idx == min_angle_frame_idx:
            # Show all three values at the minimum angle frame
            display_name = "Front Knee Flexion"
            display_value = f"{value:.1f} degrees"
            # Add additional info
            knee_angle_at_ffc = metric_info.get('knee_angle_at_ffc', 0)
            min_knee_angle = metric_info.get('min_knee_angle_in_window', 0)
            additional_info = f"@FFC: {knee_angle_at_ffc:.1f} degrees | Min: {min_knee_angle:.1f} degrees"
        else:
            display_value = f"{value:.1f} degrees"
            additional_info = None
    else:
        # Special handling for bound flight time
        if metric_name == 'bound_flight_time':
            frame_timings = metric_info.get('frame_timings', [])
            first_frame = metric_info.get('first_frame', 0)
            last_frame = metric_info.get('last_frame', 0)
            
            # Check if frame_timings is a list of dictionaries (back view format) or list of floats (side view format)
            if frame_timings and isinstance(frame_timings[0], dict):
                # Back view format: list of dictionaries with 'frame', 'elapsed_time', 'total_time'
                current_timing = next((t for t in frame_timings if t['frame'] == frame_idx), None)
                if current_timing:
                    elapsed_time = current_timing['elapsed_time']
                    total_time = current_timing['total_time']
                    display_value = f"{elapsed_time:.3f}s / {total_time:.3f}s"
                    display_name = "Bound Flight Time"
                else:
                    display_value = f"{value:.3f} s"
            else:
                # Side view format: list of floats (timestamps), show on bound frames
                if first_frame <= frame_idx <= last_frame:
                    # Calculate elapsed time based on frame index (start from 0)
                    frame_index_in_bound = frame_idx - first_frame
                    if frame_index_in_bound < len(frame_timings):
                        elapsed_time = frame_timings[frame_index_in_bound] - frame_timings[0]  # Start from 0
                        display_value = f"{elapsed_time:.3f}s / {value:.3f}s"
                        display_name = "Bound Flight Time"
                    else:
                        display_value = f"{value:.3f} s"
                else:
                    display_value = f"{value:.3f} s"
        # Special handling for FFC to release time
        elif metric_name == 'ffc_to_release_time':
            frame_timings = metric_info.get('frame_timings', [])
            ffc_frame = metric_info.get('ffc_frame', 0)
            release_frame = metric_info.get('release_frame', 0)
            
            # Check if frame_timings is a list of dictionaries (back view format) or list of floats (side view format)
            if frame_timings and isinstance(frame_timings[0], dict):
                # Back view format: list of dictionaries with 'frame', 'elapsed_time', 'total_time'
                current_timing = next((t for t in frame_timings if t['frame'] == frame_idx), None)
                if current_timing:
                    elapsed_time = current_timing['elapsed_time']
                    total_time = current_timing['total_time']
                    display_value = f"{elapsed_time:.3f}s / {total_time:.3f}s"
                    display_name = "FFC to Release Time"
                else:
                    display_value = f"{value:.3f} s"
            else:
                # Side view format: list of floats (timestamps), show on frames between FFC and release
                if ffc_frame <= frame_idx <= release_frame:
                    # Calculate elapsed time based on frame index (start from 0)
                    frame_index_in_sequence = frame_idx - ffc_frame
                    if frame_index_in_sequence < len(frame_timings):
                        elapsed_time = frame_timings[frame_index_in_sequence]
                        display_value = f"{elapsed_time:.3f}s / {value:.3f}s"
                        display_name = "FFC to Release Time"
                    else:
                        display_value = f"{value:.3f} s"
                else:
                    display_value = f"{value:.3f} s"
        else:
            if value == 'data missing':
                display_value = 'data missing'
            elif value is None:
                display_value = "N/A"
            elif isinstance(value, dict):
                display_value = str(value)
            elif isinstance(value, (list, tuple)):
                display_value = f"({int(value[0])}, {int(value[1])})"
            elif isinstance(value, str):
                display_value = value  # Handle string values
            elif 'angle' in metric_name or 'tilt' in metric_name or 'slot' in metric_name or 'abduction' in metric_name:
                if value is not None:
                    display_value = f"{value:.1f} degrees"
                else:
                    display_value = "N/A"
            elif 'speed' in metric_name:
                if value is not None:
                    # Linear velocities should be in km/hr (converted by unit_conversion.py)
                    display_value = f"{value:.1f} km/hr"
                else:
                    display_value = "N/A"
            elif 'length' in metric_name or 'offset' in metric_name or 'drop' in metric_name:
                if value is not None:
                    display_value = f"{value:.1f} px"
                else:
                    display_value = "N/A"
            elif 'time' in metric_name:
                if value is not None:
                    display_value = f"{value:.3f} s"
                else:
                    display_value = "N/A"
            else:
                if value is not None:
                    display_value = f"{value:.1f}"
                else:
                    display_value = "N/A"
        additional_info = None
    
    text_y_offset = 50 + (metric_index * 40)
    text_x = 20
    text_y = 100 + text_y_offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text = f"{display_name}: {display_value}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    rect_x1 = text_x - 5
    rect_y1 = text_y - text_size[1] - 5
    rect_x2 = text_x + text_size[0] + 5
    rect_y2 = text_y + 5
    
    # Skip text display for release_wrist_finger_midpoint
    if metric_name != 'release_wrist_finger_midpoint':
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, 2)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Add additional info if available
    if additional_info:
        additional_y = text_y + 25
        additional_text_size, _ = cv2.getTextSize(additional_info, font, font_scale, thickness)
        additional_rect_x1 = text_x - 5
        additional_rect_y1 = additional_y - additional_text_size[1] - 5
        additional_rect_x2 = text_x + additional_text_size[0] + 5
        additional_rect_y2 = additional_y + 5
        cv2.rectangle(frame, (additional_rect_x1, additional_rect_y1), (additional_rect_x2, additional_rect_y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (additional_rect_x1, additional_rect_y1), (additional_rect_x2, additional_rect_y2), color, 2)
        cv2.putText(frame, additional_info, (text_x, additional_y), font, font_scale, color, thickness, cv2.LINE_AA)
    # Per-metric drawing logic for each view
    if coordinates == 'data missing':
        print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        return # Skip drawing if coordinates are missing

    def safe_point(pt):
        return pt is not None and None not in pt
    # Side view (reuse existing logic)
    if view == 'side':
        if metric_name == 'stride_length_at_ffc':
            if coordinates is not None:
                try:
                    left_ankle, right_ankle = coordinates
                    if safe_point((left_ankle['x'], left_ankle['y'])) and safe_point((right_ankle['x'], right_ankle['y'])):
                        # Draw line between ankles
                        cv2.line(frame, (int(left_ankle['x']), int(left_ankle['y'])), 
                                (int(right_ankle['x']), int(right_ankle['y'])), color, 3)
                        # Draw circles at ankles
                        cv2.circle(frame, (int(left_ankle['x']), int(left_ankle['y'])), 8, color, -1)
                        cv2.circle(frame, (int(right_ankle['x']), int(right_ankle['y'])), 8, color, -1)
                        # Add text showing stride length
                        cv2.putText(frame, f"Stride: {value:.1f}px", 
                                  (int((left_ankle['x'] + right_ankle['x'])/2) - 30, 
                                   int((left_ankle['y'] + right_ankle['y'])/2) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'release_height':
            # Show only the release height value (no lines)
            if coordinates is not None:
                try:
                    wrist_x = keypoints_data[frame_idx]['keypoints'].get('right_wrist', {}).get('x') or \
                             keypoints_data[frame_idx]['keypoints'].get('left_wrist', {}).get('x')
                    wrist_y = keypoints_data[frame_idx]['keypoints'].get('right_wrist', {}).get('y') or \
                             keypoints_data[frame_idx]['keypoints'].get('left_wrist', {}).get('y')
                    if wrist_x is not None and wrist_y is not None:
                        # Draw circle at wrist
                        cv2.circle(frame, (int(wrist_x), int(wrist_y)), 8, color, -1)
                        # Add text showing height value only
                        cv2.putText(frame, f"Release Height: {value:.1f}px", (int(wrist_x) + 10, int(wrist_y) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        print(f"[WARN] Missing wrist keypoint for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'bound_height':
            # Show only the bound height value (no lines)
            if coordinates is not None:
                try:
                    # Use hip center for text positioning
                    left_hip = keypoints_data[frame_idx]['keypoints'].get('left_hip', {})
                    right_hip = keypoints_data[frame_idx]['keypoints'].get('right_hip', {})
                    if left_hip and right_hip:
                        hip_x = (left_hip['x'] + right_hip['x']) / 2
                        hip_y = (left_hip['y'] + right_hip['y']) / 2
                        # Add text showing bound height value only
                        cv2.putText(frame, f"Bound Height: {value:.1f}px", (int(hip_x) + 10, int(hip_y) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        print(f"[WARN] Missing hip keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'front_knee_extension_velocity':
            # Highlight the front knee at release with velocity info
            if coordinates is not None:
                try:
                    knee = coordinates  # coordinates should be knee position
                    if safe_point((knee[0], knee[1])):
                        # Draw circle at knee
                        cv2.circle(frame, (int(knee[0]), int(knee[1])), 10, color, -1)
                        cv2.circle(frame, (int(knee[0]), int(knee[1])), 12, color, 2)
                        # Add text showing velocity
                        cv2.putText(frame, f"Ext Vel: {value:.1f} deg/s", (int(knee[0]) + 15, int(knee[1])), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        print(f"[WARN] Missing knee coordinates for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'peak_forward_flexion_velocity':
            # Highlight trunk at peak flexion velocity frame
            if coordinates is not None:
                try:
                    mid_shoulder, mid_hip = coordinates
                    if safe_point(mid_shoulder) and safe_point(mid_hip):
                        # Draw trunk line
                        cv2.line(frame, (int(mid_shoulder[0]), int(mid_shoulder[1])), 
                                (int(mid_hip[0]), int(mid_hip[1])), color, 3)
                        # Draw circles at shoulder and hip midpoints
                        cv2.circle(frame, (int(mid_shoulder[0]), int(mid_shoulder[1])), 8, color, -1)
                        cv2.circle(frame, (int(mid_hip[0]), int(mid_hip[1])), 8, color, -1)
                        # Add text showing peak velocity
                        cv2.putText(frame, f"Peak Flex Vel: {value:.1f} deg/s", 
                                  (int((mid_shoulder[0] + mid_hip[0])/2) - 50, int((mid_shoulder[1] + mid_hip[1])/2) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        print(f"[WARN] Missing trunk coordinates for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'arm_extension_at_ffc':
            if coordinates is not None:
                try:
                    shoulder, wrist = coordinates
                    if safe_point((shoulder['x'], shoulder['y'])) and safe_point((wrist['x'], wrist['y'])):
                        # Draw a line from shoulder to wrist
                        cv2.line(frame, (int(shoulder['x']), int(shoulder['y'])), (int(wrist['x']), int(wrist['y'])), color, 3)
                        # Draw circles at shoulder and wrist
                        cv2.circle(frame, (int(shoulder['x']), int(shoulder['y'])), 8, color, -1)
                        cv2.circle(frame, (int(wrist['x']), int(wrist['y'])), 8, color, -1)
                        
                        # Draw a horizontal line at shoulder level for reference
                        shoulder_y = int(shoulder['y'])
                        cv2.line(frame, (int(shoulder['x']) - 50, shoulder_y), (int(shoulder['x']) + 50, shoulder_y), (0, 255, 0), 2)
                        
                        # Add text indicating the position
                        if value == 'below':
                            cv2.putText(frame, "BELOW", (int(wrist['x']) + 10, int(wrist['y']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        elif value == 'above':
                            cv2.putText(frame, "ABOVE", (int(wrist['x']) + 10, int(wrist['y']) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
    # Front view
    elif view == 'front':
        if metric_name in ['trunk_lateral_flexion_ffc', 'trunk_lateral_flexion_release']:
            # Before unpacking coordinates, check for 'data missing' or None
            if coordinates == 'data missing' or coordinates is None:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
            else:
                mid_shoulder, mid_hip = coordinates
                if mid_shoulder and mid_hip:
                    # Draw the trunk midline (shoulder to hip line)
                    cv2.line(frame, (int(mid_shoulder['x']), int(mid_shoulder['y'])), (int(mid_hip['x']), int(mid_hip['y'])), color, 3)
                    cv2.circle(frame, (int(mid_shoulder['x']), int(mid_shoulder['y'])), 7, color, -1)
                    cv2.circle(frame, (int(mid_hip['x']), int(mid_hip['y'])), 7, color, -1)
                    
                    # Draw a vertical reference line (dashed) from the mid_hip point in green/neon color
                    vertical_length = 100  # Length of vertical reference line
                    vertical_x = int(mid_hip['x'])
                    vertical_y1 = int(mid_hip['y'])
                    vertical_y2 = vertical_y1 - vertical_length
                    
                    # Draw dashed vertical line in green/neon color
                    dash_length = 10
                    for i in range(0, vertical_length, dash_length * 2):
                        y1 = vertical_y1 - i
                        y2 = max(vertical_y1 - i - dash_length, vertical_y2)
                        if y1 >= vertical_y2:
                            cv2.line(frame, (vertical_x, y1), (vertical_x, y2), (0, 255, 0), 2)  # Green/neon color
                    
                    # Draw angle arc
                    center = (int(mid_hip['x']), int(mid_hip['y']))
                    radius = 30
                    # Start angle is 270 degrees (vertical upward)
                    start_angle = 270
                    # End angle depends on trunk orientation
                    trunk_dx = mid_shoulder['x'] - mid_hip['x']
                    trunk_dy = mid_shoulder['y'] - mid_hip['y']
                    end_angle = math.degrees(math.atan2(trunk_dx, trunk_dy))
                    if end_angle < 0:
                        end_angle += 360
                    
                    # Draw angle arc
                    # cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 2)
                else:
                    print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
        elif metric_name in ['shoulder_alignment_bfc', 'shoulder_alignment_ffc']:
            if coordinates is not None:
                p1, p2 = coordinates
                if safe_point(p1) and safe_point(p2):
                    cv2.line(frame, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), color, 3)
                else:
                    print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name in ['hip_alignment_bfc', 'hip_alignment_ffc']:
            if coordinates is not None:
                p1, p2 = coordinates
                if safe_point(p1) and safe_point(p2):
                    cv2.line(frame, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), color, 3)
                else:
                    print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'shoulder_hip_separation_ffc':
            # Handle None value case first
            if value is None:
                display_value = "N/A"
            else:
                display_value = f"{value:.1f} px"
            
            shoulder_angle = metric_info.get('shoulder_angle')
            hip_angle = metric_info.get('hip_angle')
            p1, p2 = keypoints_data[frame_idx]['keypoints'].get('left_shoulder'), keypoints_data[frame_idx]['keypoints'].get('right_shoulder')
            h1, h2 = keypoints_data[frame_idx]['keypoints'].get('left_hip'), keypoints_data[frame_idx]['keypoints'].get('right_hip')
            
            # Check if all keypoints exist before accessing their properties
            if p1 is not None and p2 is not None and h1 is not None and h2 is not None:
                if safe_point((p1['x'], p1['y'])) and safe_point((p2['x'], p2['y'])) and safe_point((h1['x'], h1['y'])) and safe_point((h2['x'], h2['y'])):
                    cv2.line(frame, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), (0, 255, 0), 4)
                    cv2.line(frame, (int(h1['x']), int(h1['y'])), (int(h2['x']), int(h2['y'])), (255, 0, 0), 4)
                else:
                    print(f"[WARN] Invalid keypoint coordinates for {metric_name} at frame {frame_idx}")
            else:
                print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
        elif metric_name == 'front_foot_lateral_offset':
            if coordinates is not None:
                try:
                    back_ankle, front_ankle = coordinates
                    if safe_point((back_ankle['x'], back_ankle['y'])) and safe_point((front_ankle['x'], front_ankle['y'])):
                        cv2.line(frame, (int(back_ankle['x']), int(back_ankle['y'])), (int(front_ankle['x']), int(front_ankle['y'])), color, 3)
                        cv2.circle(frame, (int(back_ankle['x']), int(back_ankle['y'])), 8, color, -1)
                        cv2.circle(frame, (int(front_ankle['x']), int(front_ankle['y'])), 8, color, -1)
                    else:
                        print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'front_foot_landing_angle':
            if coordinates is not None:
                try:
                    ankle, toe = coordinates
                    if safe_point((ankle['x'], ankle['y'])) and safe_point((toe['x'], toe['y'])):
                        cv2.line(frame, (int(ankle['x']), int(ankle['y'])), (int(toe['x']), int(toe['y'])), color, 3)
                        cv2.circle(frame, (int(ankle['x']), int(ankle['y'])), 8, color, -1)
                        cv2.circle(frame, (int(toe['x']), int(toe['y'])), 8, color, -1)
                    else:
                        print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'bowling_arm_slot_angle_release':
            if coordinates is not None:
                try:
                    shoulder, elbow, wrist = coordinates
                    if safe_point((shoulder['x'], shoulder['y'])) and safe_point((elbow['x'], elbow['y'])) and safe_point((wrist['x'], wrist['y'])):
                        cv2.line(frame, (int(shoulder['x']), int(shoulder['y'])), (int(elbow['x']), int(elbow['y'])), color, 3)
                        cv2.line(frame, (int(elbow['x']), int(elbow['y'])), (int(wrist['x']), int(wrist['y'])), color, 3)
                        cv2.circle(frame, (int(elbow['x']), int(elbow['y'])), 8, color, -1)
                    else:
                        print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'ball_release_point':
            if value is not None:
                x, y = value
                cv2.circle(frame, (int(x), int(y)), 12, color, -1)
                cv2.putText(frame, "Release", (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif metric_name.startswith('head_lateral_sway'):
            if coordinates is not None:
                try:
                    nose, mid_hip = coordinates
                    if safe_point((nose['x'], nose['y'])) and safe_point((mid_hip['x'], mid_hip['y'])):
                        cv2.circle(frame, (int(nose['x']), int(nose['y'])), 8, color, -1)
                        cv2.circle(frame, (int(mid_hip['x']), int(mid_hip['y'])), 8, color, -1)
                        cv2.line(frame, (int(nose['x']), int(nose['y'])), (int(mid_hip['x']), int(mid_hip['y'])), color, 2)
                    else:
                        print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[WARN] Error processing coordinates for {metric_name} at frame {frame_idx}: {e}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        # Lead Arm Drop Speed
        elif metric_name == 'lead_arm_drop_speed' and arm:
            wrist = kps.get(f'{arm}_wrist')
            if safe_point(wrist):
                cv2.circle(frame, (wrist['x'], wrist['y']), 8, color, -1)
                cv2.circle(frame, (wrist['x'], wrist['y']), 10, color, 2)
                # Add text showing drop speed value in km/hr
                cv2.putText(frame, f"Drop Speed: {value:.1f} km/hr", (int(wrist['x']) + 15, int(wrist['y'])), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Front Leg Kinematics
        elif metric_name == 'front_leg_kinematics':
            # Highlight the front leg (FFC leg) with the kinematics type
            leg = metric_info.get('leg', 'left')  # Default to left if not specified
            hip, knee, ankle = kps.get(f'{leg}_hip'), kps.get(f'{leg}_knee'), kps.get(f'{leg}_ankle')
            if all(safe_point(p) for p in [hip, knee, ankle]):
                # Draw the leg lines
                cv2.line(frame, (hip['x'], hip['y']), (knee['x'], knee['y']), color, 3)
                cv2.line(frame, (knee['x'], knee['y']), (ankle['x'], ankle['y']), color, 3)
                # Draw keypoints
                cv2.circle(frame, (hip['x'], hip['y']), 8, color, -1)
                cv2.circle(frame, (knee['x'], knee['y']), 8, color, -1)
                cv2.circle(frame, (ankle['x'], ankle['y']), 8, color, -1)

        # Directional Efficiency Score
        elif metric_name == 'directional_efficiency_score':
            # Visualize the composite metric by highlighting key components
            # Draw trunk midline and arm position indicators
            left_shoulder = kps.get('left_shoulder')
            right_shoulder = kps.get('right_shoulder')
            left_hip = kps.get('left_hip')
            right_hip = kps.get('right_hip')
            wrist = kps.get('right_wrist')
            
            if all(safe_point(p) for p in [left_shoulder, right_shoulder, left_hip, right_hip, wrist]):
                # Draw trunk midline
                shoulder_center = ((left_shoulder['x'] + right_shoulder['x']) / 2, 
                                 (left_shoulder['y'] + right_shoulder['y']) / 2)
                hip_center = ((left_hip['x'] + right_hip['x']) / 2, 
                             (left_hip['y'] + right_hip['y']) / 2)
                
                # Draw trunk line
                cv2.line(frame, (int(shoulder_center[0]), int(shoulder_center[1])), 
                        (int(hip_center[0]), int(hip_center[1])), color, 3)
                
                # Draw wrist position indicator
                cv2.circle(frame, (int(wrist['x']), int(wrist['y'])), 10, color, -1)
                cv2.circle(frame, (int(wrist['x']), int(wrist['y'])), 12, color, 2)
                
                # Draw trunk center reference
                trunk_center_x = (left_shoulder['x'] + right_shoulder['x'] + left_hip['x'] + right_hip['x']) / 4
                cv2.circle(frame, (int(trunk_center_x), int(hip_center[1])), 6, (255, 255, 255), -1)

        # Trunk Forward Flexion
        elif metric_name == 'trunk_forward_flexion_at_release':
            mid_shoulder, mid_hip = coordinates
            if safe_point(mid_shoulder) and safe_point(mid_hip):
                # Draw trunk line (shoulder to hip)
                cv2.line(frame, (int(mid_shoulder[0]), int(mid_shoulder[1])), 
                        (int(mid_hip[0]), int(mid_hip[1])), color, 3)
                
                # Draw vertical line through hip midpoint
                vertical_length = 100
                vertical_x = int(mid_hip[0])
                vertical_y1 = int(mid_hip[1]) - vertical_length
                vertical_y2 = int(mid_hip[1]) + vertical_length
                cv2.line(frame, (vertical_x, vertical_y1), (vertical_x, vertical_y2), (255, 255, 255), 2)
                
                # Draw angle arc
                center = (int(mid_hip[0]), int(mid_hip[1]))
                radius = 30
                # Start angle is 270 degrees (vertical upward)
                start_angle = 270
                # End angle depends on trunk orientation
                trunk_dx = mid_hip[0] - mid_shoulder[0]
                trunk_dy = mid_hip[1] - mid_shoulder[1]
                end_angle = math.degrees(math.atan2(trunk_dy, trunk_dx))
                if end_angle < 0:
                    end_angle += 360
                
                # Draw angle arc
                cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 2)
                
                # Draw keypoints
                cv2.circle(frame, (int(mid_shoulder[0]), int(mid_shoulder[1])), 6, color, -1)
                cv2.circle(frame, (int(mid_hip[0]), int(mid_hip[1])), 6, color, -1)
        elif metric_name == 'foot_alignment_release':
            if coordinates is not None:
                ankle, toe, knee = coordinates
                if safe_point((ankle['x'], ankle['y'])) and safe_point((toe['x'], toe['y'])) and safe_point((knee['x'], knee['y'])):
                    # Draw foot line (ankle to toe)
                    cv2.line(frame, (int(ankle['x']), int(ankle['y'])), (int(toe['x']), int(toe['y'])), color, 3)
                    cv2.circle(frame, (int(ankle['x']), int(ankle['y'])), 8, color, -1)
                    cv2.circle(frame, (int(toe['x']), int(toe['y'])), 8, color, -1)
                    
                    # Draw vertical reference line through knee
                    vertical_length = 80
                    vertical_x = int(knee['x'])
                    vertical_y1 = int(knee['y']) - vertical_length
                    vertical_y2 = int(knee['y']) + vertical_length
                    cv2.line(frame, (vertical_x, vertical_y1), (vertical_x, vertical_y2), (255, 255, 255), 2)
                    
                    # Draw angle arc
                    center = (int(knee['x']), int(knee['y']))
                    radius = 25
                    # Start angle is 270 degrees (vertical upward)
                    start_angle = 270
                    # End angle depends on foot orientation
                    foot_dx = toe['x'] - ankle['x']
                    foot_dy = toe['y'] - ankle['y']
                    end_angle = math.degrees(math.atan2(foot_dx, foot_dy))
                    if end_angle < 0:
                        end_angle += 360
                    
                    # Draw angle arc
                    # cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, color, 2)
                    
                    # Draw knee point
                    cv2.circle(frame, (int(knee['x']), int(knee['y'])), 6, color, -1)
                else:
                    print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'release_wrist_finger_midpoint':
            if value is not None:
                if isinstance(value, dict) and 'x' in value and 'y' in value:
                    x, y = value['x'], value['y']
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    x, y = value
                else:
                    print(f"[WARN] Invalid value format for {metric_name} at frame {frame_idx}")
                    return
                cv2.circle(frame, (int(x), int(y)), 12, color, -1)
                cv2.circle(frame, (int(x), int(y)), 15, color, 2)
    # Back view
    elif view == 'back':
        if metric_name == 'hip_shoulder_separation_release':
            if coordinates is not None:
                left_shoulder, right_shoulder, left_hip, right_hip = coordinates
                if (safe_point(left_shoulder) and safe_point(right_shoulder) and 
                    safe_point(left_hip) and safe_point(right_hip)):
                    # Draw shoulder line (blue)
                    cv2.line(frame, (int(left_shoulder['x']), int(left_shoulder['y'])), 
                            (int(right_shoulder['x']), int(right_shoulder['y'])), (255, 0, 0), 3)
                    cv2.circle(frame, (int(left_shoulder['x']), int(left_shoulder['y'])), 7, (255, 0, 0), -1)
                    cv2.circle(frame, (int(right_shoulder['x']), int(right_shoulder['y'])), 7, (255, 0, 0), -1)
                    
                    # Draw hip line (orange)
                    cv2.line(frame, (int(left_hip['x']), int(left_hip['y'])), 
                            (int(right_hip['x']), int(right_hip['y'])), (0, 165, 255), 3)
                    cv2.circle(frame, (int(left_hip['x']), int(left_hip['y'])), 7, (0, 165, 255), -1)
                    cv2.circle(frame, (int(right_hip['x']), int(right_hip['y'])), 7, (0, 165, 255), -1)
                else:
                    print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")
        elif metric_name == 'pelvic_drop_ffc':
            y_lhip, y_rhip = metric_info.get('y_coords', (None, None))
            lx = keypoints_data[frame_idx]['keypoints'].get('left_hip', {}).get('x')
            rx = keypoints_data[frame_idx]['keypoints'].get('right_hip', {}).get('x')
            if y_lhip is not None and y_rhip is not None and lx is not None and rx is not None:
                cv2.line(frame, (lx, int(y_lhip)), (rx, int(y_rhip)), color, 3)
                cv2.circle(frame, (lx, int(y_lhip)), 8, color, -1)
                cv2.circle(frame, (rx, int(y_rhip)), 8, color, -1)
            else:
                print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
        elif metric_name == 'bowling_arm_abduction_release':
            if coordinates is not None:
                S, E = coordinates
                if safe_point((S['x'], S['y'])) and safe_point((E['x'], E['y'])):
                    # Draw arm line (shoulder to elbow)
                    cv2.line(frame, (int(S['x']), int(S['y'])), (int(E['x']), int(E['y'])), color, 3)
                    # Draw vertical reference line
                    vertical_length = 80
                    vertical_x = int(S['x'])
                    vertical_y1 = int(S['y']) - vertical_length
                    vertical_y2 = int(S['y']) + vertical_length
                    cv2.line(frame, (vertical_x, vertical_y1), (vertical_x, vertical_y2), (255, 255, 255), 2)
                    # Draw keypoints
                    cv2.circle(frame, (int(S['x']), int(S['y'])), 8, color, -1)
                    cv2.circle(frame, (int(E['x']), int(E['y'])), 8, color, -1)
                else:
                    print(f"[WARN] Missing keypoints for {metric_name} at frame {frame_idx}")
            else:
                print(f"[WARN] Missing coordinates for {metric_name} at frame {frame_idx}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python annotate_video.py <video_path> <keypoints_json_path> <view> [metrics_json_path]")
        sys.exit(1)
    video_path = sys.argv[1]
    keypoints_json_path = sys.argv[2]
    view = sys.argv[3]
    metrics_json_path = sys.argv[4] if len(sys.argv) > 4 else None
    annotate_video(video_path, keypoints_json_path, metrics_json_path=metrics_json_path, view=view)