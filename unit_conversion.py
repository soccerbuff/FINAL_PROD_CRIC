"""
unit_conversion.py

Utility module for converting pixel measurements to real-world units.
Uses reference measurements for dynamic scaling based on camera distance.
"""

import math
import numpy as np

# Reference measurements in cm
SIDE_VIEW_REFERENCE_LENGTH = 30.0  # cm (right elbow to wrist)
FRONT_BACK_REFERENCE_LENGTH = 35.0  # cm (hip to hip distance)

def calculate_pixel_distance(p1, p2):
    """Calculate pixel distance between two points."""
    if p1 is None or p2 is None:
        return None
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def get_side_view_scale_factor(data, frame_idx):
    """
    Calculate scale factor for side view using right elbow to wrist distance.
    Returns pixels per cm.
    """
    if frame_idx is None or frame_idx >= len(data):
        return None
    
    kps = data[frame_idx]['keypoints']
    right_elbow = kps.get('right_elbow')
    right_wrist = kps.get('right_wrist')
    
    if right_elbow is None or right_wrist is None:
        return None
    
    pixel_distance = calculate_pixel_distance(right_elbow, right_wrist)
    if pixel_distance is None or pixel_distance == 0:
        return None
    
    # Scale factor: pixels per cm
    scale_factor = pixel_distance / SIDE_VIEW_REFERENCE_LENGTH
    return scale_factor

def get_front_back_scale_factor(data, frame_idx):
    """
    Calculate scale factor for front/back view using hip-to-hip distance.
    Returns pixels per cm.
    """
    if frame_idx is None or frame_idx >= len(data):
        return None
    
    kps = data[frame_idx]['keypoints']
    left_hip = kps.get('left_hip')
    right_hip = kps.get('right_hip')
    
    if left_hip is None or right_hip is None:
        return None
    
    pixel_distance = calculate_pixel_distance(left_hip, right_hip)
    if pixel_distance is None or pixel_distance == 0:
        return None
    
    # Scale factor: pixels per cm
    scale_factor = pixel_distance / FRONT_BACK_REFERENCE_LENGTH
    return scale_factor

def pixels_to_cm(pixel_value, scale_factor):
    """Convert pixel measurement to centimeters."""
    if pixel_value is None or scale_factor is None or scale_factor == 0:
        return None
    return pixel_value / scale_factor

def px_per_sec_to_km_per_hr(px_per_sec, scale_factor):
    """Convert pixels per second to kilometers per hour."""
    if px_per_sec is None or scale_factor is None or scale_factor == 0:
        return None
    
    # Convert px/s to cm/s
    cm_per_sec = px_per_sec / scale_factor
    
    # Convert cm/s to km/hr
    km_per_hr = (cm_per_sec * 3600) / 100000  # (cm/s * 3600s/hr) / (100cm/m * 1000m/km)
    
    return km_per_hr

def convert_side_view_metrics(metrics, data):
    """
    Convert side view metrics using FRAME-SPECIFIC scaling.
    Uses right elbow to wrist distance as reference for each frame.
    """
    converted_metrics = {}
    
    # Calculate scale factors for all frames to show dynamic scaling
    scale_factors = []
    for i, frame in enumerate(data):
        scale_factor = get_side_view_scale_factor(data, i)
        if scale_factor is not None:
            scale_factors.append(scale_factor)
    
    if not scale_factors:
        print("Warning: Could not calculate scale factors for side view. Using original pixel values.")
        return metrics
    
    # Calculate statistics for reporting
    median_scale_factor = np.median(scale_factors)
    scale_variation = np.std(scale_factors)
    
    print(f"Side view (FRAME-SPECIFIC) scale factors:")
    print(f"  Median: {median_scale_factor:.3f} pixels/cm")
    print(f"  Range: {min(scale_factors):.3f} - {max(scale_factors):.3f} pixels/cm")
    print(f"  Std Dev: {scale_variation:.3f} pixels/cm")
    print(f"  Variation: {(scale_variation/median_scale_factor)*100:.1f}%")
    
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict) and 'value' in metric_data:
            value = metric_data['value']
            frame_idx = metric_data.get('frame', 0)
            
            # Get scale factor for this specific frame
            frame_scale_factor = get_side_view_scale_factor(data, frame_idx)
            if frame_scale_factor is None:
                frame_scale_factor = median_scale_factor  # Fallback to median
            
            # Convert pixel-based measurements to cm
            if metric_name in ['release_height', 'bound_height', 'stride_length_at_ffc']:
                if value is not None and isinstance(value, (int, float)):
                    converted_value = pixels_to_cm(value, frame_scale_factor)
                    converted_metrics[metric_name] = {
                        **metric_data,
                        'value': converted_value,
                        'unit': 'cm',
                        'original_pixels': value,
                        'frame_scale_factor': frame_scale_factor,
                        'median_scale_factor': median_scale_factor
                    }
                else:
                    converted_metrics[metric_name] = metric_data
            
            # Convert speed measurements to km/hr
            elif metric_name in ['approach_speed']:
                if value is not None and isinstance(value, (int, float)):
                    converted_value = px_per_sec_to_km_per_hr(value, frame_scale_factor)
                    converted_metrics[metric_name] = {
                        **metric_data,
                        'value': converted_value,
                        'unit': 'km/hr',
                        'original_px_per_sec': value,
                        'frame_scale_factor': frame_scale_factor,
                        'median_scale_factor': median_scale_factor
                    }
                else:
                    converted_metrics[metric_name] = metric_data
            
            # Keep angular measurements as is (degrees)
            elif metric_name in ['front_knee_angle_at_ffc', 'min_front_knee_angle_post_ffc', 
                               'front_knee_angle_at_release', 'trunk_forward_flexion_at_release',
                               'bowling_arm_hyperextension_angle', 'back_knee_angle_at_bfc',
                               'back_knee_collapse', 'front_knee_flexion', 'min_back_knee_angle_post_bfc']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'degrees'
                }
            
            # Keep angular velocity measurements as is (deg/s)
            elif metric_name in ['front_knee_extension_velocity', 'peak_forward_flexion_velocity']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'deg/s'
                }
            
            # Convert linear velocity measurements to km/hr
            elif metric_name in ['lead_arm_drop_speed']:
                if value is not None and isinstance(value, (int, float)):
                    converted_value = px_per_sec_to_km_per_hr(value, frame_scale_factor)
                    converted_metrics[metric_name] = {
                        **metric_data,
                        'value': converted_value,
                        'unit': 'km/hr',
                        'original_px_per_sec': value,
                        'frame_scale_factor': frame_scale_factor,
                        'median_scale_factor': median_scale_factor
                    }
                else:
                    converted_metrics[metric_name] = metric_data
            
            # Keep time measurements as is (seconds)
            elif metric_name in ['bound_flight_time']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'seconds'
                }
            
            # Keep categorical and composite scores as is
            elif metric_name in ['front_leg_kinematics', 'directional_efficiency_score']:
                converted_metrics[metric_name] = metric_data
            
            # Default: keep as is
            else:
                converted_metrics[metric_name] = metric_data
        else:
            converted_metrics[metric_name] = metric_data
    
    return converted_metrics

def convert_front_view_metrics(metrics, data):
    """
    Convert front view metrics using FRAME-SPECIFIC scaling.
    Uses hip-to-hip distance as reference for each frame.
    """
    converted_metrics = {}
    
    # Calculate scale factors for all frames to show dynamic scaling
    scale_factors = []
    for i, frame in enumerate(data):
        scale_factor = get_front_back_scale_factor(data, i)
        if scale_factor is not None:
            scale_factors.append(scale_factor)
    
    if not scale_factors:
        print("Warning: Could not calculate scale factors for front view. Using original pixel values.")
        return metrics
    
    # Calculate statistics for reporting
    median_scale_factor = np.median(scale_factors)
    scale_variation = np.std(scale_factors)
    
    print(f"Front view (FRAME-SPECIFIC) scale factors:")
    print(f"  Median: {median_scale_factor:.3f} pixels/cm")
    print(f"  Range: {min(scale_factors):.3f} - {max(scale_factors):.3f} pixels/cm")
    print(f"  Std Dev: {scale_variation:.3f} pixels/cm")
    print(f"  Variation: {(scale_variation/median_scale_factor)*100:.1f}%")
    
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict) and 'value' in metric_data:
            value = metric_data['value']
            frame_idx = metric_data.get('frame', 0)
            
            # Get scale factor for this specific frame
            frame_scale_factor = get_front_back_scale_factor(data, frame_idx)
            if frame_scale_factor is None:
                frame_scale_factor = median_scale_factor  # Fallback to median
            
            # Convert pixel-based measurements to cm
            if metric_name in ['step_width_ffc', 'front_foot_lateral_offset_ffc', 
                             'release_lateral_offset_release', 'pelvic_drop_ffc']:
                if value is not None and isinstance(value, (int, float)):
                    converted_value = pixels_to_cm(value, frame_scale_factor)
                    converted_metrics[metric_name] = {
                        **metric_data,
                        'value': converted_value,
                        'unit': 'cm',
                        'original_pixels': value,
                        'frame_scale_factor': frame_scale_factor,
                        'median_scale_factor': median_scale_factor
                    }
                else:
                    converted_metrics[metric_name] = metric_data
            
            # Convert coordinate measurements to cm
            elif metric_name in ['release_wrist_finger_midpoint']:
                if value is not None and isinstance(value, dict) and 'x' in value and 'y' in value:
                    converted_x = pixels_to_cm(value['x'], frame_scale_factor)
                    converted_y = pixels_to_cm(value['y'], frame_scale_factor)
                    converted_metrics[metric_name] = {
                        **metric_data,
                        'value': {'x': converted_x, 'y': converted_y},
                        'unit': 'cm',
                        'original_pixels': value,
                        'frame_scale_factor': frame_scale_factor,
                        'median_scale_factor': median_scale_factor
                    }
                else:
                    converted_metrics[metric_name] = metric_data
            
            # Keep angular measurements as is (degrees)
            elif metric_name in ['shoulder_hip_separation_ffc', 'trunk_lateral_flexion_release',
                               'foot_alignment_release', 'arm_slot_release']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'degrees'
                }
            
            # Keep angular velocity measurements as is (deg/s)
            elif metric_name in ['shoulder_hip_separation_velocity_ffc', 'peak_lateral_flexion_rate_release']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'deg/s'
                }
            
            # Default: keep as is
            else:
                converted_metrics[metric_name] = metric_data
        else:
            converted_metrics[metric_name] = metric_data
    
    return converted_metrics

def convert_back_view_metrics(metrics, data):
    """
    Convert back view metrics using FRAME-SPECIFIC scaling.
    Uses hip-to-hip distance as reference for each frame.
    """
    converted_metrics = {}
    
    # Calculate scale factors for all frames to show dynamic scaling
    scale_factors = []
    for i, frame in enumerate(data):
        scale_factor = get_front_back_scale_factor(data, i)
        if scale_factor is not None:
            scale_factors.append(scale_factor)
    
    if not scale_factors:
        print("Warning: Could not calculate scale factors for back view. Using original pixel values.")
        return metrics
    
    # Calculate statistics for reporting
    median_scale_factor = np.median(scale_factors)
    scale_variation = np.std(scale_factors)
    
    print(f"Back view (FRAME-SPECIFIC) scale factors:")
    print(f"  Median: {median_scale_factor:.3f} pixels/cm")
    print(f"  Range: {min(scale_factors):.3f} - {max(scale_factors):.3f} pixels/cm")
    print(f"  Std Dev: {scale_variation:.3f} pixels/cm")
    print(f"  Variation: {(scale_variation/median_scale_factor)*100:.1f}%")
    
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict) and 'value' in metric_data:
            value = metric_data['value']
            frame_idx = metric_data.get('frame', 0)
            
            # Get scale factor for this specific frame
            frame_scale_factor = get_front_back_scale_factor(data, frame_idx)
            if frame_scale_factor is None:
                frame_scale_factor = median_scale_factor  # Fallback to median
            
            # Convert pixel-based measurements to cm
            if metric_name in ['pelvic_drop_ffc']:
                if value is not None and isinstance(value, (int, float)):
                    converted_value = pixels_to_cm(value, frame_scale_factor)
                    converted_metrics[metric_name] = {
                        **metric_data,
                        'value': converted_value,
                        'unit': 'cm',
                        'original_pixels': value,
                        'frame_scale_factor': frame_scale_factor,
                        'median_scale_factor': median_scale_factor
                    }
                else:
                    converted_metrics[metric_name] = metric_data
            
            # Keep angular measurements as is (degrees)
            elif metric_name in ['hip_shoulder_separation_release', 'bowling_arm_abduction_release']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'degrees'
                }
            
            # Keep time measurements as is (seconds)
            elif metric_name in ['ffc_to_release_time']:
                converted_metrics[metric_name] = {
                    **metric_data,
                    'unit': 'seconds'
                }
            
            # Keep path coordinates as is (but could be converted if needed)
            elif metric_name in ['runup_path']:
                converted_metrics[metric_name] = metric_data
            
            # Default: keep as is
            else:
                converted_metrics[metric_name] = metric_data
        else:
            converted_metrics[metric_name] = metric_data
    
    return converted_metrics
