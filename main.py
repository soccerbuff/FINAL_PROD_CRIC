import sys
import os
import json
from pose_est import extract_pose_keypoints
from phase import detect_and_synchronize_phases
from side_metrics import calculate_all_side_metrics
from front_metrics import calculate_all_front_metrics
from back_metrics import calculate_all_back_metrics
from annotate_video import annotate_video
from unit_conversion import convert_side_view_metrics, convert_front_view_metrics, convert_back_view_metrics

def save_metrics_to_file(metrics, output_path):
    """Saves a metrics dictionary to a JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")

def main():
    """
    Main function to run the multi-view analysis pipeline.
    Expects three video paths as command-line arguments: side, front, and back.
    """
    # Accept 2 or 3 arguments: side, back, [front]
    if len(sys.argv) < 3:
        print("Usage: python main.py <side_video_path> <back_video_path> [front_video_path]")
        sys.exit(1)

    # Map arguments to views in order: side, back, front
    arg_views = ['side', 'back', 'front']
    video_paths = {}
    for i, view in enumerate(arg_views):
        if len(sys.argv) > i + 1:
            video_paths[view] = sys.argv[i + 1]

    # Only process views for which arguments are provided
    ordered_views = [v for v in ['side', 'back', 'front'] if v in video_paths]
    available_views = [view for view in ordered_views if os.path.exists(video_paths[view])]
    if len(available_views) < 2:
        print("Error: At least two video files must be present.")
        sys.exit(1)

    keypoints_json_paths = {}
    phased_json_paths = {}
    metrics_paths = {}
    annotated_video_paths = {}

    # --- Step 1: Pose Estimation for all views ---
    print("### Step 1: Performing Pose Estimation ###")
    # Process in the order: side, back, front, but only if the file exists
    for view in available_views:
        path = video_paths[view]
        print(f"Processing {view} view...")
        keypoints_json_paths[view] = extract_pose_keypoints(path, view)
    print("-" * 50)

    # --- Step 2: Phase Detection (from Side view if available, else first available view) and Synchronization ---
    print("### Step 2: Detecting Phases and Synchronizing ###")
    master_view = 'side' if 'side' in available_views else available_views[0]
    master_keypoints_path = keypoints_json_paths[master_view]
    all_keypoint_paths = [keypoints_json_paths[view] for view in available_views]
    phased_json_paths_list = detect_and_synchronize_phases(master_keypoints_path, all_keypoint_paths)
    for path in phased_json_paths_list:
        for view in available_views:
            if view in path:
                phased_json_paths[view] = path
    print(f"Phased JSON files created: {phased_json_paths}")
    print("-" * 50)

    # --- Step 3: Calculate View-Specific Metrics ---
    print("### Step 3: Calculating View-Specific Biomechanical Metrics ###")
    for view in available_views:
        print(f"\nCalculating {view}-view metrics...")
        
        # Load the phased data for unit conversion
        with open(phased_json_paths[view], 'r') as f:
            phased_data = json.load(f)
        
        if view == 'side':
            metrics = calculate_all_side_metrics(phased_json_paths[view])
            # Apply unit conversion for side view
            print("Applying unit conversion for side view...")
            metrics = convert_side_view_metrics(metrics, phased_data)
        elif view == 'front':
            metrics = calculate_all_front_metrics(phased_json_paths[view])
            # Apply unit conversion for front view
            print("Applying unit conversion for front view...")
            metrics = convert_front_view_metrics(metrics, phased_data)
        elif view == 'back':
            metrics = calculate_all_back_metrics(phased_json_paths[view])
            # Apply unit conversion for back view
            print("Applying unit conversion for back view...")
            metrics = convert_back_view_metrics(metrics, phased_data)
        else:
            continue
        
        base_name = os.path.splitext(os.path.basename(video_paths[view]))[0]
        metrics_paths[view] = f"{base_name}_{view}_metrics.json"
        save_metrics_to_file(metrics, metrics_paths[view])
    print("-" * 50)

    # --- Step 4: Annotate Videos ---
    print("### Step 4: Annotating Videos with Metrics ###")
    for view in available_views:
        print(f"Annotating {view} view video...")
        annotated_video_paths[view] = annotate_video(
            video_path=video_paths[view],
            keypoints_json_path=phased_json_paths[view],
            metrics_json_path=metrics_paths[view],
            view=view
        )
    print("-" * 50)

    # --- Final Summary ---
    print("\n✅ Processing Complete!")
    for view in available_views:
        print(f"\n--- {view.upper()} VIEW OUTPUTS ---")
        print(f"  Keypoints: {keypoints_json_paths[view]}")
        print(f"  Phased JSON: {phased_json_paths[view]}")
        print(f"  Metrics: {metrics_paths[view]}")
        print(f"  Annotated Video: {annotated_video_paths[view]}")

if __name__ == "__main__":
    main()
    