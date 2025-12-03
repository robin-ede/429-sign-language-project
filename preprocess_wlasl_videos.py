"""
Pre-process WLASL100 videos: Extract frames and save to disk.

This script extracts frames from all videos once, saving them as JPG files.
Training will then load images instead of videos (10-100x faster).
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Configuration
DATA_ROOT = Path(r"C:\Users\robin\OneDrive\Documents\code\daen429-project")
WLASL_ROOT = DATA_ROOT / "WLASL_100"
OUTPUT_ROOT = DATA_ROOT / "WLASL_100_frames"
NUM_FRAMES = 8  # Frames to extract per video

def extract_frames(video_path, output_dir, num_frames=8):
    """
    Extract frames from a video and save as JPG files.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        num_frames: Number of frames to extract uniformly

    Returns:
        success: True if extraction successful
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        cap.release()
        return False

    # Uniformly sample frame indices
    if frame_count >= num_frames:
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    else:
        # If video has fewer frames, repeat last frame
        indices = np.arange(frame_count)
        indices = np.pad(indices, (0, num_frames - frame_count), mode='edge')

    # Extract and save frames
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save as JPG
            frame_path = output_dir / f"frame_{i:02d}.jpg"
            Image.fromarray(frame).save(frame_path, quality=95)
        else:
            # If frame read fails, copy previous frame if available
            if i > 0:
                prev_frame = output_dir / f"frame_{i-1:02d}.jpg"
                curr_frame = output_dir / f"frame_{i:02d}.jpg"
                if prev_frame.exists():
                    Image.open(prev_frame).save(curr_frame)

    cap.release()
    return True

def process_split(split_name):
    """Process all videos in a split (train/val/test)."""
    input_dir = WLASL_ROOT / split_name
    output_dir = OUTPUT_ROOT / split_name

    if not input_dir.exists():
        print(f"Warning: {input_dir} does not exist, skipping...")
        return

    # Get all video files
    video_files = []
    classes = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    for class_dir in classes:
        for video_path in class_dir.glob('*.mp4'):
            video_files.append((video_path, class_dir.name))

    print(f"\n{split_name.upper()}: Processing {len(video_files)} videos...")

    success_count = 0
    fail_count = 0

    # Process each video
    for video_path, class_name in tqdm(video_files, desc=f"{split_name}"):
        # Output directory: WLASL_100_frames/train/class_name/video_stem/
        video_output_dir = output_dir / class_name / video_path.stem

        # Skip if already processed
        if video_output_dir.exists() and len(list(video_output_dir.glob('*.jpg'))) == NUM_FRAMES:
            success_count += 1
            continue

        # Extract frames
        if extract_frames(video_path, video_output_dir, NUM_FRAMES):
            success_count += 1
        else:
            fail_count += 1
            print(f"Failed to process: {video_path}")

    print(f"{split_name.upper()}: {success_count} success, {fail_count} failed")

def main():
    print("="*70)
    print("WLASL100 Video Preprocessing")
    print("="*70)
    print(f"Input: {WLASL_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Frames per video: {NUM_FRAMES}")
    print("="*70)

    # Create output directory
    OUTPUT_ROOT.mkdir(exist_ok=True)

    # Process each split
    for split in ['train', 'val', 'test']:
        process_split(split)

    print("\n" + "="*70)
    print("Preprocessing complete!")
    print("="*70)
    print(f"\nExtracted frames saved to: {OUTPUT_ROOT}")
    print("\nNext step: Use the updated dataset class in your notebook")

if __name__ == "__main__":
    main()
