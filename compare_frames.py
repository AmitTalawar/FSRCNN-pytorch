import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract one frame every N frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Total frames: {frame_count}")
    print(f"FPS: {fps}")
    
    # Extract frames
    frame_idx = 0
    saved_count = 0
    
    for _ in tqdm(range(frame_count), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")


def compare_frames(original_dir, upscaled_dir, output_dir, bicubic_dir=None):
    """
    Compare original and upscaled frames side by side
    
    Args:
        original_dir: Directory containing original frames
        upscaled_dir: Directory containing FSRCNN upscaled frames
        output_dir: Directory to save comparison images
        bicubic_dir: Directory containing bicubic upscaled frames (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame files in the original directory
    frame_files = [f for f in os.listdir(original_dir) if f.endswith('.png')]
    frame_files.sort()
    
    if not frame_files:
        print(f"No frame files found in {original_dir}")
        return
    
    print(f"Found {len(frame_files)} frames to compare")
    
    # Compare each frame
    for frame_file in tqdm(frame_files, desc="Comparing frames"):
        original_path = os.path.join(original_dir, frame_file)
        upscaled_path = os.path.join(upscaled_dir, frame_file)
        
        # Check if upscaled frame exists
        if not os.path.exists(upscaled_path):
            print(f"Warning: Upscaled frame {frame_file} not found")
            continue
        
        # Read frames
        original = cv2.imread(original_path)
        upscaled = cv2.imread(upscaled_path)
        
        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        
        if bicubic_dir:
            bicubic_path = os.path.join(bicubic_dir, frame_file)
            if os.path.exists(bicubic_path):
                bicubic = cv2.imread(bicubic_path)
                bicubic_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
                
                # Create a figure with three subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(original_rgb)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(bicubic_rgb)
                axes[1].set_title('Bicubic Upscaled')
                axes[1].axis('off')
                
                axes[2].imshow(upscaled_rgb)
                axes[2].set_title('FSRCNN Upscaled')
                axes[2].axis('off')
            else:
                # Create a figure with two subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                axes[0].imshow(original_rgb)
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                axes[1].imshow(upscaled_rgb)
                axes[1].set_title('FSRCNN Upscaled')
                axes[1].axis('off')
        else:
            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(original_rgb)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(upscaled_rgb)
            axes[1].set_title('FSRCNN Upscaled')
            axes[1].axis('off')
        
        # Save the comparison image
        comparison_path = os.path.join(output_dir, f"comparison_{frame_file}")
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()
    
    print(f"Comparison images saved to {output_dir}")


def create_bicubic_upscaled_frames(original_dir, output_dir, scale):
    """
    Create bicubic upscaled frames from original frames
    
    Args:
        original_dir: Directory containing original frames
        output_dir: Directory to save bicubic upscaled frames
        scale: Upscaling factor
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all frame files in the original directory
    frame_files = [f for f in os.listdir(original_dir) if f.endswith('.png')]
    frame_files.sort()
    
    if not frame_files:
        print(f"No frame files found in {original_dir}")
        return
    
    print(f"Found {len(frame_files)} frames to upscale")
    
    # Upscale each frame
    for frame_file in tqdm(frame_files, desc="Upscaling frames"):
        original_path = os.path.join(original_dir, frame_file)
        
        # Read frame
        original = cv2.imread(original_path)
        
        # Get dimensions
        height, width = original.shape[:2]
        
        # Upscale using bicubic interpolation
        bicubic = cv2.resize(original, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
        
        # Save the upscaled frame
        bicubic_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(bicubic_path, bicubic)
    
    print(f"Bicubic upscaled frames saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract frames command
    extract_parser = subparsers.add_parser('extract', help='Extract frames from a video')
    extract_parser.add_argument('--video-file', type=str, required=True, help='Path to the video file')
    extract_parser.add_argument('--output-dir', type=str, required=True, help='Directory to save extracted frames')
    extract_parser.add_argument('--frame-interval', type=int, default=30, help='Extract one frame every N frames')
    
    # Create bicubic upscaled frames command
    bicubic_parser = subparsers.add_parser('bicubic', help='Create bicubic upscaled frames')
    bicubic_parser.add_argument('--original-dir', type=str, required=True, help='Directory containing original frames')
    bicubic_parser.add_argument('--output-dir', type=str, required=True, help='Directory to save bicubic upscaled frames')
    bicubic_parser.add_argument('--scale', type=int, default=3, help='Upscaling factor')
    
    # Compare frames command
    compare_parser = subparsers.add_parser('compare', help='Compare original and upscaled frames')
    compare_parser.add_argument('--original-dir', type=str, required=True, help='Directory containing original frames')
    compare_parser.add_argument('--upscaled-dir', type=str, required=True, help='Directory containing FSRCNN upscaled frames')
    compare_parser.add_argument('--output-dir', type=str, required=True, help='Directory to save comparison images')
    compare_parser.add_argument('--bicubic-dir', type=str, help='Directory containing bicubic upscaled frames (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_frames(args.video_file, args.output_dir, args.frame_interval)
    elif args.command == 'bicubic':
        create_bicubic_upscaled_frames(args.original_dir, args.output_dir, args.scale)
    elif args.command == 'compare':
        compare_frames(args.original_dir, args.upscaled_dir, args.output_dir, args.bicubic_dir)
    else:
        parser.print_help() 