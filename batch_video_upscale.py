import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from video_upscale import upscale_video


def process_directory(input_dir, output_dir, weights_file, scale, device, extensions=('.mp4', '.avi', '.mkv', '.mov')):
    """
    Process all videos in a directory
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save output videos
        weights_file: Path to the model weights file
        scale: Upscaling factor
        device: Device to run the model on
        extensions: Tuple of valid video file extensions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files in the input directory
    video_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(extensions):
            video_files.append(file)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    start_time = time.time()
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, video_file)
        
        # Create output filename with scale factor
        filename, ext = os.path.splitext(video_file)
        output_filename = f"{filename}_x{scale}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing: {video_file}")
        upscale_video(input_path, output_path, weights_file, scale, device)
    
    total_time = time.time() - start_time
    print(f"\nAll videos processed in {total_time:.2f} seconds")
    print(f"Upscaled videos saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input videos')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output videos')
    parser.add_argument('--scale', type=int, default=3, help='Upscaling factor (2, 3, or 4)')
    args = parser.parse_args()
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    process_directory(args.input_dir, args.output_dir, args.weights_file, args.scale, device) 