import argparse
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import tempfile
import subprocess

from utils import convert_ycbcr_to_rgb, preprocess


def upscale_video_onnx(video_path, output_path, onnx_model_path, scale, num_threads=0):
    """
    Upscale a video using ONNX Runtime inference
    
    Args:
        video_path: Path to the input video
        output_path: Path to save the output video
        onnx_model_path: Path to the ONNX model
        scale: Upscaling factor
        num_threads: Number of threads for ONNX Runtime (0 means default)
    """
    # Create ONNX Runtime session
    print(f"Loading ONNX model from {onnx_model_path}")
    session_options = ort.SessionOptions()
    if num_threads > 0:
        session_options.intra_op_num_threads = num_threads
    
    # Create inference session
    session = ort.InferenceSession(
        onnx_model_path, 
        sess_options=session_options,
        providers=['CPUExecutionProvider']
    )
    
    # Get model input name
    input_name = session.get_inputs()[0].name
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new dimensions
    new_width = width * scale
    new_height = height * scale
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Path for the video without audio
        video_no_audio_path = os.path.join(temp_dir, "temp_video_no_audio.mp4")
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_no_audio_path, fourcc, fps, (new_width, new_height))
        
        # Process each frame
        start_time = time.time()
        
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess the frame
            lr, _ = preprocess(frame_rgb, 'cpu')
            
            # Convert to numpy and ensure correct shape for ONNX
            lr_np = lr.cpu().numpy()
            
            # Run inference
            outputs = session.run(None, {input_name: lr_np})
            preds = np.clip(outputs[0], 0.0, 1.0)
            
            # Convert back to image
            preds = preds.squeeze(0).squeeze(0) * 255.0
            
            # Get the bicubic upscaled frame for color information
            bicubic = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            bicubic_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
            _, ycbcr = preprocess(bicubic_rgb, 'cpu')
            
            # Combine the FSRCNN Y channel with the bicubic Cb and Cr channels
            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            
            # Convert RGB back to BGR for OpenCV
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # Write the frame to the output video
            out.write(output_bgr)
        
        # Release resources
        cap.release()
        out.release()
        
        # Copy audio from original video to the upscaled video
        print("Copying audio from original video to upscaled video...")
        
        # Check if ffmpeg is available
        try:
            # Command to copy audio from original to upscaled video
            cmd = [
                'ffmpeg',
                '-i', video_no_audio_path,  # Input video without audio
                '-i', video_path,           # Original video with audio
                '-c:v', 'copy',             # Copy video stream
                '-c:a', 'aac',              # Audio codec
                '-map', '0:v:0',            # Map video from first input
                '-map', '1:a:0',            # Map audio from second input
                '-shortest',                # Finish encoding when the shortest input stream ends
                '-y',                       # Overwrite output file if it exists
                output_path                 # Output file
            ]
            
            subprocess.run(cmd, check=True)
            print(f"Successfully copied audio to the upscaled video")
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Warning: Could not copy audio using ffmpeg: {e}")
            print("Saving video without audio...")
            # If ffmpeg fails, just copy the video without audio
            import shutil
            shutil.copy2(video_no_audio_path, output_path)
    
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time:.2f} seconds")
    print(f"Output saved to {output_path}")


def batch_upscale_videos_onnx(input_dir, output_dir, onnx_model_path, scale, num_threads=0, extensions=('.mp4', '.avi', '.mkv', '.mov')):
    """
    Process all videos in a directory using ONNX Runtime
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save output videos
        onnx_model_path: Path to the ONNX model
        scale: Upscaling factor
        num_threads: Number of threads for ONNX Runtime (0 means default)
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
        upscale_video_onnx(input_path, output_path, onnx_model_path, scale, num_threads)
    
    total_time = time.time() - start_time
    print(f"\nAll videos processed in {total_time:.2f} seconds")
    print(f"Upscaled videos saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Single video upscaling command
    single_parser = subparsers.add_parser('single', help='Upscale a single video')
    single_parser.add_argument('--onnx-model', type=str, required=True, help='Path to the ONNX model file')
    single_parser.add_argument('--video-file', type=str, required=True, help='Path to the input video file')
    single_parser.add_argument('--output-file', type=str, required=True, help='Path to save the output video')
    single_parser.add_argument('--scale', type=int, default=3, help='Upscaling factor (2, 3, or 4)')
    single_parser.add_argument('--num-threads', type=int, default=0, help='Number of threads for ONNX Runtime (0 means default)')
    
    # Batch video upscaling command
    batch_parser = subparsers.add_parser('batch', help='Upscale multiple videos in a directory')
    batch_parser.add_argument('--onnx-model', type=str, required=True, help='Path to the ONNX model file')
    batch_parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input videos')
    batch_parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output videos')
    batch_parser.add_argument('--scale', type=int, default=3, help='Upscaling factor (2, 3, or 4)')
    batch_parser.add_argument('--num-threads', type=int, default=0, help='Number of threads for ONNX Runtime (0 means default)')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        upscale_video_onnx(args.video_file, args.output_file, args.onnx_model, args.scale, args.num_threads)
    elif args.command == 'batch':
        batch_upscale_videos_onnx(args.input_dir, args.output_dir, args.onnx_model, args.scale, args.num_threads)
    else:
        parser.print_help() 