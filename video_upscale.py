import argparse
import os
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm

from models import FSRCNN
from utils import convert_ycbcr_to_rgb, preprocess


def upscale_video(video_path, output_path, weights_file, scale, device):
    # Load the FSRCNN model
    model = FSRCNN(scale_factor=scale).to(device)
    
    # Load the model weights
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    model.eval()
    
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
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Process each frame
    start_time = time.time()
    
    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess the frame
            lr, _ = preprocess(frame_rgb, device)
            
            # Upscale the frame
            preds = model(lr).clamp(0.0, 1.0)
            
            # Convert back to image
            preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            
            # Get the bicubic upscaled frame for color information
            bicubic = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            bicubic_rgb = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
            _, ycbcr = preprocess(bicubic_rgb, device)
            
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
    
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time:.2f} seconds")
    print(f"Output saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--video-file', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the output video')
    parser.add_argument('--scale', type=int, default=3, help='Upscaling factor (2, 3, or 4)')
    args = parser.parse_args()
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    upscale_video(args.video_file, args.output_file, args.weights_file, args.scale, device) 