# FSRCNN Video Upscaling

This repository contains scripts for upscaling videos using the FSRCNN (Fast Super-Resolution Convolutional Neural Network) model. The implementation is based on the paper "Accelerating the Super-Resolution Convolutional Neural Network" by Dong et al.

## Requirements

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

Additional requirements:
- OpenCV
- matplotlib (for visualization)
- ONNX and ONNX Runtime (for ONNX-based inference)
- ffmpeg (for audio preservation)

## Scripts Overview

### 1. PyTorch Video Upscaling (`video_upscale.py`)

This script upscales a single video using the FSRCNN model with PyTorch.

```bash
python video_upscale.py --weights-file weights/fsrcnn_x3.pth --video-file inputs/video.mp4 --output-file outputs/upscaled_video.mp4 --scale 3
```

Arguments:
- `--weights-file`: Path to the model weights file
- `--video-file`: Path to the input video file
- `--output-file`: Path to save the output video
- `--scale`: Upscaling factor (default: 3)

### 2. ONNX Video Upscaling (`video_upscale_onnx.py`)

This script upscales videos using the FSRCNN model with ONNX Runtime on CPU and preserves audio.

#### Single video upscaling:

```bash
python video_upscale_onnx.py single --onnx-model weights/fsrcnn_x3.onnx --video-file inputs/video.mp4 --output-file outputs/upscaled_video.mp4 --scale 3 --num-threads 4
```

Arguments:
- `--onnx-model`: Path to the ONNX model file
- `--video-file`: Path to the input video file
- `--output-file`: Path to save the output video
- `--scale`: Upscaling factor (default: 3)
- `--num-threads`: Number of threads for ONNX Runtime (default: 0, which means use default)

#### Batch video upscaling:

```bash
python video_upscale_onnx.py batch --onnx-model weights/fsrcnn_x3.onnx --input-dir inputs/videos --output-dir outputs/upscaled_videos --scale 3 --num-threads 4
```

Arguments:
- `--onnx-model`: Path to the ONNX model file
- `--input-dir`: Directory containing input videos
- `--output-dir`: Directory to save output videos
- `--scale`: Upscaling factor (default: 3)
- `--num-threads`: Number of threads for ONNX Runtime (default: 0, which means use default)

### 3. Convert PyTorch Model to ONNX (`convert_to_onnx.py`)

This script converts a PyTorch FSRCNN model to ONNX format.

```bash
python convert_to_onnx.py --weights-file weights/fsrcnn_x3.pth --output-file weights/fsrcnn_x3.onnx --scale 3
```

Arguments:
- `--weights-file`: Path to the PyTorch model weights
- `--output-file`: Path to save the ONNX model
- `--scale`: Upscaling factor (default: 3)
- `--height`: Input height for the model (default: 256)
- `--width`: Input width for the model (default: 256)

### 4. Batch Video Upscaling (`batch_video_upscale.py`)

This script processes multiple videos in a directory using PyTorch.

```bash
python batch_video_upscale.py --weights-file weights/fsrcnn_x3.pth --input-dir inputs/videos --output-dir outputs/upscaled_videos --scale 3
```

Arguments:
- `--weights-file`: Path to the model weights file
- `--input-dir`: Directory containing input videos
- `--output-dir`: Directory to save output videos
- `--scale`: Upscaling factor (default: 3)

### 5. Frame Comparison (`compare_frames.py`)

This script provides utilities for extracting frames from videos and comparing original, bicubic upscaled, and FSRCNN upscaled frames.

#### Extract frames from a video:

```bash
python compare_frames.py extract --video-file inputs/video.mp4 --output-dir frames/original --frame-interval 30
```

Arguments:
- `--video-file`: Path to the video file
- `--output-dir`: Directory to save extracted frames
- `--frame-interval`: Extract one frame every N frames (default: 30)

#### Create bicubic upscaled frames:

```bash
python compare_frames.py bicubic --original-dir frames/original --output-dir frames/bicubic --scale 3
```

Arguments:
- `--original-dir`: Directory containing original frames
- `--output-dir`: Directory to save bicubic upscaled frames
- `--scale`: Upscaling factor (default: 3)

#### Compare frames:

```bash
python compare_frames.py compare --original-dir frames/original --upscaled-dir frames/fsrcnn --output-dir frames/comparison --bicubic-dir frames/bicubic
```

Arguments:
- `--original-dir`: Directory containing original frames
- `--upscaled-dir`: Directory containing FSRCNN upscaled frames
- `--output-dir`: Directory to save comparison images
- `--bicubic-dir`: Directory containing bicubic upscaled frames (optional)

## ONNX Workflow Example

Here's a complete workflow example using ONNX:

1. Convert the PyTorch model to ONNX format:

```bash
python convert_to_onnx.py --weights-file weights/fsrcnn_x3.pth --output-file weights/fsrcnn_x3.onnx --scale 3
```

2. Upscale a video using ONNX Runtime (with audio preservation):

```bash
python video_upscale_onnx.py single --onnx-model weights/fsrcnn_x3.onnx --video-file inputs/video.mp4 --output-file outputs/video_fsrcnn_x3.mp4 --scale 3 --num-threads 4
```

3. Extract frames from both original and upscaled videos:

```bash
python compare_frames.py extract --video-file inputs/video.mp4 --output-dir frames/original --frame-interval 30
python compare_frames.py extract --video-file outputs/video_fsrcnn_x3.mp4 --output-dir frames/fsrcnn --frame-interval 30
```

4. Create bicubic upscaled frames for comparison:

```bash
python compare_frames.py bicubic --original-dir frames/original --output-dir frames/bicubic --scale 3
```

5. Compare the frames:

```bash
python compare_frames.py compare --original-dir frames/original --upscaled-dir frames/fsrcnn --output-dir frames/comparison --bicubic-dir frames/bicubic
```

## Notes

- The FSRCNN model works best with the trained scale factor (2x, 3x, or 4x).
- Video processing can be computationally intensive. Using multiple threads with ONNX Runtime can improve performance on CPU.
- For high-resolution videos, consider downscaling the video first to reduce processing time.
- The model processes the Y channel (luminance) in the YCbCr color space, while the Cb and Cr channels (chrominance) are upscaled using bicubic interpolation.
- The ONNX implementation preserves audio from the original video using ffmpeg.

## References

- Original FSRCNN paper: "Accelerating the Super-Resolution Convolutional Neural Network" by Dong et al.
- This implementation is based on the PyTorch implementation of FSRCNN. 