import argparse
import torch
import torch.onnx
import torch.backends.cudnn as cudnn

from models import FSRCNN


def convert_to_onnx(weights_file, output_file, scale, input_shape=(1, 1, 256, 256)):
    """
    Convert PyTorch FSRCNN model to ONNX format
    
    Args:
        weights_file: Path to the PyTorch model weights
        output_file: Path to save the ONNX model
        scale: Upscaling factor (2, 3, or 4)
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    # Create model
    device = torch.device('cpu')
    model = FSRCNN(scale_factor=scale).to(device)
    
    # Load weights
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    x = torch.randn(input_shape, requires_grad=False).to(device)
    
    # Export the model
    print(f"Converting FSRCNN model to ONNX format with scale factor {scale}...")
    torch.onnx.export(
        model,                  # model being run
        x,                      # model input (or a tuple for multiple inputs)
        output_file,            # where to save the model
        export_params=True,     # store the trained parameter weights inside the model file
        opset_version=12,       # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},  # variable length axes
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"ONNX model saved to {output_file}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
    except ImportError:
        print("ONNX package not installed. Skipping model verification.")
        print("To verify the model, install ONNX: pip install onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True, help='Path to the PyTorch model weights')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the ONNX model')
    parser.add_argument('--scale', type=int, default=3, help='Upscaling factor (2, 3, or 4)')
    parser.add_argument('--height', type=int, default=256, help='Input height for the model')
    parser.add_argument('--width', type=int, default=256, help='Input width for the model')
    args = parser.parse_args()
    
    cudnn.benchmark = True
    
    # Convert the model
    convert_to_onnx(
        args.weights_file, 
        args.output_file, 
        args.scale, 
        input_shape=(1, 1, args.height, args.width)
    ) 