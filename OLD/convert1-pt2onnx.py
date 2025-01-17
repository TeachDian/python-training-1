import torch
from ultralytics import YOLO  # If using YOLOv8

# Step 1: Load the saved checkpoint, which contains the model directly
checkpoint = torch.load('models/yolov8/240epoch.pt')

# Step 2: Access the model directly from the checkpoint
model = checkpoint['model']  # This directly gives you the model

# Step 3: Convert the model to full precision
model = model.float()

# Step 4: Move the model to GPU
model = model.to('cuda')

# Step 5: Set the model to evaluation mode
model.eval()

# Step 6: Define an example input and move it to GPU
dummy_input = torch.randn(1, 3, 640, 640).to('cuda')  # Adjust according to your input size

# Step 7: Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
