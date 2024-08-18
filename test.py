from ultralytics import YOLO
from openvino.runtime import Core

# Step 1: Export YOLOv8 model to OpenVINO format
model = YOLO('yolov8n.pt')
model.export(format='openvino')  # This creates 'yolov8n_openvino_model/'

# Step 2: Load the exported OpenVINO model
ie = Core()
net = ie.read_model(model='yolov8n_openvino_model/yolov8n.xml')
compiled_model = ie.compile_model(model=net, device_name='GPU')

# Step 3: Run inference
input_layer = next(iter(compiled_model.inputs))
output_layer = next(iter(compiled_model.outputs))

# Load your image here
image = 0  # Replace with code to load your image

# Run inference
results = compiled_model([image])[output_layer]
print(results)
