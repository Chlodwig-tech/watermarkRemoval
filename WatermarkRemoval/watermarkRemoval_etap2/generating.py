import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from datetime import datetime

def load_onnx_model(model_path):
    ort_session = ort.InferenceSession(model_path)
    return ort_session

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image).astype(np.float32)
    image_np = np.transpose(image_np, [2, 0, 1])  # HWC to CHW
    image_np = image_np / 255.0  # Normalize to [0, 1]
    return image_np[np.newaxis, :]  # Add batch dimension

def save_output_image(output_tensor, output_dir, original_image_path):
    output_image = output_tensor[0].clip(0, 1) * 255
    output_image = output_image.transpose(1, 2, 0).astype(np.uint8)  # CHW to HWC
    output_image = Image.fromarray(output_image)

    # Generate a unique filename
    base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_filename}_watermark_removed_{timestamp}.jpg"
    output_image_path = os.path.join(output_dir, output_filename)

    output_image.save(output_image_path)
    print(f"Result saved to {output_image_path}")

def remove_watermark(model_path, image_path, output_dir):
    ort_session = load_onnx_model(model_path)
    input_tensor = process_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_output = ort_session.run(None, ort_inputs)
    save_output_image(ort_output[0], output_dir, image_path)

model_path = ".\\results\\model_epoch_9.onnx"
image_path = ".\\results\\18.jpg"  # Replace with your image path
output_dir = ".\\results"

remove_watermark(model_path, image_path, output_dir)
