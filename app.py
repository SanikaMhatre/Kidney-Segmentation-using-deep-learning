import os
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from io import BytesIO
import matplotlib.pyplot as plt
import base64

# Define Double U-Net Model (same as in your training code)
class DoubleUNet(nn.Module):
    def __init__(self, enc_name='efficientnet-b1', classes=2, pretrain='imagenet'):
        super(DoubleUNet, self).__init__()
       
        self.unet_plus = smp.UnetPlusPlus(
            encoder_name=enc_name,
            encoder_weights=pretrain,
            in_channels=1,
            classes=classes,
        )
       
        self.unet = smp.Unet(
            encoder_name=enc_name,
            encoder_weights=pretrain,
            in_channels=1,
            classes=classes,
        )
       
        self.final_conv = nn.Conv2d(classes * 2, classes, kernel_size=1)
   
    def forward(self, x):
        out1 = self.unet_plus(x)
        out2 = self.unet(x)
        out = torch.cat((out1, out2), dim=1)
        return self.final_conv(out)

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
CHECKPOINT_PATH = 'checkpoints/best_model.pth'  # Path to your saved model
ALLOWED_EXTENSIONS = {'npz'}
IMG_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check for valid file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the trained model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleUNet(enc_name='efficientnet-b1', classes=2, pretrain=None).to(device)
   
    # Load the saved checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
   
    return model, device

# Preprocess input for inference
def preprocess_input(npz_data):
    # Create inference transform
    transform = A.Compose([
        A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])
   
    # Get the available keys in the npz file
    print(f"Available keys in NPZ file: {list(npz_data.keys())}")
    
    # Try to get the images from npz file with different possible key names
    if 'images' in npz_data:
        image = npz_data['images']
    elif 'image' in npz_data:
        image = npz_data['image']
    elif 'data' in npz_data:
        image = npz_data['data']
    else:
        # If none of the expected keys are found, use the first array in the archive
        print("Using first array in the archive as image data")
        for key in npz_data.keys():
            image = npz_data[key]
            break
   
    # Check if image data was found
    if image is None:
        raise ValueError("Could not find image data in the NPZ file. Available keys: " + str(list(npz_data.keys())))
    
    # Handle different dimensions - ensure we have a 3D volume
    if len(image.shape) == 2:  # If it's a single 2D slice
        image = np.expand_dims(image, 0)  # Make it a 3D volume with single slice
    elif len(image.shape) == 4 and image.shape[0] == 1:  # If it has an extra batch dimension
        image = image[0]  # Remove batch dimension
    
    print(f"Image shape: {image.shape}")
    
    # Process all slices in the volume
    processed_slices = []
   
    for slice_idx in range(image.shape[0]):
        image_slice = image[slice_idx].astype(np.float32)
       
        # Normalize image to [0, 1] range if not already
        if image_slice.max() > image_slice.min():  # Avoid division by zero
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
       
        # Apply transform
        transformed = transform(image=image_slice)
        tensor_slice = transformed['image']
       
        # Add batch dimension
        tensor_slice = tensor_slice.unsqueeze(0)
        processed_slices.append(tensor_slice)
   
    return processed_slices, image

# Perform inference on a volume
def predict_volume(model, processed_slices, device):
    predictions = []
   
    with torch.no_grad():
        for slice_tensor in processed_slices:
            # Move to device
            slice_tensor = slice_tensor.to(device)
           
            # Get prediction
            output = model(slice_tensor)
            pred_mask = output.argmax(dim=1).cpu().numpy()[0]
            predictions.append(pred_mask)
   
    # Stack all predictions into a volume
    return np.stack(predictions)

# Create visualization of results
def create_visualization(original_slice, pred_mask, slice_idx):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
   
    # Original image
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title(f'Original Slice {slice_idx}')
    axes[0].axis('off')
   
    # Predicted mask
    axes[1].imshow(pred_mask, cmap='viridis')
    axes[1].set_title(f'Segmentation Mask {slice_idx}')
    axes[1].axis('off')
   
    plt.tight_layout()
   
    # Save to in-memory file
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
   
    return buf

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
   
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
   
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
       
        try:
            # Load the npz file
            npz_data = np.load(filepath, allow_pickle=True)
           
            # Load model if not already loaded
            model, device = load_model()
           
            # Preprocess input
            processed_slices, original_images = preprocess_input(npz_data)
           
            # Get predictions
            predictions = predict_volume(model, processed_slices, device)
           
            # Save predictions as npz
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'pred_{filename}')
            np.savez_compressed(output_path, segmentations=predictions)
           
            # Create visualizations for sample slices
            vis_indices = [len(original_images) // 2]  # Middle slice
            visualizations = []
           
            for idx in vis_indices:
                if idx < len(original_images) and idx < len(predictions):
                    vis_buffer = create_visualization(
                        original_images[idx],
                        predictions[idx],
                        idx
                    )
                    visualizations.append(base64.b64encode(vis_buffer.getvalue()).decode('utf-8'))
           
            return jsonify({
                'success': True,
                'message': f'Prediction completed successfully for {predictions.shape[0]} slices',
                'output_file': f'pred_{filename}',
                'visualizations': visualizations
            })
           
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            return jsonify({'error': f'Error in processing: {str(e)}\n{traceback_str}'})
   
    return jsonify({'error': 'Invalid file type'})

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                     as_attachment=True)

# Serve the HTML template directly
@app.route('/template')
def template():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')