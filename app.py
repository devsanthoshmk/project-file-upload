

from flask import Flask, render_template, request
import base64
import os

def preprocess_and_classify(image_bytes):
    from PIL import Image
    from transformers import ViTFeatureExtractor, ViTForImageClassification
    import torch
    import torchvision.transforms as transforms
    import io
    # Load the feature extractor and model (pre-trained model name may vary)
    feature_extractor = ViTFeatureExtractor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model = ViTForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

    # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Replace with your model's input size if different
        transforms.ToTensor(),
    ])
    """Preprocesses the image, performs prediction, and returns results"""
    # Open the image
    image = Image.open(io.BytesIO(image_bytes))#Image.open(image_path)

    # Preprocess the image
    preprocessed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        outputs = model(preprocessed_image)

    # Get the predicted class and confidence score
    predicted_class = torch.argmax(outputs.logits).item()
    confidence_score = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()

    # Convert predicted class index to human-readable label
    labels = ["Real", "Deepfake"]
    predicted_label = labels[predicted_class]

    return predicted_label, confidence_score
        
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No image part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected image'
    image_bytes = file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    predic,accur=preprocess_and_classify(image_bytes)
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f0f0;
            }}
            .container {{
                max-width: 800px;
                margin: 50px auto;
                text-align: center;
            }}
            .image {{
            max-width: 300px; /* Adjust the width as needed */
            height: auto;
            display: block;
            margin: 0 auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .text {{
                margin-top: 20px;
            }}
            .text h2 {{
                font-size: 24px;
                color: #333;
                margin: 5px 0;
            }}
            .text p {{
                font-size: 16px;
                color: #666;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="data:image/png;base64,{image_base64}" alt="Placeholder Image" class="image">
            <div class="text">
                <h2>Predicted: Predicted: {predic}</h2>
                <h2>Accuracy: Accuracy: {accur}</h2>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
