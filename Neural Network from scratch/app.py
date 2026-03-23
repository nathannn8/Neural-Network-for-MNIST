from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import model
import os

app = Flask(__name__)

# Load the model
nn = model.NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
MODEL_PATH = "model_weights.json"

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    nn.load_weights(MODEL_PATH)
else:
    print("WARNING: Model weights not found. Please run train.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Decode Base64 image
        # Remove header "data:image/png;base64,"
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        

        img = img.resize((28, 28), Image.Resampling.BICUBIC)
        

        img = img.convert('L')
        
        img = ImageOps.invert(img)
        

        img_array = np.array(img) / 255.0
        

        img_input = img_array.reshape(1, 784)
        

        prediction, confidence = nn.predict(img_input)
        
        digit = int(prediction[0])
        conf_score = float(confidence[0])
        
        return jsonify({
            'digit': digit,
            'confidence': f"{conf_score * 100:.2f}"
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
