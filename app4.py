import tensorflow as tf
from flask import Flask, request, render_template,send_from_directory
from flask import jsonify
from pathlib import Path
import numpy as np
import json
# import tensorflow as tf
from tensorflow import keras

import pandas as pd

from keras.preprocessing import image
from werkzeug.utils import secure_filename
import csv
# import joblib
import torch
from ultralytics import YOLO
import cv2

from torchvision import transforms
from PIL import Image


app = Flask(__name__)

UPLOAD_FOLDER = Path("static/uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define label meanings
labels = [
    'besan_cheela', 'dosa', 'gulab_jamun', 'idli', 'palak_paneer', 'poha', 'samosa'
]

# nutrition_link = 'https://www.nutritionix.com/food/'
nutrition_data = pd.read_csv('nutrition101.csv')
# Loading the best saved model to make predictions.
# def load_model():
#     # Update the model loading code based on the latest TensorFlow/Keras practices
#     model_path = "./runs/detect/train/weights/best.pt"
#     model = torch.load(model_path)
#     return model

nutri = {
    "besan_cheela": [2.63,0.018,14.04,35.09,0.0],
    "dosa": [18.75,0.026,9.82,0.0,0.0],
    "gulab_jamun": [6.64,0.04,29.13,37.53,0.0013],
    "idli": [18.75,0.026,9.82,0.0,0.0],
    "palak_paneer": [18.75,0.026,9.82,0.0,0.0],
    "poha": [18.75,0.026,9.82,0.0,0.0],
    "samosa": [2.63,0.018,14.04,35.09,0.0]
}


# model_best = load_model()
model_best = YOLO('./runs/detect/train/weights/best.pt')
model_best= model_best.to('cpu')
# model_best.eval()


# Define image preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((416, 416))  # Resize image to match YOLO model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.transpose(img, (2, 0, 1))  # Change image format to (C, H, W) for PyTorch
    img = torch.from_numpy(img).float()  # Convert numpy array to torch tensor
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to perform inference and get predictions
def get_predictions(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model_best(img)
    # Process outputs to get predicted class
    # Modify this part based on your model's output format
    predicted_class = outputs.argmax(dim=1).item()
    return predicted_class



# Load nutrition data from CSV
def load_nutrition_data():
    nutrition_table = {}
    with open('nutrition101.csv', 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            name = row[1].strip()
            nutrition_table[name] = [
                {'name': 'protein', 'value': float(row[2])},
                {'name': 'calcium', 'value': float(row[3])},
                {'name': 'fat', 'value': float(row[4])},
                {'name': 'carbohydrates', 'value': float(row[5])},
                {'name': 'vitamins', 'value': float(row[6])}
            ]
    return nutrition_table

nutrition_table = load_nutrition_data()

@app.route('/')
def index():
    img = 'static/profile.jpg'
    return render_template('index.html', img=img)

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist("img")
    for idx, file in enumerate(files):
        filename = secure_filename(f"uploaded_{idx}.jpg")
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)
    return render_template('recognize.html', file_path=filename)
@app.route('/predict', methods=['POST'])
def predict():
    results = []
    
    for idx, img_file in enumerate(request.files.getlist("img")):
        file_path = UPLOAD_FOLDER / f"uploaded_{idx}.jpg"
        img_file.save(file_path)
        
        image = Image.open(file_path)

        # Assuming model_best is your YOLO model, adapt this part based on your actual model
        res = model_best.predict(source=image, stream=True, imgsz=620)
        names = model_best.names

        for r in res:
            for c in r.boxes.cls:
                pred_name = names[int(c)]

        temp = nutri[pred_name]
        new_dict = {
            "name": pred_name,
            "protein": temp[0],
            "calcium": temp[1],
            "fat": temp[2],
            "carbohydrates": temp[3],
            "vitamins": temp[4]
        }
        
        result = {
            'image': f'/static/uploads/uploaded_{idx}.jpg',
            'nutrition': new_dict,
        }
        
        results.append(result)
    with open('./static/nutrient_values.json', 'w') as file:
        json.dump(results, file)
    print(results)
    return render_template('results.html', results=results)

@app.route('/get_json_data')
def get_json_data():
    try:
        # Assuming nutrients_values.json is in the static directory
        with open('./static/nutrients_values.json') as json_file:
            data = json.load(json_file)
        return jsonify(data)
    except Exception as e:
        return str(e), 404


@app.route('/update', methods=['POST'])
def update():
    return render_template('index.html', img='static/P2.jpg')

if __name__ == "__main__":
    app.run(debug=True)
