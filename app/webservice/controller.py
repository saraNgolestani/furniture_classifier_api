import os

import torch
from flask import Flask, request, jsonify
import json
from app.trainer.data_util import load_image


class API:
    def __init__(self, model, upload_directory, img_size, label_dict):
        self.model = model
        self.app = Flask(__name__)
        self.upload_directory = upload_directory
        self.img_size = img_size
        self.label_dict = dict((v, k) for k, v in label_dict.items())
        self.allowed_files = {'png', 'jpg', 'jpeg'}
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])
        self.app.add_url_rule('/health', 'health', self.health_check, methods=['GET'])

    def predict(self):
        # get the input data from the POST request
        if 'file' not in request.files:
            return jsonify({'error': 'No file found in the request'}), 400

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No file found in the request'}), 400

        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in self.allowed_files:
            full_path = os.path.join(self.upload_directory, file.filename)
            # save the file to the upload folder
            file.save(full_path)

            img = load_image(full_path, self.img_size)
            # make a prediction using your model
            prediction = self.model(img.reshape(1, 3, self.img_size, self.img_size))
            _, prediction = torch.max(prediction.data, 1)
            prediction = self.label_dict[int(prediction.item())]
            return jsonify({
                'success': f'Image successfully classified as {prediction}',
                'class': prediction
            }), 200

        return jsonify({'error': 'Invalid file extension'}), 400



        # return the prediction as a JSON object
        return json.dumps(prediction)

    def health_check(self):
        # return a simple message indicating the API is healthy
        return 'Check'


