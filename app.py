from flask import Flask, request, jsonify
import base64
import ollama
import tensorflow as tf
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import matplotlib

app=Flask(__name__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from consuming all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) available")
    except RuntimeError as e:
        # Memory growth must be set before GPUs are initialized
        print(f"Error during GPU configuration: {e}")
else:
    print("No GPU found, using CPU")


def preprocess_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

matplotlib.use('TkAgg')
model_path = 'models/saved_model'
model = tf.saved_model.load(model_path)
   
    
@app.route('/ocr',methods=['POST','GET'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode base64 image data
        base64_image = request.json['image']
        image_data = base64.b64decode(base64_image)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    if image is None:
        return jsonify({'error': 'Failed to decode image data'}), 400
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
        detections['num_detections'] = num_detections

        boxes = detections['detection_boxes']
        classes = detections['detection_classes'].astype(np.int64)
        scores = detections['detection_scores']

        threshold = 0.5
        valid_detections = scores >= threshold
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        classes = classes[valid_detections]
        
        extracted_texts = []
        config = '--oem 3 --psm 6'
        
        for i in range(len(boxes)):
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * image.shape[1]), int(ymin * image.shape[0]))
            end_point = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))
            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            roi = image[int(ymin * image.shape[0]):int(ymax * image.shape[0]), int(xmin * image.shape[1]):int(xmax * image.shape[1])]
            preprocessed_roi = preprocess_image(roi)
            text = pytesseract.image_to_string(preprocessed_roi, config=config,lang='eng')
            extracted_texts.append(f'{text.strip()}')

            label = f'{int(classes[i])}: {scores[i]:.2f}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (start_point[0], start_point[1] - label_size[1])
            bottom_right = (start_point[0] + label_size[0], start_point[1])
            image = cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            image = cv2.putText(image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        texts = '   |||   '.join(extracted_texts)
        
        msg = "can you parse this text and give me json format version with these corresponding values: Company Name, Company Address, Customer Name, Customer Address, Invoice Number, Invoice Date, Due Date, Description, Quantity, Unit Price, Taxes, Amount, Total. If you can't find values of corresponding field then leave it empty. The text is :"
        
        response = ollama.chat(
        model="gemma2:2b",#gemma2:2b
        # model="llama3.1",
        messages=[
            {
                "role": "user",
                "content": msg+texts,
            },
        ],
        )
        ollama_out = (response["message"]["content"])

        # Split the response to extract the JSON part
        start_index = ollama_out.find('{')
        end_index = ollama_out.rfind('}') + 1

        json_part = ollama_out[start_index:end_index]
        
        return json_part
        
            
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500
            
            

if __name__ == '__main__':
    app.run(debug=True, port=9090)
    
