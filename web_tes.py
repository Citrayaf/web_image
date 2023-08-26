from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='detection_model.tflite')
interpreter.allocate_tensors()

def preprocess_image(image):
    resized_image = cv2.resize(image, (64, 64))  # Resize to match model input size
    normalized_image = resized_image / 255.0  # Normalize image
    normalized_image = normalized_image.astype(np.float32)  # Convert to FLOAT32
    return normalized_image

def classify_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    labels = ["Label 1", "Label 2", "Label 3", "Label 4"]
    label_idx = output[0].argmax()  # Get the index of the highest value in the output array
    label = labels[label_idx]
    accuracy = output[0][label_idx]
    
    return label, accuracy

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (13, 13), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        dilate = cv2.dilate(thresh, horizontal_kernel, iterations=2)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        dilate = cv2.dilate(dilate, vertical_kernel, iterations=2)

        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 20000:
                x, y, w, h = cv2.boundingRect(c)

                # Extract the detected region
                detected_image = frame[y:y+h, x:x+w]

                # Preprocess the detected image and classify it
                processed_image = preprocess_image(detected_image)
                label, accuracy = classify_image(interpreter, processed_image)

                # Draw bounding box and label
                if accuracy > 0.5:  # Set your threshold here
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)
                    cv2.putText(frame, f"{label} ({accuracy:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
