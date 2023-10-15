import cv2
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, Response
import sys

sys.path.append('/Users/khaitran/Desktop/cs/dubhacks23/WasteSortingTrain')
from resnet_model import ResNet


app = Flask(__name__)
model = ResNet()
model.load_state_dict(torch.load('resnet_weights.pth'))
model.eval()
dataset_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def classify_frame(frame):
    frame = cv2.resize(frame, (256, 256))
    frame_tensor = transforms.ToTensor()(frame)
    preprocessed_frame = frame_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(preprocessed_frame)
        predicted_class = torch.argmax(prediction, dim=1).item()

    return dataset_classes[predicted_class]

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        classification_result = "Item type: " + classify_frame(frame)
        frame = cv2.putText(frame, classification_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)
