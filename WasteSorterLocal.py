import torch
import cv2
from resnet_model import ResNet
import torchvision.transforms as transforms

# Create an instance of the model with the same architecture
model = ResNet()  # Replace num_classes with the number of classes in your dataset

# Load the model weights
model.load_state_dict(torch.load('resnet_weights.pth'))
model.eval()

dataset_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    # Preprocess the frame
    # You may need to adjust the preprocessing to match the requirements of your custom ResNet model
    # Ensure the input size and preprocessing are consistent with your trained model
    
    frame = cv2.resize(frame, (256, 256))
    frame_tensor = transforms.ToTensor()(frame)
    preprocessed_frame = frame_tensor.unsqueeze(0)

    # Perform image classification using your custom ResNet model
    with torch.no_grad():
        prediction = model(preprocessed_frame)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Display the prediction on the frame
    label = f"Class: {dataset_classes[predicted_class]}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
