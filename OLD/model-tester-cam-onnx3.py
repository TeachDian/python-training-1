import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "models/onnx/yolo 240-onnx-model/model.onnx"
session = ort.InferenceSession(model_path)

# Define class names
class_names = [
    "Aeromonas Septicemia",
    "Columnaris Disease",
    "Edwardsiella Ictaluri -Bacterial Red Disease-",
    "Epizootic Ulcerative Syndrome -EUS-",
    "Flavobacterium -Bacterial Gill Disease-",
    "Fungal Disease -Saprolegniasis-",
    "Healthy Fish",
    "Ichthyophthirius -White Spots",
    "Parasitic Disease",
    "Streptococcus",
    "Tilapia Lake Virus -TiLV-"
]

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    # Resize the frame and normalize it
    resized_frame = cv2.resize(frame, (640, 640))  # Adjust according to model input size
    normalized_frame = resized_frame / 255.0
    input_tensor = np.transpose(normalized_frame, (2, 0, 1))  # Change to CHW format
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    return input_tensor

def postprocess_output(output, threshold=0.5):
    # Model output: (batch_size, num_predictions, 85)
    # Each prediction: [x, y, w, h, objectness, class_probabilities...]

    output = output[0]  # Since batch_size is 1
    num_predictions = output.shape[0]

    boxes = []
    confidences = []
    class_ids = []

    for i in range(num_predictions):
        detection = output[i]

        # Extract box coordinates and objectness score
        box = detection[:4]  # x, y, w, h
        objectness = detection[4]

        # Extract class probabilities and find the class with the highest score
        class_scores = detection[5:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        # Filter out detections with low objectness or confidence
        if objectness > threshold and confidence > threshold and class_id < len(class_names):
            boxes.append(box)
            confidences.append(confidence)
            class_ids.append(class_id)

    return boxes, confidences, class_ids

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_tensor = preprocess_frame(frame)
    
    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    
    # Postprocess output
    boxes, confidences, class_ids = postprocess_output(outputs[0])
    
    # Draw bounding boxes and labels on the frame
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        # Extract box coordinates
        x, y, w, h = box

        # Convert center x, y to top-left x, y
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if class_id is valid (within bounds)
        if class_id < len(class_names):
            # Draw label
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Unknown Class: {confidence:.2f}"

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("ONNX Model Tester", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
