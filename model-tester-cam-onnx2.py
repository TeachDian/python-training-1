import cv2
import onnxruntime as ort
import numpy as np

# List of the 11 classes (fish diseases)
classes = [
    "Aeromonas Septicemia", "Columnaris Disease", "Edwardsiella Ictaluri -Bacterial Red Disease-", 
    "Epizootic Ulcerative Syndrome -EUS-", "Flavobacterium -Bacterial Gill Disease-", 
    "Fungal Disease -Saprolegniasis", "Healthy Fish", "Ichthyophthirius -White Spots-", 
    "Parasitic Disease", "Streptococcus", "Tilapia Lake Virus -TiLV-"
]

# Step 1: Load the ONNX model
onnx_model_path = "models/onnx/yolo 240-onnx-model/model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Step 2: Set up video capture (using the default camera)
cap = cv2.VideoCapture(0)  # 0 is the ID for the default webcam

# Step 3: Define the preprocessing function for input
def preprocess_image(frame):
    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to the input size expected by your model
    image = cv2.resize(image, (640, 640))  # Assuming the model input is 640x640
    # Convert the image to a float32 numpy array and normalize to range [0, 1]
    image = image.astype(np.float32) / 255.0
    # Transpose the image to match the ONNX model input (batch_size, channels, height, width)
    image = np.transpose(image, (2, 0, 1))
    # Add a batch dimension (1, channels, height, width)
    image = np.expand_dims(image, axis=0)
    return image

# Step 4: Loop to process frames from the camera
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break

    # Step 5: Preprocess the camera frame
    input_image = preprocess_image(frame)

    # Step 6: Run inference using ONNX Runtime
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_image})

    # Step 7: Process the output
    detections = output[0]  # Assuming this contains the detection results
    print(detections.shape)  # Debugging: print shape of the output

    detected_diseases = []

    # Loop through each detection
    for i in range(detections.shape[2]):  # Loop over 8400 detections
        detection = detections[0, :, i]  # Get the 15 elements for this detection

        # Assuming the structure is [x1, y1, x2, y2, obj_confidence, class_scores...]
        bbox = detection[:4]  # First 4 elements are the bounding box coordinates
        obj_confidence = detection[4]  # 5th element is the object confidence score
        class_scores = detection[5:]  # The rest are class scores

        # Find the class with the highest score
        class_id = np.argmax(class_scores)  # Get the index of the class with the highest score
        class_confidence = class_scores[class_id]  # Get the confidence score for the detected class

        # Optional: Set a threshold for confidence
        threshold = 0.5
        if class_confidence > threshold and class_id < len(classes):
            detected_diseases.append(classes[class_id])

    # Step 8: Display results on the frame
    if detected_diseases:
        text = f"Diseases detected: {', '.join(detected_diseases)}"
    else:
        text = "No diseases detected."

    # Step 9: Put the text on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with detection results
    cv2.imshow('Disease Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
