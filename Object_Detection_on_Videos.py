import cv2
import numpy as np

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 5)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Define your YOLOv4 file paths here
video_path = 'road1.mp4'
config_path = 'yolov4.cfg'
weights_path = 'yolov4.weights'
class_path = 'coco.names'

# Load classes
with open(class_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Use a constant red color for all bounding boxes
COLORS = [(0, 0, 255)] * len(classes)

# Load YOLOv4
net = cv2.dnn.readNet(weights_path, config_path)

# Open video capture
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    Width, Height = frame.shape[1], frame.shape[0]

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass and get output layers
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes on the image
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # Display the frame with bounding boxes
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
