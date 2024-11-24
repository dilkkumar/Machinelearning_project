import cv2
import tensorflow as tf
import numpy as np

# Load the pretrained model
model = tf.keras.models.load_model("models/pretrained_model.h5")
classes = ["Normal", "Harassment"]

def predict(frame):
    """Predict class for a given frame."""
    resized = cv2.resize(frame, (224, 224))  # Resize to model input size
    normalized = resized / 255.0  # Normalize pixel values
    reshaped = np.reshape(normalized, (1, 224, 224, 3))  # Reshape for model
    predictions = model.predict(reshaped)
    return classes[np.argmax(predictions)], max(predictions[0])

def detect_from_camera():
    """Real-time harassment detection using PC Camera."""
    cap = cv2.VideoCapture(0)  # Open PC camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict harassment or normal
        label, confidence = predict(frame)
        color = (0, 255, 0) if label == "Normal" else (0, 0, 255)  # Green for Normal, Red for Harassment
        
        # Display the label and confidence on the frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Harassment Detection", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_camera()
