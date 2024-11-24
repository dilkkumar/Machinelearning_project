from flask import Flask, render_template, Response
import cv2
from src.real_time_detection import predict

app = Flask(__name__)

@app.route("/")
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Stream video with real-time detection."""
    def gen():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            label, confidence = predict(frame)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
