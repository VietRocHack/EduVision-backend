from pyngrok import ngrok, conf
from flask import Flask, request, jsonify
from yolo_head.detect import det_head
import cv2

# init yolov5

conf.get_default().auth_token = '2ePsRAhhGv5ishskgccKSakWQ5X_3LyjJKYy69UuoRthDPmpP'

# Open a TCP ngrok tunnel to the SSH server
connection_string = ngrok.connect("22", "tcp").public_url

ssh_url, port = connection_string.strip("tcp://").split(":")
print(f" * ngrok tunnel available, access with `ssh root@{ssh_url} -p{port}`")


app = Flask(__name__)
port = "5000"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url


# Define Flask routes
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # You can perform any processing on the video file here
    process(file)
    
    return jsonify({'message': 'Video processed successfully'}), 200

def process(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append()
    det_head(frame_list)

app.run()