import sys
sys.path.insert(0,'D:/LearningHood/conda/mcgaze/MCGaze_demo/yolo_head')
import os

from pyngrok import ngrok, conf
from flask import Flask, request, jsonify
from yolo_head.detect_api import det_head
import cv2

# init yolov5

conf.get_default().auth_token = '2ePsRAhhGv5ishskgccKSakWQ5X_3LyjJKYy69UuoRthDPmpP'

# Open a TCP ngrok tunnel to the SSH server
# connection_string = ngrok.connect("22", "tcp").public_url

# ssh_url, port = connection_string.strip("tcp://").split(":")
# print(f" * ngrok tunnel available, access with `ssh root@{ssh_url} -p{port}`")


app = Flask(__name__)
port = "5000"

# Open a ngrok tunnel to the HTTP server
# public_url = ngrok.connect(port).public_url
# print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# Update any base URLs to use the public ngrok URL
# app.config["BASE_URL"] = public_url


# Define Flask routes
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    video_path = os.path.join('./', 'uploaded_video.mp4')
    file.save(video_path)
    # You can perform any processing on the video file here
    process(video_path)
    
    return jsonify({'message': 'Video processed successfully'}), 200


def delete_files_in_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # print(f"文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹中的所有文件和子文件夹
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            # 如果是文件，删除它
            os.remove(file_path)
            # print(f"del_file: {file_path}")
        elif os.path.isdir(file_path):
            # 如果是文件夹，递归删除它
            delete_files_in_folder(file_path)
    

def process(video_path):
  cap = cv2.VideoCapture(video_path)
  delete_files_in_folder("D:/LearningHood/conda/mcgaze/MCGaze_demo/result/labels/")
  delete_files_in_folder("D:/LearningHood/conda/mcgaze/MCGaze_demo/frames/")
  delete_files_in_folder("D:/LearningHood/conda/mcgaze/MCGaze_demo/new_frames/")
  frame_id = 0
  while   True:
      ret, frame = cap.read()
      if ret:
          cv2.imwrite('D:/LearningHood/conda/mcgaze/MCGaze_demo/frames/%d.jpg' % frame_id, frame)
          frame_id += 1
      else:
          break
      
  imgset = 'D:/LearningHood/conda/mcgaze/MCGaze_demo/frames/*.jpg'
  det_head(imgset)

app.run()