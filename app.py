import tensorflow as tf
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_spaces, crop_from_space, capture_spaces, capture_img, load_pickle_file
import numpy as np

MODEL = tf.keras.models.load_model('model/Carnet_transfer.h5')
(cameraMatrix, dist) = load_pickle_file(r'./calibration.pkl')
app = FastAPI()

class UserInput(BaseModel):
    camera_id: int
    user_name: str
    password: str
    ip: str

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict/')
def predict(UserInput: UserInput) -> list:
    rstp_url = f"rtsp://{UserInput.user_name}:{UserInput.password}@{UserInput.ip}/video"
    print(rstp_url)
    CAP = cv2.VideoCapture(rstp_url)
    frame = capture_img(CAP, cameraMatrix, dist)
    # cv2.imwrite("image.jpg", frame)
    spaces_dir = f"spaces/{UserInput.camera_id}.txt"
    spaces = load_spaces(spaces_dir)
    
    cropped_imgs, test_list = capture_spaces(frame, spaces)
    # for i,cropped_img in enumerate(test_list):
    #     cv2.imwrite(f"./test_folder/{i}.jpg", cropped_img)
    predictions_array = MODEL.predict(cropped_imgs)
    prd_list = []
    for pred in predictions_array:
        prd_list.append(float(pred[0]))
    # print(prd_list)

    return prd_list