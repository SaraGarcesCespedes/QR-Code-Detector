from ultralytics import YOLO
import io
import base64
from zipfile import ZipFile
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyzbar import pyzbar


# Function to run inference on an image with the trained model
def Detection(image):

    # Load the model with the best performancde
    model = YOLO("models/best.pt")

    # Load image
    img = Image.open(image)

    # Preprocess image
    test_image = preprocess_image(img)

    # Prediction
    results = model(test_image) 

    # Extract coordinates
    table = Summary_results(results)
    table['QR Code'] = range(1, 1+len(table))

    # extract details of QR
    output = pyzbar.decode(test_image)

    if len(output) == 0:
        table['QR URL'] = ""
    else:
        for qr_code in output:
            # Extract the data from the QR code
            data = qr_code.data.decode('utf-8')
            # Check if the data is a valid URL
            if data.startswith('http://') or data.startswith('https://'):
                table['QR URL'] = data
    
    
    return table


# Function to create table with coordinates of detected QR
def Summary_results(model_results):

    df = pd.DataFrame(columns=['X1', 'Y1', 'X2', 'Y2'])

    for r in model_results:
        coordinates = r.boxes.xyxy.tolist() # Boxes object containing the detection bounding boxes

        #print(coordinates)
        if len(coordinates) == 0:
            print("No QR detected")
        else:
            df = pd.concat([df, pd.DataFrame(coordinates, columns=df.columns)], ignore_index=True)

    return df

# Function to preprocess image before inference
def preprocess_image(image):
    # Preprocess image
    rgb =  Image.new('RGB', image.size)
    rgb.paste(image)
    image = rgb
    test_image = image.resize((640,640))
    
    return test_image


# Function to crop image and extract detected QR code
def crop_image(image, table_coordinates):
    # Read the image
    table_coordinates = pd.DataFrame(table_coordinates)

    list_images = []


    if table_coordinates.shape[0] == 0:
        print("No QRs in image")
    else:
        for row in table_coordinates.itertuples(index=True, name='Pandas'):
            x1, y1, x2, y2 = int(row.X1), int(row.Y1), int(row.X2), int(row.Y2)
            print("ROI coordinates defined.")
            roi = image[y1:y2, x1:x2]
            list_images.append(roi)
        
        return list_images


    




