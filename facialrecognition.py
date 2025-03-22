import numpy as np
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import os

# Load the dataset
# Dataframes consisting of their respective collections of happy, sad, and surprised faces
# These images must first be mass converted to 2D arrays using OpenCV

trainingSet = pd.DataFrame()
testingSet = pd.DataFrame()

def convert_images_to_arrays(image_folder):
    image_list = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')): # Check if the file is an image
            img_path = os.path.join(image_folder, filename)
            img = cv.imread(img_path)
            if img is not None:
                image_list.append(img)
            else:
                print(f"Error reading image: {filename}")
    return np.array(image_list)

def create_new_image_array(image_folder):
#    image_folder = 'archiveDataset/train/happy'
    image_arrays = convert_images_to_arrays(image_folder)
    if image_arrays.size > 0:
        print(f"Successfully converted {len(image_arrays)} images to arrays.")
        #plt.imshow(image_arrays[0])
        #plt.title("Example Image")
        #plt.axis('off')
        #plt.show()
        #print(image_arrays[0].shape)
        # Further processing with image_arrays (e.g., saving to a file)
    else:
        print("No images were converted.")


while True:
    response = int(input("What would you like do?\n0: Quit\n1: Create new data set"
                         "\n2: save it \n3: use it \n"))
    match response:
        case 0:
            break
        case 1:
            create_new_image_array(input("file name\n"))
        case 2:
            image_arrays = convert_images_to_arrays(input("file name\n"))
            np.save("happy_faces.npy", image_arrays)
        case 3:
            saved_array = np.load("data/happy_faces.npy")
            plt.imshow(saved_array[0])
            plt.show()
