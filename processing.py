import pandas as pd
import numpy as np
import cv2
import os

lungOpacityImages = []
normalImages = []
viralPneumonia = []

def load_and_preprocess_images(directory, label_list):
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                grayscale = cv2.convertScaleAbs(grayScale, alpha=1, beta=0)
                
                resized_img = cv2.resize(grayscale, (128, 128))
                
                image = resized_img.flatten()/255.0
                
                label_list.append(image)

load_and_preprocess_images(r'I:\Code\lungs disease\Lung X-Ray Image\Lung_Opacity', lungOpacityImages)
load_and_preprocess_images(r'I:\Code\lungs disease\Lung X-Ray Image\Normal', normalImages)
load_and_preprocess_images(r'I:\Code\lungs disease\Lung X-Ray Image\Viral Pneumonia', viralPneumonia)


# Convert lists to DataFrames
lung_opacity_df = pd.DataFrame(lungOpacityImages)
normal_images_df = pd.DataFrame(normalImages)
viral_pneumonia_df = pd.DataFrame(viralPneumonia)

lung_opacity_df['label'] = 0
normal_images_df['label'] = 1
viral_pneumonia_df['label'] = 2

data = pd.concat([lung_opacity_df, normal_images_df, viral_pneumonia_df])

print(data.shape)
print(data.head())


# Count the number of classes for imbalance checking
class_counts = data['label'].value_counts()
print("Class distribution:\n", class_counts)

data.to_csv("data.csv", index=False)

# imagPath = os.path.join(r'I:\Code\lungs disease\Lung X-Ray Image\Viral Pneumonia', "1098.jpg")
# img = cv2.imread(imagPath)

# if img is not None:
    
#     grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     grayscale = cv2.convertScaleAbs(grayscale, alpha=1, beta=0)
    
#     cv2.imshow("Image", grayscale)
    
    
#     cv2.waitKey(0)
    
#     cv2.destroyAllWindows()