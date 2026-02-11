# The data_preprocessing file is where you will perform all the necessary steps to clean and prepare your dataset for training your model.

# STEP 1: IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from PIL import Image
pd.set_option('display.max_columns', None)

df = pd.read_csv(r"C:\Users\betus\Desktop\skin_cancer\skin_cancer_project\data\extracted_files\HAM10000_metadata.csv")
image_Folder = r"C:\Users\betus\Desktop\skin_cancer\skin_cancer_project\data\extracted_files\HAM10000_images_part_1"
df.head()

# Pick one image filename (you will get filenames from the metadata CSV)
filename = 'ISIC_0027419.jpg'  # Example

# Build full path (check if it's in part 1 or part 2)
image_path = os.path.join(image_Folder, filename)

# Load the image
img = Image.open(image_path)

# If you want, resize it (CNNs usually want same size images like 224x224)
img = img.resize((224, 224))

# If you want, convert it into numpy array (for model input later)
img_array = np.array(img)

print(image_path)
print(img_array.shape)  # Example: (224, 224, 3) -> RGB image

# Display the image using matplotlib
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show(block=True)

# Append .jpg to image IDs so they match file names:
df["filename"] = df["image_id"] + ".jpg"


# List of actual image filenames in the merged folder
image_files = set(os.listdir(image_Folder))  # set method converts the list into a set, which is a collection of unique items in Python.


# Add a column that shows whether the image file exists in the folder
df['file_exists'] = df['filename'].isin(image_files)

print("Files exist:", df['file_exists'].sum()) # Exist
print("Files missing:", (~df['file_exists']).sum()) # Not exist

# So all image files have their info in csv file.


import matplotlib.pyplot as plt
from PIL import Image
import os


# STEP 2:

df["dx"].value_counts()
