import os
import shutil



# 1.1) EXTRACT ZIP FILES
# Path to the zip file
zip_file_path = '/data/archive.zip'
# Directory to extract the zip file to
extract_dir = '/data/extracted_files'
# Extract the zip file
"""
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print(f"Files extracted to {extract_dir}")
"""

# Check if the files are already extracted (only do this step once)
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"Files extracted to {extract_dir}")
else:
    print(f"Files already extracted to {extract_dir}")


# 1.2) DELETING UNNECESSARY FILES
# Path to the folder where the unnecessary files are located
folder_path = "/data/extracted_files"
# List of files to delete (files you want to remove)
files_to_delete = [
    'hmnist_28_28_L.csv',
    'hmnist_28_28_RGB.csv',
    'hmnist_8_8_L.csv',
    'hmnist_8_8_RGB.csv'
]
# Loop through the list of files and delete them
for file_name in files_to_delete:
    file_path = os.path.join(folder_path, file_name) # This line is creating the full path to the file that needs to be deleted by joining the folder path and the filename.

    # Check if the file exists
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)
        print(f"File {file_name} has been deleted.")
    else:
        print(f"File {file_name} does not exist.")

# Counting number of pictures inside Image Files.
image_Folder1 = r"C:\Users\betus\Desktop\skin_cancer\skin_cancer_project\data\extracted_files\HAM10000_images_part_1"
image_Folder2 = r"C:\Users\betus\Desktop\skin_cancer\skin_cancer_project\data\extracted_files\HAM10000_images_part_2"

# Function to count image files in a folder
def count_images(folder_path, extensions=['.jpg', '.jpeg', '.png']):
    return sum(1 for filename in os.listdir(folder_path)
               if os.path.isfile(os.path.join(folder_path, filename)) and
               os.path.splitext(filename)[1].lower() in extensions)

# Count images
count1 = count_images(image_Folder1)
count2 = count_images(image_Folder2)

print(f"Number of images in Folder 1: {count1}")
print(f"Number of images in Folder 2: {count2}")
print(f"Total number of images: {count1 + count2}")

"""
# Source and destination folders
src_folder = image_Folder2
dest_folder = image_Folder1

# Move or copy each image file from src to dest
for filename in os.listdir(src_folder):
    src_file = os.path.join(src_folder, filename)
    dest_file = os.path.join(dest_folder, filename)

    # Only copy image files
    if os.path.isfile(src_file) and os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']:
        shutil.copy2(src_file, dest_file)  # Use shutil.move(src_file, dest_file) if you want to move instead of copy

print("Files successfully merged.")
print(f"Number of images in Folder 1: {count1}")"""