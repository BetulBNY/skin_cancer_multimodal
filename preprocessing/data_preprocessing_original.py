# The data_preprocessing file is where you will perform all the necessary steps to clean and prepare your dataset for training your model.

# STEP 1: IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from PIL import Image
from glob import glob
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm  #For döngüleri sırasında işlemin ilerleyişini görsel olarak gösterir (progress bar).
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from PIL import UnidentifiedImageError

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

# STEP 2: ABOUT DATASET
df = pd.read_csv(r"C:\Users\betus\Desktop\skin_cancer\skin_cancer_project\data\extracted_files\HAM10000_metadata.csv")
df.head()
df.info()
df.shape # (10015 row, 7 columns)

# STEP 3: DATA PREPROCESSING
# STEP 3.1: In this step create a dictionary for show more readable lesion labels.
df["dx"].value_counts()
lesion_type = {
    "nv" : "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df" : "Dermatofibroma"
}
df["lesion_diagnosis"] = df["dx"].map(lesion_type)

#///// Search about that and for later order them by their danger level.

# STEP 3.2: Add image file paths to dataframe.
# Find all .jpg files in a folder
image_pathss = glob(r"C:\Users\betus\Desktop\skin_cancer\skin_cancer_project\data\extracted_files\HAM10000_images_part_1\*.jpg")

# Create Dataframe from image paths
image_df = pd.DataFrame({
    'image_id': [os.path.splitext(os.path.basename(p))[0] for p in image_pathss],
    'path': image_pathss
})

# Merge metadata with image paths
skin_df = df.merge(image_df, on='image_id', how='left')

# Resimlerin boyutu ne?
for path in image_pathss:
    if isinstance(path, str) and os.path.exists(path):
        with Image.open(path) as img:
            print(f"{path} boyutu: {img.size}")
    else:
        print(f"Geçersiz veya bulunamayan dosya yolu: {path}")

# STEP 4: DATAFRAME OVERVIEW
print("//////Head//////\n" +skin_df.head().to_string())
print("//////Shape//////\n" + str(skin_df.shape))
print("////// Missing Values //////\n" + skin_df.isnull().sum().to_string())
print("//////Data Types//////\n" +skin_df.dtypes.to_string())
print("//////Unique Values//////\n" +skin_df.nunique().to_string()) # From this output we can say almost all data is categorical

# STEP 5: DATA CLEANING
# Filling null values inside age columns with average value.
skin_df["age"].fillna((skin_df["age"].mean()), inplace= True)
skin_df["age"].isnull().sum()

# STEP 6: EDA (EXPLORATORY DATA ANALYSIS)
# In this step we'll explore dataset and show distributions and relationships.

# AGE: (Distribution by Age) From diagram we can say between 40-60 years is dangerious for this disease. We will categorize ages, then encdoing.
plt.figure(figsize=(10,5))
sns.histplot(skin_df['age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show(block=True)

# GENDER: (Distribution by Gender) There is no noticeable difference
sex_counts = skin_df['sex'].value_counts()
labels = sex_counts.index
sizes = sex_counts.values
colors = ['lightblue','pink',"purple"]  # assuming labels are like ['male', 'female']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', startangle=140)
plt.title("Gender Distribution")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show(block=True)

# Count of Patients by Age and Gender:
plt.figure(figsize=(10, 5))
for gender, color in zip(['male', 'female'], ['lightblue', 'pink']):
    sns.kdeplot(skin_df[skin_df['sex'] == gender]['age'].dropna(), label=gender, color=color, fill=True)
plt.title('Age Density by Sex')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show(block=True)

# Count of Patients by Age and Gender: We can say untile age 45 women are higher then man and after that man are higher than women there is no toom uch differecne but it seems.
plt.figure(figsize=(12,6))
sns.countplot(data=skin_df, x='age', hue='sex', palette={'male': 'lightblue', 'female': 'pink', 'unknown':'purple'})
plt.title("Count of Patients by Age and Gender")
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show(block=True)
print(skin_df['sex'].unique())
skin_df["sex"].value_counts()
skin_df["age"].value_counts()

# LESIONS: (Distribution of Lesion Types) We can say most common one is "Melanocytic nevi"
plt.figure(figsize=(10,5))
sns.countplot(data=skin_df, x='lesion_diagnosis', order=skin_df['lesion_diagnosis'].value_counts().index, palette='Set2')
plt.title("Distribution of Lesion Types")
plt.xticks(rotation=45)
plt.show(block=True)

# LESION TYPES BY GENDER: No noticable difference almost same.
plt.figure(figsize=(12,6))
sns.countplot(data=skin_df, x='lesion_diagnosis', hue='sex')
plt.title("Lesion Types by Gender")
plt.xticks(rotation=45)
plt.show(block=True)

# LOCALIZATION : We can say watch your back
plt.figure(figsize=(12, 6))
sns.countplot(data=skin_df,y='localization',order=skin_df['localization'].value_counts().index,palette='Set3')
plt.title("Lesion Localization Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Count")
plt.ylabel("Localization")
plt.tight_layout()
plt.show(block=True)

# Lesion Types by Localization: Bunu bi incele
plt.figure(figsize=(12,6))
sns.countplot(data=skin_df, y='localization', hue='lesion_diagnosis', order=skin_df['localization'].value_counts().index)
plt.title("Lesion Localization by Type")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show(block=True)

skin_df["localization"].value_counts()
# Localization reference:
# back - Upper and lower back
# lower extremity - Legs (thighs, knees, calves, feet)
# trunk - Central torso (chest, abdomen, back)
# upper extremity - Arms (shoulders, upper arms, forearms)
# abdomen - Stomach area (below chest, above pelvis)
# face - Forehead, cheeks, nose, etc.
# chest - Upper torso (above abdomen, includes breasts)
# foot - Sole, top, toes of the foot
# unknown - Not labeled or not identifiable
# neck - Front and back of neck
# scalp - Top of the head under hair
# hand - Palm, back of hand, fingers
# ear - External ear
# genital - Genital region
# acral - Palms of hands or soles of feet (acral areas)

skin_df.head()
# DIAGNOSIS IMAGES
def show_images_by_type(df, lesion_types, n=5):
    rows = len(lesion_types)
    plt.figure(figsize=(n * 2, rows * 2))
    for row_idx, lesion_type in enumerate(lesion_types):
        subset = df[df['lesion_diagnosis'] == lesion_type].sample(n)
        for col_idx, (_, row) in enumerate(subset.iterrows()):
            plt_idx = row_idx * n + col_idx + 1
            img = Image.open(row['path'])
            plt.subplot(rows, n, plt_idx)
            plt.imshow(img)
            plt.axis('off')
            plt.title(lesion_type, fontsize=8)
    plt.tight_layout()
    plt.show(block=True)
show_images_by_type(skin_df, ['Benign keratosis', 'Melanocytic nevi', 'Vascular lesions','Dermatofibroma'], n=5)
show_images_by_type(skin_df, ['Melanoma', 'Basal cell carcinoma', 'Actinic keratoses'], n=5)

# DX_TYPE: Types of diagnosis methods
skin_df["dx_type"].value_counts()
plt.figure(figsize=(8,5))
sns.countplot(data=skin_df, x='dx_type', order=['histo', 'follow_up', 'consensus', 'confocal'], palette='Set2')
plt.title("Diagnosis Confirmation Methods")
plt.xlabel("Diagnosis Type")
plt.ylabel("Count")
plt.show(block=True)

# dx_type descriptions (ordered by diagnostic reliability)

# histo: Histopathology – The most accurate and reliable method.
#        A biopsy sample is taken and examined under a microscope by a pathologist.

# follow_up: Follow-up observation – The diagnosis is confirmed by monitoring lesion changes over time,
#            without invasive procedures. Common in benign cases.

# consensus: Expert consensus – Multiple experienced dermatologists agree on a diagnosis
#            based on images and clinical data. No biopsy involved.

# confocal: Reflectance confocal microscopy – A non-invasive imaging method that captures skin layers in detail.
#           Less common and less widely used, but still useful in specific cases.

# STEP 7: STATIC FEATURE ENGINEERING:
# 2 farklı feature eng işlemimiz olacak. Bunlardan bir kısmı train-test splitten önce bir kısmı sonra uygulanacak.
# df['age_group'] = pd.cut(df['age'], ...) gibi sabit kurallara dayalı işlemleri splitten önce de yapabilirsin, çünkü veri öğrenmiyorsun.
#Ancak LabelEncoder.fit_transform() ya da scaler.fit() gibi işlemleri train setinde fit edip, sonra test setine uygularsın.


# AGE BINING: Categorizing ages.
bins = [0, 20, 40, 60, 100]
labels = ['0-20', '21-40', '41-60', '61+']
skin_df['age_group'] = pd.cut(skin_df['age'], bins=bins, labels=labels, right=False)

def create_age_group_column(df, age_col='age', new_col='age_group'):
    bins = [0, 20, 40, 60, 100]
    labels = ['0-20', '21-40', '41-60', '61+']
    df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    return df


# SEPARATING HEALTHY OR CANCEROUS:
# I searched about which ones are benign/healthy which ones are malignant/cancerous then I will classify them.
# For this classification I will use binary label mapping.

#   Benign (Non-Cancerous / Healthy-like)
# - Melanocytic nevi: common mole
# - Benign keratosis-like lesions: includes seborrheic keratoses, solar lentigines
# - Vascular lesions: like angiomas, harmless blood vessel clusters
# - Dermatofibroma: benign skin nodules

#   Malignant (Cancerous / Dangerous)
# - Melanoma: most dangerous form of skin cancer
# - Basal cell carcinoma (BCC): most common skin cancer, grows slowly
# - Actinic keratosis: considered precancerous, can evolve into squamous cell carcinoma

skin_df["lesion_diagnosis"].value_counts()

cancerous = ['Melanoma', 'Basal cell carcinoma', 'Actinic keratoses']
skin_df['lesion_risk'] = skin_df['lesion_diagnosis'].apply(
    lambda x: 'cancerous' if x in cancerous else 'benign')

skin_df.head()
skin_df["lesion_risk"].value_counts() # unbalanced -->  benign: 8061, cancerous: 1954


# STEP 8: TRAIN-TEST SPLIT
# Train-test setlerinin veri çerçevesi (DataFrame) düzeyinde ayrılması
# Stratified split ile 'lesion_risk' değişkenindeki sınıf oranları korunarak veri seti %80 eğitim, %20 test olarak bölünüyor

train_df, test_df = train_test_split(
    skin_df,
    test_size=0.2,
    stratify=skin_df["lesion_risk"],  # Hedef değişkene göre orantılı ayırma
    random_state=42
)
test_df["lesion_risk"].value_counts()
print(f"Train set boyutu: {train_df.shape}")
print(f"Test set boyutu: {test_df.shape}")

# NOT: Model eğitimi, dynamic feature engineering ve augmentation train_df'te olacak.
# "Stratified" (katmanlı) terimi, özellikle train-test split işlemlerinde kullanılır ve veri setindeki sınıf dengesini korumak anlamına gelir.
# Eğer sınıflar arasında dengesizlik varsa (örneğin, bazı sınıflar çok fazla, bazıları çok az örnek içeriyorsa), Stratified split, her sınıfın oranını eğitim ve test setinde aynı tutmaya çalışır.
# Eğer Stratify kullanmazsanız, train_test_split rastgele bölebilir ve test setinde "cancerious"" sınıfı neredeyse hiç kalmayabilir.
# Ama stratify=y derseniz, eğitim ve test setinde cancerious ve benign sınıfları oransal olarak aynı kalır.

# STEP 9: DATA AUGMENTATION
# Burada amaç az olan Cancerous verilerini çoğaltarak veri setini daha dengeli hale getirmek. Ancak bu veri ile ilgili şöyle bir problem var hem vsc hem de
# görüntü verisi içerdiği için tek bir metod ile iki veri türünü de çoğaltmak zor bi işlem. Görseller için ve csv için ayrı ayrı işlem uygulanır genellikle.
# Ancak ben burada az olan "cancerous" sınıfına ait görüntüleri çoğaltacağım. Ama CSV'deki özellikleri değiştirmeyeceğim
# sadece aynı satırı yeniden kopyalayacağım (çünkü augmentation sadece görüntü üzerinde olacak).

# PROBLEM 1: Sadece Kanserli (Malignant) Görüntüleri Artırmak İçin En Güvenli Augmentation Yöntemleri (Tıbbi görüntülerde augmentasyon yaparken dikkatli olunmalı. Çünkü bazı işlemler verinin semantik yapısını bozarak modelin yanlış öğrenmesine sebep olabilir.)
# Deri kanseri görüntüleri üzerinde lezyonun morfolojik özelliklerini (şekil, renk, doku, sınır düzensizliği) bozmadan augmentation yapmak için kullanacağım yöntemler:


"""
Uygun yöntemler:
| Yöntem                                   | Açıklama                          | Neden Uygun?                                                                  |
| ---------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------- |
| **Yatay çevirme (Horizontal Flip)**      | Görüntüyü sağ-sol olarak çevirir. | Cilt lezyonlarında çoğu zaman yön bilgisi önemli değildir.                    |
| **Dönme (Rotate - küçük açılar)**        | 0° ile ±15° arasında döndürme.    | Kamera açısındaki küçük farkları simüle eder, klinik olarak kabul edilebilir. |
| **Parlaklık / Kontrast değişimi**        | Işık koşullarını değiştirir.      | Farklı ışıkta çekilmiş görüntüleri taklit eder. Gerçekte de karşılaşılabilir. |
| **Zoom (hafif yakınlaştırma/kırpma)**    | Küçük kırpma işlemleri.           | Görüntüdeki konum değişikliklerini simüle eder.                               |
| **Gaussian noise (çok düşük düzeyde)**   | Az miktarda gürültü ekler.        | Görüntü kalitesindeki bozulmaları simüle eder, ancak abartılmamalıdır.        |
| **Color jitter (dikkatli kullanılırsa)** | Renklerde küçük değişiklikler.    | Kameralar arası renk farklarını öğrenmede yardımcı olabilir.                  |

Uygun Olmayan Yöntemler:
| Yöntem                                  | Neden Kaçınmalı?                                                        |
| --------------------------------------- | ----------------------------------------------------------------------- |
| **Dikey çevirme (Vertical Flip)**       | Cilt lezyonlarında üst-alt yön bazen önemlidir. Anlam bozulabilir.      |
| **Büyük dönüşler (90°, 180°)**          | Görüntü yapısını bozar, model alakasız şeyler öğrenebilir.              |
| **Elastic transform / grid distortion** | Doku yapısını yapay olarak bozar. Tıbbi görüntülerde gerçekçi değildir. |
| **Random crop (aşırı kırpma)**          | Lezyonun tamamı görünmeyebilir, tanı bozulur.                           |
| **Strong blur / sharpen**               | Görüntünün kalitesini gereksiz şekilde değiştirir.                      |

Kısaca:
"Gerçek dünyada karşılaşılabilecek varyasyonları simüle eden augmentasyonlar kullanılmalı, lezyonun yapısını bozanlardan kaçınılmalıdır."
"""


# PROBLEM 2: Ancak bu işlemden önce train-test split yapmak gerekiy çünkü Test verisi üzerinde augmentation uygulanmaz.
# Augmentation işlemi her zaman Train-Test Split'ten SONRA yapılmalıdır ama sadece eğitim (train) seti üzerinde uygulanmalıdır.
# Augmentation işlemi sadece train_df içerisindeki cancerous (malignant) sınıfına uygulanacaktır.

"""
| Neden                                           | Açıklama                                                                                                                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Modelin genelleme kapasitesi gelişsin diye**  | Eğitim verisini çeşitlendiririz, ama test verisi *gerçek dünyayı* temsil eder.                                                                                    |
| **Veri sızıntısı (data leakage) önlensin diye** | Eğer augmentasyonu *bölmeden önce* yaparsak, aynı görüntünün farklı versiyonları hem train hem testte olabilir. Bu da test performansını yapay şekilde yükseltir. |
| **Doğruluk değerlendirmesi adil olur**          | Test seti, eğitim sürecinde hiç görülmemiş verilerden oluşmalı.                                                                                                   |
"""
#Bu yüzden step 6 da train-test split işlemi yapıldı

# Kanserli (cancerous) sınıfa ait satırları al
cancer_df = train_df[train_df["lesion_risk"] == "cancerous"]

# Augmentation sonucu kaydedilecek klasör
augmented_dir = "augmented_images"
os.makedirs(augmented_dir, exist_ok=True)


# Görsel augmentasyon için ImageDataGenerator (yalnızca eğitim için!)
augmenter = ImageDataGenerator(
    rotation_range=15,              # Maksimum 15 derece döndürme.
    width_shift_range=0.05,         # Genişlikte %5'e kadar kaydırma.
    height_shift_range=0.05,        # Yükseklikte %5'e kadar kaydırma.
    brightness_range=[0.9, 1.1],    # Parlaklıkta değişiklik (%10).
    zoom_range=0.1,                 # Yaklaştırma (zoom).
    horizontal_flip=True,           # Yatay çevirme.
    fill_mode='nearest'             # Yeni pikselleri en yakın komşuyla doldur.
)

# Yeni verileri saklamak için liste
augmented_rows = []
n_copies = 2  #  For each original cancerous image, create n_copies new ones



# Ensure the augmented_dir exists
augmented_dir = "augmented_images" # Relative to your CWD
os.makedirs(augmented_dir, exist_ok=True)
print(f"Augmented images will be saved in: {os.path.abspath(augmented_dir)}")



for index, row in tqdm(cancer_df.iterrows(), total=cancer_df.shape[0], desc="Augmenting Cancerous Images"):
    original_image_path = row["path"]

    if not (isinstance(original_image_path, str) and os.path.exists(original_image_path)):
        print(f"Skipping invalid or non-existent original image path: {original_image_path}")
        continue

    try:
        image = load_img(original_image_path, target_size=(224, 224))
    except (FileNotFoundError, UnidentifiedImageError, Exception) as e: # Broader exception catch
        print(f"Error loading original image: {original_image_path} — {e}")
        continue

    x = img_to_array(image)
    x = x.reshape((1,) + x.shape) # Reshape to (1, height, width, channels)

    original_basename = os.path.basename(original_image_path)
    original_filename_wo_ext = os.path.splitext(original_basename)[0]

    generated_count = 0
    # Do not use save_to_dir here. We will save manually.
    for batch_img_array in augmenter.flow(x, batch_size=1):
        # batch_img_array is a NumPy array of shape (1, height, width, channels)
        augmented_image_data = batch_img_array[0] # Get the image data (height, width, channels)

        # Construct your desired new filename
        new_filename = f"{original_filename_wo_ext}_aug_{generated_count}.jpg"
        # Path where the new augmented image will be saved
        # This path will be relative to CWD, e.g., "augmented_images/ISIC_xxxx_aug_0.jpg"
        new_image_save_path = os.path.join(augmented_dir, new_filename)

        # Save the augmented image manually
        try:
            tf.keras.preprocessing.image.save_img(new_image_save_path, augmented_image_data)
        except Exception as e:
            print(f"Error saving augmented image {new_image_save_path}: {e}")
            # Decide if you want to skip adding this row or handle differently
            generated_count += 1 # Ensure loop progresses
            if generated_count >= n_copies:
                break
            continue


        # Create a new row for the DataFrame
        new_row = row.copy()
        # Store this relative path in the DataFrame.
        # It will be converted to absolute later if needed, or handled by a base_path.
        new_row["path"] = new_image_save_path
        augmented_rows.append(new_row)

        generated_count += 1
        if generated_count >= n_copies:
            break # Exit the inner loop once n_copies are generated

# Continue with creating augmented_df and combining:
augmented_df = pd.DataFrame(augmented_rows)
if not augmented_df.empty:
    print("Sample paths in augmented_df AFTER augmentation loop:")
    print(augmented_df["path"].head())
else:
    print("augmented_df is empty after augmentation loop.")

# The rest of your script for combining DFs and path transformations can follow.
# The key is that augmented_df["path"] now points to actual files you saved.
# For example, the part where you make paths absolute:

if not augmented_df.empty:
    augmented_df["path"] = augmented_df["path"].apply(lambda p: os.path.abspath(p))
    # Normalize slashes for consistency if desired (os.path.abspath usually does this well)
    augmented_df["path"] = augmented_df["path"].apply(lambda p: p.replace("\\", "/"))
    print("Sample paths in augmented_df AFTER making them absolute:")
    print(augmented_df["path"].head())



# Yeni satırları orijinal DataFrame'e ekle
augmented_df = pd.DataFrame(augmented_rows)
augmented_df.head()
augmented_df["lesion_risk"].value_counts()
combined_df = pd.concat([train_df, augmented_df], ignore_index=True)
combined_df.shape
combined_df["lesion_risk"].value_counts()

# CSV olarak kaydet (isteğe bağlı)
#combined_df.to_csv("train_augmented.csv", index=False) #/////////////////////////////////


# Augment edilmiş görsellerin tipine bakma
# Her sınıf için 3 görsel göster
unique_classes = augmented_df['lesion_diagnosis'].unique()

for lesion_type in unique_classes:
    sample_images = augmented_df[augmented_df['lesion_diagnosis'] == lesion_type].head(3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Lesion Type: {lesion_type}", fontsize=14)

    for i, (idx, row) in enumerate(sample_images.iterrows()):
        img = load_img(row['path'])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(os.path.basename(row['path']), fontsize=8)

    plt.tight_layout()
    plt.show(block=True)

"""
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img # Ensure this is imported
import os

print(f"Current Working Directory: {os.getcwd()}")
print("Verifying paths in augmented_df before plotting:")
if not augmented_df.empty:
    print(augmented_df['path'].head())
    # Check if a sample path exists
    sample_path_to_check = augmented_df['path'].iloc[0]
    print(f"Checking existence of a sample path: {sample_path_to_check} -> Exists? {os.path.exists(sample_path_to_check)}")


unique_classes = augmented_df['lesion_diagnosis'].unique()
for lesion_type in unique_classes:
    sample_images = augmented_df[augmented_df['lesion_diagnosis'] == lesion_type].head(3)
    if sample_images.empty:
        print(f"No images found for lesion type: {lesion_type} in augmented_df")
        continue

    # Adjust the number of subplots based on available samples
    num_samples_to_plot = min(3, len(sample_images))
    if num_samples_to_plot == 0:
        continue
    fig, axes = plt.subplots(1, num_samples_to_plot, figsize=(4 * num_samples_to_plot, 4))
    if num_samples_to_plot == 1: # Make axes iterable if only one subplot
        axes = [axes]
    fig.suptitle(f"Lesion Type: {lesion_type}", fontsize=14)

    for i, (idx, row) in enumerate(sample_images.iterrows()):
        # row['path'] should already be the absolute path after your transformations
        img_path = row['path']

        try:
            # --- Debugging ---
            # print(f"Attempting to load image: {img_path}")
            # print(f"Does path exist? {os.path.exists(img_path)}")
            # ---------------
            img = load_img(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(os.path.basename(img_path), fontsize=8)
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            print(f"  (Absolute path checked: {os.path.abspath(img_path)})")
            print(f"  Please verify this path and file name are correct and the file exists.")
            if i < len(axes):
                 axes[i].axis('off')
                 axes[i].set_title("File Not Found", fontsize=8)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if i < len(axes):
                 axes[i].axis('off')
                 axes[i].set_title("Error Loading", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
    plt.show(block=True)


"""

# Check for NaN values.
print(f"Shape of combinedTraindf before handling NaN in 'path': {combined_df.shape}")
print(f"Number of NaNs in 'path' before: {combined_df['path'].isnull().sum()}")

# Identify the row(s) with NaN path
nan_path_rows = combined_df[combined_df['path'].isnull()]
print("\nRows with NaN in 'path' column:")
print(nan_path_rows)

# Drop rows where 'path' is NaN
combined_df.dropna(subset=['path'], inplace=True)

print(f"\nShape of combinedTraindf after handling NaN in 'path': {combined_df.shape}")
print(f"Number of NaNs in 'path' after: {combined_df['path'].isnull().sum()}")



# STEP 10: DYNAMIC FEATURE ENGINEERING :

# AGE-LOCALIZATION RELATIONSHIP
# Copy combined_df to combinedTraindf
combinedTraindf = combined_df.copy()
combinedTraindf.head()


# 1. Calculate the mean age for each localization from the TRAINING data (combinedTraindf)
def apply_loc_mean_age(train_df, test_df, age_col='age', group_col='localization', new_col='loc_mean_age'):
    """
    Eğitim verisinden 'localization' gruplarına göre ortalama yaş hesaplar ve hem eğitim hem test verisine uygular.

    Args:
        train_df (pd.DataFrame): Eğitim verisi
        test_df (pd.DataFrame): Test verisi
        age_col (str): Yaşın bulunduğu sütun adı
        group_col (str): Gruplama yapılacak sütun
        new_col (str): Eklenecek yeni sütunun adı (default: 'loc_mean_age')

    Returns:
        tuple: (train_df, test_df, mean_age_map sözlüğü)
    """
    mean_age_map = train_df.groupby(group_col)[age_col].mean().to_dict()

    # Uygula
    train_df[new_col] = train_df[group_col].map(mean_age_map)
    test_df[new_col] = test_df[group_col].map(mean_age_map)

    return train_df, test_df, mean_age_map


combinedTraindf, test_df, mean_age_map_train = apply_loc_mean_age(combinedTraindf, test_df)


combinedTraindf.isnull().sum()


# "Age Deviation from Localization Mean" Feature
def age_dev_from_loc_mean(traindf, testdf, selectedCol1 = 'age', selectedCol2 = 'loc_mean_age', new_col='age_dev_from_loc_mean'):
    traindf[new_col] = traindf[selectedCol1] - traindf[selectedCol2]
    testdf[new_col] = testdf[selectedCol1] - testdf[selectedCol2]
    return traindf, testdf
combinedTraindf, test_df = age_dev_from_loc_mean(combinedTraindf, test_df, )


print("\ncombinedTraindf head with 'age_dev_from_loc_mean':")
print(combinedTraindf[['localization', 'age', 'loc_mean_age', 'age_dev_from_loc_mean']].head())
print("\ntest_df head with 'age_dev_from_loc_mean':")
print(test_df[['localization', 'age', 'loc_mean_age', 'age_dev_from_loc_mean']].head())


"""
| Diagnosis                | False (Younger Loc.) | True (Older Loc.) | Insight                                                               |
| ------------------------ | -------------------- | ----------------- | --------------------------------------------------------------------- |
| **Melanoma**             | 685                  | 428               | Higher absolute count in younger loc, but relative risk may differ    |
| **Actinic keratoses**    | 116                  | **211**           | **More common in older localizations** — expected due to sun exposure |
| **Basal cell carcinoma** | **284**              | 230               | Still quite common in both, slightly higher in younger loc            |

| Diagnosis            | False (Younger Loc.) | True (Older Loc.) | Insight                                                       |
| -------------------- | -------------------- | ----------------- | ------------------------------------------------------------- |
| **Melanocytic nevi** | **5624**             | 1081              | **Much more frequent in younger localizations** (as expected) |
| **Benign keratosis** | 537                  | 562               | Roughly balanced                                              |
| **Dermatofibroma**   | **91**               | 24                | Mostly in younger locs                                        |
| **Vascular lesions** | **112**              | 30                | Also skewed younger                                           |

What This Tells Us:
Actinic keratoses are notably more common in older localizations → aligns with known medical knowledge.
Melanocytic nevi, the most common (and typically benign) type, is much more frequent in younger body regions.
The difference in melanoma counts is interesting. Even though it’s higher in younger localizations, it might be due to the base rate (many more lesions overall in younger regions).

"""

# Mean age per risk & localization

"""
| Localization | Mean Age (Benign) | Mean Age (Cancerous) | Difference       |
| ------------ | ----------------- | -------------------- | ---------------- |
| **Back**     | 49.6              | 61.6                 | +12 years        |
| **Chest**    | 49.1              | 63.0                 | +14 years        |
| **Face**     | 58.0              | 66.6                 | +8.6 years       |
| **Scalp**    | 53.1              | 72.1                 | **+19 years** ⚠️ |
| **Foot**     | 42.3              | 66.6                 | **+24 years** ⚠️ |
| **Hand**     | 44.5              | 65.0                 | +20.5 years      |
| **Neck**     | 48.8              | 64.0                 | +15.2 years      |
"""

# STEP 11: CREATING TRAIN AND VALIDATION SETS FROM "combinedTraindf"
combinedTraindf.head()

full_aug_train_df = combinedTraindf.copy()

"""Bu KISIM HAKKIDNA BİR BİLGİM YoK SİLERSİN SONRA OLMADI"""
# Define paths and target from full_aug_train_df before splitting
"""
all_training_image_paths = full_aug_train_df['path'].values
all_training_tabular_features = full_aug_train_df.drop(columns=['path', 'lesion_risk', 'lesion_id', 'image_id', 'dx', 'lesion_diagnosis']) # Adjust columns to drop
all_training_labels = full_aug_train_df['lesion_risk'].values # Assuming it's not yet label encoded
"""
# Split full_aug_train_df into actual training and validation sets
# For example, 80% of full_aug_train_df for training, 20% for validation
# Stratify by 'lesion_risk' to maintain class proportions

# It's often easier to split indices or the DataFrame itself, then separate paths, tabular, and labels
final_train_df, val_df = train_test_split(
    full_aug_train_df,
    test_size=0.20, # e.g., 20% of the augmented training data for validation
    stratify=full_aug_train_df['lesion_risk'], # Stratify by your target
    random_state=42
)

print(f"Shape of final_train_df: {final_train_df.shape}")
print(f"Shape of val_df: {val_df.shape}")
print(f"Shape of test_df: {test_df.shape}") # From earlier split

# Now, you'll apply preprocessing (encoding, scaling)
# Fit on final_train_df
# Transform final_train_df, val_df, and test_df


# STEP 12: ENCODING

# STEP 12.1) Label Encoding:
# --- Encode Target Variable ---
# It's crucial that the same LabelEncoder instance is used or that the mapping is consistent.
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
TARGET_COL = 'lesion_risk'

# Use LabelEncoder directly since you've already imported it
label_encoder = LabelEncoder()
final_train_df[TARGET_COL] = label_encoder.fit_transform(final_train_df[TARGET_COL])
val_df[TARGET_COL] = label_encoder.transform(val_df[TARGET_COL])
test_df[TARGET_COL] = label_encoder.transform(test_df[TARGET_COL])


# --- Separate Image Paths, Tabular Features, and Labels for each set ---
# Separate Image Paths and Labels
image_train_paths = final_train_df['path'].values
y_train = final_train_df[TARGET_COL].values

image_val_paths = val_df['path'].values
y_val = val_df[TARGET_COL].values

image_test_paths = test_df['path'].values # from the initial split
y_test = test_df[TARGET_COL].values       # from the initial split, now encoded


# Tabular data processing
combinedTraindf.head()
combinedTraindf.nunique().sum

combinedTraindf.columns

# Why we drop 'dx','lesion_diagnosis': If you're predicting lesion_risk, and lesion_risk is derived directly from lesion_diagnosis
# (which is derived from dx), then including dx or lesion_diagnosis directly as input features when predicting lesion_risk is a form of
# data leakage or, at best, extreme multicollinearity. The model would essentially be told "if dx is 'mel', then lesion_risk is 'cancerous'".
categorical_features = ['sex', 'localization', 'age_group']
numerical_features = ['age', 'loc_mean_age', 'age_dev_from_loc_mean']

# Create copies for tabular processing
X_tabular_train_raw = final_train_df[numerical_features + categorical_features].copy()
X_tabular_val_raw = val_df[numerical_features + categorical_features].copy()
X_tabular_test_raw = test_df[numerical_features + categorical_features].copy()

# --- Preprocessing Pipelines (same as before) ---
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# --- Fit and Transform ---
# FIT the preprocessor ONLY on the FINAL training tabular data
print("Fitting preprocessor on X_tabular_train_raw...")
preprocessor.fit(X_tabular_train_raw)

import pickle
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)


# TRANSFORM all three sets using the FITTED preprocessor
X_train_tab_processed = preprocessor.transform(X_tabular_train_raw)
X_val_tab_processed = preprocessor.transform(X_tabular_val_raw)
X_test_tab_processed = preprocessor.transform(X_tabular_test_raw)

print(f"Shape of processed training tabular data: {X_train_tab_processed.shape}")
print(f"Shape of processed validation tabular data: {X_val_tab_processed.shape}")
print(f"Shape of processed test tabular data: {X_test_tab_processed.shape}")

# Now we have:
# Train: image_train_paths, X_train_tab_processed, y_train
# Val:   image_val_paths, X_val_tab_processed, y_val
# Test:  image_test_paths, X_test_tab_processed, y_test



# STEP 13: Define an Image Loading and Preprocessing Function before modeling

IMG_HEIGHT = 224 # Or your desired height
IMG_WIDTH = 224  # Or your desired width
IMG_CHANNELS = 3 # For RGB images

def load_and_preprocess_image(path):
    try:
        # Load image
        img = tf.io.read_file(path)  # Verilen dosya yolundaki görüntü dosyasını bayt (byte) biçiminde okur.
        # Decode JPEG (or PNG, etc., as appropriate)
        img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS) #JPEG (veya PNG olabilir) biçimindeki görüntü dosyasını çözerek bir tensöre (çok boyutlu dizi) dönüştürür. channels=3 sayesinde renkli görüntü (RGB) olarak yorumlanır.
        # Resize
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]) # Görüntüyü belirlediğimiz sabit boyutlara getirir (224x224). Bu, tüm görüntülerin aynı boyutta olmasını sağlar ki modelin eğitilmesi için gereklidir.
        # Normalize pixel values to [0, 1]
        img = img / 255.0  # Piksel değerleri genelde 0–255 arasındadır. Bunu 0–1 aralığına çekmek modelin daha hızlı ve stabil öğrenmesini sağlar.
        return img
    except Exception as e:
        print(f"Error loading or preprocessing image {path}: {e}")
        # Return a placeholder or raise error, depending on how you want to handle errors
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)) # Example placeholder

# Görselin önceki halinin
oldImagePath = image_train_paths[2]
raw_img = tf.io.read_file(oldImagePath)
decoded_img = tf.image.decode_jpeg(raw_img, channels=3)
print("Shape of sample before preprocession:", decoded_img)  #shape=(450, 600, 3)
plt.imshow(decoded_img)
plt.show(block = True)
# 1. Görseli orijinal haliyle oku ve çöz
sample_img_tensor = load_and_preprocess_image(image_train_paths[2])
print("Shape of sample after preprocession:", sample_img_tensor.shape)
plt.imshow(sample_img_tensor)
plt.show(block = True)


# STEP 14: Create Data Generators (or tf.data.Dataset Pipeline)

BATCH_SIZE = 32 # Adjust as needed based on your GPU memory
def create_multimodal_dataset(image_paths, tabular_data, labels, batch_size, shuffle=True, augment_fn=None):
    # Create a dataset of image paths
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

    # Map image loading and preprocessing
    # For training, you might add data augmentation here
    if augment_fn:
        image_ds = path_ds.map(lambda x: (load_and_preprocess_image(x), augment_fn(load_and_preprocess_image(x))),
                               num_parallel_calls=tf.data.AUTOTUNE)
        # Unpack the original and augmented (or just pass one if augmentation is in load_and_preprocess_image)
        # This example assumes augment_fn is separate; often augmentation is part of image loading for simplicity
        # For now, let's assume load_and_preprocess_image is sufficient for basic loading

        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    else:
        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)


    # Create a dataset of tabular features
    tabular_ds = tf.data.Dataset.from_tensor_slices(tabular_data.astype(np.float32)) # Ensure float32

    # Create a dataset of labels
    label_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.int32)) # Ensure int32 for categorical_crossentropy or binary_crossentropy

    # Zip the datasets together: ((image, tabular), label)
    # The model will expect inputs as a list or dictionary. Here, a list [image_input, tabular_input]
    dataset = tf.data.Dataset.zip(((image_ds, tabular_ds), label_ds))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths)) # Shuffle the data

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # Optimize performance

    return dataset

# Create the datasets for train, validation, and test
# For training, you might later incorporate an augmentation function
train_dataset = create_multimodal_dataset(image_train_paths, X_train_tab_processed, y_train, BATCH_SIZE, shuffle=True)
val_dataset = create_multimodal_dataset(image_val_paths, X_val_tab_processed, y_val, BATCH_SIZE, shuffle=False)
test_dataset = create_multimodal_dataset(image_test_paths, X_test_tab_processed, y_test, BATCH_SIZE, shuffle=False)

# You can inspect a batch:
# for (images, tabular_feats), labels_batch in train_dataset.take(1):
#     print("Images batch shape:", images.shape)
#     print("Tabular features batch shape:", tabular_feats.shape)
#     print("Labels batch shape:", labels_batch.shape)
#     break


# STEP 15: Build the Multimodal Neural Network Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0 # Or ResNet50, VGG16, etc.

# --- Image Branch (CNN) ---
# Using a pre-trained model (Transfer Learning) is highly recommended
# IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS defined in Step 13
base_cnn = EfficientNetB0(weights='imagenet', include_top=False,   #include_top=False: Kendi sınıflandırma katmanlarını kullanmıyoruz. Onun yerine kendi katmanlarımızı ekliyoruz.
                          input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
base_cnn.trainable = False # Freeze the pre-trained layers initially  #base_cnn.trainable = False: Bu katmanları eğitmeyeceğiz (donmuş halde). Böylece daha hızlı eğitilir ve overfitting azalır.

image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='image_input')
x = base_cnn(image_input, training=False) # training=False because base_cnn is frozen   #training=False → Batch Normalization ve Dropout gibi katmanların eğitim moduna geçmemesini sağlar.
x = Flatten(name='image_flatten')(x)
x = Dense(256, activation='relu', name='image_dense_1')(x)  #Bu katman artık bizim eğitim verimize göre öğrenme yapar.
x = Dropout(0.5, name='image_dropout')(x)  #Dropout, eğitim sırasında nöronların %50’sini rastgele geçici olarak kapatır. Amaç: Ağırlıkların tek bir nörona bağımlı hale gelmesini engelleyip overfitting’i azaltmak.
image_features = Dense(128, activation='relu', name='image_features_output')(x) # Output of image branch  #Burada 256’dan 128 nöronlu bir katmana geçiyoruz. Bu katman, görsel dalın son çıktısıdır. Bu 128 boyutlu vektör, tabular veri ile birleştirilmek üzere hazırlanır.

# --- Tabular Branch (MLP) ---
# The shape of tabular_input should match the number of columns in X_train_tab_processed
num_tabular_features = X_train_tab_processed.shape[1]   #Bu, tabular verinin kaç özellik (feature) içerdiğini verir.
tabular_input = Input(shape=(num_tabular_features,), name='tabular_input')
y = Dense(128, activation='relu', name='tabular_dense_1')(tabular_input)
y = BatchNormalization()(y) # Often helpful for MLPs
y = Dropout(0.4, name='tabular_dropout_1')(y)
y = Dense(64, activation='relu', name='tabular_dense_2')(y)
tabular_features = Dense(64, activation='relu', name='tabular_features_output')(y) # Output of tabular branch

# --- Fusion Layer ---
# Concatenate the features from both branches
combined_features = concatenate([image_features, tabular_features], name='concatenate_features')

# --- Classifier Head ---
z = Dense(128, activation='relu', name='classifier_dense_1')(combined_features)
z = Dropout(0.5, name='classifier_dropout')(z)
# Output layer: 1 neuron with sigmoid for binary classification
output_layer = Dense(1, activation='sigmoid', name='output_layer')(z)

# --- Create the Multimodal Model ---
multimodal_model = Model(inputs=[image_input, tabular_input], outputs=output_layer)

# --- Compile the Model ---
multimodal_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                         loss='binary_crossentropy',
                         metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

multimodal_model.summary()
# tf.keras.utils.plot_model(multimodal_model, show_shapes=True) # Optional: visualize model

"""
#burası görselleşitme için incele tekradan sorunlar çıkıyor chatte son kısım

import tensorflow as tf

logdir = "logs/model"
writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on(graph=True, profiler=True)
multimodal_model(tf.random.normal([1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]),
                 tf.random.normal([1, num_tabular_features]))
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)

"""






# STEP 16:
EPOCHS =  25 #30 # Start with a moderate number, can adjust based on EarlyStopping

# Callbacks (Optional but Recommended)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_auc', patience=7, mode='max', restore_best_weights=True, verbose=1)  # Amaç: Validation AUC değeri 7 epoch boyunca iyileşmezse eğitimi durdur.
# monitor 'val_loss' (mode 'min') or 'val_accuracy'/'val_auc' (mode 'max')

reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=3, min_lr=1e-6, mode='max', verbose=1)  #Amaç: Validation AUC gelişmezse, learning rate'i düşür.

# Save the best model based on validation AUC (or loss)
model_checkpoint = ModelCheckpoint('best_multimodal_model.keras', # Use .keras format     # Amaç: En yüksek val_auc elde edilen modeli dosyaya kaydetmek.
                                   monitor='val_auc', save_best_only=True,
                                   mode='max', verbose=1)


history = multimodal_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Plot training history
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1) # Adjust y-axis limit if needed
plt.title("Model Training History")
plt.show(block = True)
# Sonuç: Eğitim süreci boyunca accuracy, auc gibi metriklerin nasıl değiştiğini görselleştirirsin.



# ////////// Manually Saving the Model: //////////

# .keras format (preferred, modern)
multimodal_model.save("final_multimodal_model.keras")
# Alternative: .h5 format (old but common)
multimodal_model.save("final_multimodal_model.h5")
# Check model structure
multimodal_model.summary()

# ////////// Loading for Later Use: //////////
from tensorflow.keras.models import load_model

# Load best saved model
model = load_model("best_multimodal_model.keras")


#STEP 17: EVALUATE THE MODEL

# Load the best model saved by ModelCheckpoint
# (The 'restore_best_weights=True' in EarlyStopping might already give you the best weights in 'multimodal_model'
# but loading explicitly from the checkpoint file is safer if you ran more epochs after best was found)
print("Loading best model weights...")
multimodal_model.load_weights('best_multimodal_model.keras') # Or .h5 if you saved in that format

# ////////// Evaluating on Test Dataset: //////////
print("\nEvaluating on Test Set:")
results = multimodal_model.evaluate(test_dataset)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")
print(f"Test AUC: {results[2]:.4f}") # Assuming 'auc' was the third metric

# Get predictions for more detailed metrics
print("\nGenerating predictions on Test Set...")
y_pred_proba_test = multimodal_model.predict(test_dataset)
y_pred_test = (y_pred_proba_test > 0.5).astype(int).flatten() # Flatten if necessary

# y_true_test is y_test, but ensure it's aligned if test_dataset was batched
# For unbatched y_test:
y_true_test = y_test

from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report on Test Set:")
print(classification_report(y_true_test, y_pred_test, target_names=label_encoder.classes_)) # Use original class names

print("\nConfusion Matrix on Test Set:")
cm = confusion_matrix(y_true_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.show(block = True)



# STEP 18: FINE TUNING (Advanced Step)
# --- Optional: Fine-tuning the CNN base ---
# print("\nStarting fine-tuning...")
# base_cnn.trainable = True # Unfreeze the base model

# # It's often a good idea to only unfreeze the top layers
# # For example, for EfficientNet, let's say unfreeze last 20 layers
# # for layer in base_cnn.layers[:-20]:
# #    layer.trainable = False

# # Re-compile the model with a very low learning rate for fine-tuning
# multimodal_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Much lower LR
#                          loss='binary_crossentropy',
#                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# multimodal_model.summary() # See trainable params changed

# FINE_TUNE_EPOCHS = 10
# history_fine_tune = multimodal_model.fit(
#     train_dataset,
#     epochs=EPOCHS + FINE_TUNE_EPOCHS, # Total epochs
#     initial_epoch=history.epoch[-1] + 1, # Start from where previous training left off
#     validation_data=val_dataset,
#     callbacks=[early_stopping, reduce_lr, model_checkpoint] # Can reuse or adjust callbacks
# )
# # Then re-evaluate on test set



# STEP 19: FEATURE IMPORTANCE:


# preprocessor is your fitted ColumnTransformer
# X_tabular_train_raw is the raw tabular data fed to preprocessor.fit()

# Get feature names after one-hot encoding
try:
    # For scikit-learn >= 1.0
    processed_feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    # For older scikit-learn (less clean, might need manual construction)
    # This is a bit more involved. If you have an older version, let me know.
    # For now, let's assume you have a way to get these names.
    # One way for older versions:
    one_hot_feature_names = []
    if 'cat' in preprocessor.named_transformers_: # Check if 'cat' transformer exists
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'): # For newer OHE within ColumnTransformer
             one_hot_feature_names = cat_transformer.get_feature_names_out(categorical_features).tolist()
        elif hasattr(cat_transformer, 'categories_'): # For older OHE
            for i, col_name in enumerate(categorical_features):
                for cat_val in cat_transformer.categories_[i]:
                    one_hot_feature_names.append(f"{col_name}_{cat_val}")

    processed_feature_names = numerical_features + one_hot_feature_names
    # Ensure the order matches how ColumnTransformer concatenates them (usually numerical first, then categorical)
    # You might need to inspect preprocessor.transformers_ to be sure of the order.
    # Example:
    # numerical_part_names = [f for f, t, c in preprocessor.transformers_ if t == 'num'][0] # This is simplified
    # categorical_part_names = ...
    print("Make sure 'processed_feature_names' correctly lists all columns after one-hot encoding and scaling.")


print(f"Processed feature names (first 10): {list(processed_feature_names)[:10]}")
print(f"Total processed features: {len(processed_feature_names)}")

# Permutation Importance Function
from sklearn.metrics import roc_auc_score  # Or accuracy_score, etc.
import numpy as np

def get_permutation_importance(model, eval_image_paths, eval_X_tabular, eval_y,
                               feature_names, metric_fn=roc_auc_score, n_repeats=5):
    """
    Calculates permutation importance for a multimodal model.

    Args:
        model: The trained Keras multimodal model.
        eval_image_paths: NumPy array of image paths for evaluation.
        eval_X_tabular: NumPy array of processed tabular features for evaluation.
        eval_y: NumPy array of true labels for evaluation.
        feature_names: List of names for the columns in eval_X_tabular.
        metric_fn: Function to calculate the performance metric (e.g., roc_auc_score).
        n_repeats: Number of times to repeat the permutation for each feature.

    Returns:
        A dictionary where keys are feature names and values are their importances (drop in score).
    """

    # Create a tf.data.Dataset for efficient prediction (if not already available)
    # For this function, it's easier if we predict on the whole set at once if possible,
    # or adapt to use a dataset for prediction.

    def predict_on_data(current_X_tabular):
        # This is a simplified prediction, assumes we can create a dataset on the fly
        # or that the model.predict can handle numpy arrays directly for the tabular part
        # if eval_image_paths is relatively small, we can load all images
        # THIS PART MIGHT NEED ADJUSTMENT BASED ON DATA SIZE AND GENERATOR SETUP

        # Simple approach: make a temporary dataset for prediction
        temp_image_ds = tf.data.Dataset.from_tensor_slices(eval_image_paths).map(load_and_preprocess_image,
                                                                                 num_parallel_calls=tf.data.AUTOTUNE)
        temp_tabular_ds = tf.data.Dataset.from_tensor_slices(current_X_tabular.astype(np.float32))

        # We need to batch it, even if it's one large batch for evaluation purposes here
        # Ensure batch size can handle the full eval set if memory allows for this simplified version
        eval_batch_size = len(eval_image_paths)  # Predict all at once for simplicity here

        temp_pred_dataset = tf.data.Dataset.zip((temp_image_ds, temp_tabular_ds))
        temp_pred_dataset = temp_pred_dataset.batch(eval_batch_size)  # Process all in one batch

        predictions = model.predict(temp_pred_dataset, verbose=0)
        return predictions.flatten()

    # 1. Calculate baseline score
    baseline_predictions = predict_on_data(eval_X_tabular)
    baseline_score = metric_fn(eval_y, baseline_predictions)
    print(f"Baseline VAL Score ({metric_fn.__name__}): {baseline_score:.4f}")

    importances = {}

    for i, feature_name in enumerate(tqdm(feature_names, desc="Calculating Permutation Importance")):
        permuted_scores = []
        for _ in range(n_repeats):
            X_tabular_permuted = eval_X_tabular.copy()
            # Permute the i-th column
            np.random.shuffle(X_tabular_permuted[:, i])

            permuted_predictions = predict_on_data(X_tabular_permuted)
            score_after_permutation = metric_fn(eval_y, permuted_predictions)
            permuted_scores.append(score_after_permutation)

        avg_permuted_score = np.mean(permuted_scores)
        importance = baseline_score - avg_permuted_score  # Higher drop = more important
        importances[feature_name] = importance

    return importances


# --- Before calling, make sure you have these: ---
# multimodal_model: Your trained model
# image_val_paths: NumPy array of validation image paths
# X_val_tab_processed: NumPy array of processed validation tabular data
# y_val: NumPy array of validation labels
# processed_feature_names: List of names for columns in X_val_tab_processed

# Example usage:
# Ensure your model is loaded if not already in memory from training
# multimodal_model.load_weights('best_multimodal_model.keras') # Or wherever it's saved

# Get the full arrays (if your val_dataset batches them, you might need to iterate and concatenate)
# This is a simplification; ideally, you'd have these ready or adapt the function
# For example, if val_dataset yields ((images, tabular), labels):
# all_val_images_list = [] # This would be image paths, not processed images for this function
# all_val_tabular_list = []
# all_val_labels_list = []
# for (img_batch_paths, tab_batch), label_batch in val_dataset_unbatched_or_reconstructed: # Pseudocode
#     all_val_tabular_list.append(tab_batch.numpy())
#     all_val_labels_list.append(label_batch.numpy())
# X_val_tab_processed_full = np.concatenate(all_val_tabular_list, axis=0)
# y_val_full = np.concatenate(all_val_labels_list, axis=0)
# image_val_paths_full = image_val_paths # Assuming this is already a full array

# Check shapes
print("Shapes for permutation importance:")
print("image_val_paths:", image_val_paths.shape)
print("X_val_tab_processed:", X_val_tab_processed.shape)
print("y_val:", y_val.shape)
print("Number of processed_feature_names:", len(processed_feature_names))

if X_val_tab_processed.shape[1] != len(processed_feature_names):
    print("Error: Mismatch between number of columns in X_val_tab_processed and length of processed_feature_names!")
    print(
        f"X_val_tab_processed columns: {X_val_tab_processed.shape[1]}, Names provided: {len(processed_feature_names)}")
else:
    print("Feature name count matches processed tabular data columns. Proceeding.")
    # Calculate importances
    # Note: The predict_on_data part of the function might need tuning if your validation set is huge.
    # It currently tries to predict on the whole validation set at once after making a temp dataset.
    # If memory is an issue, model.predict should be used with the val_dataset directly,
    # but then permuting features within the dataset pipeline is more complex.

    # For now, let's assume X_val_tab_processed and image_val_paths are full numpy arrays
    # and the predict_on_data helper can handle creating a dataset from them for prediction.
    feature_importances = get_permutation_importance(
        multimodal_model,
        image_val_paths,  # Full array of image paths for validation
        X_val_tab_processed,  # Full array of processed tabular data for validation
        y_val,  # Full array of labels for validation
        processed_feature_names,
        metric_fn=roc_auc_score,  # Using AUC as the metric
        n_repeats=5
    )

    # Sort and display importances
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

    print("\nFeature Importances (Permutation Method):")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")

    # Plotting
    plt.figure(figsize=(10, max(6, len(sorted_importances) // 2)))  # Adjust figure size
    plt.barh([item[0] for item in sorted_importances[:20]],
             [item[1] for item in sorted_importances[:20]])  # Show top 20
    plt.xlabel("Importance (Drop in AUC)")
    plt.title("Top 20 Tabular Feature Importances (Permutation)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()