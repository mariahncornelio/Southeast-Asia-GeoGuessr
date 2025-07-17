![UTA-DataScience-Logo](https://github.com/user-attachments/assets/36b0607e-06da-485c-97a1-34a4f0552141)

# Southeast Asia GeoGuessr

This repository contains a deep learning project that identifies Southeast Asian countries from Google Street View images, using scene-specific coordinates and queries to gather real-world visual data.

* **IMPORTANT:** This dataset was created using the Google Street View Static API. THe actual imagery is not distributed here to comply with Google's terms of service. You must obtain your own API key and download the images using the provided coordinates and pipelines © Google Street View. Retrieved via Google Street View Static API

## OVERVIEW

  * **Background:** As a GeoGuessr player, I’ve found it especially challenging to distinguish between Southeast Asian countries due to their similar foliage, architecture, and general scenery. This inspired the idea: could a computer vision model perform better than a human at identifying which country an image is from?
  * **Project Goal:** Build an image classification model that can identify whether a given Google Street View image originates from one of four Southeast Asian countries: Indonesia, Malaysia, the Philippines, or Thailand
  * **Approach:** Google Street View Static API was used to gather real-world images based on location-specific queries on Overpass Turbo across the four countries. The problem was framed as a multi-class classification task using transfer learning. A custom CNN, MobileNetV2, and ResNet50 models were trained and compared, analyzing accuracy and generalization/confidence performance
  * **Summary of Performance** The best-performing model was **MobileNetV2**, achieving a validation accuracy of 33.75%, outperforming both the custom model (24.6%) and ResNet50 (29.3%). The results show that while classification is still difficult for this region, the model demonstrates some ability to distinguish between visually similar Southeast Asian countries

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input: An image directory of various types of street-view images, with each country being it's own class directory
    * Output: A model that outputs the predicted country, the confidence percentage, and if the prediction was correct or incorrect
  * **Source:**
    * Custom-built data set using queries from Overpass Turbo and images from Google Street View Static API
  * **Size of Classes:**
    * 886 images from Indonesia, 893 from Malaysia, 896 from the Philippines, and 898 from Thailand. 3573 total images
  * **Splits:**
    * 80% training and 20% validation
   
#### Compiling Data and Image Pre-processing

* **Data Collection:**
    * Source: Overpass Turbo (OT) and Google Street View Static API (GSVS API)
    * OT Query Strategy: Each country was queried using keywords like "beach," "mountain," "urban," and "rural" to ensure diverse scenes
    * GSVS API Image Extraction: Coordinate files were manually curated on Excel from GeoJSON's and batched to pull images from various regions within each country
* **Data Cleaning:**
    * Manual Filtering: Remove corrupted photos (containing obstructive blurs or duplicates)
        * Philippines: Image #1, 63, 79, 81
        * Thailand: Image #870, 882
        * Malaysia: Image # 319, 798, 890
        * Indonesia: Image # 678, 680, 682, 684, 689, 690, 697, 713, 850, 851, 853, 856, 859, 865
* **Image Pre-processing:**
    * Resizing: All images resized to 224x224 pixels
    * Normalization: Pixel values scaled to range [0, 1] by dividing by 255
    * Augmentation (Training Only): Random horizontal flip and random rotation (up to 10°)
    * Batching and Shuffling: Data loaded in batches of size 10 and shuffled during training to prevent learning order bias
 
#### Data Visualization

<img width="623" height="614" alt="Screenshot 2025-07-15 at 11 08 56 PM" src="https://github.com/user-attachments/assets/434de7c4-e729-4bb9-b8a4-32aef5a0f430" />

This is a 3x3 grid of random images along with their labeled country class. 

### Problem Formulation

* **Models Used:**
  * **Custom Convolutional Neural Network:** To build a lightweight baseline model from scratch and understand how well a non-transfer model could learn distinguishing features
  * **MobileNetV2 (Transfer Learning)**
  * **ResNet50 (Transfer Learning)** 
* **Loss Function & Optimizer:** SparseCategoricalCrossentropy(from_logits=True) and Adam optimizer with learning rate 0.001
* **Epochs:** Up to 20 with EarlyStopping and patience = 5
* **Callbacks:** EarlyStopping and ModelCheckpoint
 
### Training

Model training was conducted locally on a custom-built **Windows PC equipped with an AMD Ryzen 7 7700X CPU, RTX 4060 GPU, and 64 GB of DDR5 RAM**, using Jupyter Notebook. The training utilized TensorFlow/Keras along with key libraries such as numpy, matplotlib, and pandas. Training times varied by model: the **custom CNN and MobileNetV2 each took ~6 minutes**, **ResNet50 ran for ~15 minutes** (stopping at 15/20 epochs), and **early stopping** was used across all models to prevent overfitting. Total project time, including dataset curation and image cleanup, spanned about a week.

Training was guided by Categorical Crossentropy loss, the Adam optimizer, and accuracy as the main metric. Model performance was evaluated using training/validation curves and early stopping callbacks. 

**Challenges:** One major challenge was the similarity in scenery across Southeast Asian countries, making fine-grained classification difficult. Manual data cleaning was essential to reduce noise. Other challenges included spatial bias due to image sampling where many images were drawn from geographic clusters, which could limit generalization. For future iterations, broader and more randomized sampling strategies, as well as more diverse and representative images (distinguishing transport types like tricycles vs. tuk-tuks or urban vs. rural settings), would be crucial for improving the model’s learning and robustness

### Performance Comparison

#### **Custom CNN Base Model**
<img width="696" height="316" alt="Screenshot 2025-07-15 at 11 10 14 PM" src="https://github.com/user-attachments/assets/9d71cd1a-5772-4b3d-bbb1-5bf1ed1caca5" />

#### **MobileNetV2 Model**
<img width="687" height="317" alt="Screenshot 2025-07-15 at 11 10 35 PM" src="https://github.com/user-attachments/assets/9ed26cd4-821b-4fdb-93b0-3a4618224ec3" />

#### **ResNet50 Model**
<img width="697" height="287" alt="Screenshot 2025-07-15 at 11 10 51 PM" src="https://github.com/user-attachments/assets/9bf5e0a2-4696-4756-8976-4f90438dc66a" />

### Conclusions

From the experiments conducted, **MobileNet emerged as the best-performing model, achieving the highest validation accuracy of 33.75%**, outperforming both the custom CNN and ResNet50. While all models struggled due to the inherent difficulty of the task (classifying visually similar Southeast Asian countries), MobileNet demonstrated the most stable and consistent learning curves.

<img width="446" height="93" alt="Screenshot 2025-07-15 at 11 14 38 PM" src="https://github.com/user-attachments/assets/65711acf-dc70-4673-8517-21f317cb4a8d" />

The custom model underfit the data, with validation accuracy hovering around 25%, equivalent to random guessing, and showed unstable training behavior, likely due to its shallow architecture. ResNet50 offered slightly better generalization and more balanced training/validation performance but lagged slightly behind MobileNet in overall accuracy. Despite increasing validation loss in both MobileNet and ResNet, the absence of extreme overfitting suggests potential for further tuning. Overall, MobileNet proved to be the most promising baseline, but additional data variety, regularization techniques, and sampling improvements are needed to push performance beyond chance levels.

#### **MobileNet Final Model Deployment**
<img width="691" height="697" alt="Screenshot 2025-07-16 at 9 25 59 PM" src="https://github.com/user-attachments/assets/68ce19c5-e472-4d79-864d-484c90f704df" />

#### **MobileNet Final Model Deployment Performance Analysis**

<img width="315" height="114" alt="Screenshot 2025-07-16 at 9 27 15 PM" src="https://github.com/user-attachments/assets/cfa9c3ca-271a-4a81-bb46-58259929ee47" />

#### **MobileNet Final Model Deployment Performance Analysis - Average Confidence**

<img width="664" height="398" alt="Screenshot 2025-07-16 at 9 27 38 PM" src="https://github.com/user-attachments/assets/2fc2575f-606e-42f2-a3e4-fa431caa9ce8" />

**WHAT THIS MEANS:** The model shows relatively balanced confidence across all countries, with a slight tendency to overestimate predictions for Indonesia and underpredict Malaysia, while being most consistent for the Philippines and Thailand

### Future Work
 
* **Increase dataset size** to support deeper and more expressive models
* **Collect more diverse scenery samples**, especially for countries with regional visual differences (e.g., urban vs. rural Philippines)
* **Improve sampling strategy** by selecting coordinates ***randomly*** across the entire country instead of sequentially, reducing spatial bias
* **Ensure broader geographic coverage** to avoid clustering in specific types of landscapes (e.g., beaches only)
* **Incorporate finer-grained features** (types of vehicles like tricycles or tuk-tuks) that may help distinguish between countries

## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The list below follows the chronological order in which each component of the project was developed:

* **Vision_Proposal_Cornelio.pdf:** This project's proposal which includes background information and an abstract
* **GeoJSON_to_CSV.ipynb:** Includes query code to execute on Overpass Turbo website and obtain the .GeoJSON file with coordinates. API checks to see if the image for that location is available and appends the first 150 into a list
    * **Coordinates Folder:** Contains all .GeoJSON and coordinate lists for each country
* **Country_Image_Extracting.ipynb:** Uses API to extract images using country master coordinate list and zips the file  
* **seageo_feasibility_and_prototype.ipynb:** The final model pipeline, including preprocessing, finetuning, training, and analysis

### Software Setup

This project was developed and executed in Google Colab Jupyter Notebook. If you don’t already have it installed, you can download it as part of the Anaconda distribution or install it via pip "pip install notebook".

* os
* numpy
* pandas
* matplotlib
* random
* tensorflow
* keras
    * layers, models, optimizers
    * preprocessing.image (load_img, img_to_array)
    * callbacks (EarlyStopping, ModelCheckpoint)
    * applications (MobileNetV2, ResNet50)
* sklearn

### Data

* **Websites Used:**
    * **Overpass Turbo Query:** https://overpass-turbo.eu/
    * **Google Cloud Platform:** https://console.cloud.google.com
* Dataset was built using Google Street View Static API with manually curated coordinates across Indonesia, Malaysia, the Philippines, and Thailand
* Coordinates were gathered using map queries for specific sceneries (e.g., beaches, cities), reprojected, and sorted into a GeoDataFrame. API calls checked for available imagery; first 150 valid coordinates per scenery per country were used to conserve quota
    * **For reference, see GeoJSON_to_CSV.ipynb**
* Images were downloaded and organized into folders by country
    * **For reference, see Country_Image_Extracting.ipynb**
* Manual preprocessing removed broken, blurred, overly similar, or selfie-containing images

### Training

* Install required packages in notebook
* Download and prepare the data (either from scratch or above, or use files in the Coordinates folder of this directory)
* Models were trained using TensorFlow/Keras with early stopping and validation monitoring. Images were split into training and validation sets, preprocessed (resized and batched), and fed into models like MobileNetV2 and ResNet50. Training ran on a local machine with GPU support over multiple sessions

***For reference, see seageo_feasibility_and_prototype.ipynb***

#### Performance Evaluation

* After training, model performance can be evaluated using the validation set
* Run the evaluation script to compute accuracy, loss, and confidence scores per class
* Visualization tools like matplotlib can be used to plot training curves and compare actual vs predicted confidence by country

***For reference, see seageo_feasibility_and_prototype.ipynb***

## CITATIONS

[1] Andaya, Barbara Watson. “Introduction to Southeast Asia: History, Geography, and Livelihood.” Asia Society, 2025, https://asiasociety.org/education/introduction-southeast-asia#:~:text=Southeast%20Asians%20found%20it%20easier,same%20religious%20and%20cultural%20influences.

[2] Chollet, François. “Image Classification from Scratch.” Keras, 27 Apr. 2020, last modified 9 Nov. 2023, https://keras.io/examples/vision/image_classification_from_scratch/.
