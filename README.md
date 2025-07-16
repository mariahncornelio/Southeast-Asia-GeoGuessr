![UTA-DataScience-Logo](https://github.com/user-attachments/assets/36b0607e-06da-485c-97a1-34a4f0552141)

# Southeast Asia GeoGuessr

* This repository contains a deep learning project that identifies Southeast Asian countries from Google Street View images, using scene-specific coordinates and queries to gather real-world visual data.

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

### Problem Formulation

* **Models Used:**
  * **Custom Convolutional Neural Network:** To build a lightweight baseline model from scratch and understand how well a non-transfer model could learn distinguishing features
  * **MobileNetV2 (Transfer Learning):** MobileNet is lightweight and optimized for mobile vision tasks, making it ideal for efficient training and deployment
  * **ResNet50 (Transfer Learning):** A deeper model known for strong performance on image tasks through residual connections
* **Loss Function & Optimizer:** SparseCategoricalCrossentropy(from_logits=True) and Adam optimizer with learning rate 0.001
* **Epochs:** Up to 20 with EarlyStopping and patience = 5
* **Callbacks:** EarlyStopping and ModelCheckpoint
 
### Training

Model training was performed in **Jupyter Notebook** on a **MacBook Pro with an Apple M1 chip and 8 GB of memory.** The following libraries and frameworks were used throughout the training process:
* TensorFlow/Keras for building and training deep learning models (LSTM, GRU, BiLSTM, Stacked LSTM, CNN+LSTM, Transformer)
* scikit-learn for preprocessing, metrics, and tree-based models like Random Forest
* XGBoost for training a gradient-boosted decision tree classifier

Each model took **between 10 seconds to 1 minute to train** with Transformer taking the longest. In total, model training and evaluation spanned approximately **12 hours over 3 days**, including experimentation and threshold tuning. **Early stopping** was implemented for all deep learning models, with **validation loss as the monitored metric and a patience of 3 epochs, restoring the best weights** to avoid overfitting.

**Challenges:**

### Performance Comparison


### Conclusions


Overall...

### Future Work

* **Incorporate Windowed Inputs into Tree-Based Models**
  * Explore how Random Forest and XGBoost perform when fed sliding window sequences similar to those used in deep learning models. This may improve their ability to capture temporal trends
* **Feature Engineering & Expansion**
* **Build an Interactive Forecasting Dashboard**
* **Integrate into Early Warning or Emergency Systems**

## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The list below follows the chronological order in which each component of the project was developed:

* **Tabular Project Proposal MNC.pptx:** This project's proposal which includes background information and an abstract to the project
* **Tabular Project Proposal MNC.pdf:**
* **CA_Weather_Fire_Dataset_1984-2025.csv:**
* **ProjectRoughDraft.ipynb:**
* **Feasibility_Tabular_MNC.ipynb:**
* **firedf_cleaned.csv:**
* **Prototype_Tabular_MNC.ipynb:**

### Software Setup

This project was developed and executed in Google Colab Jupyter Notebook. If you don’t already have it installed, you can download it as part of the Anaconda distribution or install it via pip "pip install notebook".

* **Data Handling & Visualization:**
  * pandas, numpy, matplotlib, seaborn
* **Preprocessing & Evaluation:**
  * sklearn.preprocessing:
    * MinMaxScaler, StandardScaler, OrdinalEncoder
  * sklearn.model_selection:
    * train_test_split
  * sklearn.metrics:
    * recall_score, precision_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
  * sklearn.utils:
    * class_weight
* **Machine Learning Models:**
  * sklearn.linear_model.LogisticRegression
  * sklearn.tree.DecisionTreeClassifier
  * sklearn.ensemble.RandomForestClassifier
* **Deep Learning (TensorFlow/Keras):**
  * tensorflow
    * Sequential, Model, Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, EarlyStopping, Recall, AUC
    * MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D (used for transformer model)
* **Gradient Boosting:**
  * xgboost

### Data

* **DATA DOWNLOAD LINK:** https://zenodo.org/records/14712845
* **All preprocessing and cleanup steps are documented and executed in the Feasibility_Tabular_MNC.ipynb notebook.** This includes:
  * Handling missing values based on distribution (mean, median, or zeros)
  * Cyclical encoding of seasonal and temporal features
  * Dropping redundant or highly correlated columns
  * Scaling features using MinMax normalization
  * Encoding the target variable
  * Sequence generation using a 21-day rolling window for time series modeling (in the Prototype_Tabular_MNC.ipynb notebook)

### Training

* Install required packages in notebook
* Download and prepare the data from Zenodo and the Feasibility_Tabular_MNC.ipynb pipeline, obtaining firedf_cleaned.csv
* Train models in this order: LSTM, GRU, BiLSTM, Stacked LSTM, CNN + LSTM, Transformer, Decision Tree, XGBoost
  * All models use Binary Crossentropy loss, Adam optimizer, and EarlyStopping callback with patience of 3 and val_loss monitoring
 * After training the base model, tune thresholds to balance Recall and F1 on the **validation set**
* Use the obtained threshold value on the **test set**

***For reference, see Prototype_Tabular_MNC.ipynb***

#### Performance Evaluation

* For each model:
  * Calculate key classification metrics: precision, recall, F1-score, ROC-AUC
  * Print classification report on validation set pre-tuned
  * Print classification report on validation set post-tuned
  * Print classification report on test set post-tuned
  * Plot ROC curves and compare models visually

***For reference, see Prototype_Tabular_MNC.ipynb***

## CITATIONS

[1] Abatzoglou, J. T., & Williams, A. P. (2018). Impact of anthropogenic climate change on wildfire across western US forests. *PNAS.*

[2] Cal Fire. “Incidents.” *CAL FIRE*, California Department of Forestry and Fire Protection, https://www.fire.ca.gov/incidents/. Accessed 28 June 2025.

[3] Keith, Michael. “Exploring the LSTM Neural Network Model for Time Series.” *Towards Data Science*, 13 Jan. 2023, https://towardsdatascience.com/exploring-the-lstm-neural-network-model-for-time-series-8b7685aa8cf/.

[4] Moore, Andrew. “Explainer: How Wildfires Start and Spread.” *College of Natural Resources News*, 3 Dec. 2021, https://cnr.ncsu.edu/news/2021/12/explainer-how-wildfires-start-and-spread/.
