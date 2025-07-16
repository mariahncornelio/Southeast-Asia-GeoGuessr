![UTA-DataScience-Logo](https://github.com/user-attachments/assets/36b0607e-06da-485c-97a1-34a4f0552141)

# Southeast Asia GeoGuessr

* This repository...

## OVERVIEW

  * **Background:**
  * **Project Goal:**
  * **Approach:**
  * **Summary of Performance**

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input:
    * Output:
  * **Size:**
    * 
  * **Instances:**
    * Number of samples
   
#### Preprocessing / Clean up

* **Missing Values:**
* **Encoding:**
* **Datetime and Sorting:**
* **Feature Selection:**
* **Normalization:**
* **Other:**
 
#### Data Visualization

### Problem Formulation

* **Input/Output**
  * Input:
  * Output:
* **Models Used:**
  * **LSTM:**
  * **GRU:**
  * **Bidirectional LSTM:**
* **Loss Function & Optimizer**
* **Hyperparamters & Threshold Tuning**
 
### Training

Model training was performed in **Jupyter Notebook** on a **MacBook Pro with an Apple M1 chip and 8 GB of memory.** The following libraries and frameworks were used throughout the training process:
* TensorFlow/Keras for building and training deep learning models (LSTM, GRU, BiLSTM, Stacked LSTM, CNN+LSTM, Transformer)
* scikit-learn for preprocessing, metrics, and tree-based models like Random Forest
* XGBoost for training a gradient-boosted decision tree classifier

Each model took **between 10 seconds to 1 minute to train** with Transformer taking the longest. In total, model training and evaluation spanned approximately **12 hours over 3 days**, including experimentation and threshold tuning. **Early stopping** was implemented for all deep learning models, with **validation loss as the monitored metric and a patience of 3 epochs, restoring the best weights** to avoid overfitting.

**Challenges:**

### Performance Comparison

The primary goal of this project is to accurately detect wildfire start days, where failing to identify a true fire (false negative) can have dangerous, even deadly consequences. For this reason, I prioritized ***Recall & ROC-AUC > F1 > Precision***:
* **Recall:** Catching as many actual fire events as possible is crucial to minimize missed alarms
* **F1:** Balances recall and precision, helping prevent over-alerting while still catching fires
* **Precision:** Excessive false positives can damage trust in the system and strain emergency resources
* **ROC-AUC:** Measures the model’s ability to distinguish between fire and non-fire days, regardless of threshold

While the GRU and Transformer models showed slightly better ROC-AUC and F1 scores, the ***CNN+LSTM architecture achieved the highest recall (0.85)***, which is critical in the context of wildfire forecasting where failing to detect a fire can have severe consequences. For this reason, ***CNN+LSTM was selected as the final model for the forecasting tool.*** Note that although XGBoost performed the best along with the Decision Tree model, the main goal was to choose the best **time-series** model. Those were just for comparison and curiosity.


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
