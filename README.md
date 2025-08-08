# ECG Signal Analysis for Human Activity Recognition

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project demonstrates a complete, end-to-end machine learning pipeline for analyzing physiological time-series data. Using an electrocardiogram (ECG) signal from the MHEALTH dataset, we extract meaningful features, train a classifier to recognize human activities, and discover hidden patterns in the data through unsupervised learning.

## Project Pipeline

The project follows a standard data science workflow:
1.  **Data Acquisition**: Downloads and loads a real-world ECG dataset.
2.  **Feature Engineering**: Converts raw signal data into a structured feature set using time-domain and frequency-domain analysis.
3.  **Supervised Learning**: Trains a Random Forest model to classify 12 different physical activities.
4.  **Unsupervised Learning**: Implements K-Means clustering to discover natural groupings in the data and Isolation Forest to detect anomalies.
5.  **Data Visualization (EDA)**: Creates a suite of plots to interpret the data and model performance.

## Dataset

This project uses the **MHEALTH (Mobile Health) dataset** from the UCI Machine Learning Repository. It contains body motion and vital sign recordings for ten volunteers of diverse profiles while performing 12 physical activities.

-   **Source**: [UCI Machine Learning Repository: MHEALTH Dataset](https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset)
-   **Signal Used**: ECG (lead 1)
-   **Sampling Rate**: 50 Hz

##  Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ecg-activity-analysis.git](https://github.com/your-username/ecg-activity-analysis.git)
    cd ecg-activity-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

##  How to Run

Execute the main Python script to run the entire pipeline from data download to model training and visualization. The script will automatically download the necessary data on its first run.

```bash
python main_analysis_script.py
```

##  Analysis and Inferences

The following visualizations provide key insights into the data and the performance of our machine learning models.


### 1. Activity Distribution in the Dataset
This plot shows the number of 10-second windows recorded for each physical activity.
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/eadd36ae-e712-44d9-b5e1-510b23ede299" />



**Inference:** The plot reveals a class imbalance. Activities like "Lying" and "Sitting" are far more represented than dynamic activities like "Jumping" or "Running". This imbalance is a critical consideration for model training, justifying the use of techniques like setting `class_weight='balanced'` in our classifier.

### 2. Signal Variation (Standard Deviation) by Activity
This plot compares the distribution of the ECG signal's standard deviation for each activity.
<img width="1189" height="690" alt="image" src="https://github.com/user-attachments/assets/f5972a8f-3baa-4872-b564-fcf5e51ea949" />



**Inference:** There is a clear and strong correlation between the standard deviation of the ECG signal and the intensity of the activity. Low-intensity activities ("Lying", "Sitting", "Standing") have a very tight, low distribution of `std_dev`. In contrast, high-intensity activities ("Running", "Jumping", "Cycling") show a much higher and wider range of signal variation. This makes `std_dev` an excellent predictive feature.

### 3. Model Feature Importance
This plot ranks the features by how much they contributed to the Random Forest model's predictions.
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/fe2133a1-207c-4d55-a4bb-b07934572b84" />



**Inference:** The model relies most heavily on `dominant_freq_power`, `std_dev`, and `rate_of_change`. This tells us that the **energy and variability** of the signal are the most powerful indicators of physical activity. The model confirms that how much the signal varies (both in amplitude and frequency) is more important than its absolute average value (`mean`).

### 4. Classifier Confusion Matrix
This heatmap provides a detailed look at the model's classification performance, showing which activities were confused with others.
<img width="937" height="790" alt="image" src="https://github.com/user-attachments/assets/a9ad75b0-adab-411e-b07d-f48d3a122ad0" />



**Inference:** The strong diagonal indicates high overall accuracy. The model perfectly distinguishes distinct states like "Lying" and "Cycling". However, the off-diagonal values highlight specific points of confusion. For example, the model may occasionally misclassify "Walking" as "Standing" or vice-versa, which is an intuitive error given the similarity in the signals during transitions.

### 5. Unsupervised Clusters vs. Real Activities (PCA)
This plot shows the natural groupings (clusters) found by the K-Means algorithm, visualized in 2D using PCA.
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/e9e42f20-5503-46ed-aeb7-0a8bc3699803" />



**Inference:** The K-Means algorithm, without any labels, successfully identified distinct clusters in the data. By cross-referencing with the true labels, we can infer that these clusters correspond well to logical categories of activity intensity (e.g., a "resting" cluster, a "moderate movement" cluster, and a "high-intensity" cluster). This demonstrates that the engineered features effectively capture the underlying structure of the data.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
