# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

## 📌 Project Overview
This project is a machine learning solution designed to detect fraudulent credit card transactions. 

The dataset is highly imbalanced (only 0.17% fraud). Standard accuracy metrics are misleading, so this project focuses on **Recall** and **Precision-Recall Trade-offs**. We implemented an advanced **Ensemble Learning** approach and a custom **Threshold Optimization** technique to catch more fraud cases than standard models.

## 🚀 Live Demo
Check out the live web application here:  
👉 **[Link to your Streamlit App](https://share.streamlit.io/your-username/your-repo)** *(Replace this link after you deploy!)*

## ✨ Key Features
* **⚖️ Imbalance Handling:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) and Under-sampling to balance the training data.
* **🤖 Advanced Modeling:** Compared Logistic Regression, Neural Network, Random Forest, and Voting Classifiers.
* **🎯 Threshold Tuning:** Optimized the decision threshold (from 0.5 to ~0.1) to prioritize **Recall**, ensuring fewer fraud cases slip through.
* **📊 Interactive Dashboard:** A **Streamlit** web app allowing users to simulate transactions and visualize risk probabilities in real-time.
* **⚙️ Configurable:** Uses JSON configuration files for reproducible hyperparameter tuning.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Imbalanced-Learn
* **Deployment:** Streamlit Cloud
* **Version Control:** Git & GitHub

## 📊 Performance Results
![Comparison](Screenshots/image.png)

> **Key Insight:** By lowering the decision threshold, we achieved a **Recall of 0.97**, meaning we detect 97% of all fraud cases, which is critical for financial security.

## 📸 Screenshots

### 1. The Interactive App
![App Screenshot](Screenshots/app_demo.png)

### 2. Feature Importance
![Feature Importance](Screenshots/feature_imp.png)

## 📂 Project Structure
This project follows a production-ready directory structure:

```text
## 📂 Project Structure

The project is organized as follows:

```text
├── Dataset/                 # Folder containing the original CSV datasets
├── Screenshots/             # Images used for this README (App demo, etc.)
│
├── app.py                   # Main Streamlit application script
├── main.py                  # Orchestrator script to run the full pipeline
├── training.py              # Logic for training the Random Forest model
├── evaluation.py            # Functions for plotting and metrics
├── data_utils.py            # Data cleaning and SMOTE processing functions
├── config_utils.py          # Utilities for managing configuration settings
│
├── EDA.ipynb                # Jupyter Notebook for Exploratory Data Analysis
├── final_fraud_model.pkl    # The saved trained model (ready for deployment)
├── best_hyperparameters.json # Optimized parameters found during tuning
├── classification_reports.txt # Text file containing detailed performance metrics
├── test_sample.csv          # Sample data used by the Streamlit App for testing
│
├── requirements.txt         # List of Python dependencies
└── README.md                # Project documentation         # Project Documentation
```
## 💻 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/youssofhossam/Credit-Card-Fraud-Detection-.git](https://github.com/youssofhossam/Credit-Card-Fraud-Detection-.git)
   cd fraud-detection-app 
   ```
   
2. **Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the app
    ```bash
    streamlit run app.py
    ```

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.