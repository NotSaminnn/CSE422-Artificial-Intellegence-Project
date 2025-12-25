# CSE422 Artificial Intelligence Project: Travel Insurance Claims Prediction

## 📋 Project Overview

This project implements a comprehensive machine learning solution for predicting travel insurance claims using both supervised and unsupervised learning techniques. The analysis includes exploratory data analysis, feature engineering, multiple classification models, and customer segmentation through clustering.

**Course:** CSE422 - Artificial Intelligence  
**Problem Type:** Binary Classification (Claim: Yes/No)  
**Dataset:** Travel Insurance Claims Data (63,327 samples, 11 features)  
**Date:** August 2025

---

## 🎯 Objectives

1. **Predictive Modeling:** Build and evaluate multiple ML models to predict travel insurance claims
2. **Model Comparison:** Identify the best-performing model for deployment
3. **Customer Segmentation:** Discover natural customer groups through unsupervised learning
4. **Feature Analysis:** Identify key factors influencing insurance claims
5. **Business Intelligence:** Provide actionable insights for risk assessment

---

## 📊 Dataset Description

### Dataset Structure
- **Total Samples:** 63,327 records
- **Total Features:** 11 columns
- **Target Variable:** `Claim` (Yes/No)

### Features

**Categorical Features (7):**
1. `Agency` - Travel agency name
2. `Agency Type` - Type of agency
3. `Distribution Channel` - Online/Offline channel
4. `Product Name` - Insurance plan type
5. `Destination` - Travel destination
6. `Gender` - Customer gender
7. `Claim` - Whether claim was made (TARGET)

**Numeric Features (4):**
1. `Duration` - Trip duration in days
2. `Net Sales` - Sales amount
3. `Commision (in value)` - Commission value
4. `Age` - Customer age

### Data Characteristics
- **Class Imbalance:** Highly imbalanced dataset (majority class: No Claim)
- **Missing Values:** Some columns contain missing values (handled during preprocessing)
- **File:** `travel insurance.csv`

---

## 🔬 Methodology

### Analysis Workflow

#### Section 1: Setup & Data Loading
- Import required libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Load and inspect dataset
- Initial data quality assessment

#### Section 2: Dataset Description
- Feature type identification (numeric vs categorical)
- Target variable distribution analysis
- Statistical summaries

#### Section 3: Exploratory Data Analysis (EDA)
- Correlation analysis (numeric and encoded categorical features)
- Distribution visualizations
- Class balance analysis
- Feature relationship exploration

#### Section 4: Data Preprocessing
- **Missing Value Handling:**
  - Median imputation for numeric features
  - Most frequent category imputation for categorical features
- **Feature Encoding:**
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features (drop='first' to avoid multicollinearity)
- **Data Splitting:**
  - 70% training / 30% testing split
  - Stratified sampling to maintain class balance
  - Random state: 42 for reproducibility
- **Final Feature Space:** 181 features (after one-hot encoding)

#### Section 5: Model Training (Supervised Learning)
Four classification algorithms were trained and evaluated:

1. **Decision Tree Classifier**
   - Tree-based model with default parameters
   - Handles non-linear relationships

2. **Logistic Regression**
   - Linear model with `class_weight='balanced'` for imbalanced data
   - Interpretable coefficients

3. **Naive Bayes (Gaussian)**
   - Probabilistic classifier
   - Fast training and prediction

4. **Neural Network (MLPClassifier)**
   - Multi-layer perceptron with single hidden layer (100 neurons)
   - ReLU activation, Adam optimizer
   - Optimized for imbalanced data

#### Section 6: Model Comparison
- Comprehensive evaluation using multiple metrics
- Performance visualization and comparison
- Best model identification

#### Section 7: Unsupervised Learning
- **K-Means Clustering:** Customer segmentation (k=2)
- **PCA Analysis:** Dimensionality reduction and visualization
- **Cluster Evaluation:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz index

#### Section 8: Output Export
- Save cleaned dataset with cluster labels
- Export model performance results
- Generate visualization outputs

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Neural Network** | 98.54% | 0.00% | 0.00% | 0.00% | **0.8338** | 4.44s |
| **Logistic Regression** | 79.63% | 5.19% | **74.82%** | 9.71% | 0.8257 | 1.61s |
| **Decision Tree** | 97.09% | 5.79% | 6.47% | 6.11% | 0.5246 | 0.74s |
| **Naive Bayes** | 5.08% | 1.49% | 98.20% | 2.94% | 0.5153 | 0.14s |

### Key Findings

1. **Best Model:** **Logistic Regression** emerged as the optimal choice for this problem
   - Highest recall (74.82%) - detects approximately 3/4 of all actual claims
   - Strong ROC-AUC score (0.8257)
   - Balanced performance across metrics

2. **Neural Network:** Achieved highest ROC-AUC (0.8338) but failed to detect claims (recall=0%)

3. **Class Imbalance Challenge:** All models struggled with precision due to severe class imbalance

4. **Clustering Results:**
   - **K-Means (k=2):** Successfully segmented 63,326 customers
   - **Silhouette Score:** 0.106 (moderate cluster separation)
   - **PCA Explained Variance:** 4.5% (PC1 + PC2), indicating high dimensionality

---

## 🛠️ Technical Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` - Basic plotting
  - `seaborn` - Statistical visualizations
- **ML Algorithms:** Decision Tree, Logistic Regression, Naive Bayes, Neural Network, K-Means, PCA
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📁 Project Structure

```
CSE422-Artificial-Intellegence-Project/
├── travel insurance.csv                      # Original dataset
├── cleaned_travel_insurance.csv              # Preprocessed dataset with cluster labels
├── model_results.csv                         # Model performance comparison
├── CSE422_Travel_Insurance_Analysis.ipynb    # Main analysis notebook
├── PROJECT_DOCUMENTATION.md                  # Detailed project documentation
├── README.md                                 # This file
│
├── Visualizations/
│   ├── catagorial_feature.png                # Categorical feature analysis
│   ├── class distributio claim.png           # Class distribution
│   ├── confusion matrix.png                  # Confusion matrices
│   ├── corelation heatmap.png                # Correlation analysis
│   ├── model accuracy comparison.png         # Model comparison chart
│   ├── neumeric distribution.png             # Numeric feature distributions
│   ├── PCA.png                              # PCA visualization
│   ├── precision-recall.png                 # Precision-recall curves
│   └── ROC.png                              # ROC curves
│
└── Reports/
    ├── CSE422_FINAL_REPORT_GROUP_9.pdf      # Final project report
    ├── CSE422_Travel_Insurance_Research_Paper.tex
    └── CSE422_Travel_Insurance_Research_Paper_Concise.tex
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Usage

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook CSE422_Travel_Insurance_Analysis.ipynb
   ```

2. **Run All Cells:**
   - Execute cells sequentially from Section 1 to Section 8
   - Each section builds upon previous results

3. **Expected Outputs:**
   - Processed dataset: `cleaned_travel_insurance.csv`
   - Model results: `model_results.csv`
   - Inline visualizations in the notebook

---

## 📝 Key Design Decisions

1. **Preprocessing Strategy:**
   - Median imputation for numeric features (robust to outliers)
   - Most frequent category for categorical features
   - StandardScaler for numeric features (algorithm compatibility)
   - OneHotEncoder with drop='first' (prevents multicollinearity)

2. **Handling Class Imbalance:**
   - Stratified train-test split
   - `class_weight='balanced'` in Logistic Regression
   - Focus on recall and ROC-AUC metrics (not just accuracy)

3. **Model Selection:**
   - Evaluated multiple algorithms for comprehensive comparison
   - Prioritized recall for business context (detecting actual claims)
   - ROC-AUC as primary metric for imbalanced data

4. **Evaluation Metrics:**
   - Multiple metrics to assess different aspects of performance
   - Confusion matrices for detailed analysis
   - ROC curves for threshold optimization

---

## 🎓 Learning Outcomes

This project demonstrates:

- End-to-end machine learning pipeline implementation
- Handling imbalanced datasets
- Multiple algorithm comparison and selection
- Feature engineering and preprocessing techniques
- Unsupervised learning for customer segmentation
- Comprehensive model evaluation and interpretation
- Business-focused insights and recommendations

---

## 📊 Output Files

1. **`cleaned_travel_insurance.csv`**
   - Preprocessed dataset ready for modeling
   - Includes cluster labels from K-Means analysis

2. **`model_results.csv`**
   - Complete model performance comparison
   - Contains accuracy, precision, recall, F1-score, ROC-AUC, and training time

3. **Visualization Images**
   - Multiple PNG files showing analysis results
   - Correlation heatmaps, confusion matrices, ROC curves, etc.

---

## 👥 Author

CSE422 Student - Group 9

---

## 📄 License

This project is for academic purposes as part of CSE422 - Artificial Intelligence course.

---

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/
- Dataset: Travel Insurance Claims Data

---

**Note:** For detailed technical documentation and implementation details, refer to `PROJECT_DOCUMENTATION.md` and the Jupyter notebook.
