# CSE422 Lab Project Documentation

## Project Overview
**Dataset:** Travel Insurance Claims Data  
**Problem Type:** Binary Classification (Predicting insurance claims: Yes/No)  
**Date Started:** August 27, 2025  
**Author:** CSE422 Student  

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Section 1: Setup & Data Loading](#section-1-setup--data-loading)
3. [Section 2: Dataset Description](#section-2-dataset-description)
4. [Section 3: Exploratory Data Analysis (EDA)](#section-3-exploratory-data-analysis-eda)
5. [Section 4: Preprocessing & Train/Test Split](#section-4-preprocessing--traintest-split)
6. [Design Decisions & Rationale](#design-decisions--rationale)
7. [Next Steps](#next-steps)

---

## Dataset Overview

### Initial Data Structure
- **Total Rows:** 63,327 samples
- **Total Columns:** 11 features
- **Target Variable:** `Claim` (Yes/No)
- **File Size:** Travel insurance.csv

### Column Details
1. **Agency** - Travel agency name (categorical)
2. **Agency Type** - Type of agency (categorical)
3. **Distribution Channel** - Online/Offline (categorical)
4. **Product Name** - Insurance plan type (categorical)
5. **Claim** - Whether claim was made (TARGET - categorical)
6. **Duration** - Trip duration in days (numeric)
7. **Destination** - Travel destination (categorical)
8. **Net Sales** - Sales amount (numeric)
9. **Commision (in value)** - Commission value (numeric)
10. **Gender** - Customer gender (categorical)
11. **Age** - Customer age (numeric)

---

## Section 1: Setup & Data Loading

### What We Did:
1. **Imported Libraries:**
   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical computations
   - `matplotlib.pyplot` - Basic plotting
   - `seaborn` - Statistical visualization

2. **Configuration Setup:**
   - Set visualization style for consistent plots
   - Configured pandas display options for better data viewing
   - Set color palette for plots

3. **Data Loading:**
   - Loaded CSV file into pandas DataFrame
   - Displayed basic dataset information (shape, head, info)
   - Analyzed missing values per column
   - Showed unique value counts for each column

### Why We Did It:
- **Library Selection:** These are the standard libraries for data science in Python, providing all necessary functionality for analysis and visualization
- **Configuration:** Ensures consistent, readable output and professional-looking visualizations
- **Initial Exploration:** Critical first step to understand data structure, quality, and potential issues before analysis

### Key Findings:
- Dataset has 63,327 rows and 11 columns
- Missing values detected in some columns (needs handling)
- Mix of numeric (4) and categorical (7) features
- Large dataset suitable for machine learning

---

## Section 2: Dataset Description

### What We Did:
1. **Target Variable Identification:**
   - Identified `Claim` as the target variable
   - Analyzed target distribution (Yes/No counts and percentages)
   - Determined this is a binary classification problem

2. **Feature Type Analysis:**
   - Separated numeric features: Duration, Net Sales, Commission, Age
   - Separated categorical features: Agency, Agency Type, Distribution Channel, Product Name, Destination, Gender
   - Identified data types for each feature

3. **Problem Type Classification:**
   - Confirmed binary classification based on target having 2 unique values
   - Separated features from target for modeling preparation

4. **Data Quality Assessment:**
   - Calculated missing value statistics
   - Assessed memory usage
   - Determined complete case percentage

### Why We Did It:
- **Target Identification:** Essential to understand what we're predicting and choose appropriate algorithms
- **Feature Type Separation:** Different preprocessing steps needed for numeric vs categorical features
- **Problem Type Classification:** Determines evaluation metrics, algorithms, and preprocessing strategies
- **Quality Assessment:** Identifies data cleaning needs and potential issues

### Key Findings:
- **Problem Type:** Binary Classification (Claim: Yes/No)
- **Feature Distribution:** 4 numeric, 7 categorical features
- **Data Quality:** Some missing values need handling
- **Class Balance:** Need to check for imbalanced classes

---

## Section 3: Exploratory Data Analysis (EDA)

### What We Did:

#### 1. Enhanced Correlation Analysis
**A. Numeric Features Correlation:**
- Created correlation heatmap for 4 numeric features only
- Used masked upper triangle for cleaner visualization
- Identified strong correlations (threshold: |correlation| > 0.3)

**B. Enhanced Correlation with Encoded Variables:**
- Encoded target variable: Claim (No=0, Yes=1)
- Applied smart encoding strategy for categorical features:
  - **One-hot encoding** for low cardinality features (≤10 categories)
  - **Target encoding** for high cardinality features (>10 categories)
- Created comprehensive correlation matrix including all features
- Focused visualization on top 15 features most correlated with target

#### 2. Warning Suppression
- Added matplotlib warning suppression for emoji font issues
- Maintained emoji in titles for better readability

### Why We Did It:

#### Correlation Analysis Rationale:
1. **Numeric-Only First:** Standard correlation analysis only works with numeric data, so we showed this baseline
2. **Enhanced Analysis:** To understand ALL feature relationships, we needed to encode categorical variables
3. **Smart Encoding Strategy:**
   - **One-hot encoding** for low cardinality: Preserves all category information without creating too many features
   - **Target encoding** for high cardinality: Reduces dimensionality while capturing relationship with target
4. **Target Focus:** Prioritized features most correlated with target for better feature selection insights

#### Technical Decisions:
- **Threshold Selection (0.3):** Balance between finding meaningful relationships without being too restrictive
- **Top 15 Features:** Manageable visualization size while capturing most important relationships
- **Missing Value Handling:** Used mean imputation to prevent correlation calculation errors

### Key Insights Expected:
- Which numeric features correlate with each other (multicollinearity detection)
- Which encoded categorical features predict claims best
- Feature importance ranking for model selection
- Potential feature engineering opportunities

---

## Section 4: Preprocessing & Train/Test Split

### What We Did:

#### 1. Data Preparation
- **Target Definition:** Set `Claim` column as target variable (y)
- **Feature Definition:** All other columns as features (X)
- **Problem Classification:** Confirmed Binary Classification problem type

#### 2. Feature Pipeline Setup

**Numeric Features:** [`Duration`, `Net Sales`, `Commision (in value)`, `Age`]
- **SimpleImputer (median strategy):** Fills missing values with median for robust handling of outliers
- **StandardScaler:** Normalizes features to have mean=0 and std=1 for algorithm compatibility

**Categorical Features:** [`Agency`, `Agency Type`, `Distribution Channel`, `Product Name`, `Destination`, `Gender`]
- **SimpleImputer (most frequent strategy):** Fills missing values with most common category
- **OneHotEncoder:** Converts categories to binary columns
  - `drop='first'`: Removes first category to avoid multicollinearity
  - `handle_unknown='ignore'`: Safely handles new categories in test set

#### 3. ColumnTransformer Pipeline
- **Combined Processing:** Unified pipeline applying different preprocessing to numeric vs categorical features
- **Remainder='drop':** Ensures only specified features are processed
- **Pipeline Structure:** 
  - Numeric Pipeline: Imputation → Scaling
  - Categorical Pipeline: Imputation → One-Hot Encoding

#### 4. Train/Test Split
- **Split Ratio:** 70% training (44,328 samples) / 30% testing (18,998 samples)
- **Stratification:** Used for binary classification to maintain class balance
- **Random State:** Set to 42 for reproducible results

#### 5. Data Transformation
- **Fit on Training:** Preprocessor learned patterns from training data only
- **Transform Both Sets:** Applied same preprocessing to training and test data
- **Final Shapes:** 
  - X_train_processed: (44,328 × 181 features)
  - X_test_processed: (18,998 × 181 features)

### Why We Did It:

#### Missing Value Handling
- **Median for Numeric:** Robust to outliers, preserves distribution shape
- **Most Frequent for Categorical:** Maintains class distribution, logical default

#### Feature Scaling
- **StandardScaler:** Required for algorithms sensitive to feature magnitude (SVM, Neural Networks, Logistic Regression)
- **Prevents Feature Dominance:** Ensures all features contribute equally

#### One-Hot Encoding
- **Algorithm Compatibility:** Most ML algorithms require numeric input
- **Information Preservation:** Maintains all categorical information without imposed ordering
- **Multicollinearity Prevention:** Dropping first category avoids linear dependence

#### Stratified Sampling
- **Class Balance:** Ensures training and test sets have same proportion of Yes/No claims
- **Representative Testing:** Test performance reflects real-world class distribution
- **Reliable Metrics:** Prevents biased accuracy due to class imbalance

### Key Outcomes:
- **Clean Dataset:** No missing values, all features properly encoded
- **Model-Ready Format:** All features numeric and scaled appropriately
- **Expanded Feature Space:** 181 total features after one-hot encoding (vs original 11)
- **Balanced Split:** Both sets maintain original class distribution
- **Pipeline Reusability:** Can process new data with same transformations

---

## Design Decisions & Rationale

### 1. Target Variable Selection
**Decision:** Use `Claim` as target variable  
**Rationale:** 
- Clear binary outcome (Yes/No)
- Business relevance (predicting insurance claims)
- Sufficient samples in both classes

### 2. Feature Encoding Strategy
**Decision:** Mixed encoding approach (one-hot + target encoding)  
**Rationale:**
- **One-hot for low cardinality:** Preserves all information, prevents data leakage
- **Target encoding for high cardinality:** Reduces dimensionality, captures target relationship
- **Threshold (10 categories):** Balance between information preservation and computational efficiency

### 3. Correlation Analysis Approach
**Decision:** Two-stage correlation analysis (numeric-only + enhanced)  
**Rationale:**
- **Educational Value:** Shows standard approach first, then enhanced method
- **Comprehensive View:** Captures all feature relationships
- **Feature Selection:** Identifies most predictive features

### 4. Visualization Choices
**Decision:** Professional plots with clear titles and labels  
**Rationale:**
- **Academic Standards:** Suitable for lab report presentation
- **Interpretability:** Easy to understand for stakeholders
- **Reproducibility:** Consistent styling across all plots

### 5. Preprocessing Strategy (Section 4)
**Decision:** Pipeline-based preprocessing with ColumnTransformer  
**Rationale:**
- **Reproducibility:** Same transformations applied consistently to train/test
- **No Data Leakage:** Fit only on training data, transform both sets
- **Scalability:** Easy to apply to new data
- **Algorithm Compatibility:** All features converted to numeric format

**Decision:** Median imputation for numeric, most frequent for categorical  
**Rationale:**
- **Robust to Outliers:** Median less affected by extreme values than mean
- **Logical Defaults:** Most frequent category preserves class distribution
- **Conservative Approach:** Minimal assumptions about missing data patterns

**Decision:** StandardScaler for numeric features  
**Rationale:**
- **Algorithm Requirements:** Many algorithms assume similar feature scales
- **Convergence:** Faster training for gradient-based algorithms
- **Fair Feature Weighting:** Prevents high-magnitude features from dominating

**Decision:** OneHotEncoder with drop='first' for categorical  
**Rationale:**
- **Complete Information:** Preserves all categorical relationships
- **Linear Independence:** Dropping first prevents multicollinearity
- **Unknown Handling:** Gracefully manages new categories in test data

---

## Next Steps

### Section 5: Model Training (Planned)
1. **Algorithms to Test:**
   - Decision Tree
   - Logistic Regression (class_weight="balanced")
   - Naive Bayes
   - Neural Network (MLPClassifier)
2. **Evaluation Metrics:**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Confusion matrices and ROC curves

### Section 6: Model Comparison (Planned)
- Performance comparison across all models
- Best model identification
- Feature importance analysis

### Section 7: Unsupervised Learning (K-Means Clustering & PCA)

### K-Means Clustering
- Performed K-Means clustering to segment customers based on selected features.
- Visualized clusters using PCA-reduced 2D scatter plot.
- Added cluster labels to the cleaned DataFrame.
- **Number of Clusters:** 2 clusters identified
- **Total Samples Clustered:** 63,326 customers

### Clustering Evaluation Metrics
- Computed clustering quality metrics with the following results:
  - **Silhouette Score:** 0.106 (range: -1 to 1, higher is better)
    - *Interpretation:* Moderate cluster separation; clusters are distinguishable but have some overlap
  - **Davies-Bouldin Index:** 4.266 (lower is better)
    - *Interpretation:* Moderate clustering quality; some within-cluster scatter relative to between-cluster separation
  - **Calinski-Harabasz Index:** 1455.035 (higher is better)
    - *Interpretation:* Decent cluster density ratio; clusters are reasonably well-defined

### PCA Explained Variance Analysis
- Applied Principal Component Analysis for dimensionality reduction and visualization
- **PC1 (First Component):** 2.6% of total variance
- **PC2 (Second Component):** 1.9% of total variance
- **Cumulative Variance (PC1 + PC2):** 4.5% of total variance
- *Interpretation:* Low explained variance indicates high dimensionality in the data; many features contribute to the variation, suggesting complex customer behavior patterns that cannot be easily captured in 2D

---

## Section 8: Save Outputs
- Saved cleaned dataset with cluster labels to `cleaned_travel_insurance.csv`.
- Saved model comparison results to `model_results.csv`.
- Verified that plots are displayed inline in the notebook.

---

## Saved Models & Outputs
- **Cleaned Dataset:** `cleaned_travel_insurance.csv` (includes cluster labels)
- **Model Results:** `model_results.csv` (contains accuracy and metrics for all supervised models)
- **Clustering Metrics:** Displayed in notebook (Section 7.1)
- **PCA Explained Variance:** Displayed in notebook (Section 7.1)
- **Plots:** All visualizations are shown inline in the notebook.

---

## Learning Objectives Achieved

1. **Data Understanding:** Comprehensive dataset exploration and characterization
2. **Statistical Analysis:** Correlation analysis and relationship identification
3. **Preprocessing Preparation:** Feature type identification and encoding strategy development
4. **Visualization Skills:** Professional plot creation and interpretation
5. **Problem Formulation:** Binary classification problem setup

---

## File Structure
```
CSE422 FINAL PROJECT/
├── travel insurance.csv                    # Original dataset
├── CSE422_Travel_Insurance_Analysis.ipynb # Main analysis notebook
├── PROJECT_DOCUMENTATION.md               # This documentation file
├── cleaned_travel_insurance.csv           # (To be created)
└── model_results.csv                       # (To be created)
```

---

*This documentation will be updated after each section completion to maintain a complete record of the analysis process and decision-making rationale.*
