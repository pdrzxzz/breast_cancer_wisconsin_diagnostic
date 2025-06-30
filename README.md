# Breast Cancer Wisconsin Diagnostic Classification Project 🎗️

## Overview 🔍
This project implements a comprehensive machine learning workflow for classifying breast cancer tumors as malignant or benign using the Wisconsin Diagnostic Dataset. The solution compares multiple classification algorithms with advanced hyperparameter tuning, detailed visualizations, and robust evaluation techniques.

![Classifier Comparison](https://github.com/pdrzxzz/breast_cancer_wisconsin_diagnostic/blob/main/images/models_optimized/optimized-evaluation.png)

## Key Features ✨

### Advanced Preprocessing Pipeline ⚙️
- **PowerTransformer**: Applied Yeo-Johnson transformation to normalize feature distributions 📊
- **StandardScaler**: Standardized features to zero mean and unit variance ⚖️
- **Automated Pipeline**: Integrated transformations into a single preprocessing workflow 🔄
- **Outlier Handling**: IQR-based outlier detection and removal with configurable thresholds 🚨

### Hyperparameter Optimization 🎛️
- **Recall-Focused Tuning**: Used `make_scorer` to prioritize sensitivity (recall) for malignant cases 🎯
- **GridSearchCV**: Exhaustive search for Decision Trees and KNN 🔍
- **RandomizedSearchCV**: Efficient hyperparameter sampling for Logistic Regression 🎲
- **Custom Parameter Spaces**: Log-uniform distributions for regularization strength 📈

### Advanced Model Evaluation 📊
- **StratifiedShuffleSplit**: For reliable learning curve generation 📈
- **ROC Curves**: Visual comparison of classifier performance with AUC metrics 📉
- **Learning Curves**: Track precision, recall and F1 across training sizes 🏋️
- **Enhanced Confusion Matrices**: TN/FP/FN/TP visualization with absolute counts and percentages 🔢

### Comprehensive Visualizations 🎚
- **Correlation Analysis**: Feature-target relationships with color-coded bar plots 🌈
- **Box Plots**: Distribution analysis and outlier detection 📦
- **Histograms**: Feature distribution visualization 📊
- **Classification Heatmaps**: Styled presentation of precision, recall, and F1-scores 🔥

## Methodology 🧪

### Data Preparation 🧹
- Fetched dataset using `ucimlrepo` package 📦
- Stratified 70-30 train-test split preserving class distribution ✂️
- Removed low-correlation features (`symmetry2`, `fractal_dimension1`) 🗑️

### Model Comparison 🤖
Evaluated four base classifiers:
1. Decision Trees 🌳
2. Logistic Regression 📉
3. K-Nearest Neighbors 👥
4. Naive Bayes 🎓

### Optimization Techniques ⚡
- **Decision Trees**: Tuned max_depth, min_samples_leaf, and splitting criteria 🎚️
- **Logistic Regression**: Optimized regularization strength (C) and penalty type (L1/L2) ⚖️
- **KNN**: Adjusted neighbors count, weighting, and distance metrics 📏
- **Naive Bayes**: Calibrated var_smoothing parameter 🎚️

### Evaluation Metrics 📏
- Classification reports (precision, recall, F1-score) 📝
- Confusion matrices with detailed annotations 🔍
- AUC-ROC curves 📈
- Macro-averaged learning curves 📊

## Results 📌
The visualization system generates comparative reports including:
- Side-by-side classifier performance heatmaps 🔥
- ROC curve comparisons 📉
- Training size vs. metric learning curves 🏋️
- Before/after transformation distribution comparisons 🔄
- Feature correlation rankings 🏆

## Technologies Used 💻
- **Python** (3.10+) 🐍
- **scikit-learn** (Pipelines, Transformers, Models, Metrics) 🤖
- **pandas & numpy** (Data Manipulation) 🧮
- **matplotlib & seaborn** (Visualizations) 🎨
- **SciPy** (Statistical Functions) 📊
- **ucimlrepo** (Dataset Access) 📂

## Key Advanced Techniques 🚀
1. `PowerTransformer` for distribution normalization ⚡
2. `StratifiedShuffleSplit` for learning curve generation 📈
3. `make_scorer` with custom recall focus 🎯
4. Pipeline-based preprocessing workflow 🔄
5. Multi-metric learning curves with error bars 📊
6. Annotated confusion matrices 🔍
7. Correlation analysis with target mapping 🗺️
8. IQR-based outlier detection system 🚨
