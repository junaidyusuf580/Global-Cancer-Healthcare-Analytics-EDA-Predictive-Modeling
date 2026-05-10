# Global Cancer Healthcare Analytics - EDA & Predictive Modeling

*Analyzing Cancer Risk Factors to evaluate Severity and Survival outcomes using Data Analytics and Machine Learning*

## Table of Contents
- [Project Overview](#project-overview)
- [Project Vision](#project-vision)
- [Dataset Discription](#dataset-discription)
- [Key Objectives](#key-ojectives)
- [Technical skills](#technical-skills)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-(EDA))
- [Machine Learning Pipeline](#machine-learing-pipeline)
- [Major Insights & Findings](#major-insights--findings)
- [Key Business & Healthcare Impact](#key-business--healthcare-impact)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Skills Demonstrated](#skills-demonstrated)
- [Conclusion](#conclusion)
- [Author & Contact](#Author&Contact)

### If you found this project useful, feel free to star the repository.

---
# Project Overview
This project focuses on analyzing global cancer patient data to uncover meaningful healthcare insights, treatment disparities, survival trends, and economic burdens associated with cancer treatment.

Using a dataset of 50,000+ cancer patient records collected between 2015 and 2024, this project combines:

- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Data Visualization
- Machine Learning
- Predictive Analytics

The goal is to transform raw healthcare data into actionable insights that support better clinical understanding and healthcare decision-making reagrding cancer treatment.


---
# Project Vision
This healthcare analytics project aims to bridge the gap between raw medical data and data-driven healthcare decisions.

By analyzing patient demographics, lifestyle risks, environmental exposure, treatment costs, cancer severity and survival outcomes, the project highlights:

- Global healthcare disparities
- Economic burden of cancer treatment
- Risk factors affecting cancer severity
- Predictors of patient survival
- Country-wise variations in treatment costs

---
# Dataset Discription

## The dataset contains detailed healthcare and cancer-related information including:

### Demographic Information
- Age
- Gender
- Country/Region
- Year of Diagnosis

### Genetic & Lifestyle Risk Factors
- Genetic Risk
- Smoking
- Alcohol Use
- Obesity Level

### Environmental Exposure
- Air Pollution

### Clinical Information
- Cancer Type
- Cancer Stage
- Severity Score
- Survival Years
- Treatment Cost (USD)

---

# Key Objectives

## 1. Exploratory Data Analysis (EDA)
- Identify patterns and healthcare trends
- Analyze cancer distribution across demographics
- Visualize disparities across countries and age groups
- Explore treatment cost variations
- Study survival trends and cancer severity

## 2. Statistical & Inferential Analysis
- Examine relationships between risk factors and severity score
- Analyze early-stage diagnosis distribution
- Evaluate treatment cost burden globally
- Study the relationship between treatment cost and survival years

## 3. Predictive Analytics & Machine Learning

- Predict cancer severity using machine learning
- Identify the most influential healthcare risk factors
- Evaluate feature importance using Random Forest Regression
- Optimize model performance using GridSearchCV

# Technical skills
| Technology | Purpose |
|---|---|
| Python | Data Analytics & Machine Leaning |
| Pandas | Data Cleaning & Manipulation |
| Numpy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit Learn | Machine Learning Models |
| Jupyter Notebooks | Development Environment |
| Git & GitHub | Version Control & Project Hosting |

---

# Exploratory Data Analysis (EDA)
*The project performs extensive EDA to understand healthcare patterns and patient characteristics.*

## Visualizations Performed

*Distribution Analysis*
- Age Distribution (Histogram + KDE)
- Treatment Cost Distribution (Histogram + KDE)
- Survival Year Analysis

*Categorical Analysis*
- Gender Distribution (Bar Graph)
- Cancer Type Distribution (Bar Graph)
- Cancer Stage Distribution (Bar Graph)
- Country-wise Patient Analysis (Pie Graph)

*Comparative Analysis*
- Treatment Cost by Country & Gender
- Age Group vs Treatment Cost
- Country-wise Economic Burden Heatmap

*Correlation & Relationship Analysis*
- Risk Factors vs Cancer Severity
- Smoking vs Severity
- Obesity vs Severity
- Genetic Risk vs Severity
- Air Pollution vs Severity

---

# Machine Learning Pipeline
*The project includes a complete machine learning workflow to predict cancer severity.*

## Workflow Steps

## 1.Data Preprocessing
- Converted categorical variables into numerical values using Label Encoding
- Removed unnecessary columns
- Split dataset into training and testing data

## 2. Model Training
### Implemented:
- Random Forest Regressor

## 3. Hyperparameter Tuning
### Used:
- GridSearchCV for model optimization.

## 4. Model Evaluation
### Evaluated model performance using:
- R² Score

## 5. Feature Importance Analysis
### Identified the most important predictors influencing:
- Cancer Severity
- Survival Outcomes

---












