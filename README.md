# Global Cancer Healthcare Analytics - EDA & Predictive Modeling

*Analyzing Cancer Risk Factors to evaluate Severity and Survival outcomes using Data Analytics and Machine Learning*

## Table of Contents
- [Project Overview](#project-overview)
- [Project Vision](#project-vision)
- [Dataset Discription](#dataset-discription)
- [Key Objectives](#key-objectives)
- [Technical skills](#technical-skills)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Pipeline](#machine-learning-pipeline)
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

### 1. Exploratory Data Analysis (EDA)
- Identify patterns and healthcare trends
- Analyze cancer distribution across demographics
- Visualize disparities across countries and age groups
- Explore treatment cost variations
- Study survival trends and cancer severity

### 2. Statistical & Inferential Analysis
- Examine relationships between risk factors and severity score
- Analyze early-stage diagnosis distribution
- Evaluate treatment cost burden globally
- Study the relationship between treatment cost and survival years

### 3. Predictive Analytics & Machine Learning

- Predict cancer severity using machine learning
- Identify the most influential healthcare risk factors
- Evaluate feature importance using Random Forest Regression
- Optimize model performance using GridSearchCV

---

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

# Exploratory Data Analysis
*The project performs extensive EDA to understand healthcare patterns and patient characteristics.*

### Visualizations Performed

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

### 1. Data Preprocessing
- Converted categorical variables into numerical values using Label Encoding
- Removed unnecessary columns
- Split dataset into training and testing data

### 2. Model Training
**Implemented:**
- Random Forest Regressor

### 3. Hyperparameter Tuning
**Used:**
- GridSearchCV for model optimization.

### 4. Model Evaluation
**Evaluated model performance using:**
- R² Score

### 5. Feature Importance Analysis
**Identified the most important predictors influencing:**
- Cancer Severity
- Survival Outcomes

---

# Major Insights & Findings
### Geographic Disparities in Healthcare Costs

**Cancer treatment costs are significantly higher in developed countries such as:**
- United States
- Australia
- China

**While countries below show relatively lower treatment costs.**

- India
- Pakistan

This highlights global inequalities in healthcare affordability and accessibility.

### Gender-Based Treatment Cost Patterns
The analysis revealed minimal differences in treatment costs between male and female patients. This suggests relatively uniform healthcare pricing and treatment access across genders.

### Age-Based Increase in Treatment Costs
*Older age groups, especially patients aged 61+, experience significantly higher treatment costs.*

***Possible reasons include:***
- Intensive medical care
- Longer treatment duration
- Multiple health complications
- Higher hospitalization needs

### Role of Healthcare Systems
*Countries with stronger public healthcare systems such as:*
- Canada
- Germany
- United Kingdom
*show more stable treatment costs across age groups. This demonstrates how healthcare policies and government support can reduce financial burden on patients.*

### Risk Factors Influence Cancer Severity
*Statistical analysis, given below, indicates that all contribute to increased cancer severity:*

- Smoking
- Genetic Risk
- Obesity
- Air Pollution
- Alcohol Consumption

# Key Business & Healthcare Impact
***This project can help:***

- Healthcare analysts understand patient trends.
- Hospitals improve treatment planning.
- Governments identify healthcare inequalities.
- Researchers study risk factor impacts.
- Insurance companies evaluate healthcare costs.

---

# Project Structure

```bash
Global-Cancer-Healthcare-Analytics/
│
├── README.md
├── requirements.txt
├── data/
│   ├── raw_data.csv
│   └── Examine data.csv
│
├── Notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── machine_learning_model.ipynb
│
├── visualizations/
│   ├── correlation_heatmap.png
│   └── dashboard.png
│
└── Models/
```

---

# Future Improvements
**Potential future enhancements include:**
- Survival prediction models
- Real-time healthcare analytics
- Power BI integration
- SQL database integration

---

# Skills Demonstrated
### Data Analytics
- EDA
- Data Preprocessing
- Statistical Analysis
- Data Visualization

### Machine Learning
- Regression Modeling
- Hyperparameter Tuning
- Feature Engineering
- Model Evaluation

### Technical Skills
- Python Programming
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Git & GitHub

# Conclusion

*This project demonstrates how healthcare analytics and machine learning can uncover valuable insights from cancer patient data.*

**By combining statistical analysis, visualization, and predictive modeling, the project highlights:**
- Economic disparities in healthcare
- Risk factors influencing cancer severity
- Treatment cost patterns
- Importance of healthcare accessibility

**The analysis provides a strong foundation for future healthcare research, predictive systems, and policymaking initiatives.**

---

# Author & Contact

***Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning | Healthcare Analytics***

*junaidyusuf243@gmail.com*

### If you found this project useful, feel free to star the repository.




















