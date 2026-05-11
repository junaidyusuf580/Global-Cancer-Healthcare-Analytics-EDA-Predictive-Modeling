import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

#Load File "global_cancer_patients_2015_2024" 
health= pd.read_csv(r"C:\Users\Mohd Junaid\Downloads\global_cancer_patients_2015_2024 (1).csv")

#Data Overview
health.head()

#Overview of Data structure and dtypes 
health.info() 
health.duplicated().sum()
#Data has no null values , no duplicate values and no messy data which is now ideal for analysis. 

   
# Exploratory Data Analysis (EDA)

   
# **Statistical summary of Age**

 
health['Age'].describe()

 
**Inference on Age column dataset**
Average Mean is 54 years
Min-Max value= Mostly age is between 20 to 89 years young adult to aged person
Std deviation from mean= 20
Interquartile range= 35((Q1) to 72 (Q3))
This suggests a broad representation of both young and elderly patients in the dataset, which supports age-based comparative analysis.

 
#Visualization of Age through Histogram and KDE

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.kdeplot(data=health['Age'],fill= True, color= 'lightgreen')
plt.title("kde plot for Age")

plt.subplot(1,2,2)
sns.histplot(data=health['Age'], bins= 30, kde= False, color= 'cyan')
plt.title("Histogram plot for Age")

plt.tight_layout()
plt.show


   
# **Gender Analysis**

 
health['Gender'].value_counts()

 
#Gender visualization through Barplot
sns.barplot(x= health['Gender'].value_counts().index, y= health['Gender'].value_counts().values, 
            palette= ['blue', 'pink','green'])

for i,v in enumerate(health['Gender'].value_counts()):
    plt.text(i,v, str(v), ha= 'center', va= 'bottom')

plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")




 
#Inference
The dataset contains three gender categories(Male, Female ,other) with the most common being Male (16,796 records).
Gender dataset is uniformly distributed with minimal variance between them.

   
# **Country_Region Analysis**

 
health['Country_Region'].value_counts()

 
#Country_Region wise visualization
country_t= health['Country_Region'].value_counts()

plt.figure(figsize= [5,5])
plt.pie(x= country_t.values,
        labels= country_t.index,
        autopct= '%1.1f%%'
       )

plt.title('Region_Wise_Distribution')
plt.show()


 
#Inference
As per visualisation which shows that the Patients come from 10 different countries/regions, with Australia represent higher percentage (5,092 patients counts) and Canada showing lower. Number of data points for each country is almost same with minimum difference.This diversity enables cross-country comparison of cancer outcomes and treatment economic.


   
# **Cancer_type analysis**

 
health['Cancer_Type'].value_counts()

 
#Cancer_type visualization(category)

plt.figure(figsize=[10,5])
sns.barplot(x= health['Cancer_Type'].value_counts().index, y= health['Cancer_Type'].value_counts().values)

for i,v in enumerate(health['Cancer_Type'].value_counts()):
    plt.text(i,v, str(v), ha= 'center', va= 'bottom')

plt.title("CancerType_counts")
plt.xlabel("Cancer_type")
plt.ylabel("Counts")

 
Inference:
Above visulization shows that all cancer types in the dataset are nearly equally distributed, showing minimal variance across categories. 
This balanced distribution is ideal for analysis and the development of machine learning models, as it helps prevent bias toward any specific category.
Colon cancer has the highest occurrence, although the difference compared to other categories is marginal, while lung cancer has the lowest occurrence.
        

   
# **Cancer_Stage analysis**

 
health['Cancer_Stage'].value_counts()

 
#Cancer_Stage's visualization
plt.figure(figsize= [10,5])
sns.barplot(x=health['Cancer_Stage'].value_counts().index,
            y=health['Cancer_Stage'].value_counts().values,
            palette= ['orange', 'green', 'red', 'black', 'cyan'])

for i, v in enumerate(health['Cancer_Stage'].value_counts()):
    plt.text(i,v, str(v), ha= 'center', va='bottom')

plt.title('Stage of Cancer')
plt.xlabel('Cancer_stage')
plt.ylabel('Cancer_counts')
plt.show()

            

 
Inference:
Here the Stage II shows the highest occurrence while Stage 0 shows the low occurrence with marginal variance. 
This visualization shows that the dataset is uniformaly distributed accros all stages with minimal variances in counts. 
The balance distribution of the dataset across is ideal for analysis and machine learning, as it prevents bias toward any particular stage.”


   
# **Treatment_Cost_USD_ analysis(numerical)**

 
health['Treatment_Cost_USD'].describe()

 
#About Skewness

round(health['Treatment_Cost_USD'].skew(),4).item()


 
#Inference
The skew value is close to zero which means the dataset is almost symmetrically distributed. 

 
#plotting for Treatment_Cost_USD

plt.figure(figsize=[10,5])

plt.subplot(1,2,2)
sns.kdeplot(health['Treatment_Cost_USD'], fill=True, color= 'lightgreen', bw_adjust=1.5)
plt.title("KDE for Treatment_Cost_USD")

plt.subplot(1,2,1)
sns.histplot(health['Treatment_Cost_USD'], bins= 30, color= 'blue', edgecolor= 'black', stat='density', kde= True)
plt.title("Histogram for Treatment_Cost_USD")

plt.tight_layout()
plt.show()




 
#Inference
Above histogram shows a uniform distribution with wide spread.
While KDE confirms no skewness in the dataset or we can say platykurtic.

   
# **Analysis of risk factors(independent Variable)**

 
health.columns
risk_factors= ['Genetic_Risk','Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Treatment_Cost_USD']
summary= health[risk_factors].agg(['mean', 'std','min','max']).T
summary

 
#Inference
There are five risk factors ('Genetic_Risk','Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level', 'Treatment_Cost_USD') and these are
uniformly distributed with similar means and standard deviations, indicating a balanced dataset with consistent variabilty as shown above.
In contrast 'Treatment_Cost_USD' shows a high variabilty with a wide range from 5000 to 100000 suggesting significant difference in treatment cost
Overall, the dataset appears clean and well-suited for analysis and modeling.

   
# # Statistical & Inferential Analysis

   
# **Determine the relationship between risk factors and cancer severity.**

 
from scipy.stats import linregress

risk_factors= ['Genetic_Risk','Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
titles= ['Genetic Risk','Air Pollution', 'Alcohol Use', 'Smoking', 'Obesity Level']
colors= ["blue", "green", "orange", "red", "purple"]

plt.figure(figsize=[20,10])
for i, (factor, title, color) in enumerate(zip(risk_factors, titles, colors), start=1):
    plt.subplot(2,3,i)

    x=health[factor]
    y= health['Target_Severity_Score']
    slope, intercept, r_value, p_value, std_err= linregress(x,y)
    r_squared= r_value**2

    sns.lineplot(x= factor , y='Target_Severity_Score', data=health, color= color)
    plt.plot(x,slope*x+intercept, linewidth=2, label= "Regression line", color='black')
    plt.title(f"{title} vs Target_Severity_Score \n R2={ r_squared}, slope={slope}")
    plt.xlabel(factor)
    plt.ylabel('Target_Severity_Score')
    plt.legend()


    
plt.tight_layout()
plt.show()
    




   
# **To understand the contribution of various risk factors to cancer severity, line plots were generated for five primary variables:**  *Genetic Risk, Air Pollution, Alcohol Use, Smoking, and Obesity Level, plotted against the Target Severity Score.*
# 
# **All graphs reveal a positive relationship, indicating that as the level of a particular risk factor increases, the corresponding severity of the condition also tends to rise. However, the degree of association—measured by the slope and tightness of the confidence interval—varies across factors.**
# 
# **Genetic Risk vs Target Severity Score**
# R² = 0.23: A weak linear relationship. Only 23% of the variation in the target score is explained by Genetic Risk.
# Slope = 0.20: A positive slope indicates that as Genetic_Risk increases, the Target_Severity_Score also tends to increase. 
# 
# **Air Pollution vs Target Severity**
# R² = 0.13: A very weak relationship. Only 13% of the variation in the target score is explained Air Pollution.
# Slope = 0.15:
# A positive slope means that as air pollution increases, the severity score slightly increases. 
# 
# **Alcohol Use vs Target Severity Score**
# R² = 0.13:
# Similarly, the relationship between Alcohol_Use and Target_Severity_Score is also weak. Only 13% of the variation in the target score is explained by alcohol use.
# Slope = 0.15:
# The positive slope indicates that increased alcohol use correlates with a slight increase in target severity. 
# 
# **Smoking vs Target Severity Score**
# R² = 0.23:
# A weak relationship, similar to Genetic_Risk. Smoking explains only 23% of the variance in the target score, leaving the majority of the variation to be explained by other factors.
# Slope = 0.20:
# The positive slope implies that as smoking increases, the target severity score increases as well. 
# 
# **Obesity Level vs Target Severity Score**
# R² = 0.06:
# The weakest relationship among all factors. Only 6% of the variation in the target score is explained by obesity level, suggesting that obesity has a minimal effect on the target variable.
# Slope = 0.10:
# A positive slope, indicating a slight increase in the severity score as obesity level increases. However, due to the very low R², this is a weak and unreliable relationship.
# 

 
Key Takeaways:
Weak Linear Relationships:
The R² values for all risk factors are relatively low, ranging from 0.06 to 0.23 
Low R2 value indicates that no single factor strongly explains severity alone. Severity is likely influenced by multiple combined factor

Positive Trends:
All the slope values are positive, suggesting that as each risk factor increases, the Target_Severity_Score also tends to increase as well.
Relationship between variables is linear.Data points are closely aligned around regression line



   
# **Analyze the proportion of early-stage diagnoses by cancer type?**

 
#Unique colums name
health['Cancer_Type'].unique()

 
#Diagnosis of Lung in early stage percentage
count_stages=health[health['Cancer_Type']== 'Lung']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of lung cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Leukemia in early stage percentage
count_stages=health[health['Cancer_Type']== 'Leukemia']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of Leukemia cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Breast in early stage percentage
count_stages=health[health['Cancer_Type']== 'Breast']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of Breast cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Colon in early stage percentage
count_stages=health[health['Cancer_Type']== 'Colon']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of Colon cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Skin in early stage percentage
count_stages=health[health['Cancer_Type']== 'Skin']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of Skin cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Cervical in early stage percentage
count_stages=health[health['Cancer_Type']== 'Cervical']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of Cervical cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Prostate in early stage percentage
count_stages=health[health['Cancer_Type']== 'Prostate']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
prop= round(proportion, ndigits=2)
print(f"Proportion of Prostate cancer diagnosed at early stage 0 and stage I: {prop}")

 
#Diagnosis of Liver in early stage percentage
count_stages=health[health['Cancer_Type']== 'Liver']['Cancer_Stage'].value_counts()	
early_stage= count_stages.get('Stage 0',0)+count_stages.get('Stage I',0)
Total_count_stages= count_stages.sum()
proportion= (early_stage/Total_count_stages)*100
print(f"Proportion of Liver cancer diagnosed at early stage 0 and stage I: {proportion}")

   
# Above analysis show that early-stage diagnosis for various cancer types is relatively widespread, with most cancers having an early diagnosis rate between 38.43% and 40.61%. Liver Cancer shows the highest proportion, while Lung Cancer shows the lowest. These findings suggest that while screening and diagnostic methods are effective, improvements can still be made, particularly in lung cancer detection.
# 

   
# **Identify key predictors of cancer severity and survival years through "Karl Pearson and Spearman".**

 
#Independent variable and dependent variable
features= ['Age', 'Genetic_Risk','Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
targets= ['Survival_Years', 'Target_Severity_Score']

#calculate correlation
pearson_corr= health[features+targets].corr(method='pearson')
spearman_corr= health[features+targets].corr(method='spearman')

#slice out insight that only relationship with target variables
pearson_result= pearson_corr[targets]
spearman_result= spearman_corr[targets]

#Combine both correlation
correlation_data= pd.concat([pearson_result, spearman_result], axis=1, keys=["Pearson","Spearman"])
correlation_data

 
## Correlation Analysis Summary regarding 'Target_Severity_Score' vs 'Risk Factors'##

| Feature       | Pearson (Severity) | Spearman (Severity) | Interpretation                                               |
| ------------- | ------------------ | ------------------- | ------------------------------------------------------------ |
| Age           | -0.001             | -0.002              | No relationship → Age does not impact severity               |
| Genetic_Risk  | 0.478              | 0.472               | Strong positive → Higher genetic risk increases severity     |
| Air_Pollution | 0.367              | 0.357               | Moderate positive → Pollution contributes to higher severity |
| Alcohol_Use   | 0.363              | 0.355               | Moderate positive → Alcohol increases severity risk          |
| Smoking       | 0.484              | 0.478               | Strong positive → Smoking is a major factor in severity      |
| Obesity_Level | 0.251              | 0.243               | Weak to moderate → Obesity has some impact                   |


 
##Overall inference
The analysis shows that survival years are not strongly influenced by the selected features. However, cancer severity is moderately influenced by smoking, genetic risk, air pollution, and alcohol use. The similarity between Pearson and Spearman results suggests linear relationships with minimal impact from outliers.

   
# # Predictive Analytics & Machine Learning

   
# **_Key Identifier predictors of cancer severity through machine learning pipeline_**

 
# random forest for target severity score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

 
#converting categorical columns into numerical
catergorical_cols= ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
for cols in catergorical_cols:
    le= LabelEncoder()
    health[cols]=le.fit_transform(health[cols])


# Preparing  features and target
x= health.drop(columns= ['Patient_ID', 'Survival_Years', 'Target_Severity_Score','Treatment_Cost_USD'])
y_severity= health['Target_Severity_Score']

# Train-test split
X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(x,y_severity,  test_size= 0.2, random_state=40)

# Train the model
model=RandomForestRegressor(n_estimators= 200, max_depth= None, min_samples_split=2, min_samples_leaf=1, random_state= 40)
model.fit(X_train_s, Y_train_s)

# Evalaute
train_r2_severity= r2_score(Y_train_s, model.predict(X_train_s))
test_r2_severity= r2_score(Y_test_s, model.predict(X_test_s))






 
print(train_r2_severity)
print(test_r2_severity)


 
feature_importance_sevrity= pd.Series(model.feature_importances_, index= x.columns).sort_values(ascending= True)

#plotting of important features
plt.figure(figsize= [10,5])
feature_importance_sevrity.plot(kind= 'bar', color='skyblue')
plt.title("Feature Importance  for Target Severity Score (Random Forest)")
plt.xlabel('Features')
plt.ylabel('Important score')
plt.xticks(rotation= 45)
plt.tight_layout()
plt.show()
           

   
# **Feature	Importance	Interpretation**
# *Smoking	0.2336 most important predictor of severity score. The more a patient smokes, the higher their severity tends to be.*
# 
# **Genetic_Risk 0.2286**	*Strong genetic predisposition is nearly as important as smoking.*
# 
# **Treatment_Cost_USD 0.2133** *Higher treatment costs are associated with more severe conditions.*
# 
# **Alcohol_Use 0.1291** *Alcohol also plays a significant role.*
# 
# **Air_Pollution 0.1271** *Environmental factor—patients in more polluted areas have worse severity scores.*
# 
# **Obesity_Level	0.0573** *Has an effect, but much smaller.*
# 
# **Age to Gender	< 0.01** *Very low importance; these don’t explain much variation in severity score.*
# 
# **E.g., Smoking, Genetic Risk, AIr plollution are major influencers. These tells us if these is control on these factors that might reduce severity.**
# 

   
# # Check whether these risk-factors influence on Survival_Years apart from Target_Severity_Score.
# 

 
health.head()

 

# conveting categorical columns to numerical columns
categorical_cols= ["Gender","Country_Region","Cancer_Type","Cancer_Stage"]
for col in categorical_cols:
    le= LabelEncoder()
    health[col]=le.fit_transform(health[col])

# Preparing  features and input
X= health.drop(columns=["Patient_ID","Survival_Years","Target_Severity_Score","Treatment_Cost_USD"])
y_severity= health["Survival_Years"]

# train test split
X_train_s, X_test_s, y_train_s, y_test_s= train_test_split(X, y_severity, test_size=0.2, random_state=40)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}




 
# train the model
model= RandomForestRegressor(random_state=40)
GSC= GridSearchCV(model , param_grid, cv=3, scoring="r2", n_jobs=-1)
GSC.fit(X_train_s, y_train_s)

best_rf_severity= GSC.best_estimator_

# evalaute the model
train_r2_severity= r2_score(y_train_s , best_rf_severity.predict(X_train_s))
test_r2_severity= r2_score(y_test_s , best_rf_severity.predict(X_test_s))

 
print(train_r2_severity)
print(test_r2_severity)

 
"The above information predict that how long someone will survive that can not describe with the help these given predictors 
as per the given model result. These factors have no influention on Survival_Years.

   
# # Explore the economic burden of cancer treatment across different demographics and countries

 
health= pd.read_csv(r"C:\Users\Mohd Junaid\Downloads\global_cancer_patients_2015_2024 (1).csv")

 
health.head()

 
health['Age_group']=pd.cut(health['Age'], bins=[0,30,45,60,75,90], labels=['0-30', '31-45','46-60','61-75','76+'])

 
health.head()

 
country_age_cost=health.groupby(['Age_group', 'Age','Country_Region','Gender'], observed= False)['Treatment_Cost_USD'].mean().reset_index()

 
plt.figure(figsize=(20,10))
sns.barplot(data=country_age_cost, x='Country_Region', y='Treatment_Cost_USD', hue='Gender')
plt.title("Average cancer treatment cost by country and gender")
plt.show()

 
# Check with the heatmap
country_age_cost=health.groupby(['Age_group', 'Age','Country_Region','Gender'], observed= False)['Treatment_Cost_USD'].mean().reset_index()
health_heatmap=country_age_cost.pivot_table(index='Age_group', columns='Country_Region', values= 'Treatment_Cost_USD', aggfunc='mean')

 
plt.figure(figsize=(15,5))
sns.heatmap(data=health_heatmap, annot=True, fmt='0.0f')
plt.title("Treatment_Cost_USD burden on Demographic_region")
plt.show()

   
# **_Geographic Disparities in Economic Burden:_**
# *Cancer treatment costs are significantly higher in developed nations such as the USA, Australia, and China, revealing the heavy financial load in advanced healthcare systems. 
# Meanwhile, countries like India and Pakistan exhibit comparatively lower costs, likely due to lower healthcare pricing structures or limited access to advanced treatment. 
# People’s financial burden depends a lot on the country they live in. Some countries make treatment very expensive, while others are more affordable.*
# 
# **_Gender-Based Cost Patterns are Uniform:_**
# *The analysis shows that treatment costs for males and females are nearly equal in most countries.
# This means:
# there is no major difference in pricing between genders,
# both men and women face similar financial pressure during cancer treatment.*
# 
# **_Older People Usually Pay More for Treatment:_**
# *As age increases, treatment costs also increase, especially for people aged 61 years and above.
# This is more noticeable in countries like:Australia, United States.
# Senior citizens face higher medical expenses and need more financial support*
# 
# **_Strong Healthcare Systems Help Reduce Cost Differences:_**
# *Countries such as:Canada, Germany, United Kingdom have better public healthcare systems because of government support and insurance coverage:
# So, it can be say that Government healthcare support can reduce financial stress for cancer patients.*
#     
# 
#     

   
# # Overall Conclusion
# **The study shows that:**
# 1. Cancer treatment can become a major financial burden.
# 
# 2. Costs depend strongly on country and age.
# 
# 3. Gender has very little effect on treatment cost.
# 
# 4. Strong public healthcare systems help make treatment more affordable and fair for patients.   
# # Potential future enhancements include:
# **Survival prediction models**
# 
# **Real-time healthcare analytics**
# 
# **Power BI integration**
# 
# **SQL database integration**
# 

   
# # END

 



