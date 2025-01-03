## Project Overview
This project analyzes census data to predict individual income levels and identify key factors that influence earning potential. Using machine learning techniques, we developed a model to predict whether an individual earns more than $50,000 annually based on demographic and employment characteristics.

## Business Problem
Organizations and policymakers need data-driven insights to:
- Understand factors that most strongly influence income levels
- Identify potential areas of income inequality
- Guide policy decisions around education and workforce development
- Help employers develop fair compensation structures
- Support career development and economic mobility initiatives

## Dataset Description
The analysis uses the Adult Census Income dataset containing 48,842 instances with 15 attributes including:
- Demographic information (age, race, native country)
- Education levels
- Employment characteristics (occupation, work class, hours per week)
- Personal attributes (marital status)
- Target variable: Income (>50K or ≤50K)

## Technical Requirements
- Python
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

# Methodology
1. **Data Preprocessing**
   - Removed records with missing values in WorkClass and Occupation
   - Encoded categorical variables using one-hot encoding
   - Standardized numerical features
   - Split data into training (80%) and testing (20%) sets

2. **Class Imbalance Handling**
   - Implemented SMOTE (Synthetic Minority Over-sampling Technique)
   - Compared model performance with and without SMOTE

3. **Model Development**
   - Trained Random Forest Classifier
   - Optimized model parameters
   - Evaluated performance using multiple metrics

## Exploratory Data Analysis

### Work Class Distribution
- Private sector dominates (22,696 individuals)
- Self-employed (not incorporated): 2,541
- Government jobs (combined): ~4,351
- Very few without pay or never worked (<25)

### Education Distribution
- High school graduates: ~9,969
- Some college: 6,777
- Bachelor's degree: 5,182
- Advanced degrees: ~2,631
- Lower education levels show smaller representation

### Occupation Distribution
- Professional specialty: 4,140
- Craft-repair: 4,099
- Executive-managerial: 4,066
- Administrative-clerical: 3,770
- Sales: 3,650

### Hours Worked Distribution
- Strong peak at 40 hours/week
- Right-skewed distribution
- Significant variation in work schedules
- Some individuals working 80+ hours/week

## Model Development

### Feature Engineering
- Converted categorical variables to numeric using one-hot encoding
- Scaled numerical features using StandardScaler
- Created binary target variable (0: ≤50K, 1: >50K)

### Model Selection
Implemented Random Forest Classifier with:
- 100 estimators
- Random state: 42
- Default hyperparameters

### SMOTE Implementation
- Applied to training data only
- Created synthetic samples of minority class
- Balanced class distribution in training set

## Results and Performance

### Model Performance Without SMOTE
The initial Random Forest model without addressing class imbalance showed:
- Overall accuracy: 86%
- Strong performance in predicting the majority class (≤50K):
  - Precision: 0.89
  - Recall: 0.95
  - F1-score: 0.92
- Weaker performance in predicting the minority class (>50K):
  - Precision: 0.75
  - Recall: 0.57
  - F1-score: 0.65
- This indicates a bias towards the majority class, potentially missing many high-income individuals

### Model Performance With SMOTE
After implementing SMOTE to address class imbalance, the model showed:
- Overall accuracy: 84%
- More balanced performance for the majority class (≤50K):
  - Precision: 0.91
  - Recall: 0.89
  - F1-score: 0.90
- Improved prediction of the minority class (>50K):
  - Precision: 0.65
  - Recall: 0.70
  - F1-score: 0.67
- The slight decrease in overall accuracy was traded for better minority class detection

### Feature Importance Analysis
The Random Forest model identified the following as the top 10 most influential features:
1. Education level (0.142 importance score)
2. Age (0.128 importance score)
3. Hours per week (0.112 importance score)
4. Occupation type (0.098 importance score)
5. Work class (0.087 importance score)
6. Marital status (0.076 importance score)
7. Capital gain (0.065 importance score)
8. Capital loss (0.054 importance score)
9. Relationship status (0.048 importance score)
10. Race (0.042 importance score)

### Model Comparison Summary
- The SMOTE-enhanced model showed more balanced prediction capabilities
- While overall accuracy decreased slightly (86% to 84%), the model became more reliable for predicting high-income individuals
- The improvement in minority class recall (from 0.57 to 0.70) indicates better identification of high-income individuals
- The trade-off between precision and recall provides a more practical model for real-world applications
- Feature importance remained relatively stable across both models, confirming the significance of education and age in income prediction

These results suggest that while the SMOTE-enhanced model may have slightly lower overall accuracy, it provides more reliable and balanced predictions across both income categories, making it more suitable for practical applications where identifying high-income individuals is as important as identifying lower-income individuals.

## Key Insights

### Demographic Patterns
- Strong correlation between education and income
- Age shows significant impact on earning potential
- Work experience influences income levels
- Certain occupations associated with higher income

### Income Distribution
- Significant class imbalance (~75% earning ≤50K)
- Clear income disparities across different demographics
- Strong relationship between work hours and income

### Model Performance
- SMOTE improved minority class prediction
- Better balanced accuracy with resampling
- Trade-off between precision and recall

## Recommendations

### For Policymakers
1. Expand access to higher education
2. Develop targeted economic policies
3. Address regional income disparities

### For Educational Institutions
1. Align programs with high-income occupations
2. Develop technical and managerial skills training
3. Create adult education opportunities

### For Employers
1. Review compensation structures
2. Implement career development programs
3. Address demographic income gaps

### For Workforce Development
1. Focus on high-demand occupations
2. Create mentorship programs
3. Support self-employed individuals

## Limitations
1. Binary income classification simplifies complex income distributions
2. Dataset may contain historical biases
3. Some important real-world factors may not be captured
4. Geographic limitations in the data
