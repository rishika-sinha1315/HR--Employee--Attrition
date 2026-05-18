# HR Attrition Analysis & Prediction

## Overview
A comprehensive HR analytics project that analyzes employee attrition patterns 
and predicts which employees are at risk of leaving using Machine Learning, 
enabling HR teams to take proactive retention measures.

## Business Problems Addressed
- Employees with no promotion after 3+ years
- Salary hike less than expected
- Overtime with low compensation
- Low job satisfaction and environment dissatisfaction
- Department and role specific attrition patterns
- Young employees and single employees leaving at higher rates

## Dataset
- Source: IBM HR Analytics Employee Attrition Dataset (Kaggle)
- Records: 1470 employees
- Features: 35 columns
- Target: Attrition (Yes/No)

## Tech Stack
- Python
- Pandas and NumPy
- SQLite
- Matplotlib and Seaborn
- NLTK and Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## Project Pipeline
1. Data Loading and Exploration
2. Data Cleaning
3. SQL Analysis using SQLite
4. Python Analysis
5. Visualizations
6. NLP TF-IDF Analysis
7. Feature Engineering
8. SMOTE Class Balancing
9. Model Training and Evaluation
10. Feature Importance
11. Employee Risk Score Generation

## Key Findings
- Sales department has highest attrition rate at 20.63%
- Single employees are 2x more likely to leave than married
- Overtime employees have 3x higher attrition 30.53% vs 10.44%
- Low salary employees under 3000 have highest attrition at 28.61%
- Newly promoted employees leave most due to promotion triggering job search
- Age group 18-25 has highest attrition at 34.78%
- TF-IDF NLP revealed Sales Executive and Lab Technician roles are 
  distinctively overrepresented among employees who left

## Custom Engineered Features
| Feature | Logic | Business Reason |
|---|---|---|
| PromotionRisk | YearsSinceLastPromotion >= 3 | No growth = higher exit risk |
| HikeRisk | PercentSalaryHike <= 14% | Low hike band had highest attrition |
| OvertimeRisk | Overtime = Yes AND Income < 5000 | Compounding stress factor |

## ML Models Comparison
| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| XGBoost + SMOTE | 85.7% | 0.4167 | 0.6393 |
| Logistic Regression | 76.2% | 0.3750 | 0.6343 |
| Random Forest | 83.7% | 0.3333 | 0.6013 |

**Best Model: XGBoost + SMOTE**


## Threshold Tuning
To maximize Recall (catching more at-risk employees), prediction threshold was tuned:
- Threshold 0.3 → Higher Recall, catches more at-risk employees ✅ Selected
- Threshold 0.4 → Balanced Precision and Recall
- Threshold 0.5 → Default, higher Precision

Selected threshold 0.3 because missing a truly at-risk employee is more 
costly than a false alarm (replacing an employee costs 50-200% of annual salary).

## Employee Risk Distribution
| Risk Level | Employees | Action |
|---|---|---|
| High Risk (>60%) | 22 | Immediate HR intervention |
| Medium Risk (30-60%) | 18 | Monitor closely |
| Low Risk (<30%) | 251 | Stable |

## HR Recommendations
1. Review compensation for Sales and Lab Technician roles
2. Offer stock options to retain at risk employees
3. Create clear promotion pathways for junior employees
4. Address overtime burden with proper compensation
5. Implement engagement programs for single and young employees

## Project Structure
hr-attrition-analysis/
    HR_Attrition_Analysis.ipynb
    IBM HR Analytics.csv
    README.md

## How to Run
1. Clone the repository
   git clone https://github.com/rishika-sinha1315/hr-attrition-analysis.git

2. Install dependencies
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn nltk

3. Open the notebook
   jupyter notebook HR_Attrition_Analysis.ipynb

## Author 
Rishika Sinha
IBM Data Science Certified
GitHub: https://github.com/rishika-sinha1315
