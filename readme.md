# 🫀 Heart Disease Prediction System

A machine learning web app predicting cardiovascular disease risk using 13 clinical features.

## 📊 Dataset
**Kaggle:** Heart Disease Prediction by rishidamarla  
Link: https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction

**Features:** Age, Sex, Chest Pain Type, BP, Cholesterol, FBS over 120, EKG Results,
Max HR, Exercise Angina, ST Depression, Slope of ST, Number of Vessels Fluro, Thallium

## 🤖 Algorithms Used
| Algorithm | Type |
|---|---|
| Random Forest | Ensemble |
| Logistic Regression | Linear |
| SVM | Kernel-based |
| KNN | Instance-based |

## 🖥️ App Features
- 🔬 Real-time patient risk prediction
- 🤖 Side-by-side comparison of all 4 algorithms
- 📈 Confusion matrix + accuracy comparison (Test vs CV)
- 📉 ROC curves with AUC scores for all models
- 📋 Feature importance, correlation plots, dataset preview

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (FREE)

1. Upload your `Heart_Disease_Prediction.csv` from Kaggle into the project folder
2. Push to GitHub:
```bash
git init
git add .
git commit -m "Heart disease prediction system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git
git push -u origin main
```
3. Go to [share.streamlit.io](https://share.streamlit.io) → New App → select repo → `app.py` → Deploy 🎉

---

## 📁 Project Structure
```
heart-disease-prediction/
├── app.py                          # Main Streamlit app
├── Heart_Disease_Prediction.csv    # Dataset from Kaggle (add this!)
├── requirements.txt
└── README.md
```