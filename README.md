# 🫁 **Lung Cancer Prediction Web App**  
> 🌐 A Streamlit-based machine learning web application that predicts lung cancer risk levels using patient health and lifestyle data — with stunning visuals and intelligent insights.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📋 **Table of Contents**
- ✨ [Features](#-features)
- 🎥 [Live Demo](#-demo)
- 🛠 [Installation](#-installation)
- 🚀 [Usage](#-usage)
- 📁 [Project Structure](#-project-structure)
- 🤖 [Model Details](#-model-details)
- 📊 [Data Description](#-data-description)
- 🤝 [Contributing](#-contributing)
- 📄 [License](#-license)
- ⚠️ [Disclaimer](#%EF%B8%8F-disclaimer)
- 📞 [Support](#-support)

---

## ✨ **Features**
🧠 **AI-Powered Predictions** — Accurate lung cancer risk classification  
🎛 **Interactive Web Interface** — Built with Streamlit for simplicity and beauty  
🌲 **Multiple ML Models** — Random Forest, Gradient Boosting, SVM  
🎯 **Feature Selection** — Automated identification of the most predictive attributes  
⚙️ **Model Optimization** — Hyperparameter tuning for better accuracy  
📈 **Data Visualization** — Heatmaps, feature importance, and interactive plots  
📊 **Detailed Insights** — Probability distributions and contributing factor analysis  
📱 **Responsive Design** — Modern, clean, and mobile-friendly layout  

---

## 🎥 **Live Demo**
🔗 **Demo:** [Click here to explore the deployed app](https://lung-cancer-prediction-um7e43hxcjgkamuspjkcp8.streamlit.app/)  

---

## 🛠 **Installation**

### 🧾 **Prerequisites**
- 🐍 Python 3.7 or higher  
- 📦 pip (Python package manager)

### ⚙️ **Setup Steps**
```bash
# 1️⃣ Clone the repository
https://github.com/subhadipsinha722133/Lung-Cancer-Prediction.git
cd lung-cancer-prediction-app

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
````

---

## 🚀 **Usage**

Run the application with:

```bash
streamlit run main_app.py
```

Then open 🌐 [http://localhost:8501](http://localhost:8501) in your browser.

### 🧭 **Navigation Guide**

* 🏠 **Home:** Overview of the application
* 📊 **Data Overview:** Explore dataset and visual patterns
* 🧩 **Feature Selection:** Identify top contributing factors
* 🤖 **Model Training:** Train and evaluate models
* 🔍 **Prediction:** Input patient data to get real-time risk assessment

**Steps for Prediction:**
1️⃣ Go to the **Prediction** page
2️⃣ Adjust sliders to match patient data
3️⃣ Click **“Predict Risk Level”** 🧮
4️⃣ View risk level (Low / Medium / High) with detailed charts 📊

---

## 📁 **Project Structure**

```text
Lung-Cancer-Prediction/
│
├── main_app.py                   # 🧠 Main Streamlit application
├── requirements.txt              # 📦 Dependencies
├── README.md                     # 📘 Documentation
├──  Lung Cancer Prediction.csv   # 📂 Dataset folder
|── model.pkl
|── main.ipynb

```

---

## 🤖 **Model Details**

### 🧩 **Supported Models**

* 🌲 **Random Forest Classifier** — Ensemble of decision trees
* 🔥 **Gradient Boosting Classifier** — Sequential error correction
* ⚡ **Support Vector Machine (SVM)** — Optimal hyperplane for classification

### 🛠 **Model Optimization**

* 🧮 Hyperparameter tuning using `GridSearchCV`
* 📊 Feature selection with `SelectKBest (ANOVA F-test)`
* 📈 Standard scaling via `StandardScaler`
* 🎯 Stratified sampling for balanced data

### 🧾 **Evaluation Metrics**

* ✅ Accuracy Score
* 📉 Confusion Matrix
* 🧠 Classification Report (Precision, Recall, F1-Score)
* 🌟 Feature Importance Visualization

---

## 📊 **Data Description**

### 👩‍⚕️ **Demographic**

* **Age** 🧓 — 20–80 years
* **Gender** 🚹🚺 — Male / Female

### 🌫️ **Environmental Factors**

* Air Pollution
* Dust Allergy
* Occupational Hazards

### 🍷 **Lifestyle Factors**

* Alcohol Use
* Smoking / Passive Smoking
* Balanced Diet
* Obesity

### 🧬 **Genetic & Health Factors**

* Genetic Risk
* Chronic Lung Disease

### 😷 **Symptoms**

* Chest Pain
* Coughing of Blood
* Fatigue
* Weight Loss
* Shortness of Breath
* Wheezing
* Swallowing Difficulty
* Clubbing of Nails
* Frequent Cold
* Dry Cough
* Snoring

🎯 **Target Variable:**

* `Level` — Risk classification (**Low**, **Medium**, **High**)

---

## 🤝 **Contributing**

We ❤️ contributions! Here's how you can help improve this project:

1. 🍴 Fork the repo
2. 🌿 Create a new branch:

   ```bash
   git checkout -b feature-name
   ```
3. ✏️ Make your changes and test
4. 💾 Commit:

   ```bash
   git commit -m "Add new feature"
   ```
5. 🚀 Push:

   ```bash
   git push origin feature-name
   ```
6. 🔁 Submit a pull request

### 🔧 **Areas for Improvement**

* 🧠 Add deep learning models
* 📊 Enhance data visualization
* 📈 Include a larger dataset
* 🤖 Add deployment automation scripts
* 🌍 Add multi-language support
* 📱 Improve mobile responsiveness

---

## 📄 **License**

📜 This project is licensed under the **Boost Software License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ **Disclaimer**

> ⚕️ This application is for **educational and research purposes only**.
> It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare provider regarding medical conditions.

---

## 📞 **Support**

💬 For questions, issues, or suggestions:

* 🔍 Check existing issues
* 🐛 Open a new issue
* 📧 Contact: **[sinhasubhadip34@gmail.com](mailto:sinhasubhadip34@gmail.com)**

🩺 **Note:** For production deployment with real patient data, ensure compliance with data privacy laws such as **HIPAA (US)** or **GDPR (EU)**.

---

⭐ **If you like this project, please give it a star!** ⭐

> *Made with ❤️ by Subhadip Sinha*

```

---
