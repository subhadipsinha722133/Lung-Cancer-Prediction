import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Prediction App",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 30px;}
    .sub-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 5px;}
    .info-text {font-size: 1.1rem; line-height: 1.6;}
    .prediction-high {background-color: #fccc; padding: 20px; border-radius: 10px; border: 1px solid #ff0000;}
    .prediction-medium {background-color: #fff4; padding: 20px; border-radius: 10px; border: 1px solid #ffcc00;}
    .prediction-low {background-color: #ccffcc; padding: 20px; border-radius: 10px; border: 1px solid #00ff00;}
    .feature-importance {background-color: #f0f; padding: 15px; border-radius: 10px;}
    .stButton>button {background-color: #4CAF50; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)
# App title and description
st.markdown('<p class="main-header"><h1>🫁 Lung Cancer Prediction App</h1></p>', unsafe_allow_html=True)
st.markdown("""
<p class="info-text">
This application predicts the likelihood of lung cancer based on patient health data and lifestyle factors.
Upload a CSV file with patient data or use the input form to make predictions.
</p>
""", unsafe_allow_html=True)

st.sidebar.header("Made By Subhadip 😎")
# Load and preprocess data
@st.cache_data
def load_data():
    # For demo purposes, we'll create a more realistic dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data with meaningful relationships
    data = {
        'Age': np.random.normal(60, 12, n_samples).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'Air Pollution': np.random.randint(1, 9, n_samples),
        'Alcohol use': np.random.randint(1, 9, n_samples),
        'Dust Allergy': np.random.randint(1, 9, n_samples),
        'OccuPational Hazards': np.random.randint(1, 9, n_samples),
        'Genetic Risk': np.random.randint(1, 9, n_samples),
        'chronic Lung Disease': np.random.randint(1, 9, n_samples),
        'Balanced Diet': np.random.randint(1, 9, n_samples),
        'Obesity': np.random.randint(1, 9, n_samples),
        'Smoking': np.random.randint(1, 9, n_samples),
        'Passive Smoker': np.random.randint(1, 9, n_samples),
        'Chest Pain': np.random.randint(1, 9, n_samples),
        'Coughing of Blood': np.random.randint(1, 9, n_samples),
        'Fatigue': np.random.randint(1, 9, n_samples),
        'Weight Loss': np.random.randint(1, 9, n_samples),
        'Shortness of Breath': np.random.randint(1, 9, n_samples),
        'Wheezing': np.random.randint(1, 9, n_samples),
        'Swallowing Difficulty': np.random.randint(1, 9, n_samples),
        'Clubbing of Finger Nails': np.random.randint(1, 9, n_samples),
        'Frequent Cold': np.random.randint(1, 9, n_samples),
        'Dry Cough': np.random.randint(1, 9, n_samples),
        'Snoring': np.random.randint(1, 9, n_samples),
    }
    df = pd.DataFrame(data)
    
    # Create a more realistic target variable with meaningful relationships
    # High risk factors: Smoking, Genetic Risk, Air Pollution, Age, Coughing of Blood
    risk_score = (
        df['Smoking'] * 0.3 + 
        df['Genetic Risk'] * 0.25 + 
        df['Air Pollution'] * 0.15 +
        df['Coughing of Blood'] * 0.2 +
        (df['Age'] > 60) * 2 +
        np.random.normal(0, 1, n_samples)
    )
    
    # Convert to categories
    df['Level'] = pd.cut(risk_score, 
                         bins=[-10, 2, 4, 10], 
                         labels=['Low', 'Medium', 'High'])
    
    return df
# Function to preprocess data
def preprocess_data(df, target='Level'):
    df = df.copy()
    
    # Drop unnecessary columns if they exist
    cols_to_drop = ["index", "Patient Id"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Encode target variable
    if target in df.columns:
        le_target = LabelEncoder()
        df[target] = le_target.fit_transform(df[target])
        target_mapping = {i: label for i, label in enumerate(le_target.classes_)}
    else:
        le_target = None
        target_mapping = None
    
    return df, le_target, target_mapping

# Function for feature selection
def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features, selector.scores_

# Function to train model with hyperparameter tuning
def train_model(X, y, model_type='random_forest'):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'random_forest':
        # Hyperparameter tuning for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
    elif model_type == 'gradient_boosting':
        # Hyperparameter tuning for Gradient Boosting
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
    elif model_type == 'svm':
        # Hyperparameter tuning for SVM
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        model = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return best_model, accuracy, cm, scaler, X_train.columns

# Main app
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Home", "Data Overview", "Feature Selection", "Model Training", "Prediction"])
    
    # Load data
    df = load_data()
    
    if app_mode == "Home":
        st.markdown('<p class="sub-header">About This App</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        This application uses machine learning models to predict the risk level of lung cancer 
        based on various health and lifestyle factors. The model is trained on patient data including:
        </p>
        <ul class="info-text">
            <li>Demographic information (Age, Gender)</li>
            <li>Environmental factors (Air Pollution, Dust Allergy, Occupational Hazards)</li>
            <li>Lifestyle factors (Alcohol use, Smoking, Balanced Diet, Obesity)</li>
            <li>Genetic risk factors</li>
            <li>Health symptoms (Chest Pain, Coughing of Blood, Shortness of Breath, etc.)</li>
        </ul>
        <p class="info-text">
        Use the navigation menu to explore the data, select important features, train the model, and make predictions.
        </p>
        """, unsafe_allow_html=True)
        # Show sample data
        if st.checkbox("Show sample data"):
            st.dataframe(df.head())
    
    elif app_mode == "Data Overview":
        st.markdown('<p class="sub-header">Data Overview</p>', unsafe_allow_html=True)
        
        # Data preview
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        
        # Basic statistics
        st.write("### Basic Statistics")
        st.write(df.describe())
        
        # Data visualization
        st.write("### Data Visualization")
        
        col1, col2 = st.columns(2)
        
        

