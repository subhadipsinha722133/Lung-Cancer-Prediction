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
