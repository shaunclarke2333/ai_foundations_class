from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
from shaun_clarke_csc6313 import PatientDataGenerator, ClassifierShowdown


warnings.filterwarnings('ignore')

data_generator = PatientDataGenerator()
data = data_generator.generate_patient_data()

get_classifier = ClassifierShowdown(data)
get_classifier.preprocess_data()
get_classifier.train_models()
get_classifier.evaluate_models()
get_classifier.run_inference()