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

# Initializing the PatientDataGenerator object
data_generator: PatientDataGenerator = PatientDataGenerator()
# Generating patient data and saving it to a csv
data: pd.DataFrame = data_generator.generate_patient_data()
print(f"\nDataset saved to: classifier_patient_data.csv")
# Initializing the ClassifierShowdown object with the dataframe that was generated 
get_classifier: ClassifierShowdown = ClassifierShowdown(data)
# Splitting the data into X(inputs) and Y(targets(diganosis)) and transforming it with standard scaler
get_classifier.preprocess_data()
# training all three classification models
get_classifier.train_models()
# Generating predictions and calculating the accuracy scores of the models
get_classifier.evaluate_models()
# Visualizing the decision tree's feature importance and tree structure
get_classifier.visualize_decision_tree(data)
# Promting the user for input and then infer whether the patient is at risk or not.
# Also print the voting results.
get_classifier.run_inference()