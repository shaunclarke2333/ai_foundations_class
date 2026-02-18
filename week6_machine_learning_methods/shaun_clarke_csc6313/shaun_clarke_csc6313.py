"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Module: 6
Assignment: The Classifier Showdown (Non-Linear Methods)

This week, we move beyond linear models to explore non-linear classification and unsupervised grouping.
You will implement and compare Decision Trees, Random Forests, and k-Nearest Neighbors (k-NN).
"""

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

warnings.filterwarnings('ignore')

# This class generates patient data with realistic health patterns and save it as a csv
class PatientDataGenerator:

    def __init__(self, n_samples: int = 500, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state


    # This helper method helps generate the mean and the std used to generate each feature
    def get_mean_std(self, min_range: int, max_range: int) -> Tuple[str, str]:
        """
        This function helps generate the mean and the std used to generate each feature
        min_range: low end of the range
        max_range: hgih end of the range
        returns: returns mean and standard deviation
        """
        # List of min and max range
        range: List = [min_range, max_range]
        # Count the number of items in the range list
        count: int = len(range)
        # calculating mean of the range
        mean: int = (min_range + max_range)/count
        # calculating standard deviation. std is distance from mean
        std_dev: int = (max_range - mean)/count

        return mean, std_dev

    # This helper method saves a dataframe as csv in the cwd
    def save_to_csv(self, dataframe: pd.DataFrame, csv_filename: str) -> None:
        """
        Docstring for save_to_csv
        
        :param self: Description
        :param dataframe: Description
        :type dataframe: pd.DataFrame
        :param csv_filename: Description
        :type csv_filename: str
        """

        # Outputting the dataframe to a CSV in the cwd
        dataframe.to_csv(csv_filename, index=False)

    # This method generates the fake patient data
    def generate_patient_data(self) -> pd.DataFrame:
        """
        This function generates the fake patient data and outputs a CSV in the current working directory.

        :param n_samples: number of patients to generate
        :type n_samples: int
        :param random_state: the seed number that gives us reproducibility, meaning we start from the same number and generate the same data everytime
        :type random_state: int
        :return: Description
        :rtype: None
        """

        # Setting random seed as the random_state we defined so the same data is genearated every time
        np.random.seed(self.random_state)
        # Generating the age feature using the age range 20-80
        min_age: int = 18
        max_age: int = 90
        # Getting mean and std
        mean_age, age_std_dev = self.get_mean_std(min_age, max_age)
        age: int = np.random.normal(mean_age, age_std_dev, self.n_samples)

        # generating the bmi feature: A healthy bmi is about 18-30,
        min_bmi: int = 18.5
        max_bmi: int = 45
        mean_bmi, bmi_std_dev = self.get_mean_std(min_bmi, max_bmi)
        # print(mean_bmi, bmi_std_dev)
        bmi = np.random.normal(mean_bmi, bmi_std_dev, self.n_samples)

        # generating the blood sugar level feature: normal fasting blood sugar is 70-100 mg/dL
        min_blood_sugar: int = 70
        max_blood_sugar: int = 200
        mean_blood_sugar, blood_sugar_std_dev = self.get_mean_std(min_blood_sugar, max_blood_sugar)
        # print(mean_blood_sugar, blood_sugar_std_dev)
        blood_sugar_level: np.ndarray = np.random.normal(mean_blood_sugar, blood_sugar_std_dev, self.n_samples)
        # print(blood_sugar_level)

        # Creating a health risk score based on the features. Thanks to numpy arrays perfomring element-wise operations, everything in the generated feature arrays will be multiplied by the weights
        health_risk_score: int = ((age * 0.5) + ((bmi - 25) * 2) + ((blood_sugar_level - 90) * 0.3) + np.random.normal(0, 10, self.n_samples))
        # To generate teh diagnosis, we create an array of booleans thare greater or less than 60 and convertt the booleans to binary, 1,0
        # The idea here is, we already have a formula for health risk score with weights. lets create an array of boolean results for scores
        # That are greater or less than 60 then convert them to binary, becasue that is what the logisticregression model expects.
        diagnosis: np.ndarray = (health_risk_score > 60).astype(int)
        # print(diagnosis)
        # print(len(diagnosis))

        # Using the generated features and target to create a dataframe
        generated_data: pd.DataFrame = pd.DataFrame({
            "age": age,
            "bmi": bmi,
            "blood_sugar": blood_sugar_level,
            "diagnosis": diagnosis
        })

        # Outputting the generated data to a CSV in the cwd
        self.save_to_csv(generated_data, "classifier_patient_data.csv")

        return generated_data
    

# This class trains and compares multiple classification algorithms
class ClassifierShowdown:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        # models
        self.tree_model = None
        self.forest_model = None
        self.knn_model = None
        # prediction and scores
        self.tree_pred = None
        self.forest_pred = None
        self.knn_pred = None
        # Dict to hold accuracy scores
        self.accuracies = {}

    # This method splits the data and transforms it with StandardScaler
    def preprocess_data(self):
        """
        This method splits the data into X(inputs) and Y(targets(diganosis)) and transforms it with standard scaler
        """

        # Getting featuer inputs dataframe by select only the featuers from the original dataframe
        X: pd.DataFrame = self.data[["age", "bmi", "blood_sugar"]]
        # Getting the Y target by selecting only the diagnosis from the original dataframe
        Y: pd.DataFrame = self.data["diagnosis"]
        # print(f"This is X: {X}")
        # print(f"This is Y: {Y}")

        # Splitting features and target into train and test dataset with a 80/20 split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # Initializing StandardScaler
        self.scaler: StandardScaler = StandardScaler()
        # fitting and transforming the training data with StandardScaler
        self.X_train_scaled: np.nadarray = self.scaler.fit_transform(self.X_train)
        # Transform test data with StandardScaler. We are only transforming test and not fitting becasue we cannot expose the test data to the model.
        self.X_test_scaled: np.ndarray = self.scaler.transform(self.X_test)

    # This method trains all three classification models
    def train_models(self):
        """
        This method trains all three classification models
        """

        # Training a decision tree model with max_depth=3 and training it on the data, this depth prevents it from becoming too complexed and overfitting
        self.tree_model = DecisionTreeClassifier(max_depth=3).fit(self.X_train_scaled, self.Y_train)
        # Training random forest classifier on the data with a max depth of 3
        self.forest_model = RandomForestClassifier().fit(self.X_train_scaled, self.Y_train)
        # Training KNN on the data
        self.knn_model = KNeighborsClassifier(n_neighbors=5).fit(self.X_train_scaled, self.Y_train)
    
    # This method generates predictions and calculates accuracy scores
    def evaluate_models(self):
        """
        This method generates predictions and calculates accuracy scores
        """
        self.tree_pred: DecisionTreeClassifier = self.tree_model.predict(self.X_test_scaled)
        self.forest_pred: RandomForestClassifier = self.forest_model.predict(self.X_test_scaled)
        self.knn_pred: KNeighborsClassifier = self.knn_model.predict(self.X_test_scaled)

        self.accuracies["Decision Tree"]=accuracy_score(self.Y_test, self.tree_pred)
        self.accuracies["Random Forest"]=accuracy_score(self.Y_test, self.forest_pred)
        self.accuracies["KNN"]=accuracy_score(self.Y_test, self.knn_pred)

        # print(f"tree: {self.accuracies['Decision Tree']}")
        # print(f"forest: {self.accuracies['Random Forest']}")
        # print(f"knn: {self.accuracies['KNN']}")

        self._print_accuracy_table()

    # This private method helps to format the accuracy scores output as a table
    def _print_accuracy_table(self):
        """
        This private method helps to format the accuracy scores output as a table
        """

        print("\n" + "="*50)
        print("Model Accuracy Comparison")
        print("="*50)
        print("Decision Tree: {:.2%}".format(self.accuracies['Decision Tree'] * 100))
        print("="*25)
        print("Random Forest: {:.2%}".format(self.accuracies['Random Forest'] * 100))
        print("="*25)
        print("K-Nearest neighbors: {:.2%}".format(self.accuracies['KNN'] * 100))
        print("="*25)





