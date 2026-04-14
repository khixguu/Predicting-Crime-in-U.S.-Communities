import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# inport csv files

current_dir = os.path.dirname(__file__)

# file_path1 = os.path.join(current_dir, 'PChapelHillData.csv')
# file1 = pd.read_csv(file_path1)

# file_path2 = os.path.join(current_dir, 'PRaleighData.csv')
# file2 = pd.read_csv(file_path2)

file_path3 = os.path.join(current_dir, '22-24PLAData.csv')
file3 = pd.read_csv(file_path3)

# display data
# print("Chapel Hill Data")
# file1.info()
# print("\n\n______________\n\n")

# print("Raleigh Data")
# file2.info()
# print("\n\n______________\n\n")

# print("LA Data")
# file3.info()

keywords = ['ASSAULT', 'AGGRAVATED', 'BATTERY', 'BRANDISH WEAPON','ARSON', 'RAPE', 'CHILD ABUSE', 'SHOTS FIRED'] 

def classify_crime(crime):
    crime = crime.upper()
    for word in keywords:
        if crime in keywords:
            return 1
        return 0
    
   
#  assigns "violent" tag if the description containst any of the keywords
file3["violent"] = file3["Crm Cd Desc"].apply(classify_crime)



# Data Prep



# Defining X and y

# neighborhoods in LA
X = file3[['AREA NAME']]

# violent crimes
y = file3[['violent']]


# Encoding


file3 = pd.get_dummies(file3, columns = ['AREA NAME'], drop_first=True)


X = file3.drop(['Crm Cd Desc', 'violent'], axis=1) 
y = file3['violent']

# Splitting


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

    
# Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# Predictions and Evaluations

# Visualization