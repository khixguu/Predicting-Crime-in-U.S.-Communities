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

# inport csv files

current_dir = os.path.dirname(__file__)

file_path1 = os.path.join(current_dir, 'PChapelHillData.csv')
file1 = pd.read_csv(file_path1)

file_path2 = os.path.join(current_dir, 'PRaleighData.csv')
file2 = pd.read_csv(file_path2)

file_path3 = os.path.join(current_dir, '22-24PLAData.csv')
file3 = pd.read_csv(file_path3)

# display data
print("Chapel Hill Data")
file1.info()
print("\n\n______________\n\n")

print("Raleigh Data")
file2.info()
print("\n\n______________\n\n")

print("LA Data")
file3.info()




# Data Prep

# Encoding

# Splitting

# Random Forest Model

# Predictions and Evaluations

# Visualization