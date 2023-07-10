import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def calculate_weights_rf(data_file):
    #This data file will be state_data_2.csv
    #Need to return a list with the weights
    # Load the data
    df = pd.read_csv(data_file)

    # Normalize both axes
    scaler = StandardScaler()

    #CHRIS: Change the next line in your code
    cols_to_scale = ['normalized_smoking', 'normalized_copd', 'normalized_covid', 'normalized_drowning', 'normalized_sepsis', 'normalized_flu', 'normalized_pneumonia', 'normalized_vaccination', 'avg_normalized_incomes', 'avg_normalized_seniors', 'avg_normalized_literacy', 'vals']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    df['normalized_vaccination'] = 1 - df['normalized_vaccination']

    from sklearn.ensemble import RandomForestRegressor

    features = df[cols_to_scale[:-1]]
    target = df['vals']

    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(features, target)

    predictions = model.predict(features)

    r2 = r2_score(target, predictions)
    importances = model.feature_importances_
    return importances, r2

