import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sbs

data = pd.read_csv("used_car_price_dataset_extended.csv")
#print(data.isnull().sum()) ## service_history is missing because there are lot of them as None
#print(data["service_history"].value_counts()) ## None is not considered a word

numerical_cols = ['make_year','mileage_kmpl','engine_cc','owner_count','accidents_reported']
words_cols = ['fuel_type', 'brand','transmission', 'color', 'service_history', 'insurance_valid']

## Data analysis
def create_heatmap(data,numerical_cols):
    plt.figure()
    sbs.heatmap(data[numerical_cols].corr(), annot=True)
    plt.show()


#Model building and training

preprocessor = ColumnTransformer(transformers=[
    ('nums', StandardScaler(),numerical_cols),
    ('words', OneHotEncoder(handle_unknown='ignore'),words_cols)
])

def baseline_models(data,preprocessor):
    models = {
        'Gradient Regression' : GradientBoostingRegressor(random_state=42),
        'Random Forest Regression' : RandomForestRegressor(random_state=42),
        'Linear Regression' : LinearRegression(),
    }
    x = data.drop(columns = "price_usd")
    y = data["price_usd"]
    x_training,x_testing,y_training,y_testing = train_test_split(x,y, test_size= 0.3,random_state=42)
    
    for name,model in models.items():
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model',model)])
        final_pipeline.fit(x_training,y_training)
        prediction = final_pipeline.predict(x_testing)
        r2 = r2_score(y_testing,prediction)
        rmse = root_mean_squared_error(y_testing,prediction)
        print(f"Model {name}: r2 = {r2} ", end='')
        print(f"and rmse = {rmse}")
baseline_models(data,preprocessor)

## Results for the baseline models
# Model: Gradient Regression r2 = 0.869372964490497 and rmse = 1018.9187110323095
# Model: Random Forest Regression r2 = 0.8488226747304112 and rmse = 1096.1407685495867
# Model: Linear Regression r2 = 0.8743160195901898 and rmse = 999.4543561951389


## Let´s start tuning each model

def linear_reg_tuning(data,preprocessor):
    x = data.drop(columns = "price_usd")
    y = data["price_usd"]
    x_training,x_testing,y_training,y_testing = train_test_split(x,y, test_size= 0.35,random_state=42)