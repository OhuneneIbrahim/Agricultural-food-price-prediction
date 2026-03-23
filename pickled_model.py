import pandas as pd



# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import pickle



#import dataset
price_df = pd.read_csv("food_price_dataset")

# Extract the numeric columns
num_cols = list(price_df.select_dtypes(exclude = 'object').columns)
num_cols.pop(4)          # Remove the target variable
categorical_list= ['states','crops']

target = ['Price/Kg (Naira)']
#drop = num_cols + target

df3= price_df.copy()
X=df3[num_cols + categorical_list]
y= df3[target]

# Pipeline for numerical and categorical columns 
num_pipe = Pipeline(steps=[("scale", MinMaxScaler())])
cat_pipe = Pipeline(steps=[("encoder", OneHotEncoder())])

#Performing transformation with the pipelines
column_transform = ColumnTransformer(transformers=[("num_trans", num_pipe, num_cols), 
                                                  ("cat_trans", cat_pipe, categorical_list)],
                                     remainder="drop", n_jobs=-1)



#initiating Xgboost model
xgb_regressor = XGBRegressor(learning_rate=0.1,
    n_estimators= 150,
    max_depth= 15)

#xgb_pipeline
xgb_pipeline = Pipeline(steps= [("transformation", column_transform),
                           ("model", xgb_regressor)])



#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=1)


#Fit the model
xgb_pipeline.fit(X_train, y_train)

#Perform prediction
predicted2 = xgb_pipeline.predict(X_test)

R2 = r2_score(y_test, predicted2)
#print(f"R2-score {R2}")


MSE = mean_squared_error(y_test, predicted2, squared=False)
#print(f"Mean squared error: {MSE}")

MAE = mean_absolute_error(y_test, predicted2)
#print(f"Mean absolute error: {MAE}")



# save the model as a pickle file
model_pkl_file = "food_price_predictor_model.pkl" 

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(xgb_pipeline, file)

