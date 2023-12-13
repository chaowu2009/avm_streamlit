import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from base_dl_model import train_dl_model

st.title("AVM example with streamlit")

st.markdown("Ref: https://github.com/mahsamir/CaseStudy-KerasRegression/blob/master/Kaggle-KerassRegression.ipynb")

df = pd.read_csv("kc_house_data.csv")

st.write(f"total data length = {len(df)}")

df = df.drop(['id','date'], axis =1)

st.write(df.head(5))

comment = """
Feature Columns
id - Unique ID for each home sold
date - Date of the home sale
price - Price of each home sold
bedrooms - Number of bedrooms
bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
sqft_living - Square footage of the apartments interior living space
sqft_lot - Square footage of the land space
floors - Number of floors
waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
view - An index from 0 to 4 of how good the view of the property was
condition - An index from 1 to 5 on the condition of the apartment,
grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
sqft_above - The square footage of the interior housing space that is above ground level
sqft_basement - The square footage of the interior housing space that is below ground level
yr_built - The year the house was initially built
yr_renovated - The year of the house’s last renovation
zipcode - What zipcode area the house is in
lat - Lattitude
long - Longitude
sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors
"""

st.subheader("all columns")
st.write(sorted(df.columns))

st.subheader("df.describe()")
st.write(df.describe())

st.subheader("nul values stats")
st.write(df.isnull().sum())


st.title("Try different machine learing algorithms")

slider_values = st.slider('Select a range of values (in millions)', 0, 100, (1, 20))
lower_threshold= slider_values[0]
upper_threshold = slider_values[1]

lower_threshold = float(lower_threshold)*1e4  #using different scale here, 1e4 = 0.01 million 
upper_threshold = float(upper_threshold)*1e5  #20 equals to 20e5 = 2 million

st.write(f"selected threshold (million) = {lower_threshold/1e6}, {upper_threshold/1e6}")

f1 = df['price'] < upper_threshold
f2 = df['price'] > lower_threshold

st.write(f"before trimming: length(df)= {len(df)}")

df = df[f1 & f2]

st.write(f"after trimming: length(df)= {len(df)}")

# convert price to log10 scale
import streamlit_toggle as toggle 
log_scale = toggle.st_toggle_switch(label="log scale", 
                    key="Key1", 
                    default_value=False, 
                    label_after = False, 
                    inactive_color = '#D3D3D3', 
                    active_color="#11567f", 
                    track_color="#29B5E8"
                    )

log_scale = False
print(f"log_scale = {log_scale}")
df_original = df.copy()
if log_scale:
    df['price'] = np.log10(df_original['price'])
else:
    df['price'] = df_original['price']

X = df.drop(['price'],axis =1).values
y = df['price'].values

def prepare_training_data(df):
    X = df.drop(['price'],axis =1).values
    y = df['price'].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

    # normalize
    from sklearn.preprocessing import StandardScaler
    s_scaler = StandardScaler()
    X_train = s_scaler.fit_transform(X_train.astype(float))
    X_test = s_scaler.transform(X_test.astype(float))

    return X_train, X_test, y_train, y_test

def convert_df_to_np_data(df):
    X = df.drop(['price'],axis =1).values
    y = df['price'].values

    # normalize
    from sklearn.preprocessing import StandardScaler
    s_scaler = StandardScaler()
    X = s_scaler.fit_transform(X.astype(float))
    
    return X, y


X_train, X_test, y_train, y_test = prepare_training_data(df)

st.subheader("Linear Regression")
#Liner Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()  
lr_model.fit(X_train, y_train)

# compute PPE 10

def compute_ppe10(df_input):
    if log_scale:
        df_input['predicted'] = np.power(10, df_input['predicted'])
        df_input['soldprice'] = np.power(10, df_input['soldprice'])

    df_input['error'] = (df_input['predicted']-df_input['soldprice'])/df_input['soldprice']

    f1 = np.abs(df_input['error'])<=0.1
    
    ppe10 = round(len(df_input[f1])/len(df_input)*100,2)

    return ppe10

def compute_stat(model, X_train, X_test, y_train, y_test, dl_model_flag=0):
    if dl_model_flag==1:
        y_pred_1 = model.predict(X_train)[:,0]
        y_pred_2 = model.predict(X_test)[:,0]
    else:
        y_pred_1 = model.predict(X_train)
        y_pred_2 = model.predict(X_test)
    #st.write(y_pred)
    df_train= pd.DataFrame({'soldprice': y_train})
    df_train['predicted'] =  y_pred_1

    #evaluate the model (intercept and slope)
    
    df_test = pd.DataFrame({'soldprice': y_test})
    df_test['predicted'] = y_pred_2

    train_ppe10 = compute_ppe10(df_train)
    test_ppe10 = compute_ppe10(df_test)
    try:
        train_score = round(model.score(X_train,y_train), 2)
        test_score = round(model.score(X_test, y_test), 2)
        st.write(f"Train Score = {train_score}, Test Score = {test_score}")
    except:
        pass
    
    st.write(f"Train: PPE10 = {train_ppe10}, Test: PPE10 = {test_ppe10}")

compute_stat(lr_model,  X_train, X_test, y_train, y_test)

st.subheader("Random Forest model")
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()  
rf_model.fit(X_train, y_train)

compute_stat(rf_model,  X_train, X_test, y_train, y_test)

st.subheader("GAM model")
from pygam import LinearGAM
#gam = LinearGAM().fit(X, y)
#gam = LinearGAM().gridsearch(X_train, y_train)
gam_model = LinearGAM(n_splines=10).gridsearch(X_train, y_train)

compute_stat(gam_model,  X_train, X_test, y_train, y_test)


st.subheader("xgboost model")
def train_xgboost_model( X_train, X_test, y_train, y_test):
    import xgboost as xgb
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)
    # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "hist"}

    n = 100
    xgboost_model = xgb.train(
    params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    )

    y_pred = xgboost_model.predict(dtrain_reg)
    df_train= pd.DataFrame({'soldprice': y_train})
    df_train['predicted'] =  y_pred

    #evaluate the model (intercept and slope)
    y_pred = xgboost_model.predict(dtest_reg)
    df_test = pd.DataFrame({'soldprice': y_test})
    df_test['predicted'] = y_pred

    train_ppe10 = compute_ppe10(df_train)
    test_ppe10 = compute_ppe10(df_test)

    try:
        train_score = round(xgboost_model.score(X_train,y_train), 2)
        test_score = round(xgboost_model.score(X_test, y_test), 2)
        st.write(f"Train Score = {train_score}, Test Score = {test_score}")
    except:
        pass

    st.write(f"Train: PPE10 = {train_ppe10}, Test: PPE10 = {test_ppe10}")

train_xgboost_model( X_train, X_test, y_train, y_test)


st.subheader("Catboost model")

from catboost import CatBoostRegressor
catboost_model = CatBoostRegressor(iterations=20,
                          learning_rate=0.01)
# Fit model
catboost_model.fit(X_train, y_train)
# Get prediction errors

compute_stat(catboost_model,  X_train, X_test, y_train, y_test)

st.subheader("Deep Learing Model")

foundation_dl_model = train_dl_model(X_train, X_test, y_train, y_test)

compute_stat(foundation_dl_model,  X_train, X_test, y_train, y_test, 1)

st.subheader("using the pretrained foundation model")
st.markdown("saving the model to files")
model_json = foundation_dl_model.to_json()
with open("foundation_model.json","w") as json_file:
    json_file.write(model_json)

foundation_dl_model.save_weights("foundation_model.h5")

st.markdown("loading model from files")
from tensorflow.keras.models import model_from_json
with open("foundation_model.json","r") as fp:
    loaded_model_json = fp.read()

loaded_model =model_from_json(loaded_model_json)
loaded_model.load_weights("foundation_model.h5")
loaded_model.compile(optimizer='adam', loss='mse')

st.markdown("# train on a specific zipcode")

df_2 = df[df['zipcode']==98103]

X_train_2, y_train_2 = convert_df_to_np_data(df_2)

loaded_model.fit(X_train_2, y_train_2, epochs=200, batch_size=64)

#compute_stat(loaded_model, X_train_2, y_train_2, X_test, y_test, 1)
y_pred_1 = loaded_model.predict(X_train_2)[:,0]

df_train= pd.DataFrame({'soldprice': y_train_2})
df_train['predicted'] =  y_pred_1

train_ppe10 = compute_ppe10(df_train)
st.write(f"new train PPE10 = {train_ppe10}")



