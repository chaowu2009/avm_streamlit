import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


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

st.subheader('look at house price distribution')
fig = plt.figure()

fig.add_subplot(2,2,1)
plt.hist(df['price'], bins=20)
plt.xlabel('price')
plt.grid(True)
plt.title("histogram")

fig.add_subplot(2,2,2)
sns.displot(df['price'])
plt.xlabel('price')
plt.grid(True)
plt.title("distribution")

fig.add_subplot(2,2,3)
plt.hist(np.log10(df['price']), bins=20)
plt.xlabel('log(price)')
plt.title("histogram ( log10(price)")
plt.grid(True)

fig.add_subplot(2,2,4)
sns.displot(np.log10(df['price']))
plt.xlabel('log(price)')
plt.title("distribution")
plt.grid(True)
plt.tight_layout()

st.pyplot(fig) 


st.subheader("Check correlation matrix")
st.write(df.corr()['price'].sort_values(ascending=False))


# feature with higher correlation
st.subheader("scatterplot sqft_living versus price")
fig = plt.figure()
fig.add_subplot(2,1,1)
plt.scatter( df['price'], df['sqft_living'])
plt.grid(True)
plt.grid(True)
plt.xlabel('sqft_living')
plt.ylabel("price")

fig.add_subplot(2,1,2)
plt.scatter(np.log10(df['price']), df['sqft_living'] )
plt.grid(True)
plt.xlabel('sqft_living')
plt.ylabel("log10(price)")

plt.tight_layout()

st.pyplot(fig)

# fig = plt.figure(figsize=(15,10))
# sns.scatterplot(x='long',y='lat',data=df,hue='price')
# st.pyplot(fig)


st.title("Try different machine learing algorithms")

slider_values = st.slider('Select a range of values (in millions)', 0, 10, (1, 6))
lower_threshold= slider_values[0]
upper_threshold = slider_values[1]

lower_threshold = float(lower_threshold)*1e5
upper_threshold = float(upper_threshold)*1e6

st.write(f"selected threshold (million) = {lower_threshold/1e6}, {upper_threshold/1e6}")


f1 = df['price'] < upper_threshold
f2 = df['price'] > lower_threshold
df = df[f1 & f2]

st.write(f"length(df)= {len(df)}")

X = df.drop(['price'],axis =1).values

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
print(f"log_scale = {log_scale}")
if log_scale:
    df['price'] = np.log10(df['price'])


y = df['price'].values

#splitting Train and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=888)

# normalize
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(float))
X_test = s_scaler.transform(X_test.astype(float))


st.subheader("Linear Regression")
#Liner Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()  
lr_model.fit(X_train, y_train)

# compute PPE 10
def compute_ppe10(df):
    df['predicted'] = np.power(10, df['predicted'])
    df['soldprice'] = np.power(10, df['soldprice'])

    df['error'] = (df['predicted']-df['soldprice'])/df['soldprice']
    
    ppe10 = round(len(df[np.abs(df['error'])<=0.1])/len(df)*100,2)

    return ppe10

def compute_stat(model):
    y_pred = model.predict(X_train)
    #st.write(y_pred)
    df_train= pd.DataFrame({'soldprice': y_train})
    df_train['predicted'] =  y_pred

    #evaluate the model (intercept and slope)
    y_pred = model.predict(X_test)
    df_test = pd.DataFrame({'soldprice': y_test})
    df_test['predicted'] = y_pred

    train_ppe10 = compute_ppe10(df_train)
    test_ppe10 = compute_ppe10(df_test)
    try:
        train_score = round(model.score(X_train,y_train), 2)
        test_score = round(model.score(X_test, y_test), 2)
        st.write(f"Train Score = {train_score}, Test Score = {test_score}")
    except:
        pass
    
    st.write(f"Train: PPE10 = {train_ppe10}, Test: PPE10 = {test_ppe10}")

compute_stat(model = lr_model)

st.subheader("Random Forest model")
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor()  
rf_model.fit(X_train, y_train)

compute_stat(model = rf_model)

st.subheader("Linear GAM model")
from pygam import LinearGAM
#gam = LinearGAM().fit(X, y)
#gam = LinearGAM().gridsearch(X_train, y_train)
gam_model = LinearGAM(n_splines=10).gridsearch(X_train, y_train)

compute_stat(model = gam_model)


st.subheader("xgboost model")
def xgboost_model():
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

xgboost_model()


st.subheader("Catboost model")

from catboost import CatBoostRegressor
catboost_model = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2)
# Fit model
catboost_model.fit(X_train, y_train)
# Get predictions

compute_stat(model=catboost_model)

st.subheader("Deep Learing Model")
# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

import os
import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dl_model = Sequential()

input_dimesion = 18

dl_model.add(Dense(input_dimesion, activation='relu'))
dl_model.add(Dense(input_dimesion, activation='relu'))
dl_model.add(Dense(input_dimesion, activation='relu'))
dl_model.add(Dense(input_dimesion, activation='relu'))
dl_model.add(Dense(1))

dl_model.compile(optimizer='adam',loss='mse')

#st.write(X_train.shape)

dl_model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=20)


loss_df = pd.DataFrame(dl_model.history.history)
fig = plt.figure()
plt.plot(loss_df)
plt.grid(True)
plt.title('loss function')
st.pyplot(fig)

compute_stat(model=dl_model)


