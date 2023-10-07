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
sns.distplot(df['price'])
plt.xlabel('price')
plt.grid(True)
plt.title("distribution")

fig.add_subplot(2,2,3)
plt.hist(np.log10(df['price']), bins=20)
plt.xlabel('log(price)')
plt.title("histogram ( log10(price)")
plt.grid(True)

fig.add_subplot(2,2,4)
sns.distplot(np.log10(df['price']))
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

price_threshold = st.slider("choose a price cut off (maximum) to remove outlier in millions", 
                            min_value=2, 
                            max_value=10, 
                            value=5)
price_threshold = float(price_threshold)*1.0e5

st.write(f"selected threshold = {price_threshold}")

f1 = df['price'] < price_threshold
df = df[f1]

X = df.drop(['price'],axis =1).values

# convert price to log10 scale
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
lr = LinearRegression()  
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)
df_train= pd.DataFrame({'soldprice': y_train, 'predicted': y_pred})

#evaluate the model (intercept and slope)
y_pred = lr.predict(X_test)
df_test = pd.DataFrame({'soldprice': y_test, 'predicted': y_pred})

#st.write(df_test.head(10))

# compute PPE 10
def compute_ppe10(df):
    df['predicted'] = np.power(10, df['predicted'])
    df['soldprice'] = np.power(10, df['soldprice'])

    df['error'] = (df['predicted']-df['soldprice'])/df['soldprice']
    
    ppe10 = round(len(df[np.abs(df['error'])<=0.1])/len(df)*100,2)

    return ppe10

train_ppe10 = compute_ppe10(df_train)
test_ppe10 = compute_ppe10(df_test)

train_score = round(lr.score(X_train,y_train), 2)
test_score = round(lr.score(X_test, y_test), 2)

st.write('Linear Regression Model:')
st.write(f"Train Score = {train_score}, Test Score = {test_score}")
st.write(f"Train: PPE10 = {train_ppe10}, Test: PPE10 = {test_ppe10}")


st.subheader("Random Forest")
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()  
rf.fit(X_train, y_train)

y_pred = rf.predict(X_train)
df_train= pd.DataFrame({'soldprice': y_train, 'predicted': y_pred})

#evaluate the model (intercept and slope)
y_pred = rf.predict(X_test)
df_test = pd.DataFrame({'soldprice': y_test, 'predicted': y_pred})

train_ppe10 = compute_ppe10(df_train)
test_ppe10 = compute_ppe10(df_test)

train_score = round(lr.score(X_train,y_train), 2)
test_score = round(lr.score(X_test, y_test), 2)

st.write('Random Forest Regresion Model:')
st.write(f"Train Score = {train_score}, Test Score = {test_score}")
st.write(f"Train: PPE10 = {train_ppe10}, Test: PPE10 = {test_ppe10}")


st.subheader("Deep Learing Model")

# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

import os
import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = Sequential()

input_dimesion = 18

model.add(Dense(input_dimesion, activation='relu'))
model.add(Dense(input_dimesion, activation='relu'))
model.add(Dense(input_dimesion, activation='relu'))
model.add(Dense(input_dimesion, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

#st.write(X_train.shape)

model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128,epochs=20)


loss_df = pd.DataFrame(model.history.history)
fig = plt.figure()
plt.plot(loss_df)
plt.grid(True)
plt.title('loss function')
st.pyplot(fig)

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

st.write(f"Train: PPE10 = {train_ppe10}")
st.write(f"Test: PPE10 = {test_ppe10}")

