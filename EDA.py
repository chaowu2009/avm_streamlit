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

df['log_price'] = np.log10(df['price'])

st.subheader("all columns")
st.write(sorted(df.columns))

st.subheader("df.describe()")
st.write(df.describe())

st.subheader("nul values stats")
st.write(df.isnull().sum())

st.header('look data distribution and distplot')
st.subheader("continuous values")
for col in df.columns:
      
    all_values = set(df[col])
    if len(all_values)<20:
        continue

    st.subheader(f"{col}")
    st.write(df[f'{col}'].describe())

    fig = plt.figure(figsize=(16,12))
    fig.add_subplot(2,2,1)
 #   plt.hist(df['price'], bins=100, rwidth= 0.5)
    sns.histplot(data=df, x=f"{col}", kde=True)

    plt.xlabel(col)
    plt.grid(True)
    plt.title(f"histogram of {col}")

    fig.add_subplot(2,2, 2)
    try:
        sns.histplot(data=df, x=f"{col}", kde=True, log_scale=True)
        plt.xlabel(f'log({col})')
        plt.grid(True)
        plt.title(f"histogram of {col}")
    except:
        pass

    fig.add_subplot(2, 2, 3)
    sns.regplot(data=df, x=f"{col}", y='price')
    plt.xlabel(f'{col}')
    plt.grid(True)
    plt.title(f"displot of {col}")
    
    fig.add_subplot(2, 2, 4)
    sns.regplot(data=df, x=f"{col}", y='log_price', ci=99, marker="x", color=".3", line_kws=dict(color="r"))
    plt.xlabel(f'{col}')
    plt.grid(True)
    plt.title(f"displot of {col}")

    plt.tight_layout()

    st.pyplot(fig) 

st.subheader("discrete values")
for col in df.columns:
      
    all_values = set(df[col])
    if len(all_values)>20:
        continue

    st.subheader(f"{col}")
    st.write(df[f'{col}'].describe())

    st.write(df.groupby([col]).count())

    fig = plt.figure(figsize=(16,12))
    fig.add_subplot(2,2,1)
 #   plt.hist(df['price'], bins=100, rwidth= 0.5)
    sns.histplot(data=df, x=f"{col}", kde=True)

    plt.xlabel(col)
    plt.grid(True)
    plt.title(f"histogram of {col}")

    fig.add_subplot(2,2, 2)
    try:
        sns.histplot(data=df, x=f"{col}", kde=True, log_scale=True)
        plt.xlabel(f'log({col})')
        plt.grid(True)
        plt.title(f"histogram of {col}")
    except:
        pass

    fig.add_subplot(2, 2, 3)
    sns.regplot(data=df, x=f"{col}", y='price')
    plt.xlabel(f'{col}')
    plt.grid(True)
    plt.title(f"displot of {col}")
    
    fig.add_subplot(2, 2, 4)
    sns.regplot(data=df, x=f"{col}", y='log_price', ci=99, marker="x", color=".3", line_kws=dict(color="r"))
    plt.xlabel(f'{col}')
    plt.grid(True)
    plt.title(f"displot of {col}")

    plt.tight_layout()

    st.pyplot(fig) 



st.subheader("Check correlation matrix")
st.write(df.corr()['price'].sort_values(ascending=False))


# feature with higher correlation
st.header("long/lat display")
fig = plt.figure(figsize=(15,10))
sns.scatterplot(x='long',y='lat',data=df, hue='price')
st.pyplot(fig)


