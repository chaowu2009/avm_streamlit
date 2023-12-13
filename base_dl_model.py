import streamlit as st
import pandas as pd
import matplotlib.pylab as plt


def train_dl_model(X_train, X_test, y_train, y_test):

    # Creating a Neural Network Model
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout


    import os
    import tensorflow as tf
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    dl_model = Sequential()

    input_dimesion = 20

    dl_model.add(Dense(input_dimesion, activation='relu'))
    dl_model.add(Dropout(0.2))
                 
    dl_model.add(Dense(input_dimesion, activation='relu'))
    dl_model.add(Dropout(0.2))
                 
    dl_model.add(Dense(input_dimesion, activation='relu'))
    dl_model.add(Dropout(0.2))
                 
    dl_model.add(Dense(input_dimesion, activation='relu'))
    dl_model.add(Dense(1))

    dl_model.compile(optimizer='adam',loss='mse')

    #st.write(X_train.shape)

    dl_model.fit(x=X_train,y=y_train,
            validation_data=(X_test,y_test),
            batch_size=64, 
            epochs=20)

    loss_df = pd.DataFrame(dl_model.history.history)
    fig = plt.figure()
    plt.plot(loss_df)
    plt.grid(True)
    plt.title('loss function')
    st.pyplot(fig)

    return dl_model