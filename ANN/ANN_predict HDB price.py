import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

rs_flat_1 = pd.read_csv('/Users/yesenia/Desktop/ResaleFlatPrices/Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv',on_bad_lines="skip")
rs_flat_2 = pd.read_csv('/Users/yesenia/Desktop/ResaleFlatPrices/Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv',on_bad_lines="skip")
rs_flat_3 = pd.read_csv('/Users/yesenia/Desktop/ResaleFlatPrices/Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv',on_bad_lines="skip")
rs_flat_df = pd.concat([rs_flat_1, rs_flat_2, rs_flat_3], ignore_index=False, axis=0)

sns.pairplot(rs_flat_df)
#plt.show()

#3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING
rs_flat_df.drop(['flat_type', 'block', 'street_name', 'storey_range','flat_model', 'lease_commence_date','month'], inplace=True, axis=1)
X = rs_flat_df
y = rs_flat_df['resale_price']
print(y.shape)

unique_towns = rs_flat_df['town'].unique()
print(unique_towns)

df = pd.DataFrame({'town': [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
    'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
    'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
    'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
    'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL'
]})

# Create a mapping dictionary
town_mapping = {town: idx + 1 for idx, town in enumerate(df['town'].unique())}

# Apply the mapping
df['town_encoded'] = df['town'].map(town_mapping)
# Ensure your original dataset (X) has a 'town' column
X['town_encoded'] = X['town'].map(town_mapping)
X.drop(['town','resale_price'], inplace=True, axis=1)
print(X)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
(scaler.data_max_)
(scaler.data_min_)
print(X_scaled[:,0])
print(y.shape)
y = y.values.reshape(-1,1)
y.shape
y_scaled = scaler.fit_transform(y)
print(y_scaled)

#4: training the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)
import tensorflow as tf
import gc
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
gc.collect()
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),  # Change input_shape=(4,)
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer
])
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())
model.compile(optimizer='adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)

#5: EVALUATING THE MODEL
print(epochs_hist.history.keys())
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

#columns: floor_area_sqm, town_encoded
X_Testing = np.array([[123, 1]])
y_predict = model.predict(X_Testing)
y_predict = scaler_y.inverse_transform(y_predict)  # Convert back to original scale

y_predict.shape
print('Expected Purchase Amount=', y_predict[:,0])

