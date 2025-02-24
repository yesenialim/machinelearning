import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from prophet import Prophet
#aim to train the data and later compare it with the actual result to see if it matches

rs_flat_1 = pd.read_csv('/Users/yesenia/Desktop/ResaleFlatPrices/Resale Flat Prices (Based on Approval Date), 1990 - 1999.csv',on_bad_lines="skip")
rs_flat_2 = pd.read_csv('/Users/yesenia/Desktop/ResaleFlatPrices/Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012.csv',on_bad_lines="skip")
rs_flat_3 = pd.read_csv('/Users/yesenia/Desktop/ResaleFlatPrices/Resale Flat Prices (Based on Registration Date), From Mar 2012 to Dec 2014.csv',on_bad_lines="skip")
rs_flat_df = pd.concat([rs_flat_1, rs_flat_2, rs_flat_3], ignore_index=False, axis=0)
(rs_flat_df.head())
'''
If the heatmap is too cluttered, consider:
Transposing the DataFrame (rs_flat_df.T) if there are too many rows.
Reducing the dataset (rs_flat_df.sample(100)) if it's too large.
'''
#The sns.heatmap(rs_flat_df.isnull(), cbar = False, cmap = 'YlGnBu') is generating a heatmap that visualizes missing (null) values in your dataset rs_flat_df.
#Yellow shades (lighter): These represent False values, meaning the data point is not missing (there is a value in that cell).
plt.figure(figsize=(12,6))
sns.heatmap(rs_flat_df.isnull(), cbar = False, cmap = 'YlGnBu')
#plt.show()
(rs_flat_df.columns)
rs_flat_df['ds'] = pd.to_datetime(rs_flat_df['month'] + '-01', format='%Y-%m-%d')
#rs_flat_df[['Year', 'Month']] = rs_flat_df['month'].str.split('-', expand=True)
rs_flat_df.drop(['flat_type', 'block', 'street_name', 'storey_range','flat_model', 'lease_commence_date','month'], inplace=True, axis=1)
#print(rs_flat_df.head())


plt.figure(figsize=[25,12])
sns.countplot(x = 'town', data = rs_flat_df)
plt.xticks(rotation = 45)
plt.grid(True)
#plt.show()
rs_flat_prophet_df = rs_flat_df[['ds', 'resale_price']]
(rs_flat_prophet_df)

#STEP 3: MAKE PREDICTIONS
rs_flat_prophet_df = rs_flat_prophet_df.rename(columns={'resale_price':'y'})
#print(rs_flat_prophet_df)


m = Prophet()
m.fit(rs_flat_prophet_df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
print(forecast.columns)


figure = m.plot(forecast, xlabel='Date', ylabel='Price')
#plt.show()
figure3 = m.plot_components(forecast)
#plt.show()

town_count=rs_flat_df['town'].value_counts()
#print(town_count)



#part_2 by resampling the data to only wanting sqm >=100
rs_flat_df_sample = rs_flat_df[rs_flat_df['floor_area_sqm']>=100]
(rs_flat_df_sample)
plt.figure(figsize=(10,10))
sns.set(style="whitegrid")
ax=sns.boxplot(x='town', y='floor_area_sqm', data=rs_flat_df_sample)

# Set labels and title
plt.title('Distribution of Floor Area by Region')
plt.xlabel('Region')
plt.ylabel('Floor Area (sqm)')
# Show the plot
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
labels = ax.get_xticklabels()
ax.set_xticklabels(labels, horizontalalignment='right')
for i, label in enumerate(ax.get_xticklabels()):   # Manually adjust position of labels to shift them left
    label.set_x(label.get_position()[0] - 0.1)
plt.tight_layout()
plt.show()
