import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from prophet import Prophet

chicago_df_1 = pd.read_csv('/Users/yesenia/Desktop/Chicago_Crimes_2001_to_2004.csv', on_bad_lines='skip')
chicago_df_2 = pd.read_csv('/Users/yesenia/Desktop/Chicago_Crimes_2005_to_2007.csv', on_bad_lines='skip')
chicago_df_3 = pd.read_csv('/Users/yesenia/Desktop/Chicago_Crimes_2008_to_2011.csv', on_bad_lines='skip')
chicago_df = pd.concat([chicago_df_1, chicago_df_2,chicago_df_3], ignore_index=False, axis=0)
print(chicago_df)

#data analysis
### ensure all headers/columns are intact
##chicago_df.tail() #by default last 5

#drop null/unrelated values
chicago_df.drop(['Unnamed: 0', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)
chicago_df
#changing date format
chicago_df.Date = pd.to_datetime(chicago_df.Date, format='%m/%d/%Y %I:%M:%S %p')

#print(chicago_df.columns)

'''
#identify top 15 crime cases
top 15=chicago_df['Primary Type'].value_counts().iloc[:15]
#plotting to visualise it5]
print(top_15)
plt.figure(figsize=(15, 10))
sns.countplot(y='Primary Type', data=chicago_df.loc[chicago_df['Primary Type'].isin(top_15.index)])
plt.show() #to plot the total number of cases over the year

#resampling the cases by date
# setting the index to be the date
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)
chicago_df.resample('YE').size() #resample by year

plt.plot(chicago_df.resample('YE').size())
plt.title('Crimes Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.show()
'''
# there is a significant drop in crime rate since last of 2009 till 2018
# would this trend give analyst the idea that crime rate is deemed to drop in the upcoming years?
'''
####to group the types per year 
# Ensure 'Date' column is in datetime format and set as index
chicago_df['Date'] = pd.to_datetime(chicago_df['Date'])
chicago_df.set_index('Date', inplace=True)

# Ensure crime types are correctly formatted
chicago_df['Primary Type'] = chicago_df['Primary Type'].str.strip()

# Get the top 15 crime types
top_15 = chicago_df['Primary Type'].value_counts().index[:15]

# Filter dataset to only include the top 15 crimes
filtered_df = chicago_df[chicago_df['Primary Type'].isin(top_15)]

# Group by year and crime type, then count occurrences
crime_trends = (
    filtered_df
    .groupby([filtered_df.index.year, 'Primary Type'])  # Group by year and crime type
    .size()  # Count occurrences
    .unstack(fill_value=0)  # Convert crime types to columns, fill missing years with 0
)

# Check if data is correct
print(crime_trends.head())  # Inspect first few rows
# Plot the trends for each crime type over the years
plt.figure(figsize=(15, 8))
crime_trends.plot(kind='line', ax=plt.gca(), colormap='tab20')

# Add labels and title
plt.title("Trends of Top 15 Crime Cases Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Cases")
plt.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.tight_layout()
plt.show()
'''
## can do per month too
###

# preparing data
'''
chicago_prophet = chicago_df.resample('ME').size().reset_index()
print(chicago_prophet)
chicago_prophet.columns = ['Date', 'Crime Count']
chicago_prophet
chicago_prophet_df = pd.DataFrame(chicago_prophet)
'''
#
print(chicago_df.columns)  # Make sure 'Date' is in the columns
chicago_df['Date'] = pd.to_datetime(chicago_df['Date'], errors='coerce')  # Convert to datetime
chicago_df.set_index('Date', inplace=True)  # Set the Date column as the index
print(chicago_df.head())  # Check first few rows to see 'Date' as index
print(chicago_df.index)

# Resample by month ('M' instead of 'ME' which is incorrect)
chicago_prophet = chicago_df.resample('ME').size().reset_index()
print(chicago_prophet)
#chicago_prophet.columns = ['Date', 'Crime Count']


# Step 1: Convert Date column to datetime and set it as index
#chicago_df['Date'] = pd.to_datetime(chicago_df['Date'])
#chicago_df.set_index('Date', inplace=True)
#print(chicago_df.columns)
'''
# Step 2: Filter for top 15 crimes
top_15_crimes = chicago_df['Primary Type'].value_counts().index[:15]
filtered_df = chicago_df[chicago_df['Primary Type'].isin(top_15_crimes)]

# Step 3: Aggregate crime counts by month & crime type
crime_counts = filtered_df.groupby([pd.Grouper(freq='M'), 'Primary Type']).size().reset_index()
crime_counts.columns = ['ds', 'crime_type', 'y']

# Step 4: Train a separate Prophet model for each crime type
future_predictions = []

for crime in top_15_crimes:
    crime_data = crime_counts[crime_counts['crime_type'] == crime][['ds', 'y']]

    if len(crime_data) < 2:  # Ensure there is enough data for forecasting
        continue

    # Initialize and train the model
    model = Prophet()
    model.fit(crime_data)

    # Create future dates for the next 12 months
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Store results
    forecast['crime_type'] = crime
    future_predictions.append(forecast[['ds', 'crime_type', 'yhat']])

# Step 5: Concatenate all forecasts and find the highest predicted crime type each month
forecast_df = pd.concat(future_predictions)
most_frequent_crime = forecast_df.loc[forecast_df.groupby('ds')['yhat'].idxmax()]

# Step 6: Plot results
plt.figure(figsize=(12, 6))
for crime in top_15_crimes:
    crime_forecast = forecast_df[forecast_df['crime_type'] == crime]
    plt.plot(crime_forecast['ds'], crime_forecast['yhat'], label=crime)

plt.title('Predicted Crime Cases for the Next Year')
plt.xlabel('Date')
plt.ylabel('Predicted Crime Cases')
plt.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Step 7: Print most frequent predicted crime type for each month
print(most_frequent_crime[['ds', 'crime_type', 'yhat']])
'''

#part 4- predicting
chicago_prophet_df.columns
chicago_prophet_df_final = chicago_prophet_df.rename(columns={'Date':'ds', 'Crime Count':'y'})
chicago_prophet_df_final
m = Prophet()
m.fit(chicago_prophet_df_final)
# Forcasting into the future
future = m.make_future_dataframe(periods=365) #365days ahead/1year
forecast = m.predict(future)
figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
figure3 = m.plot_components(forecast)