import pandas as pd
bike_data = pd.read_csv('sample_citibike_2023.csv', dtype={'start_station_id': 'str', 'end_station_id': 'str'})
print(bike_data)
bike_data.dropna(inplace=True)

import matplotlib.pyplot as plt

# Ensure datetime format for 'started_at' and 'ended_at'
bike_data['started_at'] = pd.to_datetime(bike_data['started_at'])
bike_data['ended_at'] = pd.to_datetime(bike_data['ended_at'])

# Extract hours from 'started_at' and 'ended_at'
bike_data['start_hour'] = bike_data['started_at'].dt.hour
bike_data['end_hour'] = bike_data['ended_at'].dt.hour

# Group by start hour and end hour to get the count of rides
start_activity = bike_data.groupby('start_hour').size().reset_index(name='start_ride_count')
end_activity = bike_data.groupby('end_hour').size().reset_index(name='end_ride_count')

# Merge the start and end activity counts for consistent plotting
hourly_activity = pd.merge(start_activity, end_activity, left_on='start_hour', right_on='end_hour', how='outer').fillna(0)
hourly_activity['start_hour'] = hourly_activity['start_hour'].astype(int)
hourly_activity['end_hour'] = hourly_activity['end_hour'].astype(int)

# Plot both start and end hourly activity
"""
plt.figure(figsize=(10, 6))
plt.plot(hourly_activity['start_hour'], hourly_activity['start_ride_count'], marker='o', label='Start Activity')
plt.plot(hourly_activity['end_hour'], hourly_activity['end_ride_count'], marker='s', label='End Activity')
plt.title('Hourly Start and End Activity Throughout the Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rides')
plt.xticks(range(0, 24))  # Set x-ticks for each hour of the day
plt.legend()
plt.grid(True)
plt.show()
"""

from sklearn.cluster import KMeans
num_stations = bike_data[['start_station_id', 'start_lat', 'start_lng']]
print(num_stations) #997021
# Extract unique stations based on 'start_station_id', 'start_lat', and 'start_lng' to avoid duplicate station entries
unique_stations = bike_data[['start_station_id', 'start_lat', 'start_lng']].drop_duplicates()
print(unique_stations) #399340
# Run KMeans to create 30 clusters (regions) based on station coordinates
kmeans = KMeans(n_clusters=20, random_state=0)
unique_stations['region'] = kmeans.fit_predict(unique_stations[['start_lat', 'start_lng']])
print(kmeans)
print(unique_stations['region'].sum()) #399340
print(unique_stations['region'])

# Parse 'started_at' as datetime if needed
bike_data['started_at'] = pd.to_datetime(bike_data['started_at'], errors='coerce')

# Filter for trips that start between 7 a.m. and 8 a.m.
morning_demand_data = bike_data[(bike_data['started_at'].dt.hour == 7)]

# Add region information to morning demand data by merging with unique_stations DataFrame
morning_demand_data = morning_demand_data.merge(unique_stations[['start_station_id', 'region']],
                                                left_on='start_station_id',
                                                right_on='start_station_id',
                                                how='left')

# Extract the date part and group by region and date to get the daily count of trips per region
morning_demand_data['date'] = morning_demand_data['started_at'].dt.date
region_daily_morning_demand = morning_demand_data.groupby(['region', 'date']).size().reset_index(name='daily_count')

# Calculate the average daily demand per region for 7-8 a.m.
demand = region_daily_morning_demand.groupby('region')['daily_count'].mean().reset_index(name='demand').astype(int)
print(demand)
print(demand['demand'].sum()) #43526
import matplotlib.pyplot as plt

# Data from your code
regions = demand['region'].to_numpy()  # Assuming 'region' is a column name in your 'demand' DataFrame
demand_values = demand['demand'].to_numpy()  # Assuming 'demand' is a column name in your 'demand' DataFrame

# Create a bar chart
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.bar(regions, demand_values, color='skyblue')  # You can change the color

# Add labels and title
plt.xlabel('Region')
plt.ylabel('Average Daily Morning Demand (7-8 AM)')
plt.title('Average Daily Morning Demand by Region')

# Show the plot
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability if many regions
plt.tight_layout()
plt.show()

# Parse 'ended_at' as datetime
bike_data['ended_at'] = pd.to_datetime(bike_data['ended_at'], errors='coerce')

# Create a "supply_date" column to assign trips ending between 11 p.m. and 4 a.m. to the next day's date
bike_data['supply_date'] = bike_data['ended_at'].apply(lambda x: x.date() + pd.Timedelta(days=1) if x.hour >= 23 else x.date())

# Filter for trips ending between 11 p.m. and 4 a.m.
supply_hours_data = bike_data[((bike_data['ended_at'].dt.hour >= 23) | (bike_data['ended_at'].dt.hour < 4))]

# Add region information to supply data by merging with unique_stations DataFrame
supply_hours_data = supply_hours_data.merge(unique_stations[['start_station_id', 'region']],
                                            left_on='end_station_id',
                                            right_on='start_station_id',
                                            how='left')

# Group by region and supply date to get the daily supply count per region
region_daily_supply = supply_hours_data.groupby(['region', 'supply_date']).size().reset_index(name='daily_supply_count')

# Calculate the average daily supply per region
supply = region_daily_supply.groupby('region')['daily_supply_count'].mean().reset_index(name='supply').astype(int)

print(supply)
print(supply['supply'].sum()) #56135
import matplotlib.pyplot as plt

# Data from your code
regions = supply['region'].to_numpy()  # Assuming 'region' is a column name in your 'supply' DataFrame
supply_values = supply['supply'].to_numpy()  # Assuming 'supply' is a column name in your 'supply' DataFrame

# Create a bar chart
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.bar(regions, supply_values, color='skyblue')  # You can change the color

# Add labels and title
plt.xlabel('Region')
plt.ylabel('Average Daily Supply (11 PM - 4 AM)')
plt.title('Average Daily Supply by Region (11 PM - 4 AM)')

# Show the plot
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability if many regions
plt.tight_layout()
plt.show()

# Calculate the regional capacity based on the number of unique stations in each region
# Assume each station has a capacity of 20

# Count the number of unique stations in each region
reg_cap = unique_stations.groupby('region')['start_station_id'].nunique().reset_index(name='station_count')

# Calculate regional capacity
reg_cap['regional_capacity'] = reg_cap['station_count'] * 20

print(reg_cap)

# Merge demand, supply, and station capacity data on 'region'
demand_supply_capacity = demand.merge(supply, on='region', how='inner')\
                                             .merge(reg_cap, on='region', how='inner')

print(demand_supply_capacity)
