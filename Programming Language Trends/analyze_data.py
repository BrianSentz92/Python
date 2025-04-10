import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the data from the CSV
df = pd.read_csv('QueryResults.csv', names=['DATE', 'TAG', 'POSTS'], header=0)

# Step 2: Convert the 'DATE' column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Step 3: Pivot the DataFrame so that each TAG becomes a column
reshaped_df = df.pivot(index='DATE', columns='TAG', values='POSTS')

# Step 4: Fill NaN values with 0
reshaped_df.fillna(0, inplace=True)

# Step 5: Apply rolling mean with a window size of 6
roll_df = reshaped_df.rolling(window=6).mean()

# Step 6: Plotting the smoothed data
plt.figure(figsize=(16, 10))  # Make the chart larger
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Posts', fontsize=14)
plt.ylim(0, 35000)

# Plot each programming language in the smoothed DataFrame
for column in roll_df.columns:
    plt.plot(roll_df.index, roll_df[column], linewidth=3, label=column)

# Add legend
plt.legend(fontsize=16)

# Show the plot
plt.show()
