import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

"""
1. Load the auto.csv dataset again using the pandas.read function and remember to
remove the missing values in the dataset, indicated by ‘?’, and then make sure the
corresponding columns are casted to a numerical type. """

#Load the dataset using pandas.read()
df = pd.read_csv('SDU_indiv/Exercise2/auto.csv') 

# Remove ? and cast corresponding columns to a numerical type
df2 = df[df['horsepower'] != '?']
#print(df2)
#print(df2.dtypes)
df3 = df2.astype({'horsepower' : 'int64'})
print(df3.head())
#print(df3.dtypes)

""""
2. Inspect the data. Plot the relationships between the different variables and mpg. Use
for example the matplotlib.pyplot scatter plot. Do you already suspect what features
might be helpful to regress the consumption? Save the graph. """

# Plot the relationships between the different variables and mpg 
#   - use matplotlib.pyplot scatter plot
# I need to choose all the columns except for mpg and name and plot them against mpg


x_column = df3.columns[0]
y_columns = df3.columns[1:8]

plt.figure(figsize=(15,10))

for y_column in y_columns:
    plt.scatter(df3[x_column], df3[y_column], alpha=0.5, label=y_column)

plt.title('Scatter plot against {}'.format(x_column))
plt.xlabel(x_column)
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.savefig('mpg_plot.png')
plt.show
