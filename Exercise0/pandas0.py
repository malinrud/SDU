import pandas as pd

# Read the file ‘auto.csv’, df is dataframe
df = pd.read_csv('auto.csv')
print(df)

#Remove all rows with ‘mpg’ lower than 16.
df2 = df[df['mpg'] >= 16]
print(df2)

#Get the first 7 rows of the columns’ weights’ and ‘acceleration’.
df3 = df2[['weight', 'acceleration']].head(7)
print(df3)

#Remove rows in ‘horsepower’ col with value ‘?’+ convert col to ‘int’ type instead of ‘string’
df4 = df[df['horsepower'] != '?']
print(df4)

print(df4.dtypes)

# df4['horsepower'] = df4['horsepower'].astype(int)

#Calculate the averages of every column, except for ‘name’.