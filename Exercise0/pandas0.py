import pandas as pd

# Read the file ‘auto.csv’, df is dataframe
df = pd.read_csv('Exercise0/auto.csv')
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

#df4.loc[:, 'horsepower'] = df4.loc[:, 'horsepower'].astype(int)
df5 = df4.astype({'horsepower' : 'int64'})
print(df5)
print(df5.dtypes)

#Calculate the averages of every column, except for ‘name’.
df6 = df5.drop(columns ='name')

averages = df6.mean()
print(averages)