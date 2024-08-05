import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = np.array([1,1,2,3,5,8,13,21,34])
b = np.array([1,8,28,56,70,56,28,8,1])

# Make a plot with two lines, using ‘a’ and ‘b’.
# Name the first axis ‘epochs’ and the 2nd axis ‘accuracy’. 
# Call the line made from ‘a’ for ‘training accuracy’ and the line made from ‘b’ for ‘validation accuracy’.
plt.figure(figsize=(10, 6))
plt.plot(a, label='training accuracy')
plt.plot(b, label='validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()