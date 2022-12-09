import os 
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


df = pd.read_csv(os.path.join("sign_lang_mnist", 'sign_mnist_test.csv'), usecols = ['label']) 
data_len = df.shape[0]

letters = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
height = [0] * 26
for i in range(26):
    if i == 9 or i == 25:
        continue
    height[i] = df.value_counts()[i]/data_len

x_pos = np.arange(len(letters))
plt.bar(x_pos, height)
plt.title('Distribution of Training Data')
plt.xlabel("Character")
plt.ylabel('Proportion')
plt.xticks(x_pos, letters)
plt.show()
