

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_datafile(file_name):
    data = pd.read_csv(file_name,delimiter=',', encoding="utf-8-sig")
    return data


# data = read_datafile('winequality-red.csv')
data = read_datafile('cars.csv')
print(data['brand'])

# fig = plt.figure()

# ax1 = fig.add_subplot(111)

# ax1.set_title("Mains power stability")
# ax1.set_xlabel('time')
# ax1.set_ylabel('Mains voltage')

# ax1.plot(data['fixed acidity'],data['free sulfur dioxide'], 'bo')

# leg = ax1.legend()

# plt.show()
