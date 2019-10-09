#%%

import numpy as np
x = np.array([[1,2,3], [4,5,6]])
print("x:\n{}".format(x))

#%%

from scipy import sparse

eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

#%%


sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

#%%

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))


#%%

import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")

#%%

import pandas as pd
from IPython.display import display

data = {'Name': ["John", "Anna", "Peter", "Linda"],
'Location': ["New York", "Paris", "Berlin", "London"],
'Age': [24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)
display(data_pandas)
#%%

display(data_pandas[data_pandas.Age > 30])
