import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def his_plot(col1,col2):
    model = LinearRegression()
    x = col1.values.reshape(-1, 1)
    y = col2.values
    model.fit(x,y)
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

            
    plt.figure(figsize=[10, 8])
    plt.hexbin(col1,col2,bins='log',)
    #plt.colorbar(col1,col2,bins='log')
    plt.plot(x_range,y_range,"r--")
    plt.show()