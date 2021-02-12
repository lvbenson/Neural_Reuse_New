
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
#print(os.getcwd())
#Reuse data

reused_2x5 = np.load("./Combined/4T_2x5/Data/reused_prop.npy")
#print(reused_2x5)
reused_2x3 = np.load("./Combined/4T_2x3/Data/reused_prop.npy")
reused_2x10 = np.load("./Combined/4T_2x10/Data/reused_prop.npy")
reused_2x20 = np.load("./Combined/4T_2x20/Data/reused_prop.npy")
reused_seq = np.load("./Combined/Sequential/Data/reused_prop.npy")

#Specialized data
special_2x5 = np.load("./Combined/4T_2x5/Data/special_prop.npy")
special_2x3 = np.load("./Combined/4T_2x3/Data/special_prop.npy")
special_2x10 = np.load("./Combined/4T_2x10/Data/special_prop.npy")
special_2x20 = np.load("./Combined/4T_2x20/Data/special_prop.npy")
special_seq = np.load("./Combined/Sequential/Data/special_prop.npy")

#mpg = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(mpg, column=['hwy', 'cty'], by="class", ylim='own', figsize=(14,10))

# Decoration
plt.title('Joy Plot of City and Highway Mileage by Class', fontsize=22)
plt.show()