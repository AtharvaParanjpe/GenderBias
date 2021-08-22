import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt

m = pd.read_excel("./Temporary Distribution/Data_Points_For_Accuracy_1.xlsx")
f = pd.read_excel("./Temporary Distribution/Data_Points_For_Accuracy_0.xlsx")

minValsMale = m.min()
minValsFemale = f.min()
maxValsMale = m.max()
maxValsFemale = f.max()
meanValsMale = m.mean()
meanValsFemale = f.mean()

## Dataframe objects
df_male = pd.concat([minValsMale, maxValsMale, meanValsMale], axis=1)
df_female = pd.concat([minValsFemale, maxValsFemale, meanValsFemale], axis=1)
df_male.columns = ['Min_Male', 'Max_Male', 'Mean_Male']
df_female.columns = ['Min_Female', 'Max_Female', 'Mean_Female']

## Plot distributions Male
kwargs = dict(alpha=0.5)
fig, axs = plt.subplots(2, 2)
fig.suptitle("Male")
plt.xlim([0.0, 1.0])
axs[0, 0].hist(m.values[:,0], **kwargs, color="g", bins=15, label="Linear")
axs[0, 0].set_title('Linear')
axs[0, 1].hist(m.values[:,1], **kwargs, color="r", bins=15, label="Logistic")
axs[0, 1].set_title('Logistic')
axs[1, 0].hist(m.values[:,2], **kwargs, color="b", bins=15, label="SVM")
axs[1, 0].set_title('SVM')
axs[1, 1].hist(m.values[:,3], **kwargs, color="grey", bins=15, label="Decision Tree")
axs[1, 1].set_title('Decision Tree')

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)
plt.subplot_tool()
for ax in axs.flat:
    ax.set(xlabel='Accuracy', ylabel='Frequency')
    ax.set_xlim(0.0, 1.0)

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.show()


## Plot distributions female
kwargs = dict(alpha=0.5)
fig, axs = plt.subplots(2, 2)
plt.xlim([0.0, 1.0])
fig.suptitle("Female")
axs[0, 0].hist(f.values[:,0], **kwargs, color="g", bins=15, label="Linear")
axs[0, 0].set_title('Linear')
axs[0, 1].hist(f.values[:,1], **kwargs, color="r", bins=15, label="Logistic")
axs[0, 1].set_title('Logistic')
axs[1, 0].hist(f.values[:,2], **kwargs, color="b", bins=15, label="SVM")
axs[1, 0].set_title('SVM')
axs[1, 1].hist(f.values[:,3], **kwargs, color="grey", bins=15, label="Decision Tree")
axs[1, 1].set_title('Decision Tree')


for ax in axs.flat:
    ax.set(xlabel='Accuracy', ylabel='Frequency')
    ax.set_xlim(0.0, 1.0)
 
# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)
plt.subplot_tool()
plt.show()