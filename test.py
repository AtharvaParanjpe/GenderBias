import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# patientsFile = pd.read_csv('./output_1/csv/patients.csv',sep = ',', low_memory=False)
# print(list(patientsFile.columns))
# input()
# for x in list(patientsFile.columns):
#     print(x)
#     print(patientsFile[x].values[5])
#     print("---------------------------------------------------------------------------")
#     input()

import csv

# modifiedDataset = []
# columns = []
# with open('./csv9/patients.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     count=0
#     for row in readCSV:
#         count+=1
#         if(count==1):
#             columns = row
#             continue
#         if(len(row)==17 and int(row[1].split("-")[0])>2000):
#             modifiedDataset.append(row)

# df = pd.DataFrame(modifiedDataset,columns = columns)
# df.to_csv('./csv9/NewPatients.csv')


################### COMBINING ALL THE FILES ##########################

def reader(filename):
    df = pd.read_csv(filename)
    df = df[df['SUICIDE_AGE'] < 30]
    # df = df[df['SUICIDE_AGE'] > 10]
    df = shuffle(df)
    return df

df =  reader('./Negative Samples/neg 1')
for i in range(2,13):
    df2 = reader('./Negative Samples/neg '+str(i))
    df = df.append(df2, ignore_index=False)
    print("Done " + str(i))

df = df.drop('Unnamed: 0', axis =1)
df = df.sample(n=200)
print(df.shape)
# final = pd.ExcelWriter('./For_25.xlsx')
# input()
df.to_excel('./Neg_For_30.xlsx', index=False, header=True)

