import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import csv


def reader(filename):
    df = pd.read_csv(filename)
    df = df[df['SUICIDE_AGE'] < 30]  ## replace 30 for upper limit
    ## Uncomment below for applying lower limit
    # df = df[df['SUICIDE_AGE'] > 10]
    df = shuffle(df)
    return df

## Do for both positive and negative samples to get the resulting files
df =  reader('./Filtered Negatives/neg 1')
for i in range(2,13):
    df2 = reader('./Filtered Negatives/neg '+str(i))
    df = df.append(df2, ignore_index=False)
    print("Done " + str(i))

df = df.drop('Unnamed: 0', axis =1)
df = df.sample(n=200)
print(df.shape)
# final = pd.ExcelWriter('./For_25.xlsx')
# input()
df.to_excel('./neg/Neg_For_30.xlsx', index=False, header=True)

