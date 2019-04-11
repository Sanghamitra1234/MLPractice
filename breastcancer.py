import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.drop(columns=['Unnamed: 32'], inplace=True))
