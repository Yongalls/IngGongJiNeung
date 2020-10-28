import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import f1_score

dataset = pd.read_csv("test.tsv", delimiter='\t', names=['image','plant','disease'])
combined_label = [(3,5), (3,20), (4,2), (4,7), (4,11),(5,8), (7,20), (8,6), (8,9), (10,20), (11,14), (13,1), (13,6), (13,9), (13,15), (13,16), (13,17), (13,18), (13,20), (7,1)]

print("index", "plant", "disease", "answer")

prediction = []
label = []

for i in range(3940):
    plant = dataset.iloc[i,1]
    disease = dataset.iloc[i,2]
    prediction.append(combined_label.index((plant,disease)))
    label.append(i % 20)
    if (plant, disease) != combined_label[i % 20]:
        print(i, plant, disease, combined_label[i % 20])

for i in range(57):
    plant = dataset.iloc[i + 3940, 1]
    disease = dataset.iloc[i + 3940, 2]
    prediction.append(combined_label.index((plant,disease)))
    label.append(i % 19)
    if (plant, disease) != combined_label[i % 19]:
        print(i+3940, plant, disease, combined_label[i % 19])

f1 = f1_score(prediction, label, average = 'macro')
print(f1)
