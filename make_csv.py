import numpy as np
import csv
import sys

with open('./data/predictions.npy', 'rb') as f:
    predictions = np.load(f)

with open('./data/answers.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, lineterminator='\n')

    writer.writerow(['name','invasive'])

    for i in range(len(predictions)):
        writer.writerow([i+1,predictions[i]])

print('DONE!')

