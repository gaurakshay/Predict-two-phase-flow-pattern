import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier

CLASS_COL = 14
data = np.genfromtxt('mfdb.csv', delimiter=',', skip_header=1)
# rand_rows = np.random.choice(data.shape[0], size=8000, replace=False)
# data = data[rand_rows, :]
TRAIN_ROWS = math.floor(data.shape[0] * 0.8)
print(data.shape)
print('Depth, Trees, Accuracy')
for depth in [5, 10, 20, 40]:
    for n_trees in [400]:
        rfc = RandomForestClassifier(n_estimators=n_trees, criterion='gini', max_depth=depth)
        rfc.fit(data[:TRAIN_ROWS, :CLASS_COL], data[:TRAIN_ROWS, CLASS_COL])
        prediction = rfc.predict(data[TRAIN_ROWS:, :CLASS_COL])
        # print(prediction)
        accuracy = data[TRAIN_ROWS:, CLASS_COL] - prediction
        # print(accuracy)
        # print(len(accuracy))
        # print(np.count_nonzero(accuracy))
        print('{}, {}, {}'.format(depth, n_trees, float((len(accuracy) - np.count_nonzero(accuracy)) * 100) / float(len(accuracy))))
        # print('No of trees: {}'.format(n_trees))
        # print('Accuracy: {}'.format(float((len(accuracy) - np.count_nonzero(accuracy)) * 100) / float(len(accuracy))))

