import numpy as np

data = np.genfromtxt('mfdb.csv', delimiter=',', skip_header=1)
print(data.shape)
d1 = data[data[:, 14] == 1, :]
d2 = data[data[:, 14] == 2, :]
d3 = data[data[:, 14] == 3, :]
d4 = data[data[:, 14] == 4, :]
d5 = data[data[:, 14] == 5, :]
d6 = data[data[:, 14] == 6, :]
print(np.unique(d1[:, 14]))

r1 = np.random.choice(d1.shape[0], size=100, replace=False)
r2 = np.random.choice(d2.shape[0], size=100, replace=False)
r3 = np.random.choice(d3.shape[0], size=100, replace=False)
r4 = np.random.choice(d4.shape[0], size=100, replace=False)
r5 = np.random.choice(d5.shape[0], size=100, replace=False)
r6 = np.random.choice(d6.shape[0], size=100, replace=False)

d1 = d1[r1, :]
d2 = d2[r2, :]
d3 = d3[r3, :]
d4 = d4[r4, :]
d5 = d5[r5, :]
d6 = d6[r6, :]

d_fin = np.vstack((d1, d2, d3, d4, d5, d6))
print(d_fin.shape)
print(np.unique(d_fin[:, 14]))
np.random.shuffle(d_fin)

np.savetxt('mfdb_2.csv', d_fin, delimiter=',')


rand_rows = np.random.choice(data.shape[0], size=8000, replace=False)