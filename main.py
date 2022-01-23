import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

stars = pd.read_csv('Stars.csv')
stars.head()

le = LabelEncoder()
le.fit(stars.Color)
stars['Color_label'] = le.transform(stars.Color)
le.fit(stars.Spectral_Class)
stars['Spectral_class_label'] = le.transform(stars.Spectral_Class)

X = stars[['Temperature', 'L', 'R', 'A_M', 'Color_label', 'Spectral_class_label']]
y = stars['Type']

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

knn.fit(X_train, y_train)
score=knn.score(X_test, y_test)
print("Accuracy:", "{:.0%}".format(round(score,2)))



k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9), cmap=cmap)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['Temperature'], X['L'], X['R'], c=y, marker='o', s=100)

ax.set_xlabel('Temperature')
ax.set_ylabel('L')
ax.set_zlabel('R')

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])

plt.show()
