import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import pylab

stars = pd.read_csv('Stars.csv')
stars.head()

print(stars)

print("Total number of attributes:", stars.shape[1], "\n")
print("Data set attributes: ")
print(list(stars.columns))
print("\n")

Type_label = {0: 'Red Dwarf', 1: 'Brown Dwarf',
              2: 'White Dwarf', 3: 'Main Sequence', 4: 'Super Giants', 5: 'Hyper Giants'}


le = LabelEncoder()
le.fit(stars.Color)
stars['Color_label'] = le.transform(stars.Color)
le.fit(stars.Spectral_Class)
stars['Spectral_class_label'] = le.transform(stars.Spectral_Class)

X = stars[['Temperature', 'L', 'R', 'A_M', 'Color_label', 'Spectral_class_label']]
y = stars['Type']

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

knn.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

print("kNN using 'Euclidean Distance'")
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print("Accuracy :", "{:.0%}".format(round(score, 2)))

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# create and show temperature and color bar plot
fig, ax = plt.subplots()
fig.canvas.manager.set_window_title('Temperature & Color')
stars.groupby('Color')['Temperature'].mean().plot.bar()
plt.gcf().subplots_adjust(bottom=0.25, left=0.22)
plt.ylabel('Average Temperature (K)')

plt.show()
