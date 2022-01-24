import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

knn.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

print(stars['Color'].unique())

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.style.use('_mpl-gallery')

fig, ax = plt.subplots()
fig1, bx = plt.subplots()

bx.bar(stars['Type'].unique(), len(stars['Type'].unique()), width=1, edgecolor="white", linewidth=0.7)

# temperature and color scatter plot
ax.scatter(stars['Temperature'], stars['Color'])
plt.gcf().subplots_adjust(bottom=0.15, left=0.15)

plt.show()
