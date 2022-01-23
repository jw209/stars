import pandas as pd
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

knn.fit(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
