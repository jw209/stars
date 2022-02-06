import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statistics import mode

stars = pd.read_csv('Stars.csv')
cols_to_norm = ['Temperature', 'L', 'R', 'A_M']
stars[cols_to_norm] = stars[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
stars.head()
print(stars.duplicated())

print("Total number of attributes:", stars.shape[1]-1, "\n")
print("Data set attributes: ")
print(list(stars.columns))
print("\n")

# create data frames of specific star types
red_dwarf = stars.where(stars['Type'] == 0)
brown_dwarf = stars.where(stars['Type'] == 1)
white_dwarf = stars.where(stars['Type'] == 2)
main_sequence = stars.where(stars['Type'] == 3)
super_giants = stars.where(stars['Type'] == 4)
hyper_giants = stars.where(stars['Type'] == 5)

# print instance and class
print("Stars instance class:")
print("Red Dwarf:",red_dwarf['Type'].count())
print("Bronw Dwarf:",brown_dwarf['Type'].count())
print("White Dwarf:",white_dwarf['Type'].count())
print("Main Sequence:",main_sequence['Type'].count())
print("Super Giants:",super_giants['Type'].count())
print("Hyper Giants:",hyper_giants['Type'].count(),"\n")

Type_label = {0: 'Red Dwarf', 1: 'Brown Dwarf',
              2: 'White Dwarf', 3: 'Main Sequence', 4: 'Super Giants', 5: 'Hyper Giants'}

le = LabelEncoder()
le.fit(stars.Color)
stars['Color_label'] = le.transform(stars.Color)
le.fit(stars.Spectral_Class)
stars['Spectral_class_label'] = le.transform(stars.Spectral_Class)

X = stars[['Temperature', 'L', 'R', 'A_M', 'Color_label', 'Spectral_class_label']]
y = stars['Type']

knn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')

knn.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.35, random_state=42)

print("kNN using 'Manhattan Distance'")
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print("Accuracy :", "{:.0%}".format(round(score, 2)))

k_range = range(1, 20)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

# create and show accuracy vs k-value plot
plt.plot(k_range, scores_list)
plt.xlabel('Value of k for kNN')
plt.ylabel('Testing Accuracy')

# create and show confusion matrix for star classifier
for k in range(1, 4):
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    disp.ax_.set(xlabel='Predicted', ylabel='True', title=f"Star Type Classifier Confusion Matrix knn={k}")

# create and show temperature and color bar plot
fig, ax = plt.subplots()
stars.groupby('Color')['Temperature'].mean().plot.bar()
plt.gcf().subplots_adjust(bottom=0.40, left=0.22)
plt.ylabel('Average Temperature (K)')

# create temperature and star type bar plot
fig1, cx = plt.subplots()
stars.groupby('Type')['Temperature'].mean().plot.bar()
plt.gcf().subplots_adjust(bottom=0.25, left=0.22)
plt.ylabel('Average Temperature (K)')

# create lumonsity and star type bar plot
fig2, dx = plt.subplots()
stars.groupby('Type')['L'].mean().plot.bar()
plt.gcf().subplots_adjust(bottom=0.25, left=0.22)
plt.ylabel('Relative lumonisty (W)')

# create radius and star type bar plot
fig3, ex = plt.subplots()
stars.groupby('Type')['R'].mean().plot.bar()
plt.gcf().subplots_adjust(bottom=0.25, left=0.22)
plt.ylabel('Relative Radius (m)')

# create absolute magnitude and star type bar plot
fig4, fx = plt.subplots()
stars.groupby('Type')['A_M'].mean().plot.bar()
plt.gcf().subplots_adjust(bottom=0.25, left=0.22)
plt.ylabel('Absolute Magnitude (Mv)')

# create and show class distribution
plt.figure()
bx = sns.scatterplot(data=stars, x=X['Temperature'], y=X['A_M'], hue='Label', size=X['R'])
bx.set(xlabel='Temperature (K)', ylabel='Absolute Magnitude')

# create data frames of specific star types
red_dwarf = stars.where(stars['Type'] == 0)
brown_dwarf = stars.where(stars['Type'] == 1)
white_dwarf = stars.where(stars['Type'] == 2)
main_sequence = stars.where(stars['Type'] == 3)
super_giants = stars.where(stars['Type'] == 4)
hyper_giants = stars.where(stars['Type'] == 5)

print('\nDescribing red dwarf stars: \n', red_dwarf.describe())
print('\nDescribing brown dwarf stars: \n', brown_dwarf.describe())
print('\nDescribing white dwarf stars: \n', white_dwarf.describe())
print('\nDescribing main sequence stars: \n', main_sequence.describe())
print('\nDescribing super giant stars: \n', super_giants.describe())
print('\nDescribing hyper giant stars: \n', hyper_giants.describe())
# print(hyper_giants.mode(numeric_only=True))

# testing real world star (we are trying to see if Betelgeuse is a super giant)
# spectral type: M1-M2, A_M: -5.85, Temperature: 3600+-200, L: 126,000, R: 764

unknown = pd.DataFrame([[3600, 126000, 764, -5.85, 'Red', 'M']], columns=['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class'])
le.fit(unknown.Color)
unknown['Color'] = le.transform(unknown.Color)
le.fit(unknown.Spectral_Class)
unknown['Spectral_Class'] = le.transform(unknown.Spectral_Class)

star_prediction = knn.predict(unknown)
print('The unknown star you are predicting is apart of: ', Type_label[star_prediction[0]], ' star type')
print(knn.predict_proba(unknown))

unknown2 = pd.DataFrame([[3150, 0.00035, 0.14, 15, 'Red', 'M']], columns=['Temperature', 'L', 'R', 'A_M', 'Color', 'Spectral_Class'])
le.fit(unknown2.Color)
unknown2['Color'] = le.transform(unknown2.Color)
le.fit(unknown2.Spectral_Class)
unknown2['Spectral_Class'] = le.transform(unknown2.Spectral_Class)

star_prediction = knn.predict(unknown2)
print('The unknown star you are predicting is apart of: ', Type_label[star_prediction[0]], ' star type')
print(knn.predict_proba(unknown2))

plt.show()
