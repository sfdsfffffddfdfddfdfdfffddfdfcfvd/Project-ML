import pandas as pd 
df = pd.read_csv('adult.csv')
import matplotlib.pyplot as plt 
df.drop(['workclass','fnlwgt','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country'], axis = 1, inplace = True)

print('Гипотезы: 1. Люди, которымм больше 30 зарабатывают больше.  2. Людей, которые отучились на бакалавра зарабатывают больше.')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
sc = StandardScaler()

def fill_h(age):
    if age >= 30:
        return 1
    return 0
df['age'] = df['age'].apply(fill_h)
df['age'] = df['age'].apply(int)

def fill_act(education):
    if education == 'Bachelors':
        return 1
    return 0
df['education'] = df['education'].apply(fill_act)
df['education'] = df['education'].apply(int)

print(df.info())
x = df.drop('income', axis = 1)
y = df['income']
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.25) 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('Процент правильно предсказанных исходов:',accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print(df.info())


