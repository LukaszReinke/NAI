from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd


data_frame = pd.read_csv("dataset_54_vehicle.csv")

collumns_to_encode = []  

le = LabelEncoder()
#https://www.openml.org/d/54
data_frame['Class'] = le.fit_transform(data_frame['Class'])
#van -> 3, saab -> 2, bus -> 0, opel -> 1


X_train, X_test, y_train,y_test = train_test_split(data_frame.drop(["Class"], axis=1), data_frame["Class"], test_size=1 / 3)
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Dla danych")
print(X_test.iloc()[0])
print("wynik:")
print(y_pred[0])
print("-"*48)
print("Dla danych")
print(X_test.iloc()[1])
print("wynik:")
print(y_pred[1])
print("-"*48)
print("Dla danych")
print(X_test.iloc()[2])
print("wynik:")
print(y_pred[2])
print("-"*48)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

