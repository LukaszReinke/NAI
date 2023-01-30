from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd

#https://www.openml.org/d/31
data_frame = pd.read_csv("dataset_31_credit-g.csv")
collumns_to_encode = ['checking_status',  'credit_history', 'purpose',
        'savings_status', 'employment',
       'installment_commitment', 'personal_status', 'other_parties',
       'property_magnitude', 'other_payment_plans',
       'housing', 'job', 'own_telephone',
       'foreign_worker', 'class'] ##bad -> 0; good -> 1

le = LabelEncoder()
for col in collumns_to_encode:
    print(data_frame[col])
    data_frame[col] = le.fit_transform(data_frame[col])
    print(data_frame[col])


X_train, X_test, y_train,y_test = train_test_split(data_frame.drop(["class"], axis=1), data_frame["class"], test_size=1 / 3)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Dla danych:")
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