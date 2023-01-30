from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
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

mlp= MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,solver='sgd', verbose=10, random_state=1,learning_rate_init=.1)
mlp2= MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,solver='sgd', verbose=10, random_state=1,learning_rate_init=.1)


mlp.fit(X_train, y_train)
mlp2.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
y_pred2 = mlp2.predict(X_test)

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
print("confusion matrix")
print("-"*48)

metrics.plot_confusion_matrix(mlp, X_test ,y_test)
plt.show()
metrics.plot_confusion_matrix(mlp2, X_test ,y_test)
plt.show()
print("-"*48)