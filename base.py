import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sbn
# Assuming 'train.csv' is in the current directory
path = 'train.csv'
path_test = 'test.csv'
data = pd.read_csv(path)
data_test = pd.read_csv(path_test)
# Define columns to exclude
columns_to_exclude = ['PassengerId', 'Name', 'Age', 'Fare', 'Ticket', 'Cabin', 'Survived']
categorical_columns = ['Pclass', 'Sex', 'Embarked']  # Assuming these are categorical columns

# Create a new DataFrame 'x' by dropping specified columns
x = data.drop(columns=columns_to_exclude)
y = data['Survived']
# Perform one-hot encoding for each categorical column
for column in categorical_columns:
    x = pd.get_dummies(x, columns=[column], prefix=column, dtype='int')


x.to_csv('x_data.csv')
x_train, x_left, y_train, y_left = train_test_split(x, y, train_size=0.6, test_size=0.4, random_state=40)

x_cv, x_test, y_cv, y_test = train_test_split(x_left, y_left, train_size=0.5, test_size=0.5, random_state=40)


model = Sequential([
    Dense(units=128, input_shape=(x_train.shape[1],), activation='relu'),
    Dense(units=128, activation='relu'),
    Dropout(0.25),
    Dense(units=1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
batch_sizes = 300
epochs = 10
model.fit(x_train, y_train, batch_size=batch_sizes, epochs=epochs, validation_data=(x_cv, y_cv))
cv_loss, cv_acc = model.evaluate(x_cv, y_cv)
# print("The loss is: {:.4f} and the accuracy is: {:.4f}".format(cv_loss, cv_acc))


y_pred_old = model.predict(x_test)
y_pred = (y_pred_old > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"The accuracy: {accuracy}")

cfmx = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(15,20))


sbn.heatmap(cfmx, fmt='d', annot=True, ax=ax,cmap='Blues')
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title("Confusion matrix")
plt.show()