# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.'pandas': For handling and analyzing data.

2.'load_iris': Loads the Iris dataset, a built-in dataset in scikit-learn.

3.'SGDClassifier': Implements Stochastic Gradient Descent (SGD) for classification.

4.'train_test_split': Splits the dataset into training and testing sets.

5.'accuracy_score', 'confusion_matrix', 'classification_report': Evaluate model performance.

6.The Iris dataset is loaded.

7.The dataset is converted into a pandas DataFrame with feature names as column labels.

8.The target column (species labels: 0, 1, 2) is added.

9.The first few rows are printed to inspect the data.

10.'x' (features): All columns except target.

11.'y' (target variable): The 'target' column containing class labels.

12.80% of the data is used for training ('x_train', 'y_train').

13.20% of the data is used for testing ('x_test', 'y_test').

14.'random_state=42' ensures reproducibility (same split every time).

15.'SGDClassifier' is initialized with: 'max_iter=1000': Runs up to 1000 iterations to optimize weights. 'tol=1e-3': Stops early if the loss improvement is below '0.001'.

16.The classifier is trained on the training dataset using 'fit()'.

17.The trained model predicts labels ('y_pred') for 'x_test' using 'predict()'.

18.'accuracy_score(y_test, y_pred)' compares predictions with actual values.

19.The accuracy (fraction of correct predictions) is printed.

20.The Confusion Matrix is printed to analyze how many predictions were correct or misclassified.

21.The Classification Report includes: Precision: How many positive predictions were actually correct? Recall: How many actual positives were correctly predicted? F1-score: Harmonic mean of precision and recall. Support: Number of actual occurrences of each class.

## Program:
Program to implement the prediction of iris species using SGD Classifier.
Developed by: KEERTHANA
RegisterNumber: 212224220046
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Iris dataset
iris = load_iris()
# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Display the first few rows of the dataset
print(df.head())
# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
``` 
## Output:
<img width="1409" height="402" alt="image" src="https://github.com/user-attachments/assets/510d5b72-dc0e-4671-af37-19192f9f4b36" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
