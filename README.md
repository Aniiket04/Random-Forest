# Random-Forest

## Project Overview 

**Project Title :Random Forest**
The goal of this project is to develop a machine learning model using the Random Forest algorithm to accurately classify images of handwritten digits from the scikit-learn digit dataset. This dataset contains images of digits (0–9) represented as 8x8 grayscale pixel grids.

## Objectives
**1.Data Exploration**: Understand and visualize the dataset, including pixel intensity distributions and digit samples.
**2.Model Training**: Train a Random Forest classifier on the dataset.
**3.Evaluation**: Assess the model's accuracy and performance using metrics such as precision, recall, F1-score, and confusion matrices.
**4.Insights and Learning**: Draw insights about feature importance (e.g., which pixels are most important for classification).

## Project Structure

### 1. Importing Libraries and Loading the digits dataset
pandas for data manipulation
from sklearn.datasets load the digits dataset directly into our python environment
matplotlib for data visualization.
```python
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
%matplotlib inline
digits=load_digits()
digits
```
%matplotlib inline is a jupyter notebook command which is used to display plots directly in the notebook output cells

### 2. Data processing
```python
digits.data
digits.data[0]
```

### 3. Ploting the graph
```python
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])
```
plt.matshow(digits.images[i]) is used to visualize a specific digit image from the scikit-learn digits dataset.

### 4. Data processing
```python
dir(digits)
digits.target
digits.target_names
digits.data
df=pd.DataFrame(digits.data)
df
df["target"]=digits.target
df
x=df.drop('target',axis='columns')
x
y=df.target
y
```

### 5. Train/Test Split
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
len(x_train)
len(y_train)
```
test_size=0.2 means 20% of the data will be used for testing, while 80% will be used for training.

### 6.Model Training
```python
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)
model.fit(x_train,y_train)
```
'n_estimators=50' in the context of a Random Forest Classifier in scikit-learn specifies the number of decision trees to include in the ensemble. By setting it to 50, we are instructing the model to build 50 individual decision trees during training.

### 7. Model Prediction
Predictions on the digits dataset.
```python
model.score(x_test,y_test)
y_predicted=model.predict(x_test)
y_predicted
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm
```
Visualize this confusion matrix using seaborn's heatmap.
```python
import seaborn as sn
import matplotlib.pyplot as plt 
%matplotlib inline
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('y_predicted')
plt.ylabel('truth')
```

## Conclusion
In this project, we successfully built and evaluated a Random Forest classifier to recognize handwritten digits using the famous scikit-learn digit dataset. The Random Forest classifier with 50 estimators achieved high accuracy on the test set, demonstrating its effectiveness in handling the dataset. The confusion matrix highlighted that the model performed exceptionally well across most classes, with only a few misclassifications, indicating strong generalization.

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]
