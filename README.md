# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Naveen R
RegisterNumber: 212225040276 

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts ()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le. fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
"time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x. head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt. fit(x_train, y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics. accuracy_score(y_test, y_pred)
accuracy

0.9843333333333333

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
<img width="1343" height="398" alt="Screenshot 2026-02-14 152037" src="https://github.com/user-attachments/assets/abcc7950-c8cd-420e-ac48-e56f38ac63fe" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

