import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn 
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


if __name__ == "__main__":
    
    df = pd.read_csv("C:/Users/91958/Downloads/Titanic+Data+Set.csv")
    
    df = df.drop(['PassengerId','Name','Ticket','Cabin',], axis = 'columns')
    
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    
    X = df.drop('Survived', axis=1)
    
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    with mlflow.start_run():   

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        
        
        accuracy = metrics.accuracy_score(y_test,y_predict)
        precision = metrics.precision_score(y_test,y_predict)
        recall = metrics.recall_score(y_test,y_predict)
        f1 = metrics.f1_score(y_test,y_predict)

        print(accuracy)
        print(precision)
        print(recall)
        print(f1)

        mlflow.log_param("accuracy", accuracy)
        mlflow.log_param("precision", precision)
        mlflow.log_param("recall", recall)
        mlflow.log_param("f1", f1)

        #mlflow.log_metric("accuracy", accuracy)
        #mlflow.log_metric("precision", precision)
        #mlflow.log_metric("recall", recall)
        #mlflow.log_metric("f1", f1)