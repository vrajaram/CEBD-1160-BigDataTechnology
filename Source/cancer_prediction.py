# Import libraries and sklearn libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.datasets import load_breast_cancer

def graph():
    df = pd.read_csv('../data/wdbc.data',
                     sep=',',
                     header=None)
    df.columns = ['ID', 'Diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area',
                  'mean smoothness', 'mean compactness', 'mean concavity',
                  'mean concave points', 'mean symmetry', 'mean fractal dimension',
                  'radius error', 'texture error', 'perimeter error', 'area error',
                  'smoothness error', 'compactness error', 'concavity error',
                  'concave points error', 'symmetry error', 'fractal dimension error',
                  'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                  'worst smoothness', 'worst compactness', 'worst concavity',
                  'worst concave points', 'worst symmetry', 'worst fractal dimension']

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    axes[0][0].scatter(df['Diagnosis'], df['mean radius'], alpha='0.2', marker=",", c='blue')
    axes[0][1].scatter(df['Diagnosis'], df['mean perimeter'], alpha='0.2', marker="o", c='green')
    axes[0][2].scatter(df['Diagnosis'], df['mean smoothness'], alpha='0.2', marker="v", c='cyan')
    axes[1][0].scatter(df['Diagnosis'], df['mean compactness'], alpha='0.2', marker=".", c='yellow')
    axes[1][1].scatter(df['Diagnosis'], df['mean concavity'], alpha='0.2', marker=",", c='red')
    axes[1][2].scatter(df['Diagnosis'], df['mean concave points'], alpha='0.2', marker="o", c='magenta')
    axes[2][0].scatter(df['Diagnosis'], df['mean symmetry'], alpha='0.2', marker="v", c='black')
    axes[2][1].scatter(df['Diagnosis'], df['mean fractal dimension'], alpha='0.2', marker=".", c='blue')
    axes[2][2].scatter(df['Diagnosis'], df['mean texture'], alpha='0.2', marker=".", c='green')

    axes[0][0].set_xlabel('Diagnosis')
    axes[0][1].set_xlabel('Diagnosis')
    axes[0][2].set_xlabel('Diagnosis')
    axes[1][0].set_xlabel('Diagnosis')
    axes[1][1].set_xlabel('Diagnosis')
    axes[1][2].set_xlabel('Diagnosis')
    axes[2][0].set_xlabel('Diagnosis')
    axes[2][1].set_xlabel('Diagnosis')
    axes[2][2].set_xlabel('Diagnosis')

    axes[0][0].set_ylabel('mean radius')
    axes[0][1].set_ylabel('mean perimeter')
    axes[0][2].set_ylabel('mean smoothness')
    axes[1][0].set_ylabel('mean compactness')
    axes[1][1].set_ylabel('mean concavity')
    axes[1][2].set_ylabel('mean concave points')
    axes[2][0].set_ylabel('mean symmetry')
    axes[2][1].set_ylabel('mean fractal dimension')
    axes[2][2].set_ylabel('mean texture')

    axes[0][0].set_title('Diagnosis Vs mean radius')
    axes[0][1].set_title('Diagnosis Vs mean perimeter')
    axes[0][2].set_title('Diagnosis Vs mean smoothness')
    axes[1][0].set_title('Diagnosis Vs mean compactness')
    axes[1][1].set_title('Diagnosis Vs mean concavity')
    axes[1][2].set_title('Diagnosis Vs mean concave points')
    axes[2][0].set_title('Diagnosis Vs mean symmetry')
    axes[2][1].set_title('Diagnosis Vs mean fractal dimension')
    axes[2][2].set_title('Diagnosis Vs mean texture')

    plt.tight_layout()
    plt.savefig('../plots/Diagnosis.png', dpi=300)

def main():
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    columns_names = cancer.feature_names

    print(columns_names)

    # Splitting features and target datasets into: train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    # Training a Linear Regression model with fit()
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Predicting the results for our test dataset
    predicted_values = lr.predict(X_test)

    # Printing the residuals: difference between real and predicted
    for (real, predicted) in list(zip(y_test, predicted_values)):
        print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

    # Printing accuracy score(mean accuracy) from 0 - 1
    print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

    # Printing the classification report
    print('Classification Report')
    print(classification_report(y_test, predicted_values))

    # Printing the classification confusion matrix (diagonal is true)
    print('Confusion Matrix')
    print(confusion_matrix(y_test, predicted_values))

    print('Overall f1-score')
    print(f1_score(y_test, predicted_values, average="macro"))

    # Graph for accuracy of the test data
    plt_array = np.arange(0, predicted_values.size)

    actual = np.zeros(predicted_values.size)
    for x in plt_array:
        if predicted_values[x]==y_test[x]:
            actual[x] = 1
        else:
            actual[x] = 0

    plt.figure(figsize=(5,5))
    plt.plot(plt_array, actual, 'gv')
    plt.xlabel('Number of test iteration')
    plt.ylabel('Correct Predicted value')
    plt.title('Accuracy of test')

    plt.legend()
    plt.savefig('../plots/performance.png', dpi=300)

    # Generate Graph
    graph()

if __name__ == "__main__":
    main()