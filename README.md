# CEBD-1160-BigDataTechnology
Final Project to introduction to big data technology

| Name | Date |
|:-------|:---------------|
|Vignesh Rajaram|March 28, 2020|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: `network_analysis.py`
- Results figure/saved file:  `figures/`
- Dockerfile for your experiment: `Dockerfile`
- runtime-instructions in a file named RUNME.md

-----

## Research Question

Based on the geometrical characteristics of tumor to determine whether they are malignant or benign tumor?

### Abstract

In this project, based on the data available from Wisconsin Diagnostic Breast Cancer(WDBC): We are creating a model to predict the type of tumor based on the geometrical characteristics of tumor. This model will help in better and quick decision making on treatment for either malignant or benign tumor, with initial data on geometrical characteristics of tumor. Using logistic regression model available in sklearn library a model will be created to predict the behavior. Based on the F1 score of the regressor the initial test provided encouraging result, but due to its application in medical field the F1 score must be improved further.

### Introduction

There are two different types of tumor:
Malignant tumor - which may invade surrounding tissue or spread around the body
Benign tumor - which does not affect the surrounding tissue or spread around the body. [1]

The dataset obtained from the University of Wisconsin, describes the features of cell nuclei present in the digitized image. Based on the features obtained the type of tumor is predicted. The graphs are obtained from this dataset.

### Methods

The target in our dataset is to predict whether the tumor is either malignant or benign. Therefore this model falls under the classification model of the supervised machine learning. Hence, we use the Logistic Regressor built in scikit-learn for simplicity and applicability to the problem in hand. Psuedocode for the regressor can be found in this link (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

To train the model [2] and to obtain the metrics of the model [3], methods are available in the scikit learn. The graph is generated to compare the mean values of the features with type of tumor.

![matrix](./plots/Diagnosis.png)

### Results

Brief (2 paragraph) description about your results. Include:

At least 1 figure
At least 1 "value" that summarizes either your data or the "performance" of your method
A short explanation of both of the above

The performance of the regressor was an R^2 value of 0.661. The figure below shows the performance on the testing set.

![performange figure](./figures/performance.png)

We can see that in general, our regressor seems to underestimate our edgeweights. In cases where the connections are small, the regressor performs quite well, though in cases where the strength is higher we notice that the
performance tends to degrade.

### Discussion

Brief (no more than 1-2 paragraph) description about what you did. Include:

interpretation of whether your method "solved" the problem
suggested next step that could make it better.

The method used here does not solve the problem of identifying the strength of connection between two brain regions from looking at the surrounding regions. This method shows that a relationship may be learnable between these features, but performance suffers when the connection strength is towards the extreme range of observed values. To improve this, I would potentially perform dimensionality reduction, such as PCA, to try and compress the data into a more easily learnable range.

### References
[1] https://study.com/academy/lesson/benign-vs-malignant-definition-characteristics-differences.html
[2] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
[3] https://scikit-learn.org/stable/modules/model_evaluation.html
-------
