#--- Load packages for datasets---
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

#--- Load packages for logistic regression and random forest---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#--- Load packages for train/test split---
from sklearn.model_selection import train_test_split

# Now, we will start to train logistic regression models on the Iris and Wine datasets.

# TODO: Load the Iris dataset using sklearn.
X, y = ...
# Split train/test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)

# TODO: Initialize a logistic regression model for the Iris dataset.
# Here, you only need to tune the inverse regularization parameter `C`. 
# Please set `random_state` to 3.
lr = ...

# Start training.
lr.fit(X_train, y_train)

# Print the training error.
1 - lr.score(X_train, y_train)
# TODO: Print the testing error.
...


# TODO: Load the Wine dataset.
X, y = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)

# TODO: Initialize a logistic regression model.
# Here, you only need to tune the inverse regularization parameter `C`. 
# Please set `random_state` to 3.
lr = ...

# Start training.
lr.fit(X_train, y_train)

# Print the training error.
1 - lr.score(X_train, y_train)
# TODO: Print the testing error.
...

# Now, we will start to train random forest models on the Iris and Breast Cancer datasets.

# Load the Iris dataset for training a random forest model.
X, y = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 1)

# Initialize a random forest model using sklearn.
# Here, you need to take turns to tune max_depth/max_samples for showing cases of underfitting/overfitting.
# Note that when you tune max_depth, please leave max_samples unchanged!
# Similarly, when you tune max_samples, leave max_depth unchanged!
# Please set `random_state` to 3 and feel free to set the value of `n_estimators`.
rf = ...

# Start training.
rf.fit(X_train, y_train)

# Print the training error.
1 - rf.score(X_train, y_train)
# TODO: Print the testing set error.
...

# Load the Breast Cancer dataset.
X, y = ...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 3)

# TODO: Initialize a random forest model for the Breast Cancer dataset.
rf = ...

# Start training.
rf.fit(X_train, y_train)

# Print the training error.
1 - rf.score(X_train, y_train)
# TODO: Print the testing error.
...

