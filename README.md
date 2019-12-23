# Supervised Classification Scikit-Learn

This function serves to implement, evaluate and visualize supervised learning classifier performances.
The classifiers contained in the function:

The classifiers contained in the function:

- Logistic regression (solver = lbfgs, max iter= 400)
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Support Vector Machine
- Neural Network (layer structure: 10,20,10, activation function: relu)
- Bagging Support Vector Machine (kernel: linear, sample size = 1%, estimators = 10)

In its current form it assumes the following variables:

- X_train_new 
- y_train
- X_test_new
- y_test

(X_test0)
(y_test0)

X_train_new: contains the data on which the model trains, excluding the target variable y (variable to be predicted)

y_train: contains only the true labels

X_test_new: contains the data on which the model will be tested, excluding the target variable y 

y_test contains the true labels

-----

X_test0 and y_test0 can be manually deleted from the function if not needed.

This function was developed for a dataset split into two CSV files. The first CSV contained week1 data and the second CSV week2 data. In fact, an 80%-20% split was performed on the week1 data on which the classifiers were trained. In detail, 80% of the data of week1 (80 % of X_train_new) was used for training exclusively, and the other 20% served as test data (20% of X_train_new) --> creating X_test0 and y_test0. Finally, week2 data was solely used for model testing purposes (X_test_new and y_test).



# Example 1:


## Performing the split: 

from sklearn.model_selection import train_test_split


X_train, X_test0, y_train, y_test0 = train_test_split(
    train_df_b.drop(['class'], axis=1), 
   train_df_b['class'], 
    test_size=0.2
)


## Perparing test data from second csv file (week2):

X_test = test_df_b.drop('class', axis=1)
y_test = test_df_b['class']



## Performing scaling for numerical features: 


from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler().fit(X_train[numeric_cols])

X_train[numeric_cols] = standard_scaler.transform(X_train[numeric_cols])
X_test0[numeric_cols] = standard_scaler.transform(X_test0[numeric_cols])
X_test[numeric_cols] = standard_scaler.transform(X_test[numeric_cols])



## Calling the classifiers: 

logistic_reg(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)

decision_tree(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)

randomforest(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)

gaussiannb(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)

SupportVectorMachine(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)

BaggingSVM(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)

NeuralNetwork(X_train_new, y_train, X_test_new, y_test, X_test0, y_test0)


--------


The generic version (without X_test0 and y_test0):

Same functionality as above described; however, the function takes only the following inputs:



- X_train 
- y_train
- X_test
- y_test


Foremost, for the classification Report to work, the number of classes predicted needs to be either 2, 3, or 5. More classes can be predicted by changing the try and except statements in the wrapper of the code.

# Example 2:

## Performing the split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train_df_b.drop(['class'], axis=1), 
   train_df_b['class'], 
    test_size=0.2
)


## Perparing test data from second csv file (week2):

X_test = test_df_b.drop('class', axis=1)
y_test=test_df_b['class']


## Performing scaling for numerical features:

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler().fit(X_train[numeric_cols])

X_train[numeric_cols] = standard_scaler.transform(X_train[numeric_cols])

X_test[numeric_cols] = standard_scaler.transform(X_test[numeric_cols])



## Calling the classifiers: 


LogisticReg(X_train_num, y_train, X_test_num, y_test)

DecisionTree(X_train_num, y_train, X_test_num, y_test)

RandomForest(X_train_num, y_train, X_test_num, y_test)

GaussianNb(X_train_num, y_train, X_test_num, y_test)

SupportVectorMachine(X_train_num, y_train, X_test_num, y_test)

BaggingSVM(X_train_num, y_train, X_test_num, y_test)

NeuralNetwork(X_train_num, y_train, X_test_num, y_test)
