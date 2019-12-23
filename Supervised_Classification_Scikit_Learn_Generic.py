import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


from sklearn.metrics import (
    accuracy_score, 
    recall_score,
    precision_score,
    confusion_matrix
)











def Classification(func):
    def wrapper(X_train, y_train, X_test, y_test):
        t0 = time.time()
        func(X_train, y_train, X_test, y_test)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        print("elapsed time = %.2f" % (time.time()-t0))
        print("ACCURACY Test: ", accuracy_score(y_pred_test, y_test))
        print("ACCURACY Train: ", accuracy_score(y_pred_train, y_train))
        
        print('')
        print("--------------Classification Report Test-------------------")
        print('')
        try:
            print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1']))
        except:
            try:
                print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1','Class 2']))
            except:
                print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1','Class 2','Class 3','Class 4']))
        print('')
        print("--------------Classification Report Train-------------------")
        print('')
        try:
            print(classification_report(y_train, y_pred_train, target_names=['Class 0', 'Class 1']))
        except:
            try:
                print(classification_report(y_train, y_pred_train, target_names=['Class 0', 'Class 1','Class 2']))
            except:
                print(classification_report(y_train, y_pred_train, target_names=['Class 0', 'Class 1','Class 2','Class 3','Class 4']))
  
        print('')
        print("--------------CONFUSION-MATRIX-------------------")
        print('')

        conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
        print('Confusion matrix:\n', conf_mat)

        labels = ['Class 0', 'Class 1','Class 2','Class 3','Class 4']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True Value')
        plt.show()
        
    return wrapper

@Classification
def LogisticReg(X_train, y_train, X_test, y_test):
    global clf
    clf = LogisticRegression(solver='lbfgs', max_iter=400, verbose=1,n_jobs=-1)
    
    
    
@Classification
def DecisionTree(X_train, y_train, X_test, y_test):
    global clf
    clf = DecisionTreeClassifier()
    
    

@Classification
def RandomForest(X_train, y_train, X_test, y_test):
    global clf
    clf = RandomForestClassifier(verbose=1,n_jobs=-1)
    
    
    
@Classification
def GaussianNb(X_train, y_train, X_test, y_test):
    global clf
    clf = GaussianNB()
       

@Classification
def SupportVectorMachine(X_train, y_train, X_test, y_test):
    global clf
    clf = SVC(verbose=1)
        

@Classification
def NeuralNetwork(X_train, y_train, X_test, y_test):
    global clf
    clf = MLPClassifier(hidden_layer_sizes=(10,20,10), activation='relu',verbose=1)

@Classification
def BaggingSVM(X_train, y_train, X_test, y_test):
    global clf
    clf = BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples= 0.01, n_estimators=10,n_jobs=-1, verbose = 1)