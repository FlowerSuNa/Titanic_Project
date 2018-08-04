
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train_modify.csv', index_col=['PassengerId'])
test = pd.read_csv('test_modify.csv', index_col=['PassengerId'])
submission = pd.read_csv('gender_submission.csv')

train.columns
test.columns


## Divid data
def divid_data(df, feat):
    X_train, X_valid, y_train, y_valid = train_test_split(df[feat], df['Survived'], 
                                                          random_state=0, test_size=0.2)
    
    print("X_train : " + str(X_train.shape))
    print("X_valid : " + str(X_valid.shape))
    print("y_train : " + str(y_train.shape))
    print("y_valid : " + str(y_valid.shape) + '\n')
    
    print("download frequency of y_train : ")
    print(y_train.value_counts())
    
    print("download frequency of y_valid : ")
    print(y_valid.value_counts())
    
    return X_train, X_valid, y_train, y_valid



def logistic(train_df, test_df, feat, c=1):
    print("When C=%.2f :" %c)    
    
    ## Divid Dataset
    X_train, X_valid, y_train, y_valid = divid_data(train_df, feat)
    
    ## Train the model
    log = LogisticRegression(C=c)
    log.fit(X_train, y_train)
        
    ## Evaluate the model
    train_score = log.score(X_train, y_train)
    valid_score = log.score(X_valid, y_valid)
    
    print("Accuracy of train data : %.2f" % train_score)
    print("Accuracy od valid data : %.2f" % valid_score)
    
        
    ## Predict target variable
    pred = log.predict(test_df[feat])      
    
    return pred

feat = ['Sex', 'Mr', 'Mrs', 'Miss', 'Master', 
        'E_Cherbourg', 'E_Queenstown', 'E_Southampton']
pred = logistic(train, test, feat)
submission['Survived'] = pred
submission.to_csv('submission/logistic_1.csv', index=False)

del submission['is_attributed']
