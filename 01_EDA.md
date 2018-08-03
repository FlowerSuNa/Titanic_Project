##### kaggle 'Titanic : Machine Learning from Disaster'

# 1. EDA and Preprocessing

### Import ibrary and load data


```python
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('train.csv', index_col=['PassengerId'])
test = pd.read_csv('test.csv', index_col=['PassengerId'])
```

<br>

### Explor the data

* train shape : (891, 12)
* test shape : (418, 11)

<br>

* train columns : Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
* test columns : Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

<br>

* train info : <br>
Survived    891 non-null int64
Pclass      891 non-null int64
Name        891 non-null object
Sex         891 non-null object
Age         714 non-null float64
SibSp       891 non-null int64
Parch       891 non-null int64
Ticket      891 non-null object
Fare        891 non-null float64
Cabin       204 non-null object
Embarked    889 non-null object
* test info : <br>
Pclass      418 non-null int64
Name        418 non-null object
Sex         418 non-null object
Age         332 non-null float64
SibSp       418 non-null int64
Parch       418 non-null int64
Ticket      418 non-null object
Fare        417 non-null float64
Cabin       91 non-null object
Embarked    418 non-null object

<br>

* train missing values : <br>
Age : 177
Cabin : 687
Embarked : 2
* test missing values : <br>
Age : 86
Fare : 1
Cabin : 327

<br>

* train data Pclass :
1 : 216
2: 184
3 : 491
* test data Pclass :
1 : 107
2 : 93
3 : 218

<br>

* train data Embarked :
S : 644
C : 168
Q : 77
* test data Embarked :
S : 270
Q : 46
X : 102

<br>

* train data SibSp :
0 : 608
1 : 209
2 : 28
3 : 16
4 : 18
5 : 5
8 : 7
* test data SibSp :
0 : 283
1 : 110
2 : 14
3 : 4
4 : 4
5 : 1
8 : 2

<br>

* train data Parch :
0 : 678
1 : 118
2 : 80
3 : 5
4 : 4
5 : 5
6 : 1
* test data Parch :
0 : 324
1 : 52
2 : 33
3 : 3
4 : 2
5 : 1
6 : 1
9 : 2

<br>

* train data Survived :
0 : 549
1 : 342

---

### Merge train data and test data

```python
train['dataset'] = 'train set'
test['dataset'] = 'test set'

merged = pd.concat([train, test])
```

* merged data shape : (1309, 12)
* merged data columns : Age, Cabin, Embarked, Fare, Name, Parch, Pclass, Sex, SibSp, Survived, dataset
