
#
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns


#
train = pd.read_csv('train.csv', index_col=['PassengerId'])
test = pd.read_csv('test.csv', index_col=['PassengerId'])


#
print('train shape : ', train.shape)    # (891, 12)
print('test shape : ', test.shape)      # (418, 11)

print('train columns : ', train.columns)
print('test columns : ', test.columns)

print('train head : \n', train.head(10))
print('test head : \n', test.head(10))

print('train info : \n', train.info())
print('test info: \n', test.info())

print('train missing value : \n', train.isnull().sum())
# Age : 177, Cabin : 687, Embarked : 2

print('test missing value : \n', test.isnull().sum())
# Age : 86, Fare : 1, Cabin : 327

print('train describe : \n', train.describe())
print('test describe : \n', test.describe())

print('train data Pclass : \n', train['Pclass'].value_counts(sort=False))
print('test data Pclass : \n', test['Pclass'].value_counts(sort=False))

print('train data Embarked : \n', train['Embarked'].value_counts(sort=False))
print('test data Embarked : \n', test['Embarked'].value_counts(sort=False))

print('train data SibSp : \n', train['SibSp'].value_counts(sort=False))
print('test data SibSp : \n', test['SibSp'].value_counts(sort=False))

print('train data Parch : \n', train['Parch'].value_counts(sort=False))
print('test data Parch : \n', test['Parch'].value_counts(sort=False))

print('train data Survived : \n', train['Survived'].value_counts(sort=False))


# -----------------------------------------------------------------------------
#
train['dataset'] = 'train set'
test['dataset'] = 'test set'

merged = pd.concat([train, test])

print('merged data shape : ', merged.shape)     # (1309, 12)
print('merged data columns : ', merged.columns)
print('merged data head : \n', merged.head())
print('merged data tail : \n', merged.tail())


#
del merged['Cabin']


#
def to_number(data):
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1
    
    print(data['Sex'].head())
    
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 2
    
    print(data['Embarked'].head())
    
    return data
    
merged = to_number(merged)
print(merged[['Sex','Embarked']].head())
print(merged[['Sex','Embarked']].tail())
    

#
def title_name(name):
    n = re.findall(', .{1,15}\.', name)
    return ' '.join(n)[2:]

merged['title_name'] = np.nan
merged['title_name'] = merged['Name'].apply(lambda x: title_name(x))

print(merged[['Name', 'title_name']].head())
print(merged[['Name', 'title_name']].tail())
del merged['Name']


#
def ticket_str(name):
    n = re.findall('[a-zA-Z]', name)
    return ' '.join(n)

merged['ticket_str'] = np.nan
merged['ticket_str'] = merged['Ticket'].apply(lambda x: ticket_str(x))
merged['ticket_str'].value_counts()


#
def ticket_int(name):
    n = re.findall('[0-9]', name)
    return ''.join(n)

merged['ticket_int'] = np.nan
merged['ticket_int'] = merged['Ticket'].apply(lambda x: ticket_int(x))
merged['ticket_int'].value_counts()


# -----------------------------------------------------------------------------
#
def Pclass_bar(data, hue):
    plt.figure(figsize=(10,7))
    ax = sns.countplot(x='Pclass', hue=hue, data=data, palette='husl')                          
    plt.xlabel('')
    plt.xticks([0,1,2], ('1st', '2nd', '3rd'))
    plt.title('Pclass Frequency', fontsize=15)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(p.get_height()), 
                fontsize=12, ha='center', va='top', color='white')

Pclass_bar(merged, 'dataset')
plt.savefig('graph/bar_Pclass.png')
plt.show()

Pclass_bar(merged, 'Survived')
plt.savefig('graph/bar_Pclass_Survivd.png')
plt.show()


#
def Sex_bar(data, hue):
    plt.figure(figsize=(10,7))
    ax = sns.countplot(x='Sex', hue=hue, data=data, palette='husl')
    plt.xlabel('')
    plt.xticks([0,1], ('male', 'female'))
    plt.title('Sex frequency', fontsize=15)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(p.get_height()), 
                fontsize=12, ha='center', va='top', color='white')

Sex_bar(merged, 'dataset')
plt.savefig('graph/bar_Sex.png')
plt.show()

Sex_bar(merged, 'Survived')
plt.savefig('graph/bar_Sex_Survived.png')
plt.show()


#
def Embarked_bar(data, hue):
    plt.figure(figsize=(10,7))
    ax = sns.countplot(x='Embarked', hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
    plt.title('Embarked frequency', fontsize=15)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(p.get_height()), 
                fontsize=12, ha='center', va='top', color='white')
    
Embarked_bar(merged, 'dataset')
plt.savefig('graph/bar_Embarked.png')
plt.show()

Embarked_bar(merged, 'Survived')
plt.savefig('graph/bar_Embarked_Survived.png')
plt.show()


#
def countplot(x, hue, **kwargs):
    sns.countplot(x=x, hue=hue, palette='husl')
    
def bar(data, x, hue, col):
    g =  sns.FacetGrid(data, col=col, size=7)
    g = g.map(countplot, x, hue)
            
bar(merged, 'Pclass', 'Sex', 'Survived')
plt.xticks([0,1,2], ('1st', '2nd', '3rd'))
plt.legend(('male','female')).set_title('Sex')
plt.savefig('graph/bar_Pclass_Sex_Survived.png')
plt.show()

bar(merged, 'Embarked', 'Sex', 'Survived')
plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
plt.legend(('male','female')).set_title('Sex')
plt.savefig('graph/bar_Embarked_Sex_Survived.png')
plt.show()


#
def SibSp_bar(data, hue):
    plt.figure(figsize=(10,7))
    ax = sns.countplot(x='SibSp', hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.title('SibSp frequency', fontsize=15)
    plt.legend(loc='upper right').set_title(hue)
    
    for p in ax.patches:
        a = p.get_height()
        
        if np.isnan(a):
            a = 0

        ax.text(p.get_x() + p.get_width()/2., 
                a, '%d' % int(a), 
                fontsize=12, ha='center', va='bottom', color='#C39BD3')

SibSp_bar(merged, 'dataset')
plt.savefig('graph/bar_SibSp.png')
plt.show()

SibSp_bar(merged, 'Survived')
plt.savefig('graph/bar_SibSp_Survived.png')
plt.show()


#
def Parch_bar(data, hue):
    plt.figure(figsize=(10,7))
    ax = sns.countplot(x='Parch', hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.title('Parch frequency', fontsize=15)
    plt.legend(loc='upper right').set_title(hue)
    
    for p in ax.patches:
        a = p.get_height()
        
        if np.isnan(a):
            a = 0

        ax.text(p.get_x() + p.get_width()/2., 
                a, '%d' % int(a), 
                fontsize=12, ha='center', va='bottom', color='#C39BD3')

Parch_bar(merged, 'dataset')
plt.savefig('graph/bar_Parch.png')
plt.show()

Parch_bar(merged, 'Survived')
plt.savefig('graph/bar_Parch_Survived.png')
plt.show()


#
def Name_bar(data, hue):
    plt.figure(figsize=(15,7))
    ax = sns.countplot(x='title_name', hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.xticks(rotation=30)
    plt.legend(loc='upper right')
    plt.title('Title Name frequency', fontsize=15)
    
    for p in ax.patches:
        a = p.get_height()
        
        if np.isnan(a):
            a = 0
            
        ax.text(p.get_x() + p.get_width()/2., 
                a, '%d' % int(a), 
                fontsize=12, ha='center', va='bottom', color='#C39BD3')    
    

Name_bar(merged, 'dataset')
plt.savefig('graph/bar_Name.png')
plt.show()

Name_bar(merged, 'Survived')
plt.savefig('graph/bar_Name_Survived.png')
plt.show()


# -----------------------------------------------------------------------------
#
def Name_to_number(data):
    data['Mr'] = 0
    data.loc[data['title_name'] == 'Mr.', 'Mr'] = 1
    
    data['Mrs'] = 0
    data.loc[data['title_name'] == 'Mrs.', 'Mrs'] = 1
    
    data['Miss'] = 0
    data.loc[data['title_name'] == 'Miss.', 'Miss'] = 1
    
    data['Master'] = 0
    data.loc[data['title_name'] == 'Master.', 'Master'] = 1
    
    return data

merged = Name_to_number(merged)


#
temp = pd.get_dummies(merged.Embarked)
temp.columns = ['E_Cherbourg', 'E_Queenstown', 'E_Southampton']
merged = pd.concat([merged, temp], axis=1)


# -----------------------------------------------------------------------------
#
def Age_hist(**kwargs):
    sns.distplot(a='Age', hist=True, palette='husl')
    
def hist(data, hue, col):
    g =  sns.FacetGrid(data, hue=hue, col=col, size=7)
    g = g.map(Age_hist)
    
hist(merged, 'dataset', 'Survived')
plt.savefig('graph/hist_Age.png')
plt.show()



merged['Ticket'].value_counts()
merged['SibSp'].value_counts()
merged['Parch'].value_counts()

merged.loc[merged['Age'].isnull()]

train.to_csv('train_modify.csv', index=False)

