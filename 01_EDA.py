
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

print('train describe : \n', train.describe())
print('test describe : \n', test.describe())


#
train['dataset'] = 'train set'
test['dataset'] = 'test set'

merged = pd.concat([train, test])

print('merged data shape : ', merged.shape)     # (1309, 11)
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
    

#
def title_name(name):
    n = re.findall(', .{1,15}\.', name)
    return ' '.join(n)[2:]

merged['title_name'] = np.nan
merged['title_name'] = merged['Name'].apply(lambda x: title_name(x))

print(merged['title_name'].head())


#
def Pclass_bar(data, hue):
    ax = sns.countplot(x='Pclass', hue=hue, data=data, palette='husl')                          
    plt.xlabel('')
    plt.xticks([0,1,2], ('1st', '2nd', '3rd'))
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(p.get_height()), 
                fontsize=12, ha='center', va='top', color='white')

plt.figure(figsize=(10,7))
Pclass_bar(merged, 'dataset')
plt.title('Pclass Frequency', fontsize=15)
plt.savefig('graph/bar_Pclass.png')
plt.show()

plt.figure(figsize=(10,7))
Pclass_bar(merged, 'Survived')
plt.title('Pclass Frequency', fontsize=15)
plt.savefig('graph/bar_Pclass_Survivd.png')
plt.show()


#
def Sex_bar(data, hue):
    ax = sns.countplot(x='Sex', hue=hue, data=data, palette='husl')
    plt.xlabel('')
    plt.xticks([0,1], ('male', 'female'))
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(p.get_height()), 
                fontsize=12, ha='center', va='top', color='white')

plt.figure(figsize=(10,7))
Sex_bar(merged, 'dataset')
plt.title('Sex frequency', fontsize=15)
plt.savefig('graph/bar_Sex.png')
plt.show()

plt.figure(figsize=(10,7))
Sex_bar(merged, 'Survived')
plt.title('Sex frequency', fontsize=15)
plt.savefig('graph/bar_Sex_Survived.png')
plt.show()


#
def Embarked_bar(data, hue):
    ax = sns.countplot(x='Embarked', hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(p.get_height()), 
                fontsize=12, ha='center', va='top', color='white')
    
plt.figure(figsize=(10,7))
Embarked_bar(merged, 'dataset')
plt.title('Embarked frequency', fontsize=15)
plt.savefig('graph/bar_Embarked.png')
plt.show()

plt.figure(figsize=(10,7))
Embarked_bar(merged, 'Survived')
plt.title('Embarked frequency', fontsize=15)
plt.savefig('graph/bar_Embarked_Survived.png')
plt.show()


# ***
def Name_bar(data, hue):
    ax = sns.countplot(x='title_name', hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.xticks(rotation=30)
    plt.legend(loc='upper right')
    
    for p in ax.patches:
        a = p.get_height()

            
        ax.text(p.get_x() + p.get_width()/2., 
                p.get_height(), '%d' % int(a), 
                fontsize=12, ha='center', va='bottom')    
    
plt.figure(figsize=(10,7))
Name_bar(merged, 'dataset')
plt.title('Title Name frequency', fontsize=15)
plt.savefig('graph/bar_Name.png')
plt.show()

'%d' % int(0 if 0 else 1)

#
def bar(data, x, hue, col):
    ax = sns.countplot(x=x, hue=hue, col=col, data=data, palette='husl')                            
    plt.xlabel('')
    plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
    
    for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., 
                    p.get_height(), '%d' % int(p.get_height()), 
                    fontsize=12, ha='center', va='top', color='white')


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



train.to_csv('train_modify.csv', index=False)

