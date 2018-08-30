
# Import library and load data
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_rows = 999

train = pd.read_csv('train.csv', index_col=['PassengerId'])
test = pd.read_csv('test.csv', index_col=['PassengerId'])


# Explor the data
print('train shape : ', train.shape)
print('test shape : ', test.shape)

print('train columns : ', train.columns)
print('test columns : ', test.columns)

print('train head : \n', train.head(10))
print('test head : \n', test.head(10))

print('train info : \n', train.info())
print('test info: \n', test.info())

print('train missing values : \n', train.isnull().sum())
print('test missing values : \n', test.isnull().sum())

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
# Merge train data and test data
train['dataset'] = 'train set'
test['dataset'] = 'test set'

merged = pd.concat([train, test])

print('merged data shape : ', merged.shape)
print('merged data columns : ', merged.columns)
print('merged data head : \n', merged.head())
print('merged data tail : \n', merged.tail())


# -----------------------------------------------------------------------------
# Define graph functions
def count_bar(data, x, hue):
    plt.figure(figsize=(10,7))
    ax = sns.countplot(x=x, hue=hue, data=data, palette='husl')
    plt.title(x + ' Frequency', fontsize=15)
    plt.legend(loc='upper right').set_title(hue)
    
    for p in ax.patches:
        a = p.get_height()
        va = 'top'
        color = 'white'
        
        if np.isnan(a):
            a = 0
            
        if a < 30:
            va = 'bottom'
            color = '#C39BD3'
            
        ax.text(p.get_x() + p.get_width()/2., a, '%d' % int(a), 
                fontsize=12, ha='center', va=va, color=color)
        
def countplot(x, hue, **kwargs):
    sns.countplot(x=x, hue=hue, palette='husl')
    
def bar(data, x, hue, col):
    g =  sns.FacetGrid(data, col=col, size=7)
    g = g.map(countplot, x, hue)
    
def box(data, x, y, hue=None):
    plt.figure(figsize=(10,7))
    sns.boxplot(x=x, y=y, hue=hue, data=data, palette='husl')
    
def hist(data, a, hue, col=None):
    g =  sns.FacetGrid(data, hue=hue, col=col, size=7, palette='husl')
    g = g.map(sns.distplot, a, bins=10, hist_kws={'alpha':0.2})


# -----------------------------------------------------------------------------
# Change the string value to an integer value
# Sex
merged.loc[merged['Sex'] == 'male', 'Sex'] = 0
merged.loc[merged['Sex'] == 'female', 'Sex'] = 1

print(merged['Sex'].head())


# Embarked
merged.loc[merged['Embarked'] == 'C', 'Embarked'] = 0
merged.loc[merged['Embarked'] == 'Q', 'Embarked'] = 1
merged.loc[merged['Embarked'] == 'S', 'Embarked'] = 2

print(merged['Embarked'].head())


# -----------------------------------------------------------------------------
# Draw bar graphs
# Pclass
count_bar(merged, 'Pclass', 'dataset')
plt.xticks([0,1,2], ('1st', '2nd', '3rd'))
plt.savefig('graph/bar_Pclass.png')
plt.show()

count_bar(merged, 'Pclass', 'Survived')
plt.xticks([0,1,2], ('1st', '2nd', '3rd'))
plt.legend(['Deceased','Survived']).set_title('Survived')
plt.savefig('graph/bar_Pclass_Survivd.png')
plt.show()


# Sex
count_bar(merged, 'Sex', 'dataset')
plt.xticks([0,1], ('male', 'female'))
plt.savefig('graph/bar_Sex.png')
plt.show()

count_bar(merged, 'Sex', 'Survived')
plt.xticks([0,1], ('male', 'female'))
plt.legend(['Deceased','Survived']).set_title('Survived')
plt.savefig('graph/bar_Sex_Survived.png')
plt.show()


# Embarked
count_bar(merged, 'Embarked', 'dataset')
plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
plt.savefig('graph/bar_Embarked.png')
plt.show()

count_bar(merged, 'Embarked', 'Survived')
plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
plt.legend(['Deceased','Survived']).set_title('Survived')
plt.savefig('graph/bar_Embarked_Survived.png')
plt.show()


# SibSp
count_bar(merged, 'SibSp', 'dataset')
plt.savefig('graph/bar_SibSp.png')
plt.show()

count_bar(merged, 'SibSp', 'Survived')
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/bar_SibSp_Survived.png')
plt.show()


# Parch
count_bar(merged, 'Parch', 'dataset')
plt.savefig('graph/bar_Parch.png')
plt.show()

count_bar(merged, 'Parch', 'Survived')
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/bar_Parch_Survived.png')
plt.show()


# ----------------------------------------------------------------------------- 
# Makes a feature : family
merged['family'] = merged['SibSp'] + merged['Parch'] + 1
print(merged[['SibSp','Parch','family']].head())

count_bar(merged, 'family', 'dataset')
plt.savefig('graph/bar_family.png')
plt.show()

count_bar(merged, 'family', 'Survived')
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/bar_family_Survived.png')
plt.show()


# Split family size
merged['family_class'] = np.nan
merged.loc[merged['family'] == 1, 'family_class'] = 1
merged.loc[merged['family'] > 1, 'family_class'] = 2
merged.loc[merged['family'] > 4, 'family_class'] = 3

count_bar(merged, 'family_class', 'dataset')
plt.savefig('graph/bar_family_class.png')
plt.show()

count_bar(merged, 'family_class', 'Survived')
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/bar_family_class_Survived.png')
plt.show()


# -----------------------------------------------------------------------------
# Draw bar graphs per Survived       
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

bar(merged, 'family_class', 'Sex', 'Survived')
plt.legend(('male', 'female')).set_title('Sex')
plt.savefig('graph/bar_family_class_Sex_Survived.png')
plt.show()


# -----------------------------------------------------------------------------
# Extract the title name
def title_name(name):
    n = re.findall(', .{1,15}\.', name)
    return ' '.join(n)[2:]

merged['title_name'] = np.nan
merged['title_name'] = merged['Name'].apply(lambda x: title_name(x))

print(merged[['Name', 'title_name']].head())
print(merged[['Name', 'title_name']].tail())


# Draw title name bar graphs
def Name_bar(data, x, hue):
    plt.figure(figsize=(15,7))
    ax = sns.countplot(x=x, hue=hue, data=data, palette='husl')                            
    plt.xlabel('')
    plt.xticks(rotation=30)
    plt.legend(loc='upper right').set_title(hue)
    plt.title('Title Name frequency', fontsize=15)
    
    for p in ax.patches:
        a = p.get_height()
        
        if np.isnan(a):
            a = 0
            
        ax.text(p.get_x() + p.get_width()/2., 
                a, '%d' % int(a), 
                fontsize=12, ha='center', va='bottom', color='#C39BD3')    
    

Name_bar(merged, 'title_name', 'dataset')
plt.savefig('graph/bar_Name.png')
plt.show()

Name_bar(merged, 'title_name', 'Survived')
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/bar_Name_Survived.png')
plt.show()


#
title_name_map = {'Mr.':0, 'Mrs.':1, 'Miss.':2, 'Master.':3, 'Don.':0, 'Rev.':0,
                  'Dr.':4, 'Mme.':1, 'Ms.':1, 'Major.':0, 'Lady.':1, 'Sir.':0,
                  'Mlle.':2, 'Col.':0, 'Capt.':0, 'the Countess.':1, 
                  'Jonkheer.':0, 'Dona.':1}

merged['title_name_map'] = merged['title_name'].map(title_name_map)
print(merged[['title_name','title_name_map']].head())
print(merged[['title_name','title_name_map']].tail())

Name_bar(merged, 'title_name_map', 'dataset')
plt.xticks((0,1,2,3,4), ('Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.'))
plt.savefig('graph/bar_Name_map.png')
plt.show()

Name_bar(merged, 'title_name_map', 'Sex')
plt.xticks((0,1,2,3,4), ('Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.'))
plt.savefig('graph/bar_Name_map_Sex.png')
plt.show()

Name_bar(merged, 'title_name_map', 'Survived')
plt.xticks((0,1,2,3,4), ('Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.'))
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/bar_Name_map_Survived.png')
plt.show()


# -----------------------------------------------------------------------------
# Fill the missing value of Embarked
print(merged.loc[merged['Embarked'].isnull()])
print(merged.loc[merged['Fare'] == 80, 'Embarked'])
merged.loc[merged['Pclass'] == 1, 'Embarked']

merged.loc[merged['Embarked'].isnull(), 'Embarked'] = 2


# Do one-hot encoding
# Pclass
temp = pd.get_dummies(merged.Pclass)
temp.columns = ['P_1st', 'P_2rd', 'P_3rd']
merged = pd.concat([merged, temp], axis=1)


# Embarked
temp = pd.get_dummies(merged.Embarked)
temp.columns = ['E_Cherbourg', 'E_Queenstown', 'E_Southampton']
merged = pd.concat([merged, temp], axis=1)


# title_name_map
temp = pd.get_dummies(merged.title_name_map)
temp.columns = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr']
merged = pd.concat([merged, temp], axis=1)


# -----------------------------------------------------------------------------
#
box(merged, 'dataset', 'Age')
plt.savefig('graph/box_Age.png')
plt.show()
    
box(merged, 'Sex', 'Age')
plt.xticks((0,1),('male','female'))
plt.savefig('graph/box_Age_Sex.png')
plt.show()

box(merged, 'title_name_map', 'Age')
plt.xticks((0,1,2,3,4),('Mr', 'Mrs', 'Miss', 'Master', 'Dr'))
plt.savefig('graph/box_Age_titme_name_map.png')
plt.show()

box(merged, 'title_name', 'Age')
plt.savefig('graph/box_Age_titme_name.png')
plt.show()


# -----------------------------------------------------------------------------
# Fill 
def age_loss(temp):
    loss = merged['Age'] - temp
    loss = np.power(loss, 2)
    loss = np.sum(loss) / len(merged.loc[merged['Age'].notnull()])
    loss = np.sqrt(loss)
    return loss
    
temp = merged.groupby(['title_name'])['Age'].median()
print(temp)

temp = merged.groupby(['title_name_map'])['Age'].median()
print(temp)

temp1 = merged.groupby(['title_name'])['Age'].transform('median')
temp2 = merged.groupby(['title_name_map'])['Age'].transform('median')


loss1 = age_loss(temp1)
loss2 = age_loss(temp2)

print(loss1, loss2)

merged['Age_fill1'] = merged['Age'].fillna(temp1)
merged['Age_fill2'] = merged['Age'].fillna(temp2)

temp1 = merged.groupby(['title_name'])['Age'].median()
temp2 = merged.groupby(['title_name'])['Age_fill1'].median()
temp3 = merged.groupby(['title_name'])['Age_fill2'].median()
temp = pd.concat([temp1,temp2,temp3], axis=1)
print(temp)

temp1 = merged.groupby(['title_name_map'])['Age'].median()
temp2 = merged.groupby(['title_name_map'])['Age_fill1'].median()
temp3 = merged.groupby(['title_name_map'])['Age_fill2'].median()
temp = pd.concat([temp1,temp2,temp3], axis=1)
print(temp)


temp = merged[['Age_fill1', 'Age_fill2']].mean(axis=1)
loss1 = age_loss(temp)


merged['Age_fill'] = merged['Age'].fillna(temp)



#
box(merged, 'dataset', 'Age_fill')
plt.savefig('graph/box_Age_fill.png')
plt.show()
    
box(merged, 'Sex', 'Age_fill')
plt.xticks((0,1),('male','female'))
plt.savefig('graph/box_Age_fill_Sex.png')
plt.show()

box(merged, 'title_name_map', 'Age_fill')
plt.xticks((0,1,2,3,4),('Mr', 'Mrs', 'Miss', 'Master', 'Dr'))
plt.savefig('graph/box_Age_fill_titme_name_map.png')
plt.show()

box(merged, 'title_name', 'Age_fill')
plt.savefig('graph/box_Age_fill_titme_name.png')
plt.show()



    
hist(merged, 'Age', 'dataset')
plt.legend(['train','test']).set_title('dataset')
plt.ylim((0, 0.05))
plt.savefig('graph/hist_Age.png')
plt.show()

hist(merged, 'Age_fill1', 'dataset')
plt.legend(['train','test']).set_title('dataset')
plt.savefig('graph/hist_Age_fill1.png')
plt.show()

hist(merged, 'Age_fill2', 'dataset')
plt.legend(['train','test']).set_title('dataset')
plt.savefig('graph/hist_Age_fill2.png')
plt.show()

hist(merged, 'Age_fill', 'dataset')
plt.legend(['train','test']).set_title('dataset')
plt.savefig('graph/hist_Age_fill.png')
plt.show()

hist(merged, 'Age_fill', 'Survived')
plt.legend(['Deceased','Survived'], loc=1).set_title('Survived')
plt.savefig('graph/hist_Age_fill_Survived.png')
plt.show()

hist(merged, 'Age_fill', 'Sex', 'Survived')
plt.legend(['male','female']).set_title('Sex')
plt.savefig('graph/hist_Age_fill_Sex.png')
plt.show()


#
box(merged, 'Survived', 'Age_fill')
plt.savefig('graph/box_Age_fill_Survived.png')
plt.show()

box(merged, 'Sex', 'Age_fill', 'Survived')
plt.xticks((0,1),('male','female'))
plt.legend(('Deceased','Survived')).set_title('Survived')
plt.savefig('graph/box_Age_fill_Sex_Survived.png')
plt.show()


# -----------------------------------------------------------------------------
#
box(merged, 'dataset', 'Fare')
plt.savefig('graph/box_Fare.png')
plt.show()

box(merged, 'Survived', 'Fare')
plt.savefig('graph/box_Fare_Survived.png')
plt.show()

box(merged, 'Pclass', 'Fare')
plt.xticks((0,1,2), ('1st', '2nd', '3rd'))
plt.savefig('graph/box_Fare_Pclass.png')
plt.show()

box(merged, 'Embarked', 'Fare')
plt.xticks([0,1,2], ('Cherbourg', 'Queenstown', 'Southampton'))
plt.savefig('graph/box_Fare_Embarked.png')
plt.show()

hist(merged, 'Fare', 'dataset')
plt.legend(['train','test']).set_title('dataset')
plt.savefig('graph/hist_Fare.png')
plt.show()


#
merged.loc[merged['Fare'].isnull()]


# -----------------------------------------------------------------------------
#
merged.columns

train = merged.iloc[:891]
train.to_csv('train_modify.csv')

test = merged.iloc[891:]
test.to_csv('test_modify.csv')

merged.to_csv('merged_modify.csv')


# -----------------------------------------------------------------------------
#
#
def cabin(value):
    return value[0]

merged['Cabin_value'] = np.nan
merged.loc[merged['Cabin'].notnull(),'Cabin_value'] = merged.loc[merged['Cabin'].notnull(),'Cabin'].apply(lambda x: cabin(x))

merged.groupby(['Cabin_value', 'Pclass', 'Fare'])['Age'].mean()
merged.groupby(['Cabin_value', 'Pclass'])['Fare'].count()
merged.groupby(['Pclass','Cabin_value'])['Cabin_value'].count()
merged.groupby(['Pclass','Cabin_value', 'Ticket'])['Ticket'].count()
temp = merged['Ticket'].value_counts()
temp.head()


# -----------------------------------------------------------------------------
#
def ticket_str(name):
    n = re.findall('[a-zA-Z]', name)
    return ' '.join(n)

merged[['Ticket', 'Cabin','Fare', 'Pclass']]
merged[merged['Ticket'] == '19950']

merged.groupby(['Ticket', 'Cabin', 'Pclass'])['Fare'].count()

merged['ticket_str'] = np.nan
merged['ticket_str'] = merged['Ticket'].apply(lambda x: ticket_str(x))
merged['ticket_str'].value_counts()


#
def ticket_int(name):
    n = re.findall('[0-9]+', name)
    return n[-1]

merged['ticket_int'] = np.nan
merged['ticket_int'] = merged['Ticket'].apply(lambda x: ticket_int(x))
merged['ticket_int'].value_counts()
merged['Ticket']


# -----------------------------------------------------------------------------
#
corr = train.corr(method='pearson')
corr = corr.round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(15,15))
sns.heatmap(corr, vmin=-1, vmax=1,
            mask=mask, cmap=cmap, annot=True, linewidth=.5, cbar_kws={'shrink':.6})
plt.savefig('graph/heatmap.png')
plt.show()

corr = train.corr(method='spearman')
corr = corr.round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(15,15))
sns.heatmap(corr, vmin=-1, vmax=1,
            mask=mask, cmap=cmap, annot=True, linewidth=.5, cbar_kws={'shrink':.6})
plt.savefig('graph/heatmap_spear.png')
plt.show()