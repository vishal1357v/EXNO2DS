# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
         from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/DATA/

from ast import increment_lineno
import cufflinks as cf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

titan = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/titanic_dataset.csv')

titan.head()

titan.isnull()

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titan,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titan,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=titan,palette='rainbow')

sns.displot(titan['Age'].dropna(),kde=False,color='darkred',bins=40)

titan['Age'].hist(bins=30,alpha=0.3)

sns.countplot(x='SibSp',data=titan)

titan['Fare'].hist()

titan['Fare'].iplot()

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=titan,palette='winter')

def impute_age(cols):
  Age=cols[0]
  Pclass=cols[1]
  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass == 2:
      return 29
    else:
      return 24
  else:
    return Age

titan['Age'] = titan[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(titan.isnull(),yticklabels=False,cbar=False,cmap='viridis')

titan.drop('Cabin',axis=1,inplace=True)

titan.head()

titan.dropna(inplace=True)

titan.info()

pd.get_dummies(titan['Embarked'],drop_first=True).head()

sex=pd.get_dummies(titan['Sex'],drop_first=True)
embark=pd.get_dummies(titan['Embarked'],drop_first=True)

titan.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

titan.head()

titan=pd.concat([titan,sex,embark],axis=1)

titan.head()

titan.drop('Survived',axis=1).head()

titan['Survived'].head()

from sklearn.model_selection import train_test_split

X_titan,X_test,Y_titan,Y_test = train_test_split(titan.drop('Survived',axis=1),titan['Survived'],test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(X_titan,Y_titan)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(Y_test,predictions)

accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,predictions)
accuracy

predictions

output:
![image](https://github.com/user-attachments/assets/aa6bcd0f-e9bc-4bb4-b2ca-cf34d1b55505)
![image](https://github.com/user-attachments/assets/26729081-731f-4a73-b322-d4f6158d47d5)
![image](https://github.com/user-attachments/assets/9c195273-98bd-4a39-b944-f8411c16b577)
![image](https://github.com/user-attachments/assets/c73076fb-5f1c-4ca6-a073-e7ac34350406)
![image](https://github.com/user-attachments/assets/245f198d-cc4b-438d-a395-e524f99eda6b)
![image](https://github.com/user-attachments/assets/b96deae4-fad1-4044-8a47-2de30d975b06)
![image](https://github.com/user-attachments/assets/e8575149-2ec7-4701-b93c-6382b3459d98)
![image](https://github.com/user-attachments/assets/1dbd8118-d66e-483e-8682-1cbae6616ab7)
![image](https://github.com/user-attachments/assets/57b54a66-d482-43d2-99fd-31b244dfab81)
![image](https://github.com/user-attachments/assets/b84ea0e8-ac37-412e-8b62-341e45ba8bf5)
![image](https://github.com/user-attachments/assets/762f15e6-c38c-4be7-a96b-e63f84a97682)
![image](https://github.com/user-attachments/assets/f20995d0-d29a-43e2-8fdf-797a7d15e9af)
![image](https://github.com/user-attachments/assets/cc721e38-7263-4276-8dd1-a436abbb22b5)
![image](https://github.com/user-attachments/assets/beac101e-8d9f-4afb-85b9-2e7baa8f4822)
![image](https://github.com/user-attachments/assets/4baba2c3-a326-4c45-9457-bddba1533fbd)
![image](https://github.com/user-attachments/assets/e9ab40bf-6c8f-4eeb-b7f4-b1bfca814463)
![image](https://github.com/user-attachments/assets/b678045a-94ad-4b6f-8a91-f25085962b55)
![image](https://github.com/user-attachments/assets/7fdad3f7-2070-4eb8-a9e2-111af9c5d096)
![image](https://github.com/user-attachments/assets/ae90dbac-7fb9-4758-a118-285479c490bb)
![image](https://github.com/user-attachments/assets/de561e85-b6ef-464e-8b39-278a5c804924)
![image](https://github.com/user-attachments/assets/204e6b12-fd85-4f36-8b18-1dc82dc9e426)
![image](https://github.com/user-attachments/assets/f434a3bb-b8b8-4054-a5b9-f21e370d87bc)
![image](https://github.com/user-attachments/assets/2ad65c2b-4ddc-44cc-b3be-5c74d0f17d50)



# RESULT
Data analysis was finished
        
