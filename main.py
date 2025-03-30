import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import random
# 处理空值
data=pd.read_csv('train.csv')
missing_values = data.isnull().sum()
missing_values=missing_values[missing_values>0]
data.dropna(inplace=True,subset=['Embarked'],axis=0)
age_int=[int(i)  if pd.notna(i) else -1 for i in data['Age']]
# 删除无意义值
data.drop(['Cabin'],axis=1,inplace=True)
data.drop('Ticket',axis=1,inplace=True)
data.drop('Name',axis=1,inplace=True)

for i in range(len(age_int)):
    if age_int[i]==-1:
        age_int[i]=random.randint(30,42)
    # 五岁为一个分界点
    age_int[i]=age_int[i]//5*5
sex=[1 if i=='male' else 0 for i in data['Sex']]
embarked=[1 if i=='S' else 0 for i in data['Embarked']]
fare=[int(i) for i in data['Fare']]
data.loc[:,'Age']=age_int
data.loc[:,'Sex']=sex
data.loc[:,'Embarked']=embarked
data.loc[:,'Fare']=fare
X=[]
y=[]


for i in range(len(data)):
    X.append([
        data.iloc[i]['Pclass'],  # 使用 .loc 或 .iloc 访问具体行的值
        data.iloc[i]['Age'],
        data.iloc[i]['SibSp'],
        data.iloc[i]['Parch'],
        data.iloc[i]['Sex'],
        data.iloc[i]['Embarked'],
        data.iloc[i]['Fare']
    ])    
    y.append(data.iloc[i]['Survived'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
models = {
    'Logistic Regression': LogisticRegression(penalty='l2', C=0.5, random_state=1, solver='lbfgs', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=1),
    'Random Forest': RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=6, random_state=1),
    'Support Vector Machine': SVC(kernel='linear', C=1.0, random_state=1)
}

# 评估模型
results = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_std, y_train, cv=5, scoring='accuracy')
    results[model_name] = scores.mean()

# 打印结果
for model_name, accuracy in results.items():
    print(f'{model_name}: {accuracy:.2f}')

# 选择最佳模型
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_model.fit(X_train_std, y_train)
y_pred = best_model.predict(X_test_std)

# 输出最佳模型的准确率
print(f'Best Model: {best_model_name}')
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')