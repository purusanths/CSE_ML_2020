---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.1
  kernelspec:
    display_name: myenv
    language: python
    name: myenv
---

### <font color='blue'> Import library </font>

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
```

### <font color='blue'> Read the data</font>

```python
data=pd.read_csv('/home/trabeya/cse_ml/data/train.csv')
```

### <font color='blue'> Data understanding and preprocessing</font>

```python
#data.head()
```

```python
data.info()
```

### <font color="blue">From the above descriptive statistic we can see that some attribute have missing values missing values are there </font>


### <font color='blue'> Drop Missing values</font>

```python
data=data.dropna(axis=0,how='any')
```

### <font color='blue'> Data type casting </font>

```python
data['pickup_time'] = pd.to_datetime(data['pickup_time'])
data['drop_time'] = pd.to_datetime(data['drop_time'])
```

```python
data.info()
```

### <font color='blue'> Day of the week annd hour of the day are two importand features for this problem

```python
data['hour_of the_day_pickpup']=data['pickup_time'].dt.hour
data['hour_of the_day_drop']=data['drop_time'].dt.hour
data['day_of_the_week_pickup']=data['pickup_time'].dt.dayofweek
```

```python
data.head()
```

### <font color='blue'> Plotting libraries</font>

```python
%matplotlib inline  
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
#sns.pairplot(data, hue = 'label')
```

### <font color='blue'> Convert label to one and zero </font>

```python
le = preprocessing.LabelEncoder()
```

```python
le.fit(['incorrect','correct'])
```

```python
data['label']=le.transform(data.label)
```

### <font color='blue'> Drop the columns that are  important for the problem</font>

```python
data=data.drop(['drop_time','pickup_time','tripid'],axis=1)
data.reset_index(drop=True, inplace=True)
```

### <font color='blue'> An API to measure the distance </font>

```python
from geopy.distance import distance 
```

```python
data['distance']=0
for i in range(len(data)):
    data['distance'][i]=distance((data['pick_lat'][i],data['pick_lon'][i]),(data['drop_lat'][i],data['drop_lon'][i])).km
# data['distance']=np.abs(data['pick_lat'][i]-data['pick_lon'][i])+np.abs(data['drop_lat'][i]-data['drop_lon'][i])
```

### <font color='blue'> Drop the columns that are  important for the problem</font>

```python
data=data.drop(['pick_lat','pick_lon','drop_lat','drop_lon'],axis=1)
data.reset_index(drop=True, inplace=True)
```

### <font color='blue'> Split the data into train and val set </font>

```python
y=data[['label']].values
```

```python
X=data.drop('label',axis=1).values
```

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
```

```python
X_train.shape,  X_val.shape,  y_train.shape,  y_val.shape
```

### <font color='blue'> Save the clean data so for other notebooks we can use directly </font>

```python
data.to_csv('/home/trabeya/cse_ml/data/train_prep.csv',index=False)
```

### <font color='blue'> XGBClassifier </font>

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score
```

```python
model = XGBClassifier(learning_rate=0.1,n_estimators=10000)
```

```python
model.fit(X_train,y_train)
```

```python
predict_val = model.predict(X_val)
```

```python
accuracy_val = accuracy_score(y_val,predict_val)
print('\naccuracy_score on val dataset : ', accuracy_val)
```

### <font color='blue'> Train on whole training set</font>

```python
model = XGBClassifier(learning_rate=0.1,n_estimators=500)
```

```python
model.fit(X,y)
```

```python
predict_train = model.predict(X)
```

```python
accuracy_train = accuracy_score(y,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)
```

## <font color='blue'> Test set</font>


### <font color='blue'> We have to do all the preprocessing for test set thant we did for the training set</font>

```python
test=pd.read_csv('/home/trabeya/cse_ml/data/test.csv')
```

```python
#test.head()
```

```python
test['pickup_time'] = pd.to_datetime(test['pickup_time'])
test['drop_time'] = pd.to_datetime(test['drop_time'])
```

```python
test['hour_of the_day_pickpup']=test['pickup_time'].dt.hour
test['hour_of the_day_drop']=test['drop_time'].dt.hour
# data['minit_of_hour_pickup']=data['pickup_time'].dt.minute
# data['mini_of_hour_drop']=data['drop_time'].dt.minute
test['day_of_the_week_pickup']=test['pickup_time'].dt.dayofweek
```

```python
data_1=test.drop(['drop_time','pickup_time','tripid'],axis=1)
```

```python
data_1.reset_index(drop=True, inplace=True)
```

```python
data_1['distance']=0
for i in range(len(data_1)):
    data_1['distance'][i]=distance((data_1['pick_lat'][i],data_1['pick_lon'][i]),(data_1['drop_lat'][i],data_1['drop_lon'][i])).km
```

```python
data_1=data_1.drop(['pick_lat','pick_lon','drop_lat','drop_lon'],axis=1)
```

### <font color='blue'> Save the preprocessed test dataset so that we can use if for other notebooks </fon>

```python
data_1.to_csv('/home/trabeya/cse_ml/data/test_prep.csv',index=False)
```

### <font color='blue'> Predict the labels for the test set </font>

```python
predict_test = model.predict(data_1.iloc[:,:].values)
```

```python
submission=pd.DataFrame(predict_test,columns=['prediction'])
```

### <font color="blue" > label encoder conver "correct" as 0 and "incorrect" as 1 but the submission expecting other way </font>

```python
for i in range(len(submission)):
    if submission['prediction'][i]==0:
        submission['prediction'][i]=1
    else:
        submission['prediction'][i]=0
```

```python
submission['tripid']=test.tripid
```

```python
final_submition=submission[['tripid','prediction']]
```

```python
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_xgboost.csv',index=False)
```
