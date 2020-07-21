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

### <font color='blue'> Import the libraries </font>

```python
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
```

### <font color='blue'> Read the preprocessed data </font>

```python
data=pd.read_csv('/home/trabeya/cse_ml/data/train_prep.csv')
```

### <font color="blue"> Split the data into train set and val set</font>

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
```

```python
train_split=pd.DataFrame(X_train)
```

```python
train_split['label']=y_train
```

```python
train_split.columns=['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
       'meter_waiting_till_pickup', 'fare', 'hour_of the_day_pickpup',
       'hour_of the_day_drop', 'day_of_the_week_pickup', 'distance','label']
```

```python
val_split=pd.DataFrame(X_val)
```

```python
val_split['label']=y_val
```

```python
val_split.columns=['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare',
       'meter_waiting_till_pickup', 'fare', 'hour_of the_day_pickpup',
       'hour_of the_day_drop', 'day_of_the_week_pickup', 'distance','label']
```

```python
h2o.init()
```

```python
traindf=h2o.H2OFrame(train_split)
```

```python
valdf=h2o.H2OFrame(val_split)
```

```python
y='label'
```

```python
x=list(traindf.columns)
```

```python
x.remove(y)
```

```python
traindf[y]=traindf[y].asfactor()
```

```python
valdf[y]=valdf[y].asfactor()
```

```python
aml=H2OAutoML(max_runtime_secs=600)
```

```python
aml.train(x=x,y=y,training_frame=traindf)
```

```python
predict_val=aml.predict(valdf)
```

```python
predict_val=predict_val.as_data_frame()
```

```python
accuracy_val = accuracy_score(y_val,predict_val.iloc[:,0])
print('\naccuracy_score on validation dataset : ', accuracy_val)
```

```python
h2o.shutdown(prompt=False)
```

### <font color='blue' > Train on whole training data</font>

```python
h2o.init()
```

```python
train=pd.read_csv('/home/trabeya/cse_ml/data/train_prep.csv')
```

```python
test=pd.read_csv('/home/trabeya/cse_ml/data/test_prep.csv')
```

```python
train=h2o.H2OFrame(train.iloc[:,:])
```

```python
test=h2o.H2OFrame(test.iloc[:,:])
```

```python
train[y]=train[y].asfactor()
```

```python
aml=H2OAutoML(max_runtime_secs=180)
```

```python
aml.train(x=x,y=y,training_frame=train)
```

```python
predict_test=aml.predict(test)
```

```python
predict_test=predict_test.as_data_frame()
```

```python
submission=predict_test[['predict']]
```

```python
submission.columns=['prediction']
```

```python
submission.shape
```

```python
h2o.shutdown(prompt=False)
```

```python
for i in range(len(submission)):
    if submission['prediction'][i]==0:
        submission['prediction'][i]=1
    else:
        submission['prediction'][i]=0
```

```python
test=pd.read_csv('/home/trabeya/cse_ml/data/test.csv')
```

```python
submission['tripid']=test.tripid
```

```python
final_submition=submission[['tripid','prediction']]
```

```python
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_h2o.csv',index=False)
```
