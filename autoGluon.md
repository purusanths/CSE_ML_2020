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
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from autogluon import TabularPrediction as task
```

### <font color='blue'> Read the preprocessed data </font>

```python
data=pd.read_csv('/home/trabeya/cse_ml/data/train_prep.csv')
```

```python
data=data.drop(['meter_waiting_till_pickup'],axis=1)
```

```python
data['fare']=data['fare']-data['additional_fare']
```

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
file_path='../data'
```

```python
data = task.Dataset(file_path=f'{file_path}/train_prep.csv') 
```

```python
data.head()
```

```python
label_column='label'
```

```python
dir = 'model' # specifies folder where to store trained models
predictor = task.fit(train_data=data, label=label_column, output_directory=dir,num_bagging_folds=10)
```

```python
test_data = task.Dataset(file_path=f'{file_path}/test_prep.csv')
```

```python
predict_test = predictor.predict(test_data)

```

```python
submission=pd.DataFrame(predict_test,columns=['prediction'])
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
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_autoGluon_v2.csv',index=False)
```

```python

```
