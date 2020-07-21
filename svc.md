---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

### <font color="blue"> Import libraries </font>

```python
from sklearn.svm import SVC
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

### <font color="blue"> Split the data into train val and test set</font>

```python
y=data[['label']].values
```

```python
X=data.drop('label',axis=1).values
```

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
```

### <font color='blue'> Intansciate the SVC classifiers </font>

```python
model =SVC(kernel='rbf')
```

### <font color='blue'> Fit the SVC classifier </font>

```python
model.fit(X_train,y_train )
```

### <font color='blue'> Predicting the labels for validation set </font>

```python
val_predictions = model.predict(X_val)
```

```python
accuracy_val = accuracy_score(y_val,val_predictions)
print('\naccuracy_score on validation dataset : ', accuracy_val)
```

### <font color='blue'> Training the data with full data </font>

```python
test_prep=pd.read_csv('/home/trabeya/cse_ml/data/test_prep.csv')
```

```python
model =SVC(kernel='rbf')
```

```python
model.fit(X,y)
```

### <font color='blue'> Predict the label for test set</font>

```python
test_predictions = model.predict(test_prep)
```

```python
submission=pd.DataFrame(test_predictions,columns=['prediction'])
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
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_random_svs.csv',index=False)
```
