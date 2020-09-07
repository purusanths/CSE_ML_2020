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

### <font color='blue'> Import the libraries </font>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
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
accuracy=[]
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_val)
    accuracy.append(accuracy_score(y_val,pred_i))
```

```python
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('K Value vs Accuracy')
plt.xlabel('K Value')
plt.ylabel('Acccuracy')
```

### <font color='blue'> K=5 is the bes choice for K </font>


### <font color='blue'> Training the data with full data </font>

```python
test_prep=pd.read_csv('/home/trabeya/cse_ml/data/test_prep.csv')
```

```python
classifier = KNeighborsClassifier(n_neighbors=5)
```

```python
classifier.fit(X,y)
```

```python
test_predictions = classifier.predict(test_prep)
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
submission.head()
```

```python
test=pd.read_csv('/home/trabeya/cse_ml/data/test.csv')
```

```python
test.head()
```

```python
submission['tripid']=test.tripid
```

```python
final_submition=submission[['tripid','prediction']]
```

```python
final_submition.head()
```

```python
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_knn.csv',index=False)
```
