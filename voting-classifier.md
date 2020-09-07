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

### <font color='blue'> Import library</font>

```python
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
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

### <font color='blue'> Intansciate the classifiers </font>

```python
logistis_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(n_estimators=350)
svm_clf = SVC(kernel='rbf')
knn = KNeighborsClassifier(n_neighbors=5)
```

### <font color='blue'> Intansciate the Voting classifier </font>

```python
voting_classifier = VotingClassifier(
estimators=[('lr', logistis_clf), ('rf', rnd_clf), ('svc', svm_clf),('knn',knn)],
voting='hard'
)
```

### <font color="blue"> Fit the voting classifier </font>

```python
voting_classifier.fit(X_train,y_train )
```

### <font color='blue'> Predicting the labels for validation set </font>

```python
val_predictions = voting_classifier.predict(X_val)
```

```python
accuracy_val = accuracy_score(y_val,val_predictions)
print('\naccuracy_score on validation dataset : ', accuracy_val)
```

### <font color='blue'> Training the data with full data </font>

```python
voting_classifier = VotingClassifier(
estimators=[('lr', logistis_clf), ('rf', rnd_clf), ('svc', svm_clf),('knn',knn)],
voting='hard'
)
```

```python
voting_classifier.fit(X,y)
```

### <font color='blue'> Read the preprocessed test data</font>

```python
test_prep=pd.read_csv('/home/trabeya/cse_ml/data/test_prep.csv')
```

### <font color='blue'> Predict the label for test set</font>

```python
test_predictions = voting_classifier.predict(test_prep)
```

```python
submission=pd.DataFrame(test_predictions,columns=['prediction'])
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
test=pd.read_csv('/home/trabeya/cse_ml/data/test.csv')
```

```python
submission['tripid']=test.tripid
```

```python
final_submition=submission[['tripid','prediction']]
```

```python
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_voting.csv',index=False)
```
