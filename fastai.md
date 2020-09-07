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
from fastai.tabular import * 
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
```

### <font color='blue'> Read the preprocessed data </font>

```python
train=pd.read_csv('/home/trabeya/cse_ml/data/train_split.csv')
```

```python
val=pd.read_csv('/home/trabeya/cse_ml/data/val_split.csv')
```

```python
val.head()
```

```python
val_label=val[['label']]
```

```python
cont_names = [i for i in train.columns]
```

```python
val = TabularList.from_df(val, cont_names=cont_names)
```

```python
data = (TabularList.from_df(train, path='.',cont_names=cont_names,)
                        .split_by_idx(list(range(0,200)))
                        .label_from_df(cols = 'label')
                        .add_test(val)
                        .databunch())
```

```python
learn = tabular_learner(data, layers=[100, 50, 5], metrics=accuracy, emb_drop=0.8, callback_fns=ShowGraph)
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

```python
learn.fit_one_cycle(1, max_lr=slice(1e-03))
```

```python
predictions,_=learn.get_preds(DatasetType.Test)
```

```python
labels = np.argmax(predictions, 1)
```

```python
accuracy_val = accuracy_score(val_label,labels)
print('\naccuracy_score on val dataset : ', accuracy_val)
```

### <font color='blue'> Train on whole training dataset</font>

```python
train=pd.read_csv('/home/trabeya/cse_ml/data/train_prep.csv')
```

```python
train.head()
```

```python
test_prep=pd.read_csv('/home/trabeya/cse_ml/data/test_prep.csv')
```

```python
test_prep.head()
```

```python
test = TabularList.from_df(test_prep,cont_names=cont_names)
```

```python
data = (TabularList.from_df(train, path='.',cont_names=cont_names,)
                        .split_by_idx(list(range(0,200)))
                        .label_from_df(cols = 'label')
                        #.add_test(test)
                        .databunch())
```

```python
learn = tabular_learner(data, layers=[100, 50, 5], metrics=accuracy, emb_drop=0.8, callback_fns=ShowGraph)
```

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

```python
learn.fit_one_cycle(5, max_lr=slice(1e-04))
```

```python
predictions,_=learn.get_preds(test)
```

```python
predictions=predictions[-8576:]
```

```python
test_labels = np.argmax(predictions, 1)
```

```python
len(test_labels)
```

```python
submission=pd.DataFrame(test_labels,columns=['prediction'])
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
submission['tripid']=test[['tripid']]
```

```python
final_submition=submission[['tripid','prediction']]
```

```python
final_submition.to_csv('/home/trabeya/cse_ml/data/final_submition_fastai_v2.csv',index=False)
```
