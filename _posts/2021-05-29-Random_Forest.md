---
layout: post
title:  "[ML] 랜덤 포레스트 구현 및 비교"
---

# [ML] 랜덤 포레스트 구현 후 사이킷런의 랜덤 포레스트와 비교하기


**랜덤 포레스트를 구현하기 위해**
1.   데이터셋 불러오기
2.   다수의 결정트리 집계
3.   랜덤 포레스트 구현
4.   구현된 랜덤 포레스트와 사이킷런의 랜덤 포레스트 비교



**공통 모듈 임포트하기**


```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import os
```



### **데이터 불러오기**

**`make_moons`를 사용해 데이터셋을 생성**

`n_samples = 2000, noise = 0.5` 옵션을 주었습니다.

동일한 결과를 출력하기 위해 `random_state = 42`으로 고정합니다.


```python
from sklearn.datasets import make_moons #moons 데이터를 불러오기 위한 임포트
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples = 2000, noise = 0.5, random_state=42)
```

**이를 `train_test_split()`을 사용하여 훈련 세트와 테스트 세트로 나눕니다.**

테스트 세트의 크기 30%로 설정


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```



### 다수의 결정트리 집계

**결정트리 분류기의 최적의 매개변수를 찾기위해 교차 검증과 그리드 탐색을 수행합니다.**


```python
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
```



    [Out]
    Fitting 3 folds for each of 294 candidates, totalling 882 fits
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 882 out of 882 | elapsed:    2.1s finished
    GridSearchCV(cv=3, error_score=nan,
                 estimator=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort='deprecated',
                                                  random_state=42,
                                                  splitter='best'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)



그리드 탐색을 통해 찾은 최적의 매개변수는

`max_leaf_nodes = 4`

`min_samples_leaf = 1`

`min_samples_split = 2`

으로 나타났고 최적의 매개변수는 `make_moons` 데이터셋의 표본 크기나 noise에 따라 달라지며 모델의 성능에도 영향을 미치는 것으로 나타납니다.


```python
grid_search_cv.best_estimator_
```




    [Out]
    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=4,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=42, splitter='best')



찾은 매개변수를 사용하여 전체 훈련 세트에 대해 모델을 훈련시킨 후 테스트 세트에서 성능을 측정한 결과

모델의 정확도가 0.815로 81.5%의 성능을 보여주었습니다.


```python
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
```




    [Out]
    0.815

모델의 정확도를 높이기 위해서는 데이터 세트의 `n_samples`데이터의 `noise`가 낮아야 하며, `noise`가 상대적으로 높은 비율을 차지할 때, `n_samples`데이터가 많으면 많을수록 정확도가 떨어지는 것으로 관측됩니다.



### 랜덤 포레스트 구현
**2000개 서브셋을 생성하여 100개의 샘플을 무작위로 array에 담습니다.**

`ShuffleSplit`을 사용하여 무작위 샘플 생성 


```python
from sklearn.model_selection import ShuffleSplit

n_trees = 2000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```

이전 그리드 탐색을 통해 찾은 최적의 매개변수를 사용하여 각 서브셋에 결정트리를 훈련시킵니다.

테스트 세트로 2000개의 결정트리를 `sklearn`의 `clone`을 사용하여 동일한 파라미터를 가지고 새로운 추정치를 만들어 평가합니다.

결과는 0.782965로 약 78%의 결과가 나타났으며, 이는 더 작은 데이터셋에서 훈련되었기 때문에 앞서 만든 결정 트리보다 성능이 떨어진 것으로 나타납니다.



```python
from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)
```




    [Out]
    0.782965



이제 각 테스트 세트 샘플에 대하여 2000개의 결정트리 예측을 만들고 다수로 나온 예측만을 취합니다.

`scipy`의 `mode` 함수를 사용하여 테스트 세트에 대한 다수결 예측을 만들 수 있습니다.


```python
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)
```


```python
from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
```



마지막으로 테스트 세트에 대한 예측을 평가합니다.
결과는 0.8233...4로 약 82%의 결과가 나타났습니다.

앞서 만든 결정트리의 결과인 78%보다 4%정도 높은 정확도를 얻게 되었습니다.


```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
```




    [Out]
    0.8233333333333334



### 구현된 랜덤 포레스트와 사이킷런의 랜덤 포레스트 비교

`sklearn.ensemble`의 `RandomForestClassifier`을 이용하여 튜닝된 훈련 세트를 랜덤 포레스트 예측 분류 모델을 생성하고 성능을 측정합니다.

구현한 랜덤 포레스트의 매개변수와 동일한 값으로 설정하여 정확도를 비교합니다.

`n_estimators = 2000` : 결정트리 2000개

`max_leaf_nodes = 4` : 리프 노드의 최대수 4


```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators = 2000, max_leaf_nodes = 4)
rnd_clf.fit(X_train, y_train)
```




    [Out]
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=4, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=2000,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
y_pred_rf = rnd_clf.predict(X_test)
```


```python
rnd_clf.score(X_test, y_test)
```




    [Out]
    0.82

직접 구현한 랜덤 포레스트의 정확도는 0.82333..4이며, 사이킷런의 랜덤 포레스트 분류 모델의 정확도는 0.82로 정확도의 차이가 거의 없는 결과를 나타냅니다.