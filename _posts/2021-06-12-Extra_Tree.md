---
layout: post
title:  "[ML] 엑스트라 트리 직접 구현 및 랜덤 포레스트 모델과 비교하기"
---

# [ML] 엑스트라 트리 직접 구현하기

**엑스트라 트리를 구현하기 위해**
1.   데이터셋 불러오기
2.   결정트리를 통한 무작위 분할
3.   그리드 탐색을 통한 최적의 선택
4.   엑스트라 트리 구현
5.   지난번 구현한 랜덤 포레스트와 성능 비교





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

지난번 구현한 랜덤 포레스트와 성능 비교를 위해 지난 랜덤 포레스트의 `make_moons`데이터 옵션을 동일하게 주엇습니다.


```python
from sklearn.datasets import make_moons #moons 데이터를 불러오기 위한 임포트

X, y = make_moons(n_samples = 2000, noise = 0.5, random_state=42)
```



**분할 역시 `train_test_split()`을 사용하여 훈련 세트와 테스트 세트로 나눕니다.**

테스트 세트의 크기 30%로 설정


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
```



### 결정트리를 통한 무작위 분할

**엑스트라 트리 구현을 위해 결정트리의 분할 옵션 `splitter='random'`옵션을 부여합니다.**


```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(splitter='random')
dtc
```


    [Out]
    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='random')



### 그리드 탐색을 통한 최적의 선택

**결정트리에서 무작위 분할을 실시한 후 `max_leaf_nodes` 및 `min_samples_split` 옵션을 지난 랜덤포레스트 구현과 같이 하여 최적값을 찾기위해 그리드 탐색을 수행합니다..**


```python
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(dtc, params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)
```

    [Out]
    Fitting 3 folds for each of 294 candidates, totalling 882 fits
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 882 out of 882 | elapsed:    1.1s finished
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
                                                  random_state=None,
                                                  splitter='random'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)



그리드 탐색을 통해 찾은 최적의 파라미터는

`max_leaf_nodes = 42`

`min_samples_leaf = 1`

`min_samples_leaf = 2`

으로 나타났고 이는 결정트리를 수행할 때마다 계속해서 변동하며, 최종 성능에도 영향을 끼칩니다.


```python
grid_search_cv.best_estimator_
```


    [Out]
    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=42,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=3,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='random')



그리드 탐색을 통해 찾은 매개변수를 사용하여 훈련 세트에 대해 모델을 훈련시킵니다.

이후 테스트 세트에서 성능을 측정합니다.

그 결과, 모델의 정확도는 0.778로 약 77.8%의 성능으로 나타났습니다.

이는 지난번 구현한 랜덤 포레스트의 결과와는 약 0.37 정도 차이나며 랜덤 포레스트 모델보다 좋지 못한 성능으로 나타났습니다.

**지난 실습의 랜덤 포레스트 구현에서의 결정 트리 모델 결과 : 0.815(81.5%)**

하지만, 결정트리 분할을 여러번 시도 해보면 더 좋은 결과가 나타나기도 합니다.


```python
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)
```


    [Out]
    0.7783333333333333



### 엑스트라 트리 구현
**2000개 서브셋을 생성하여 100개의 샘플을 무작위로 array에 담습니다.**

`ShuffleSplit`을 사용하여 무작위 샘플 생성 

또한 극단적인 랜덤을 부여하기 위해 `random_state`는 고정하지 않았습니다.


```python
from sklearn.model_selection import ShuffleSplit

n_trees = 2000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```



이제, 그리드 탐색을 통해 찾은 최적의 매개변수를 사용하여 각 서브셋에 결정트리를 훈련시킵니다.

테스트 세트로 2000개의 결정트리를 `sklearn`의 `clone`을 사용하여 동일한 파라미터를 가지고 새로운 추정치를 만들어 평가합니다.

결과는 약 0.731로 약 73.1%의 결과가 나타났으며, 이 또한 앞서 만든 결정트리보다 성능이 떨어진 것으로 나타나고 있습니다.


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
    0.7308583333333334



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

결과는 0.8083으로 약 80.8%의 성능을 보입니다.

지난번 구현한 랜덤 포레스트와 비교하면 0.14 정도 낮은 수치입니다.


```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
```


    [Out]
    0.8083333333333333



## 지난번 지난번 구현한 랜덤 포레스트와 성능 비교

**자세한 내용은**

https://sungho95.github.io/2021/05/29/Random_Forest.html

**위 링크에서 확인할 수 있습니다.**




**엑스트라 트리 모델**

그리드 탐색을 통한 최적의 매개값의 결정트리 성능 : 약 0.778 (77.8%)

엑스트라 트리 모델 성능 : 약 0.808 (80.8%)



**랜덤 포레스트 모델**

그리드 탐색을 통한 최적의  매개값의 결정트리 성능 : 0.815 (81.5%)

랜덤 포레스트 모델 성능 : 약 0.823 (82.3%)

로 결과가 나타났으며, 엑스트라 트리 모델의 성능이 랜덤 포레스트 모델의 성능보다 좋게 나타날 수도 있습니다.

또한, 모델의 최적의 임곗값을 찾는 과정에서 랜덤 포레스트보다 훨씬 빠른 성능을 보였으며, 성능도 뒤쳐지지 않는 결과를 보여주었습니다.