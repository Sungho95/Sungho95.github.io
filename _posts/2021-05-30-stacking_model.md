```
layout: post
title:  "[ML] 스태킹 모델 구현 및 비교"
```

# [ML] 스태킹 모델 구현 후 사이킷런의 스태킹 모델과 비교하기


**스태킹 모델을 구현하기 위해**
1.   데이터셋 불러온 후 훈련, 검증, 테스트 세트로 나누기
2.   여러 종류의 분류기를 훈련(랜덤 포레스트, 엑스트라 트리, SVM 등)
3.   간접 또는 직접 투표 분류기를 사용하여 앙상블로 연결
4.   스태킹 앙상블 모델 구현
5.   사이킷런의 스태킹 앙상블 모델과 비교



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



### 데이터셋 불러온 후 훈련, 검증, 테스트 세트로 나누기

**MNIST 데이터를 불러들여 훈련 세트, 검증 세트, 테스트 세트로 나눕니다.**

ex) 훈련에 40000개 샘플, 검증에 10000개 샘플, 테스트에 10000개 샘플

**`sklearn.datasets`의 `fetch_openml`을 사용하여 MNIST 데이터셋을 불러옵니다.**


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
```

****

**`sklearn.model_selection`의 `train_test_split`을 사용하여 검증세트와 테스트 세트의 크기를 10000개로 지정합니다.**


```python
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)
```



### 여러 종류의 분류기를 훈련

분류 모델인 랜덤 포레스트 분류기, 엑스트라 트리 분류기, SVM, neural_nerwork, k-nn 분류 모델 다섯 가지를 모두 사용하여 분류기를 훈련시켜 보겠습니다.

* k-nn(k-최근접 이웃) 분류 모델을 포함하여 훈련 시킬 경우 오랜 시간이 걸립니다.(k=3의 경우 각 훈련 마다 50분 이상 소요되는 것으로 판단 됨)


```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
```



**각 모델의 분류기들을 사용하여 데이터를 훈련 시킵니다.**


```python
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
mlp_clf = MLPClassifier(random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
```


```python
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf, knn_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)
```

    [Out]
    Training the RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)
    Training the ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                         criterion='gini', max_depth=None, max_features='auto',
                         max_leaf_nodes=None, max_samples=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=100,
                         n_jobs=None, oob_score=False, random_state=42, verbose=0,
                         warm_start=False)
    Training the LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=100,
              multi_class='ovr', penalty='l2', random_state=42, tol=20, verbose=0)
    Training the MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                  hidden_layer_sizes=(100,), learning_rate='constant',
                  learning_rate_init=0.001, max_fun=15000, max_iter=200,
                  momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                  power_t=0.5, random_state=42, shuffle=True, solver='adam',
                  tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)
    Training the KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=-1, n_neighbors=1, p=2,
                         weights='uniform')



**각각의 분류기들의 성능결과**

RandomForest 분류기 : 0.9692

ExtraTrees 분류기 : 0.9715

SVC 분류기 : 0.8662

MLP 분류기 : 0.9639

knn 분류기 : 0.9715

선형 SVM 분류기의 성능이 가장 낮게 나왔습니다.


```python
[estimator.score(X_val, y_val) for estimator in estimators]
```


    [Out]
    [0.9692, 0.9715, 0.8662, 0.9639, 0.9715]



### 간접 또는 직접 투표 분류기를 사용하여 앙상블로 연결

**분류기들을 앙상블로 연결하기 위해 `VotingClassifier`를 사용합니다.**


```python
from sklearn.ensemble import VotingClassifier
```



각각의 분류기들을 파이프라인으로 연결하여 직접 투표 방식으로 모델을 훈련합니다.


```python
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
    ("knn_clf", knn_clf),
]
```



`VotingClassifier`를 사용할 때 Voting의 파라미터를 hard 또는 soft로 줄 수 있습니다.

hard = 직접 투표 방식

soft = 간접 투표 방식

default = hard 


```python
voting_clf = VotingClassifier(named_estimators)
```


```python
voting_clf.fit(X_train, y_train)
```


    [Out]
    VotingClassifier(estimators=[('random_forest_clf',
                                  RandomForestClassifier(bootstrap=True,
                                                         ccp_alpha=0.0,
                                                         class_weight=None,
                                                         criterion='gini',
                                                         max_depth=None,
                                                         max_features='auto',
                                                         max_leaf_nodes=None,
                                                         max_samples=None,
                                                         min_impurity_decrease=0.0,
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1,
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,
                                                         n_estimators=100,
                                                         n_jobs...
                                                nesterovs_momentum=True,
                                                power_t=0.5, random_state=42,
                                                shuffle=True, solver='adam',
                                                tol=0.0001, validation_fraction=0.1,
                                                verbose=False, warm_start=False)),
                                 ('knn_clf',
                                  KNeighborsClassifier(algorithm='auto',
                                                       leaf_size=30,
                                                       metric='minkowski',
                                                       metric_params=None,
                                                       n_jobs=-1, n_neighbors=1,
                                                       p=2, weights='uniform'))],
                     flatten_transform=True, n_jobs=None, voting='hard',
                     weights=None)



SVM 분류기를 포함한 직접 투표 방식의 결과는 0.9753으로 약 97.5%의 성능을 보이며, 이는 단일 분류기들의 정확도보다 높게 측정되었습니다.


```python
voting_clf.score(X_val, y_val)
```


    [Out]
    0.9753



개별 단일 분류기들의 정확도는 아래와 같습니다. 비교해보면 직접 투표 방식의 결과가 더 높습니다.


```python
[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]
```


    [Out]
    [0.9692, 0.9715, 0.8662, 0.9639, 0.9715]



**앞서 가장 낮은 성능을 보인 SVM을 제외하여 직접 투표 방식의 앙상블 모델을 훈련 시켜봅니다.**

voting 분류기의 `set_params` 옵션을 `svm_clf = None`을 주어 SVM 예측기를 제외하여 모델을 훈련할 수 있습니다.

만약, knn의 예측기를 제외하고 싶다면 옵션에 `knn_clf = None`를 입력하면 됩니다.


```python
voting_clf.set_params(svm_clf=None)
```


    [Out]
    VotingClassifier(estimators=[('random_forest_clf',
                                  RandomForestClassifier(bootstrap=True,
                                                         ccp_alpha=0.0,
                                                         class_weight=None,
                                                         criterion='gini',
                                                         max_depth=None,
                                                         max_features='auto',
                                                         max_leaf_nodes=None,
                                                         max_samples=None,
                                                         min_impurity_decrease=0.0,
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1,
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,
                                                         n_estimators=100,
                                                         n_jobs...
                                                nesterovs_momentum=True,
                                                power_t=0.5, random_state=42,
                                                shuffle=True, solver='adam',
                                                tol=0.0001, validation_fraction=0.1,
                                                verbose=False, warm_start=False)),
                                 ('knn_clf',
                                  KNeighborsClassifier(algorithm='auto',
                                                       leaf_size=30,
                                                       metric='minkowski',
                                                       metric_params=None,
                                                       n_jobs=-1, n_neighbors=1,
                                                       p=2, weights='uniform'))],
                     flatten_transform=True, n_jobs=None, voting='hard',
                     weights=None)



예측기 목록에 SVM 모델이 없음을 확인할 수 있습니다.

`('svm_clf', None)`


```python
voting_clf.estimators
```


    [Out]
    [('random_forest_clf',
      RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                             criterion='gini', max_depth=None, max_features='auto',
                             max_leaf_nodes=None, max_samples=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100,
                             n_jobs=None, oob_score=False, random_state=42, verbose=0,
                             warm_start=False)),
     ('extra_trees_clf',
      ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=42, verbose=0,
                           warm_start=False)),
     ('svm_clf', None),
     ('mlp_clf',
      MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=(100,), learning_rate='constant',
                    learning_rate_init=0.001, max_fun=15000, max_iter=200,
                    momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                    power_t=0.5, random_state=42, shuffle=True, solver='adam',
                    tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)),
     ('knn_clf',
      KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=-1, n_neighbors=1, p=2,
                           weights='uniform'))]



**하지만 훈련된 예측기 목록에는 SVM 예측기가 그대로 남아 있습니다.**


```python
voting_clf.estimators_
```


    [Out]
    [RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='auto',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=100,
                            n_jobs=None, oob_score=False, random_state=42, verbose=0,
                            warm_start=False),
     ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                          criterion='gini', max_depth=None, max_features='auto',
                          max_leaf_nodes=None, max_samples=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=42, verbose=0,
                          warm_start=False),
     LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
               intercept_scaling=1, loss='squared_hinge', max_iter=100,
               multi_class='ovr', penalty='l2', random_state=42, tol=20, verbose=0),
     MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                   beta_2=0.999, early_stopping=False, epsilon=1e-08,
                   hidden_layer_sizes=(100,), learning_rate='constant',
                   learning_rate_init=0.001, max_fun=15000, max_iter=200,
                   momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                   power_t=0.5, random_state=42, shuffle=True, solver='adam',
                   tol=0.0001, validation_fraction=0.1, verbose=False,
                   warm_start=False),
     KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                          metric_params=None, n_jobs=-1, n_neighbors=1, p=2,
                          weights='uniform')]



이를 제거하고 싶다면 `del voting_clf.estimators[i]` 명령어를 통해 제거할 수 있습니다.

i 값은 예측기의 인덱스 값으로 분류기의 인덱스 값을 대입해주면 됩니다.


```python
del voting_clf.estimators_[2]
```



예측기와 훈련된 예측기 모두 svm 모델을 제거하였으므로 다시 검증을 수행합니다.

그 결과 svm이 포함된 0.9753 보다 조금 향상된 0.9777의 정확도를 나타내고 있습니다. 


```python
voting_clf.score(X_val, y_val)
```


    [Out]
    0.9777



**다음은 직접 투표 방식이 아닌 간접 투표 방식 모델의 정확도를 알아봅니다.**

간접 투표 방식 모델의 정확도 : 0.9813. 98.13%


```python
voting_clf.voting = "soft"
```


```python
voting_clf.score(X_val, y_val)
```


    [Out]
    0.9813



그 결과 간접 투표 방식이 더 좋은 결과를 나타냈습니다.

직접 투표 방식 모델의 정확도 : 0.9719. 97.19%


```python
voting_clf.voting = "hard"
voting_clf.score(X_test, y_test)
```


    0.9719



직접 또는 간접 투표 방식의 모델들은 각각의 개별 모델보다 좋은 성능을 보여주고 있습니다.


```python
[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]
```


    [0.9692, 0.9715, 0.9639, 0.9715]



### 스태킹 앙상블 모델 구현

앞서 구현한 개별 분류기를 실행하여 검증 세트에대한 예측을 하고, 그 결과 예측으로 새로운 훈련 세트를 만듭니다.


```python
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)
```



이미지에 대한 예측으로 만든 새로운 훈련세트의 샘플들은 각 하나의 이미지에 대한 전체 분류기의 예측을 담은 벡터값으로 나타내며, 그 타깃은 이미지의 클래스입니다.


```python
X_val_predictions
```


    [Out]
    array([[5., 5., 5., 5., 8.],
           [8., 8., 8., 8., 8.],
           [2., 2., 2., 2., 2.],
           ...,
           [7., 7., 7., 7., 7.],
           [6., 6., 6., 6., 6.],
           [7., 7., 7., 7., 7.]], dtype=float32)



이제 블렌더를 통해 예측으로 구성된 새로운 훈련 세트를 학습해 최종 예측 값을 얻을 수 있도록 합니다.

랜덤 포레스트 블렌더 뿐만 아니라 MLP 블렌더 또는 SVM 블렌더와 같이 다른 모델의 블렌더를 사용할 수 있습니다.

또한 교차 검증을 통해 여러 개의 블렌더를 사용하여 좋은 성능의 블렌더를 채택하는 방법도 있습니다.

여기서는 랜덤 포레스트 블렌더를 사용하도록 하겠습니다.


```python
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)
```


    [Out]
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_jobs=None, oob_score=True, random_state=42, verbose=0,
                           warm_start=False)


```python
rnd_forest_blender.oob_score_
```


    [Out]
    0.9758



이제 테스트 세트에 앙상블을 평가할 수 있습니다.


```python
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)
```


```python
y_pred = rnd_forest_blender.predict(X_test_predictions)
```


```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(y_test, y_pred)
```


    [Out]
    0.9721



구현한 스태킹 앙상블의 정확도는 0.9721로 97.21%의 성능을 나타내고 있습니다.

 앞서 구현한 직접 투표 분류기 0.9719 보다는 좋은 성능을,

 간접 투표 분류기 0.9813보다 낮은 성능을 보입니다.

또한 개별 분류기를 통해 예측한 모델 중 랜덤 포레스트, k-최근접 이웃 모델의 예측 성능보다 낮은 결과를 보여줍니다.

### 사이킷런의 스태킹 앙상블 모델과 비교

이전에 구현한 스태킹 앙상블 모델과 성능을 비교하기 위해 개별 분류기로 구현한 다섯 가지 분류기 모두 사용하여 스태킹 모델을 훈련 시킵니다.


```python
from sklearn.ensemble import StackingClassifier

estimators = [("random_forest_clf", random_forest_clf),
              ("extra_trees_clf", extra_trees_clf),
              ("svm_clf", svm_clf),
              ("mlp_clf", mlp_clf),
              ("knn_clf", knn_clf)]
```



마찬가지로 `final_estimator` 옵션 또한 랜덤 포레스트 블렌더를 사용하도록 합니다.


```python
stacking_clf = StackingClassifier(estimators=estimators,
                                  final_estimator=rnd_forest_blender)
```

이제 스태킹 모델 훈련이 완료 되었으므로 테스트 세트에 대한 예측 성능을 확인합니다.

기존에 구현한 스태킹 앙상블 모델의 성능 : 0.9721. 약 97%
사이킷런을 통해 훈련한 스태킹 앙상블 모델의 성능 : 0.9796. 약 98%
사이킷런의 스태킹 앙상블 모델이 0.065 더 높은 정확도로 측정이 되었으나,
두 모델의 성능에는 큰 차이가 없는 것으로 판단됩니다.


```python
stacking_clf.fit(X_train, y_train).score(X_test, y_test)
```


    [Out]
    0.9796



번외로 `sklearn`을 활용하여 `final_estimator`의 모델을 `MLP`모델로 사용하여 성능을 확인해 보겠습니다.


```python
stacking_clf = StackingClassifier(estimators=estimators,
                                  final_estimator=MLPClassifier())
```

MLP 모델로 사용한 스태킹 앙상블 모델은 0.9743으로 랜덤 포레스트 블렌더를 통해 예측한 성능보다는 조금 떨어지는 성능을 나타내고 있습니다.

여러가지 블렌더를 사용하여 예측하고 싶을 때, 교차검증을 통해 모델을 훈련하는 방법이 있습니다.


```python
stacking_clf.fit(X_train, y_train).score(X_test, y_test)
```

    /usr/local/lib/python3.7/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:571: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      % self.max_iter, ConvergenceWarning)
    
    [Out]
    0.9743

