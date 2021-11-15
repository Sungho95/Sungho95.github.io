---
layout: post
title:  "[Keras] MNIST 데이터셋을 이용한 합성곱신경망 모델(CNN) 구현하기"
toc: true
---

## 합성곱신경망 모델(CNN) 구현

`Conv2D`와 `MaxPooling2D`를 이용하여 합성곱 신경망을 구현하고자 합니다.


VGG16 모델을 참고하여 직접 각 층마다 파라미터를 조정하여 합성곱신경망 모델을 만듭니다.

총 4개의 컨볼루션 블록이 있는 합성곱 신경망으로 구성해 보았습니다.

**`Conv2D`**
* 필터(filters) : 32개부터 512개까지 맥스풀링 후 점진적으로 늘립니다.
* 필터 크기(kernel_size) : 3으로 고정합니다.
* 보폭(strides) : (1, 1)로 설정하여 한 칸씩 이동하도록 합니다.
* 패딩(padding) : "same"으로 하여 컨볼루션 후 MNIST 이미지 크기가 변하지 않도록 합니다.
(보폭 1, 패딩 1과 설정이 같음)
* 활성화함수(activation) : relu 함수를 사용합니다.

**`MaxPooling2D`**
* 크기(pool_size) : 2로 주어 필터크기를 2 x 2로 사용합니다.
* 보폭(strides) = : (2, 2)로 2칸씩 이동하도록 합니다.

따라서 2 x 2 필터를 사용하며 보폭은 2로주어 특성맵의 크기를 절반으로 축소시킵니다.

**`Dropout`**

과대적합 방지를 하기 위해 드롭아웃을 사용합니다.

드롭아웃은 출력층 바로 전에 사용합니다.

**`Outputs`**
* 활성화함수(activation) : softmax 함수를 사용하여 다중클래스 분류를 하도록 설정합니다. 1부터 10가지의 10개의 클래스가 존재하기 때문에 다중 클래스 분류가 적합합니다.

```python
!nvidia-smi
```
```
Mon Nov 15 12:17:32 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 495.44       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   50C    P8    29W / 149W |      0MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```




```python
from tensorflow import keras
from tensorflow.keras import layers

# 입력층
inputs = keras.Input(shape=(28, 28, 1))

# 은닉층
x = layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)
x = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(x)
x = layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2, strides=(2, 2))(x)
x = layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)
x = layers.Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding="same", activation="relu")(x)

x = layers.Flatten()(x) # 1차원 텐서로 변환
x = layers.Dropout(0.5)(x)
# 출력층
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```


```python
model.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                     
     conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                     
     conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
     )                                                               
                                                                     
     conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                     
     conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0         
     2D)                                                             
                                                                     
     conv2d_4 (Conv2D)           (None, 7, 7, 128)         73856     
                                                                     
     conv2d_5 (Conv2D)           (None, 7, 7, 128)         147584    
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 3, 3, 128)        0         
     2D)                                                             
                                                                     
     conv2d_6 (Conv2D)           (None, 3, 3, 256)         295168    
                                                                     
     conv2d_7 (Conv2D)           (None, 3, 3, 256)         590080    
                                                                     
     flatten (Flatten)           (None, 2304)              0         
                                                                     
     dropout (Dropout)           (None, 2304)              0         
                                                                     
     dense (Dense)               (None, 10)                23050     
                                                                     
    =================================================================
    Total params: 1,194,730
    Trainable params: 1,194,730
    Non-trainable params: 0
    _________________________________________________________________


## MNIST 데이터셋 전처리

MNIST 데이터셋이 훈련 모델에 적합하도록 하기 위해 데이터를 불러온 후 전처리 과정을 진행합니다.

훈련 이미지는 60000개 데이터로 이루어 졌으며 크기는 28 x 28로 통일시킵니다.

테스트 이미지는 10000개의 데이터로 이루어진 28 x 28 크기의 이미지 데이터셋입니다.

이들은 컬러 이미지가 아니므로 1로 설정합니다.


따라서,

훈련 이미지는 (60000, 28, 28, 1)

테스트 이미지는 (10000, 28, 28, 1)

입니다.


```python
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    11501568/11490434 [==============================] - 0s 0us/step


## 훈련 및 평가

모델을 컴파일 하기 위해 옵티마이저를 설정한 후, 손실 함수를 결정합니다.

훈련에 사용된 손실 함수는 다중 분류 손실 함수(sparse_categorical_crossentropy) 입니다.

이후 예측 정확도와 함께 모델을 훈련시킵니다.

에포크는 5, 배치 사이즈는 64로 설정하였습니다.


```python
model.compile(optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

history = model.fit(train_images,
                    train_labels,
                    epochs=20,
                    batch_size=64)
```

    Epoch 1/20
    938/938 [==============================] - 52s 23ms/step - loss: 0.2050 - accuracy: 0.9345
    Epoch 2/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0555 - accuracy: 0.9851
    Epoch 3/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0477 - accuracy: 0.9882
    Epoch 4/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0444 - accuracy: 0.9896
    Epoch 5/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0411 - accuracy: 0.9900
    Epoch 6/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0411 - accuracy: 0.9910
    Epoch 7/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0430 - accuracy: 0.9905
    Epoch 8/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0446 - accuracy: 0.9912
    Epoch 9/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0526 - accuracy: 0.9915
    Epoch 10/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0431 - accuracy: 0.9909
    Epoch 11/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0523 - accuracy: 0.9917
    Epoch 12/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0642 - accuracy: 0.9918
    Epoch 13/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0472 - accuracy: 0.9907
    Epoch 14/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0439 - accuracy: 0.9913
    Epoch 15/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0468 - accuracy: 0.9912
    Epoch 16/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0577 - accuracy: 0.9911
    Epoch 17/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0513 - accuracy: 0.9906
    Epoch 18/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0802 - accuracy: 0.9912
    Epoch 19/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0571 - accuracy: 0.9910
    Epoch 20/20
    938/938 [==============================] - 21s 22ms/step - loss: 0.0890 - accuracy: 0.9911



```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
```

    313/313 [==============================] - 3s 9ms/step - loss: 0.0549 - accuracy: 0.9909
    Test accuracy: 0.991


훈련셋에 대한 예측 정확도는  0.9911,

테스트셋에 대한 성능은 0.991로

좋은 성능을 나타내는 것으로 나타났습니다.


```python
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
loss = history.history["loss"]

epochs = range(1, len(accuracy) + 1)

# 에포크별 정확도 및 손실값 그래프
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, loss, "b", label="Training loss")
plt.title("Training accuracy and Training loss")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f68f4ba4650>




![output_12_1](https://user-images.githubusercontent.com/80394894/141779419-a4d68cf1-66f4-4f4e-a9b8-8c2325c19a73.png)


## 결론
* 먼저 VGG16 모델을 대부분 적용시켜 보았으나 만족하는 결과를 얻을 수 없었습니다.
* 추측하건데 MNIST 데이터셋은 1개의 채널을 가지고 있으나 VGG16 모델의 경우 RGB 3개의 채널을 가진 입력을 모델에 적용하도록 최적화 되어있기 때문이지 않을까 합니다.
* 따라서 기존 합성곱신경망 예제에 VGG16 모델의 설정을 참고하여 신경망의 깊이를 줄여가면서 적용 했습니다.
* 결과적으로 기존 예제보다 살짝 좋은 성능을 발휘하는 모델을 만들 수 있었습니다.
