---
layout: post
title:  "[TensorFlow] 텐서(Tensor)란 무엇인가?"
---

# [TensorFlow] 텐서(Tensor)란 무엇인가?

**바로가기**

[텐서(Tensor)](#텐서tensor)

[텐서의 기초](#텐서의-기초)

[텐서의 연산](#텐서의-연산)

[형상 정보](#형상-정보)

[인덱싱](#인덱싱)

[단일 축 인덱싱](#단일-축-인덱싱)

[다중 축 인덱싱](#다중-축-인덱싱)

[형상 조작](#형상-조작)

[DTypes](#dtypes)

[브로드캐스팅](#브로드캐스팅broadcasting)

[tf.convert_to_tensor](#tf-convert-to-tensor)

[비정형 텐서](#비정형-텐서)

[문자열 텐서](#문자열-텐서)

[희소 텐서](#희소-텐서)




## 텐서(Tensor)

텐서플로우의 텐서는 `dtype`이라 불리는 일관된 타입을 갖는 다차원 배열입니다.

`dtype`에 관한 정보는 [tf.dtypes.DType](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)링크를 통해 확인이 가능합니다.

```python
import tensorflow as tf		#텐서플로우 텐서
import numpy as np 			#파이썬의 넘파이
```

텐서플로우의 텐서는 파이썬 NumPy의 `np.array`와 비슷합니다.

모든 텐서는 Python 숫자 및 문자열과 같이 변경할 수 없습니다.

텐서의 내용을 업데이트할 수 없으며 새로운 텐서만 만들 수 있습니다.




## 텐서의 기초

텐서는 기본적으로 int32의 default 형식을 제공하고 있습니다.

print()함수를 통해 텐서를 출력하게 되면 `tf.Tesor(값, shape=(형상), dtype=유형)`형태로 제공합니다.

기본 텐서를 만드는 방법은 다음과 같습니다.



**0차원 텐서(스칼라)** 

0차원 텐서는 스칼라라고도 하며, 단일 값을 포함하며 축(axis)이 존재하지 않습니다.


```python
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
```

    tf.Tensor(4, shape=(), dtype=int32)



**1차원 텐서(벡터)**

1차원 텐서는 벡터라고도 하며 리스트 값을 가집니다. 이는 하나의 축이 존재합니다.

또한, 텐서 값을 float 형태의 부동소수 값을 입력하면 자동으로 `dtype`이 `float32`로 변경된 것을 확인할 수 있습니다.


```python
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)
```

    tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)



**2차원 텐서(행렬)**

2차원 텐서는 행렬이라고도 하며, 두 개의 축이 존재합니다.

또한, 텐서의 dtype을 변경하기 위해 값을 선언할 때 `dtype=tf.float16`과 같이 타입을 선언하여 구체화할 수 있습니다.


```python
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)
```

    tf.Tensor(
    [[1. 2.]
     [3. 4.]
     [5. 6.]], shape=(3, 2), dtype=float16)




<table>
<tr>
  <th>스칼라, 형상: <code>[]</code> </th>
  <th>벡터, 형상: <code>[3]</code> </th>
  <th>행렬, 형상: <code>[3, 2]</code> </th>
</tr>
<tr>
  <td>    <img src="images/tensor/scalar.png" alt="A scalar, the number 4">
</td>
  <td>    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/vector.png?raw=true" alt="The line with 3 sections, each one containing a number.">   </td>
  <td>    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/matrix.png?raw=true" alt="A 3x2 grid, with each cell containing a number.">   </td>
</tr>
</table>


**3차원 텐서**

3차원 텐서는 세 개의 축이 존재하며 이를 통해 축을 차원이라고도 할 수 있습니다.

텐서에는 더 많은 축(차원)이 존재할 수 있습니다.


```python
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
                    
print(rank_3_tensor)
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)



축이 두 개 이상인 텐서를 시각화하는 방법에는 여러 가지가 있습니다.

<table>
<tr>
  <th colspan="3">3축 텐서, 형상: <code>[3, 2, 5]</code> </th>
</tr>
<tr>
</tr>
<tr>
  <td>    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/3-axis_numpy.png?raw=true">   </td>
  <td>    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/3-axis_front.png?raw=true">   </td>
  <td>    <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/3-axis_block.png?raw=true">   </td>
</tr>
</table>


**텐서를 넘파이 어레이로 변환**

`np.arry(텐서)`메서드를 사용하여 넘파이 array로 변환


```python
np.array(rank_2_tensor)
```


    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)

`텐서.numpy()`메서드를 통한 넘파이 array로 변환


```python
rank_2_tensor.numpy()
```


    array([[1., 2.],
           [3., 4.],
           [5., 6.]], dtype=float16)



텐서는 float와 int 자료형 이외에도 다음과 같은 다른 유형이 존재합니다.

- 복소수(complex number)
- 문자열(string)

기본 `tf.Tensor` 클래스에서는 텐서가 각 축을 따라 모든 요소의 크기가 같은 "직사각형"이어야 합니다.

그러나 다양한 형상을 처리할 수 있는 특수 유형의 텐서가 있습니다.

- 비정형 텐서(Ragged Tensor)																						(참조 : [비정형 텐서](#비정형 텐서))
- 희소 텐서(Sparse Tensor)  																					       (참조 : [희소 텐서](#희소 텐서))



## 텐서의 연산

텐서는 덧셈, 요소별 곱셈 및 행렬 곱셈을 포함하여 기본적인 산술 연산을 수행할 수 있습니다.



`tf.add(), tf.multiply(), tf.matmul()`메서드를 통한 연산


```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # b는 tf.ones([2,2])라고도 선언할 수 있다.

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")
```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 



`+, *, @`를 통한 직접적인 연산 수행


```python
print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication
```

    tf.Tensor(
    [[2 3]
     [4 5]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[1 2]
     [3 4]], shape=(2, 2), dtype=int32) 
    
    tf.Tensor(
    [[3 3]
     [7 7]], shape=(2, 2), dtype=int32) 


​    

텐서는 모든 종류의 연산(ops)에 사용할 수 있습니다.


```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# c에서 가장 큰 값 찾기
print(tf.reduce_max(c))
# c에서 가장 큰 값의 인덱스 찾기
print(tf.argmax(c))
# 소프트맥스 함수 계산
print(tf.nn.softmax(c))
```

    tf.Tensor(10.0, shape=(), dtype=float32)
    tf.Tensor([1 0], shape=(2,), dtype=int64)
    tf.Tensor(
    [[2.6894143e-01 7.3105854e-01]
     [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)



## 형상 정보

텐서는 형상이 있습니다. 사용되는 일부 용어는 다음과 같습니다.

- **형상** : 텐서의 각 차원의 길이(요소의 수)
- **순위** : 텐서 축의 수입니다. 스칼라는 순위가 0이고 벡터의 순위는 1이며 행렬의 순위는 2입니다.
- **축** 또는 **차원** : 텐서의 특정 차원
- **크기 **: 텐서의 총 항목 수, 곱 형상 벡터

참고: 텐서는 텐서의 2차원 공간의에 대한 참조가 있을 수 있지만, 2차원 텐서(행렬)는 일반적으로 2차원 공간을 설명하지 않습니다.



텐서 및 `tf.TensorShape` 객체에는 다음에 액세스하기 위한 편리한 속성이 있습니다.


```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

<table>
<tr>
  <th colspan="2">순위-4 텐서, 형상: <code>[3, 2, 4, 5]</code> </th>
</tr>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/shape.png?raw=true" alt="A tensor shape is like a vector.">     </td>
<td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/4-axis_block.png?raw=true" alt="A 4-axis tensor">   </td>
  </tr>
</table>


**형상 정보 출력**

텐서는 다음과 같이 텐서의 유형, 차원의 수, 형상, 축에대한 정보, 요소의 수 등을 출력하여 확인할 수 있습니다.


```python
print("모든 요소의 유형 :", rank_4_tensor.dtype)
print("차원의 수 :", rank_4_tensor.ndim)
print("텐서의 형상(모양) :", rank_4_tensor.shape)
print("텐서의 0축에 있는 요소 :", rank_4_tensor.shape[0])
print("텐서의 마지막 축에 있는 요소 :", rank_4_tensor.shape[-1])
print("모든 요소의 수(3*2*4*5) :", tf.size(rank_4_tensor).numpy())
```

    모든 요소의 유형 : <dtype: 'float32'>
    차원의 수 : 4
    텐서의 형상(모양) : (3, 2, 4, 5)
    텐서의 0축에 있는 요소 : 3
    텐서의 마지막 축에 있는 요소 : 5
    모든 요소의 수(3*2*4*5) : 120



축은 대부분 인덱스를 통해 참조하지만, 항상 각 축의 의미를 추적해야 합니다.

축은 일반적으로 전역에서 로컬 순서로 정렬됩니다.

배치(Batch) 축이 먼저 오고 그 다음에 공간 차원과 각 위치의 특성(Features)이 마지막에 옵니다.

이러한 방식으로 특성 벡터는 연속적인 메모리 영역입니다.

<table>
<tr>
<th>일반적인 축 순서</th>
</tr>
<tr>
    <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/shape2.png?raw=true" alt="Keep track of what each axis is. A 4-axis tensor might be: Batch, Width, Height, Features">   </td>
</tr>
</table>


## 인덱싱

텐서의 인덱싱은 단일 축 인덱싱과 다중 축 인덱싱이 있습니다.



### 단일 축 인덱싱

텐서플로우는 파이썬의 [리스트 또는 문자열 인덱싱](https://docs.python.org/3/tutorial/introduction.html#strings)과 마찬가지로 표준 파이썬 인덱싱 규칙과 numpy 인덱싱의 기본 규칙을 따릅니다.

- 인덱스는 `0`에서 시작
- 음수 인덱스는 끝에서부터 역순으로 계산
- 콜론`:`은 부분 추출을 위한 슬라이싱 `start:stop:step`에 사용됩니다.



1차원 텐서(벡터)를 이용하여 단일 축 인덱싱을 할 수 있습니다.

```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())
```

    [ 0  1  1  2  3  5  8 13 21 34]



스칼라 값 사용하여 인덱싱하면 축은 제거되며 인덱스의 값을 사용합니다.


```python
print("First :", rank_1_tensor[0].numpy())
print("Second :", rank_1_tensor[1].numpy())
print("Last :", rank_1_tensor[-1].numpy())
```

    First : 0
    Second : 1
    Last : 34



`:` 슬라이싱을 사용하여 인덱싱하면 축이 유지된 상태로 사용이 가능합니다.


```python
print("모든 인덱스 출력 :", rank_1_tensor[:].numpy())
print("0~4 인덱스 값 출력 :", rank_1_tensor[:4].numpy())
print("4번 인덱스 값 부터 출력 :", rank_1_tensor[4:].numpy())
print("2~6번 인덱스 값 출력 :", rank_1_tensor[2:7].numpy())
print("2step씩 이동하면서 출력 :", rank_1_tensor[::2].numpy())
print("역순으로 출력 :", rank_1_tensor[::-1].numpy())
```

    모든 인덱스 출력 : [ 0  1  1  2  3  5  8 13 21 34]
    0~4 인덱스 값 출력 : [0 1 1 2]
    4번 인덱스 값 부터 출력 : [ 3  5  8 13 21 34]
    2~6번 인덱스 값 출력 : [1 2 3 5 8]
    2step씩 이동하면서 출력 : [ 0  1  3  8 21]
    역순으로 출력 : [34 21 13  8  5  3  2  1  1  0]



### 다중 축 인덱싱

더 높은 차원의 텐서는 여러 인덱스를 전달하여 인덱싱합니다.

단일 축의 경우에서와 정확히 같은 규칙이 각 축에 독립적으로 적용됩니다.



2차원 텐서(배열)이상의 텐서 사용하여 다중 축 인덱싱을 진행할 수 있습니다.


```python
print(rank_2_tensor.numpy())
```

    [[1. 2.]
     [3. 4.]
     [5. 6.]]



각 인덱스에 정수를 전달하면 결과는 스칼라입니다.


```python
# 2차원 텐서에서의 단일 값 추출
print(rank_2_tensor[1, 1].numpy())
```

    4.0



정수와 슬라이싱을 조합하여 인덱싱할 수 있습니다.


```python
# 행과 열 텐서 추출하기
print("2행 출력 :", rank_2_tensor[1, :].numpy())
print("2열 출력 :", rank_2_tensor[:, 1].numpy())
print("마지막 행 출력 :", rank_2_tensor[-1, :].numpy())
print("마지막 열의 0번째 값 출력 :", rank_2_tensor[0, -1].numpy())
print("1행을 제외하여 출력 :", rank_2_tensor[1:, :].numpy(), "\n")
```

    2행 출력 : [3. 4.]
    2열 출력 : [2. 4. 6.]
    마지막 행 출력 : [5. 6.]
    마지막 열의 0번째 값 출력 : 2.0
    1행을 제외하여 출력 :
    [[3. 4.]
     [5. 6.]] 


​    

다음은 3차원 텐서의 다중 축 인덱싱 입니다.


```python
print(rank_3_tensor[:, :, 4])
```

    tf.Tensor(
    [[ 4  9]
     [14 19]
     [24 29]], shape=(3, 2), dtype=int32)


<table>
<tr>
<th colspan="2">배치에서 각 예의 모든 위치에서 마지막 특성 선택하기</th>
</tr>
<tr>
    <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/index1.png?raw=true" alt="A 3x2x5 tensor with all the values at the index-4 of the last axis selected.">   </td>
      <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/index2.png?raw=true" alt="The selected values packed into a 2-axis tensor.">   </td>
</tr>
</table>
[텐서 슬라이싱 가이드](https://tensorflow.org/guide/tensor_slicing)에서 인덱싱을 적용하여 텐서의 개별 요소를 조작하는 방법을 알아볼 수 있습니다.



## 형상 조작

텐서의 형상을 바꾸는 것은 매우 유용합니다. 각 형상을 이용하여 여러 가지 방법으로 변환할 수 있습니다.

```python
# shape를 이용하여 각 축의 크기를 표시하는 TensorShape 객체를 반환할 수 있습니다.
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)
```

    (3, 1)



```python
# 객체를 리스트로 변환하여 출력할 수도 있습니다.
print(var_x.shape.as_list())
```

    [3, 1]



텐서를 새로운 형상으로 바꿀 수 있습니다. 기본 데이터를 복제할 필요가 없으므로 재구성이 빠르고 저렴합니다.


```python
# 텐서를 리스트 형태로 전달하여 새로운 형상으로 reshape가 가능합니다.
reshaped = tf.reshape(var_x, [1, 3])
```


```python
print(var_x.shape)
print(reshaped.shape)
```

    (3, 1)
    (1, 3)



데이터의 레이아웃은 메모리에서 유지되고 요청된 형상이 같은 데이터를 가리키는 새 텐서가 작성됩니다.

 TensorFlow는 C 스타일 "행 중심" 메모리 순서를 사용합니다.

여기에서 가장 오른쪽에 있는 인덱스를 증가시키면 메모리의 단일 단계에 해당합니다.


```python
print(rank_3_tensor)
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
    
     [[10 11 12 13 14]
      [15 16 17 18 19]]
    
     [[20 21 22 23 24]
      [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)



`tf.reshape(tensor, shape, name=none)`

텐서를 평평하게 하면 어떤 순서로 메모리에 배치되어 있는지 확인할 수 있습니다.

또한, shape의 인수가  -1이면 전체 크기가 일정하게 유지되도록 차원의 크기가 계산됩니다.


```python
print(tf.reshape(rank_3_tensor, [-1]))
```

    tf.Tensor(
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29], shape=(30,), dtype=int32)



일반적으로, `tf.reshape`의  올바른 사용 방법은 인접한 축을 결합하거나 분할하는 것입니다.

또는 `1`을 추가하거나 제거하는 방법도 있습니다.

즉, 위 텐서(3x2x5)의 경우, 슬라이스가 혼합되지 않으므로 (3x2)x5 형태 또는 3x (2x5) 형태로 재구성하는 것이 합리적입니다.


```python
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n") # [3, 2*5] 형태로도 가능
print(tf.reshape(rank_3_tensor, [3, -1]))
```

    tf.Tensor(
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]
     [20 21 22 23 24]
     [25 26 27 28 29]], shape=(6, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32)

<table>
<th colspan="3">몇 가지 좋은 reshape 방법</th>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/reshape-before.png?raw=true" alt="A 3x2x5 tensor">   </td>
  <td>   <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/reshape-good1.png?raw=true" alt="The same data reshaped to (3x2)x5">   </td>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/reshape-good2.png?raw=true" alt="The same data reshaped to 3x(2x5)">   </td>
</tr>
</table>



형상을 변경하면 요소 수가 같은 새로운 형상에 대해서는 잘 작동하지만, 축의 순서를 고려하지 않으면 작동이 되지 않습니다.

따라서 `tf.reshape`에서 축 교환이 작동하지 않으면, `tf.transpose`를 수행해야 합니다.

```python
# 좋지 못한 예

# reshape를 사용하여 축을 재정렬할 수 없게 되는 경우
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# 전혀 다른 결과물을 출력하게 되는 경우
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# 잘 작동하지 않게 됩니다.
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

    tf.Tensor(
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]]
    
     [[15 16 17 18 19]
      [20 21 22 23 24]
      [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) 
    
    tf.Tensor(
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]
     [24 25 26 27 28 29]], shape=(5, 6), dtype=int32) 
    
    InvalidArgumentError: Input to reshape is a tensor with 30 values, but the requested shape requires a multiple of 7 [Op:Reshape]


<table>
<th colspan="3">몇 가지 잘못된 재구성</th>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/reshape-bad.png?raw=true" alt="You can't reorder axes, use tf.transpose for that">   </td>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/reshape-bad4.png?raw=true" alt="Anything that mixes the slices of data together is probably wrong.">   </td>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/reshape-bad2.png?raw=true" alt="The new shape must fit exactly.">   </td>
</tr>
</table>
완전히 지정되지 않은 형상 전체에 걸쳐 실행할 수 있습니다. 형상에 `None`(축 길이를 알 수 없음)이 포함되거나 전체 형상이 `None`(텐서의 순위를 알 수 없음)입니다.

[tf.RaggedTensor](#ragged_tensors)를 제외하고, TensorFlow의 상징적인 그래프 빌딩 API의 컨텍스트에서만 발생합니다.

- [tf.function](function.ipynb)
- [keras 함수형 API](keras/functional.ipynb)




## `DTypes`

`tf.Tensor`의 데이터 유형을 검사하려면, `Tensor.dtype` 속성을 사용해야 합니다.

파이썬에서  `tf.Tensor`객체를 만들 때, 선택적으로 데이터 유형을 지정할 수 있습니다.

데이터를 지정하지 않으면, TensorFlow는 자체적으로 데이터 유형을 Default로 선택합니다.

텐서플로우는 파이썬 정수는 `tf.int32`,  부동 소수점은  `tf.float32`로 변환합니다.

이 외의 경우 NumPy가 배열로 변환할 때 사용하는 것과 같은 규칙을 사용합니다.

유형별로 캐스팅할 수 있습니다.

| 자료형     | 해석        |
| ---------- | ----------- |
| tf.float16 | 16비트 실수 |
| tf.float32 | 32비트 실수 |
| tf.float64 | 64비트 실수 |
| tf.int16   | 16비트 정수 |
| tf.int32   | 32비트 정수 |
| tf.int64   | 64비트 정수 |
| tf.string  | 문자열      |
| tf.bool    | boolean     |


```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)

# uint8로 선언하면 부동소수점의 소수점 부분을 없앱니다.
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)
```

    tf.Tensor([2 3 4], shape=(3,), dtype=uint8)



## 브로드캐스팅(Broadcasting)

브로드캐스팅은 [NumPy의 특성](https://numpy.org/doc/stable/user/basics.html)에서 기초한 개념입니다.

이는 특정 조건에서 작은 텐서를 대상으로 결합 연산을 실행할 때 더 큰 텐서에 맞게 자동으로 확장되는 것입니다.

가장 간단하고 일반적인 경우로 스칼라에 텐서를 곱하거나 추가하려고 할 때, 스칼라는 다른 인수와 같은 형상으로 브로드캐스팅됩니다. 


```python
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])

# 모두 같은 결과를 나타냅니다.
print(tf.multiply(x, 2))
print(x * y)
print(x * z)
```

    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)
    tf.Tensor([2 4 6], shape=(3,), dtype=int32)



크기가 1인 축은 다른 인수와 일치하도록 확장할 수 있으며, 두 인수 모두 같은 계산으로 확장할 수 있습니다.

아래의 경우, 3x1 행렬에 요소별로 1x4 행렬을 곱하여 3x4 행렬을 만드는 것으로 위의 예제와 같은 연산입니다.

하지만 x\*y, x\*z는 같은 결과를 나타내지 않습니다.


```python
# 위의 예시와 같은 연산입니다.
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))
```

    tf.Tensor(
    [[1]
     [2]
     [3]], shape=(3, 1), dtype=int32) 
    
    tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 
    
    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)


<table>
<tr>
  <th>추가 시 브로드캐스팅: <code>[1, 4]</code>와 <code>[3, 1]</code>의 곱하기는 <code>[3,4]</code>입니다.</th>
</tr>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/broadcasting.png?raw=true" alt="Adding a 3x1 matrix to a 4x1 matrix results in a 3x4 matrix">   </td>
</tr>
</table>


**브로드캐스팅을 사용하지 않는 경우**


```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)
```

    tf.Tensor(
    [[ 1  2  3  4]
     [ 2  4  6  8]
     [ 3  6  9 12]], shape=(3, 4), dtype=int32)

브로드캐스팅은 브로드캐스트 연산으로 메모리에 확장된 텐서를 구체화하지 않습니다.

따라서 메모리를 효율적으로 관리할 수 있으며 시간도 단축됩니다.



`tf.broadcast_to`를 사용하여 브로드캐스팅이 어떤 모습인지 알 수 있습니다.


```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
```

    tf.Tensor(
    [[1 2 3]
     [1 2 3]
     [1 2 3]], shape=(3, 3), dtype=int32)

예를 들어, `broadcast_to`는 수학적인 연산과는 달리 메모리를 절약하기 위해 특별한 연산을 수행하지 않습니다. 

따라서 메모리에 텐서를 구체화(할당)하기 때문에 훨씬 더 복잡해질 수 있습니다.

Jake VanderPlas의 저서 *Python Data Science Handbook*의 [해당 섹션](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html)(NumPy)에서는 더 많은 브로드캐스팅 트릭을 보여줍니다.



## tf.convert_to_tensor

`tf.matmul` 및 `tf.reshape`와 같은 대부분의 연산은 클래스 `tf.Tensor`의 인수를 사용합니다.

그러나 위의 경우, 텐서 형상의 Python 객체가 수용됨을 알 수 있습니다.

전부는 아니지만 대부분의 연산은 텐서가 아닌 인수에 대해 `convert_to_tensor`를 호출합니다.

변환 레지스트리가 존재하여 NumPy의 `ndarray`, `TensorShape` , Python 리스트 및 `tf.Variable`과 같은 대부분의 객체 클래스는 모두 자동으로 변환됩니다.



자세한 내용은 `tf.register_tensor_conversion_function`을 참조할 수 있습니다.

여기서 자신만의 유형이 있으면 자동으로 텐서로 변환할 수 있습니다.



## 비정형 텐서

어떤 축을 따라 다양한 수의 요소를 가진 텐서를 "비정형(ragged)"이라고 합니다. 비정형 데이터에는 `tf.ragged.RaggedTensor`를 사용합니다.

예를 들어, 비정형 텐서는 정규 텐서로 표현할 수 없습니다.

<table>
<tr>
  <th>`tf.RaggedTensor`, 형상: <code>[4, None]</code> </th>
</tr>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/ragged.png?raw=true" alt="A 2-axis ragged tensor, each row can have a different length.">   </td>
</tr>
</table>


```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]
```


```python
try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")
```

    ValueError: Can't convert non-rectangular Python sequence to Tensor.



대신 `tf.ragged.constant`를 사용하여 `tf.RaggedTensor`를 작성할 수 있습니다.


```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
```

    <tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>



`tf.RaggedTensor`의 형상에는 알 수 없는 길이의 축이 포함됩니다.


```python
print(ragged_tensor.shape)
```

    (4, None)



## 문자열 텐서

문자열 텐서의 `dtype`은 `tf.string`이며, 텐서에서 문자열과 같은 데이터를 나타낼 수 있습니다.

문자열은 원자성이므로 Python 문자열과 같은 방식으로 인덱싱할 수 없습니다.

문자열의 길이는 텐서의 축 중의 하나가 아닙니다.

문자열을 조작하는 함수에 대해서는 `tf.strings`를 참조할 수 있습니다.



**스칼라 문자열 텐서**


```python
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)
```

    tf.Tensor(b'Gray wolf', shape=(), dtype=string)



**문자열의 벡터 텐서**

<table>
<tr>
  <th>문자열의 벡터, 형상: <code>[3,]</code> </th>
</tr>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/strings.png?raw=true" alt="The string length is not one of the tensor's axes.">   </td>
</tr>
</table>


```python
# 길이가 다른 세 가지의 문자열 텐서
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])

# 형상은 (3, )이며 문자열의 길이가 포함되지 않습니다.
print(tensor_of_strings)
```

    tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)

위의 출력에서 `b` 접두사는 `tf.string` dtype이 유니코드 문자열이 아니라 바이트 문자열입니다.

 TensorFlow에서 유니코드 텍스트를 처리하는 자세한 내용은 [유니코드 튜토리얼](https://www.tensorflow.org/tutorials/load_data/unicode)에서 참조할 수 있습니다.



유니코드 문자를 전달하면 UTF-8로 인코딩됩니다.


```python
tf.constant("🥳👍")
```


    <tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>



문자열이 있는 일부 기본 함수는 `tf.strings`을 포함하여 `tf.strings.split`에서 찾을 수 있습니다.


```python
# split을 통해 문자열을 텐서로 분할할 수 있습니다.
print(tf.strings.split(scalar_string_tensor, sep=" "))
```

    tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)



```python
# 하지만 문자열 텐서를 split하면 비정형 텐서로 변환합니다.
# 따라서 문자열은 각 다른 수의 텐서로 분할될 수 있습니다.
print(tf.strings.split(tensor_of_strings))
```

    <tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>


<table>
<tr>
  <th>세 개의 분할된 문자열, 형상: <code>[3, None]</code> </th>
</tr>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/string-split.png?raw=true" alt="Splitting multiple strings returns a tf.RaggedTensor">   </td>
</tr>
</table>


`tf.string.to_number`:


```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))
```

    tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)



`tf.cast`를 사용하여 문자열 텐서를 숫자로 변환할 수는 없지만, 바이트로 변환한 다음 숫자로 변환할 수 있습니다.


```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings :", byte_strings)
print("Bytes :", byte_ints)
```

    Byte strings : tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)
    Bytes : tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)



```python
# 유니코드로 split 후 디코딩
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)
```


    Unicode bytes: tf.Tensor(b'\xe3\x82\xa2\xe3\x83\x92\xe3\x83\xab \xf0\x9f\xa6\x86', shape=(), dtype=string)
    
    Unicode chars: tf.Tensor([b'\xe3\x82\xa2' b'\xe3\x83\x92' b'\xe3\x83\xab' b' ' b'\xf0\x9f\xa6\x86'], shape=(5,), dtype=string)
    
    Unicode values: tf.Tensor([ 12450  12498  12523     32 129414], shape=(5,), dtype=int32)

`tf.string` dtype은 TensorFlow의 모든 원시 바이트 데이터에 사용됩니다.

 `tf.io` 모듈은 이미지 디코딩 및 csv 구문 분석을 포함하여 데이터를 바이트로 변환하거나 바이트에서 변환하는 함수가 포함되어 있습니다.



## 희소 텐서

넓은 임베딩 공간처럼 데이터가 희소할 경우가 있습니다.

텐서플로우는 `tf.sparse.SparseTensor` 및 관련 연산을 지원하여 희소 데이터를 효율적으로 저장할 수 있습니다.

<table>
<tr>
  <th>`tf.SparseTensor`, 형상: <code>[3, 4]</code> </th>
</tr>
<tr>
  <td> <img src="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/images/tensor/sparse.png?raw=true" alt="An 3x4 grid, with values in only two of the cells.">   </td>
</tr>
</table>


```python
# 희소 텐서는 메모리 효율적인 방식으로 인덱스 별로 값을 저장합니다.
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# 희소 텐서를 dense로 변환
print(tf.sparse.to_dense(sparse_tensor))
```

    SparseTensor(indices=tf.Tensor(
    [[0 0]
     [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 
    
    tf.Tensor(
    [[1 0 0 0]
     [0 0 2 0]
     [0 0 0 0]], shape=(3, 4), dtype=int32)

[맨 위로](#\[tensorflow\]-텐서tensor란-무엇인가?)

