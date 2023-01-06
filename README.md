# Electra_for_Domain_Adaptation

### 프로젝트 소개
- Generator와 Descriminator 학습을 위한 구조 및 🤗Transformers를 활용해 Electra에 대한 Domain Adaptation 수행

- Domain Adaptation은 Electra-Base를 활용했으며 데이터/컴퓨터 과학 분야의 도서 데이터를 학습

- Electra의 학습 방식을 이해할 수 있도록 모델의 예측 결과 제공(ELECTRA 학습 데이터 시각화 예시 참고)

- Electra 학습 구조와 학습 과정에 대한 설명은 [ELECTRA 모델 구현 및 Domain Adaptation 방법 정리](https://yangoos57.github.io/blog/DeepLearning/paper/Electra/electra/) 참고

- Domain Adaptation에 대한 설명은 [[NLP] Domain Adaptation과 Finetuning 개념 정리](https://yangoos57.github.io/blog/DeepLearning/paper/Finetuning/Finetuning/) 참고

<br/>

### 이런 경우 활용하면 좋습니다.

- Electra에 대한 이론은 아는데, 이를 구현하는데 어려움을 겪는 경우
- Electra를 활용해 Domain Adaptation 또는 처음부터 모델을 학습시키고 싶은 경우
- Huggingface Trainer를 이해하고 직접 활용할 예제가 필요한 경우

<br/>

### ELECTRA 학습 데이터 시각화 예시

```python

0번째 epoch 진행 중 ------- 12번째 step 결과

원본 문장 :  특히 안드로이드 플랫폼 기반의 (웹)앱과 (하이)브드리앱에 초점을 맞추고 있다
가짜 문장 :  특히 안드로이드 플랫폼 기반의 (마이크로)앱과 (이)브드리앱에 초점을 맞추고 있다



21개 토큰 중 0개 예측 실패 -------- 2개 가짜 토큰 중 2개 판별

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>문장 위치</th>
      <th>실제 토큰</th>
      <th>(가짜)토큰</th>
      <th>실제</th>
      <th>예측</th>
      <th>정답</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>웹</td>
      <td>(마이크로)</td>
      <td>fake</td>
      <td>fake</td>
      <td>O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>하이</td>
      <td>(이)</td>
      <td>fake</td>
      <td>fake</td>
      <td>O</td>
    </tr>
  </tbody>
</table>

```python

Combined Loss 2.225 -- Generator Loss : 0.576 -- Discriminator Loss : 0.033
```

<br/>
<br/>

```python
0번째 epoch 진행 중 ------- 22번째 step 결과

원본 문장 :  친절한 이론 설명은 (누구) (##나) 어렵지 않게 기초를 다 (##질) 수 있도록 안내하며 감각적인 실무 예제는 여러분의 디자인 잠재력을 깨워줄 수 있을 것입니다
가짜 문장 :  친절한 이론 설명은 (이렇) (##이) 어렵지 않게 기초를 다 (##할) 수 있도록 안내하며 감각적인 실무 예제는 여러분의 디자인 잠재력을 깨워줄 수 있을 것입니다.



43개 토큰 중 3개 예측 실패 -------- 3개 가짜 토큰 중 2개 판별

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>문장 위치</th>
      <th>실제 토큰</th>
      <th>(가짜)토큰</th>
      <th>실제</th>
      <th>예측</th>
      <th>정답</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>누구</td>
      <td>(이렇)</td>
      <td>fake</td>
      <td>fake</td>
      <td>O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>##나</td>
      <td>(##이)</td>
      <td>fake</td>
      <td>fake</td>
      <td>O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>##질</td>
      <td>(##할)</td>
      <td>fake</td>
      <td>-</td>
      <td>X</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>다</td>
      <td>다</td>
      <td>-</td>
      <td>fake</td>
      <td>X</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>감각</td>
      <td>감각</td>
      <td>-</td>
      <td>fake</td>
      <td>X</td>
    </tr>
  </tbody>
</table>

```python

Combined Loss 14.246 -- Generator Loss : 2.97 -- Discriminator Loss : 0.226

```

<br/>

### 구동환경

```python
  torch == 1.12.1
  pandas == 1.4.3
  transformers == 4.20.1
  datasets == 2.8.0
```

### 참고 라이브러리
* [Pytorch-Electra](https://github.com/lucidrains/electra-pytorch) 
* [koELECTRA](https://github.com/monologg/KoELECTRA)

