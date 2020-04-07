---
title: Joint Optimization Framework for Learning with Noisy Labels
categories:
- Noisy Label
tags:
- Noisy Label
use_math: true
---

오늘 리뷰할 논문은 CVPR 2018에 발표된 **Joint Optimization Framework for Learning with Noisy Labels**이라는 논문입니다 ([논문 링크](https://arxiv.org/abs/1803.11364)).  Noisy label과 관련된 첫 번째 리뷰이고, 앞으로 이 주제의 논문들을 제가 이해한 범위에서 꾸준히 리뷰할 계획입니다.

일반적으로 Large dataset을 만들기 위해 Crowd sourcing이나 Web crawling 등의 방법을 이용하는데, 이러한 방법들은 부정확하게 라벨링된 데이터를 만들게 됩니다.

예를 들면, 강아지 사진들의 품종 label을 얻기 위해 Amazon Mechanical Turk (AMT)를 이용하는 상황을 가정해봅시다. 많은 작업자들이 아래 그림과 같이 허스키와 말라뮤트를 잘 구별하지 못한다면 AMT를 통해 얻은 label이 noisy 할 수 밖에 없습니다. 그리고 잘못 달린 label이 진실이라고 철썩같이 믿고 학습된 모델은 마찬가지로 허스키와 말라뮤트를 잘 구별할 수 없을 것입니다.

<p align="center">
<a href='https://photos.google.com/share/AF1QipNl74a6QByFgFZIQUmpnSaVihpnn_97cEVAoG-_EusA2pE8k78mYFUqWh19lzMx6A?key=b1JwWFNhR1JBVlA5T3VvbFBqTzUtUllrRHdwRkFB&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/zmYLKLM7Ar4BSGFkhe_d4cmq93hU4iiT4F2ArgPzPmmJJgop45LHicPH7QyddH83_X5hahvZfd-oggIECrmY1IHklgikqwBJpXLnP7QdP0JbIDW156JMtrSX3D0gnPcx_NaQL73ffg=w2400' width="50%" height="80%"></a>
<br>
<em>Label이 noisy해질 수 있는 이미지 사례; source: <a href="https://1boon.kakao.com/petxlab/58e1d992ed94d200019b4c4f">petxlab</a></em>
</p>



보통 이런 상황에서는 label noise가 특정 label로만 발생하기 때문에 이를 **Asymmetric label noise**라고 부릅니다. 반대로 label noise가 완전히 random한 label로 발생하는 경우 이를 **Symmetric label noise**라고 부릅니다.

또 다른 예를 들면, 꽃을 분류하는 모델을 학습시키기 위한 데이터를 구하기 위해 구글에서 "아이리스"를 검색하면 아이리스 꽃 뿐만 아니라 드라마 아이리스 포스터도 나옵니다. 사실 이런 경우는 noisy label도 되지만 데이터 자체가 우리가 관심있는 영역 (ex. 꽃) 밖에 있어서 이러한 데이터가 미치는 나쁜 영향을 제거하는 것은 엄밀히는 또 다른 문제인 것 같습니다.

<p align="center">
<a href='https://photos.google.com/share/AF1QipMay8K5uvt6k-X_DcmQGRz13LacvhtMF2tqLy9TtQNYSunFn2rUTc8Trm9kBjEOCA?key=ZEJHbUZLSkF1bFlDa3lIX19PQ0M3T1lOc21xemVn&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/OzW0jhVNEHNvPZZAbuHb7xY6t8ykGEAxf2qb-3aKNGhIH8rL_n69ntcU9f0ly_MhfMchtUHJ205OVMctag1WjEF8mFE320bTbjggTM06Zi0fKssZ4xSXDJOXqdvPg-1OI0oI1IVHFQ=w2400' width="80%" height="80%"></a>
<br>
<em>원하지 않는 데이터가 crawling되는 경우; 구글 이미지 검색</em>
</p>


결국 우리가 label이 noisy한 데이터셋을 가지고 있을 때, 이러한 데이터에서 최대한 좋은 성능을 내는 딥러닝 모델을 학습키기 위한 기법들이 연구되어 왔고, 이 논문 또한 noisy label에 robust한 모델을 학습시키는 기법을 소개하고 있습니다.

## Introduction

[YFCC100M](http://projects.dfki.uni-kl.de/yfcc100m/)이나 [Clothing1M](https://www.floydhub.com/lukasmyth/datasets/clothing1m) 같은 dataset들은 Web crawling을 통해 생성되어 incorrect noisy label을 포함하고 있습니다. DNN은 어떠한 데이터도 학습하고, 기억할 수 있으므로 결국 noisy data에 overfitting하게 됩니다.

### Contribution
이 논문은 noisy label dataset을 위한 optimization framework를 제안하는데, 핵심 아이디어는 noisy label을 고정된 것으로 취급하지 않고 **label 자체를 최적화**하겠다는 것입니다. 즉, DNN의 parameter와 noisy label을 동시에 optimize하면서 classifier의 performance와 noisy label의 correction이라는 두 마리 토끼를 잡겠다는 것입니다. 이 논문의 main contribution을 정리하면 다음과 같습니다.
* Joint optimization framework를 제안하여 **network parameter와 class label을 번갈아가며 optimize** 합니다.
*  이렇게 학습된 DNN은 더이상 noisy label을 기억하지 않으면서도 clean data의 성능이 큰 learning rate에서 높게 유지됩니다. 이는 **DNN이 먼저 단순한 sample을 학습하고, 나중에 noisy data를 학습한다**는 Arpit *et al.* ([논문링크](https://arxiv.org/abs/1706.05394); 조만간 읽어봐야겠습니다) 의 주장을 뒷받침합니다.
* CIFAR-10 dataset에서 SOTA의 성능을 보이고, Clothing1M dataset에서는 comparable한 성능을 보였습니다. 

<center><a href='https://photos.google.com/share/AF1QipOYJ6tCGHykdNZnoigGw_mYb_Ult46bwoXO3qv4RxyZR07VCXNm5tVduvgDM0gYIg?key=M1VudWtXbFhSWXdLZktHZVB6QmFoVEhjX3daZW53&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/m1Ca6CV4YTNLW98BeoQDZ2Hl_c9r8uCgtv2uzijrvk6i2L2o3jAgNqFPF2QXg0rzYL3hh-aCc4pExpS05Sdd8_Pe3MEH36fm7TLoM6UVkoNFsSjEUKq26GJY4Fdn-21o6D5TvEHojQ=w2400' /></a></center>

여기까지 읽고 Figure 1을 보면 일단 model parameter와 label을 번갈아가며 업데이트 하겠다는 것 같은데.. 아직까지는 loss $\mathcal{L}$ 과 probability $s$가 무엇인지는 알 수가 없습니다. 그럼 계속 읽어보겠습니다.

## Related Works
### Generalization abilities of DNNs
DNN의 generalization과 memorization 성능에 대한 연구가 진행되면서, DNN은 label이 완전히 random이어도 학습이 가능하다는 것을 알아내게 됩니다. 이러한 DNN의 memorization ability로 인해 발생하는 문제는 다음과 같습니다.
* DNN이 noisy label도 완전히 학습할 수 있기 때문에 training data에 noisy data가 있으면 test performance가 감소할 수 밖에 없습니다.
* Learning ability가 완전하다면 어떤 label이 noisy한 지 가려내기가 어렵습니다.

그리고 저자들은 현재까지 noisy label에 대한 논문들이 DNN의 memorization ability에 집중하지 않았다고 합니다.

### Existing methods
기존에는 weight decay, dropout, adversarial training, mixup과 같은 regularization 기법을 통해 이러한 noise에 robust한 DNN을 학습시키려 하였습니다. 하지만, 이러한 regularization 기법들은 explicit하게 noisy label을 다루지는 않습니다. 그래서 learning curve를 그려보면 test performance가 증가하다가 epoch이 지남에 따라 성능이 점점 감소하는 모습을 보입니다. 게다가 noisy data에 대한 loss도 전체 training loss에 포함되어 있어서 training loss로 clean data의 성능을 판단하기도 어렵습니다.

또 다른 방법으로는 confusion matrix와 같은 prior knowledge를 활용하는 방법이 있기는 한데, 실제로 활용하기는 어렵다고 합니다. 결국 새로운 optimization framework가 필요한 상황입니다.

그 외에도 noise transition matrix, noise-tolerant loss function, bootstrapping scheme 등의 다양한 방법들이 있지만, 추후 리뷰하면서 자세히 설명하는 것이 이해에 더 도움이 될 것 같아서 생략하였습니다.

### Self-training and pseudo-labeling

Pesudo-labeling은 self-training의 한 종류로 semi-supervised learning에서 자주 사용되고 있습니다. Semi-supervised learning에서 pseudo-labeling은 unlabeled data에 model의 prediction을 assign하여 initialize하고, DNN 모델을 clean label과 pseudo-label 둘 다 이용하여 학습하면서도 더 좋은 pseudo-label을 찾는 방식으로 학습됩니다.

Semi-supervised learning은 어떤 데이터가 unlabeled data인지 알고 있는 반면에 noisy label은 어떤 데이터가 noise data인지 모른다는 문제가 있습니다. 그래서 모든 데이터가 noise data일 수도 있다는 의심을 가지고 있어야 합니다.

Noisy label을 위해 self-training scheme ([논문링크](https://arxiv.org/abs/1412.6596); 조만간 읽어봐야겠습니다) 이 제안되기는 했지만, 학습이 끝날 때까지 original noisy label을 사용하고, noise rate가 높은 경우 noisy label의 효과가 남아 성능을 감소시킵니다. 그런데 본 논문에서는 self-training scheme과 다르게 모든 label을 pseudo-label로 대체하였습니다.

## Classification with Label Optimization

일반적인 supervised learning은 다음과 같은 optimization problem으로 formulation 됩니다.

$$\min_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}|X,Y) \tag{1}$$

여기서 $\mathcal{L}$은 cross entropy loss이고, 데이터셋에 clean data만 있다고 가정하면 위와 같은 optimization problem을 통해 DNN의 parameter가 잘 학습됩니다.

저자들은 Training dataset에 noisy label이 있는 상황에서 learning rate가 높으면 DNN이 training data를 완전히 memorize하는 것을 막을 수 있음을 실험으로 확인했다고 합니다. 그러면, 높은 learning rate로 학습된 DNN은 noise label을 잘 배울 수 없을 것이고, 식 (1)의 loss 값이 clean data에서는 낮고 noise data에서는 높을 것이라 가정합니다. 그러면 이 높아진 loss를 label을 optimize 함으로써 noise label을 correction 할 수 있게 됩니다. 

이러한 아이디어로 noisy label이 있는 상황에서의 supervised learning 문제를 DNN의 parameter와 label을 joint optimize하는 문제로 아래와 같이 formulation하게 됩니다.

$$\min_{\boldsymbol{\theta},Y}\mathcal{L}(\boldsymbol{\theta},Y|X) \tag{2}$$

이렇게 제안된 loss function은 다시 세개의 term으로 나눌 수 있습니다.

$$\mathcal{L}(\boldsymbol{\theta},Y|X) =\mathcal{L}_c(\boldsymbol{\theta},Y|X) + \alpha \mathcal{L}_p(\boldsymbol{\theta}|X)+\beta\mathcal{L}_e(\boldsymbol{\theta}|X) \tag{3}$$

$\mathcal{L}_c$는 classification loss로 다음과 같이 label $\boldsymbol{y}_{i}$와 final layer의 output $\boldsymbol{s}(\boldsymbol{\theta}, \boldsymbol{x}_i)$ 간의 KL divergence로 계산합니다.

$$\mathcal{L}_c(\boldsymbol{\theta},Y|X) = {1 \over n}\sum_{i=1}^n D_{KL}(\boldsymbol{y}_i || \boldsymbol{s}(\boldsymbol{\theta}, \boldsymbol{x}_i)) \tag{4}$$

$$D_{KL}(\boldsymbol{y}_i || \boldsymbol{s}(\boldsymbol{\theta}, \boldsymbol{x}_i)) = \sum_{j=1}^c y_{ij}\log{ \left( {y_{ij}\over s_j} (\boldsymbol{\theta},\boldsymbol{x}_i) \right) } \tag{5}$$

여기서 $c$는 class 수 입니다. 따라서 $\boldsymbol{y}_i$는 one-hot vector이고, $\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)$는 softmax output인 상황에서 이 두 벡터의 차이 (KL divergence)를 loss로 정의하였습니다. (아마 $\boldsymbol{y}_i$와 $\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)$의 분포 간 차이를 loss로 한다는 철학에서 우리가 일반적으로 알고있는 Cross-entropy 대신 KL divergence를 택한 것 같습니다.) 나머지 두 loss에 대해 설명하기 위해서는 label의 optimization을 먼저 이야기해야 하기 때문에 뒤에서 다루도록 하겠습니다.

### Alternating Optimization
본 논문에서 제안하는 learning framework는 DNN의 parameter $\boldsymbol{\theta}$와 class label $Y$를 번갈아가며 update하게 됩니다. Update rule은 다음과 같습니다.

* $Y$를 고정하고  $\boldsymbol{\theta}$를 update
	- 식 (3)의 모든 term은 $\boldsymbol{\theta}$로 미분가능합니다.
	- 따라서 stochastic gradient descent (SGD)로 $\boldsymbol{\theta}$를 update할 수 있습니다.
* $\boldsymbol{\theta}$를 고정하고  $Y$를 update
	* $Y$는 식 (3)의 첫 번째 term인 $\mathcal{L}_c(\boldsymbol{\theta},Y \vert X)$과만 관련이 있습니다.
	* 식 (4)의 optimization 문제는 각 data point의 label $\boldsymbol{y}_i$을 optimize하는 문제로 쪼개집니다.

Label을 optmize하는 것은 label과 DNN의 output 간의 KL divergence를 줄이는 것으로 볼 수 있고, label의 output 분포를  은 두 가지로 생각해 볼 수 있습니다.

#### Hard-label method

Hard-label method는 현재 DNN의 output을 보고 **one-hot label로 $Y$를 update**하는 방법이라고 볼 수 있습니다. 수식으로 나타내면 다음과 같습니다.

$$y_{ij} = 
\begin{cases}
  1  & \text{if} \ \ \  j=\arg \max_{j'}{s_{j'}(\boldsymbol{\theta}, \boldsymbol{x}_i)}\\
  0  & \text{otherwise} \tag{6}
\end{cases}
$$

수식을 살펴보면 softmax output이 가장 큰 output node를 1로 하고, 나머지 output node를 0으로 하는 one-hot label로 update하게 됩니다.

#### Soft-label method

Soft-label method는 **softmax output을 그대로 label로 update**하는 방식입니다. 수식으로 나타내면 다음과 같습니다.

$$\boldsymbol{y}_i = \boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i) \tag{7}$$

저자들이 두 가지 방법 모두 실험해 보았는데 soft-label method가 더 좋은 결과를 보였다고 합니다.

### Regularization Terms

이제 식 (3)의 뒤의 두 term인 $\mathcal{L}_p(\boldsymbol{\theta} \vert X)$와 $\mathcal{L}_e(\boldsymbol{\theta} \vert X)$을 마저 설명하겠습니다. 

#### Regularization loss $\mathcal{L}_p$

$\mathcal{L}_p$는 모든 label이 하나의 class로 쏠리는 상황을 방지하기 위한 regularization loss 입니다. 식 (4)를 다시 잘 보면 label $\boldsymbol{y}_i$에 $\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)$를 따라가도록 update하면서, $\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)$가 $\boldsymbol{y}_i$을 따라가도록 DNN의 parameter를 학습합니다. 그러면 $\boldsymbol{y}_i$가 고정된 일반적인 supervised learning과는 달리 $\boldsymbol{y}_i$와 $\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)$ 모두 변할 수 있으므로 지지대(?)가 없어서, 모든 데이터의 $\boldsymbol{y}_i$와 $\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)$가 같은 class로 쏠리더라도 식 (4)의 loss는 작게 유지될 수 있습니다. 

따라서 class의 prior distribution $\boldsymbol{p}$를 도입하여 $s_{ij}$를 데이터 방향으로 평균낸 **$\bar{s}_j$가 prior $\boldsymbol{p}$의 분포를 따라가게끔 하자**는 아이디어로 제안한 regularization loss 입니다. 이를 KL divergence를 이용하여 다음과 같이 나타낼 수 있습니다.

$$\mathcal{L}_p = D_{KL}(\boldsymbol{p} || \bar{\boldsymbol{s}}(\boldsymbol{\theta},X))=\sum_{j=1}^c{\log{p_j \over \bar{s}_j(\boldsymbol{\theta},X)}} \tag{8}$$

$\bar{\boldsymbol{s}}(\boldsymbol{\theta},X)$는 전체 $n$개의 데이터에서 계산하지 않고 $\mathcal{B}$개의 mini-batch로 approximation 하였습니다.

$$\bar{\boldsymbol{s}}(\boldsymbol{\theta},X)={1 \over n} \sum_{i=1}^n{\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x}_i)} \approx {1 \over |\mathcal{B}|} \sum_{\boldsymbol{x} \in \mathcal{B}}{\boldsymbol{s}(\boldsymbol{\theta},\boldsymbol{x})} \tag{9}$$

이러한 approximation이 class 수가 많거나 class imbalance가 심한 경우에는 잘 동작하지 않습니다. 그런데 실험에서 사용한 CIFAR-10이나 Clothing1M 데이터에서는 잘 동작했다고 합니다.

#### Regularization loss $\mathcal{L}_e$

$\mathcal{L}_e$는 soft-label method를 사용할 때 필요합니다. 식 (7)로 $Y$를 update할 때 $\boldsymbol{\theta}$와 $Y$가 local minima에 빠져서 학습이 진행되지 않을 때가 있습니다. 이러한 상황을 방지하고자 **soft-label의 entropy를 낮추어** deterministic해지는 방향의 regularization loss를 도입하였습니다.

$$\mathcal{L}_e = H(\boldsymbol{s}(\boldsymbol{\theta}, \boldsymbol{x}_i))- {1 \over n} \sum_{i=1}^n{\sum_{j=1}^c{s_j(\boldsymbol{\theta}, \boldsymbol{x}_i) \log{s_j(\boldsymbol{\theta}, \boldsymbol{x}_i)}}} \tag{10}$$

그래서 전체 framework를 알고리즘으로 정리하면 다음과 같습니다.
<center><a href='https://photos.google.com/share/AF1QipMG-NV8RVSiYgztsuc2_O82V3_bE88uYk-HAd4V0EPjj3fDfNsepvPpsc1QGIl5pw?key=YWRuVVJaY1U0bndUdGhNV19EWEVJLUQtWVRZSzBB&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/sWDomE34I5LXQR_l-D377Qp7d_TEWu8d3tuvkRJefWrW0hoLUcB64GZNt2WosMoCwfRR740PAeI3BaUfSPLyxmSwjUJuSV4al6R5piH-cJxkXS6ERLMxIQ95mI8rd8T_3QQBV2alxQ=w2400' width="80%" /></a></center>

위 알고리즘의 Eq. (8), (9)가 각각 식 (6), (7) 입니다.

## Experiments

### Datasets

실험을 위해 두 가지 데이터셋을 사용했습니다.

#### CIFAR-10

- **Symmetric Noise CIFAR-10 (SN-CIFAR)**
	- Symmetric noise를 구현하기 위해 $r$의 확률로 랜덤한 one hot label을 줌으로써 noisy label을 만든 데이터셋입니다.
- **Asymmetric Noise CIFAR-10 (AN-CIFAR)**
	- Asymmetric noise를 구현하기 위해 $r$의 확률로 정말 헷갈릴법한 class로 noisy label을 준 데이터셋입니다. (ex. 트럭 $\to$ 자동차, 새 $\to$ 비행기, 사슴 $\to$ 말)
- **Pseudo Label CIFAR-10 (PL-CIFAR)**
	- Transfer learning 환경을 구현하기 위해 $r$의 확률로 pseudo-label을 준 데이터셋입니다. Pseudo-label은 ImageNet 데이터로 pretrain된 ResNet-50의 pool5 layer에서 k-means++를 통해 부여하였습니다. Pseudo-label의 전체 정확도는 62.50% 입니다.

#### Clothing1M

Clothing1M 데이터셋은 실제로 온라인 쇼핑몰에서 Crawling한 백만장의 이미지 데이터셋이고 14개의 class로 구성되어 있습니다. 각 이미지의 label은 이미지 주변 글씨나 판매자가 제공하는 정보를 이용하여 달았기 때문에 오류가 많습니다. 따라서 실제 noisy label 상황을 잘 보여주는 좋은 예시 데이터입니다.


### Generalization and Memorization

<center><a href='https://photos.google.com/share/AF1QipP-TdoW56reekC6vgiQZ_rGniK46El_IjnYWr03ug2vStg8Jkj44DT6SXiZ16NRdw?key=SmhYQkFRaDd3WDR6YmlVdHVpbTZWeHp2VzZ3OGJ3&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/C4Zbxit7yh2xJTu6PKS5rm40cHjjxJoOLurO9y-VoZPdJ7dxLzO-JZw_KMdN-KWctEkzLrMx-0QO-42dySm6nun44SeKpejyohRgdFH74N-t-NIgXFUPxG0FXyPmmuIqQEarennmig=w2400' width="70%"/></a></center>

Generalization에 관한 실험으로 method 첫 부분에서 언급했던 learning rate가 높을수록 noisy label에 robust하다는 주장을 뒷받침하는 실험입니다.

<center><a href='https://photos.google.com/share/AF1QipOwh0t0mZDW6FOkTOWNgHd0200Tcdc5u7U4BabPkpf5GtiQevAkK_AR6H1sbti0jw?key=czdlUHl5aDJXWHBWUW16VWRqaWJLbnBRLVpuUWdR&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/bRIqHCMEAx2gdvn8KYL1sUtK5uD54PVo4sv3YH94CGBxStxhgC8b9tI05UJc9tTRXpf_gZciWpP7KesIjrqcOz2N2aT3HvtCES2xarAUAebYf6wyJM2_SELzVjVErhD7E3nfwMx0vQ=w2400' width="70%"/></a></center>

Memorization에 관한 실험으로 noisy label rate $r$이 0.9, 즉 90%의 label이 noisy label 이어도 DNN의 training loss가 0이 되어 noisy label을 모두 memorization 해버린다는 것을 확인할 수 있는 실험입니다.

### Hard-Label vs. Soft-Label

<center><a href='https://photos.google.com/share/AF1QipPqHmI74qu2E0bC9YS4C4I5FiIxe-hD50bYMw7YrAd80Jdlp_J8zidj-PUB4GexOA?key=cWtxUDY5YUhCcEJLVVNaS1dDbE04OXZfVzNKUXBB&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/IhZNgQNBcEhqhXIV5_BOFJMGrhQEJpQn9BJxhovw8_OrxviNqaRdTw6L7aUVzXXPS_GP8SOvERa3XwSSap6cHcOT2R3y-RvR3Vmgz9wV51RQ-cEQ-dczLHVYd9XeGrgYef7V3taRuQ=w2400' width="70%" /></a></center>

모든 label을 Hard-Label method로 update한 경우와 softmax output이 현재 label과 가장 다른것부터 top 50개, 500개, 5000개만 Hard-Label method로 update한 경우, 그리고 모든 label을 Soft-label method로 update한 경우 update된 label과 model output 간의 recovery accuracy를 측정한 실험 결과입니다. 

Soft-label method가 가장 수렴이 빠르고 높은 recovery accuracy를 보임을 확인할 수 있습니다. 부가적으로 Soft-label은 hard-label과 달리 학습된 network의 confidence를 보여주어 noisy label을 학습하는데 중요한 성질을 가지고 있다고 볼 수 있습니다.

### Experiment on the CIFAR-10 dataset

<center><a href='https://photos.google.com/share/AF1QipOsweOqhWjJx8k4q1Tdc4xvdo0qz_7BZjdlOlsQH14nQOs5WuiWUHiGUTP_Ml-WCw?key=ODNqSVNtUWllYXNzV0FVWEdwSUtlWFVBVHMybEx3&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/eJA9ysbe9SNMeU8XwuOfYM7313M8lSPaprLfxiqLHj_ZNEpqvs2gUpmcAqbh6kmvyOSXZM3wVNaCLVgS8LeLXbR5uRtxA0eNYtedoatCn5Hoqyj-O1hvnwR0FS9bKSwl9ypNOwJSdw=w2400' /></a></center>

SN-CIFAR와 AN-CIFAR 실험 결과를 보면 다른 방법들과 비교하였을 때 test accuracy와 recovery accuracy의 best 값은 다른 방법과 비슷하거나 근소하게 높지만 마지막 epoch (last)에서의 성능이 월등히 좋은 것을 확인할 수 있습니다.

<center><a href='https://photos.google.com/share/AF1QipM5Hy-sdvKXZpnbwErqnD74qP1FuE_mncFyzEuBMRWR2NE4sbVg0epirq1xkCpstA?key=cG16WDFMc25XVHF0Z19fNVFDLWxmUVlRdTR2eXpn&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/meFuNIVCp-C6sP273BeN4qpTU4KZpKR-VA_X-uENflYuuLPR5_CcIqK0w7bwYNO6zSWoZXfin6gnleaSzWnd1hG4wlAaw5QkhrQl2WEPM4Bj4o2fsdGMYQOO4XA6ke8e2KNLdxNShA=w2400' /></a></center>

PL-CIFAR에 관한 실험 결과를 보면 pseudo-label은 ResNet-50에 의해 생성되었으므로 이미 DNN에 의해 optimize된 label이라 생각할 수 있습니다. 따라서 pseudo-label은 잘 update 되지 못하고 noise rate와 training loss가 더 낮음에도 불구하고 더 낮은 test accuracy를 보이게 된다고 합니다. 



### Experiment on the Clothing1M dataset
<center><a href='https://photos.google.com/share/AF1QipPQTus6x2pMH9hodEp55jZv0V6eNfvbxfP0cMrpIVVFEAafbCzeZyyjXGDILAxbPw?key=V1NzeXVxeXpNeG45N0Zpa0V6TldPclc1WUUtRmVn&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/9iuFWiaVn6ggXz1JIM3fxqYq1iHT4gFLvngCpx7f2isrWZ-3KnCFBj6cwGnmFZ1zU4ZUZNlUHgZqUPUn1Ib9SuKAhSZxQtC0h9FZyUfk9QPheA7IRivio67WCtnC_mkQndmVHcEbow=w2400' width="70%" /></a></center>

Clothing1M dataset에서는 Forward라는 기법을 제안한 논문에서 제시한 실험 결과를 실험 1, 2로 가져오고 cross entropy loss를 reproduce한 결과와 제안하는 방법을 비교하였습니다. 그 결과 월등하지는 않지만 약간 우세한 성능을 보임을 확인할 수 있었습니다.

<center><a href='https://photos.google.com/share/AF1QipMbUNyN49OQdRibt3dvVSZXXiPfPDiLKtPpkevnY89XIis8eek86jAtNYcYJxdiKA?key=UjV2eHBvTkNwVVZOTFlzWkQ0cmxudDlnY0hiWWh3&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/m_4byLwc-5tBNgyyzAoZwhiyeJXcihgDzXRG1NZ30snd_uMuWgqMsrxMsZX4oPCnN3nkV8FW3PQEDfLvTzYiyCIEXd8vispizI-vdpKMiEyXtFOEWubJ0TI0lNf8unxu3wLDAhqFfQ=w2400' width="70%" /></a></center>

마지막으로 label이 Hoodie이었던 이미지가 T-shirt로 변한 경우의 Top-2, Bottom-2 class probability와 반대로 T-shirt에서 Hoodie로 변한 경우의 이미지를 확인하였습니다. Top-2의 경우에는 정말 noisy label이었음이 확실한 이미지들이 보이고, Bottom-2는 원래 클래스가 오히려 잘 맞는 느낌이 있습니다. 이는 soft-label을 이용하면 class의 confidence를 확인할 수 있다는 것을 잘 보여주는 실험입니다.

## Conclusion

이 논문에서는 noisy label data를 학습하기 위해 DNN의 parameter와 label 전체를 번갈아가며 update하는 joint optimization framework를 제시합니다. 이 방법을 통해 DNN이 noisy label을 memorize하는 것을 막아 noisy label이 있을 때 SOTA의 성능을 얻었습니다. 실험이 굉장히 빵빵한 논문이긴 한데 Ablation study가 있었으면 더 좋았을 것 같습니다. 그리고 왜 loss를 그렇게 제안하였는지, 제안하는 loss를 번갈아 optimize하면 정말 좋은 DNN parameter와 loss로 수렴이 가능한지를 증명해 줬으면 더 좋았을 것 같습니다. 또 DNN의 output space에 soft-label이 크게 영향을 받을 것 같은데 Batch normalization이나 Spectral Normalization과 같이 DNN에 Lipschitz constraint을 거는 방법들이 soft-label에 어떻게 영향을 주는지도 실험해보았다면 더 재밌었을 것 같습니다.

<p align="center">
<a href='https://photos.google.com/share/AF1QipNl74a6QByFgFZIQUmpnSaVihpnn_97cEVAoG-_EusA2pE8k78mYFUqWh19lzMx6A?key=b1JwWFNhR1JBVlA5T3VvbFBqTzUtUllrRHdwRkFB&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/zmYLKLM7Ar4BSGFkhe_d4cmq93hU4iiT4F2ArgPzPmmJJgop45LHicPH7QyddH83_X5hahvZfd-oggIECrmY1IHklgikqwBJpXLnP7QdP0JbIDW156JMtrSX3D0gnPcx_NaQL73ffg=w2400' width="50%" height="80%"></a>
<br>
    <em>asdf</em>
</p>