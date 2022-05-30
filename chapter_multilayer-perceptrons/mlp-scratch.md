# Çok Katmanlı Algılayıcıların Sıfırdan Uygulanması
:label:`sec_mlp_scratch`

Artık çok katmanlı algılayıcıları (MLP'ler) matematiksel olarak nitelendirdiğimize göre, birini kendimiz uygulamaya çalışalım. Softmax regresyonu (:numref:`sec_softmax_scratch`) ile elde ettiğimiz önceki sonuçlarla karşılaştırmak için Fashion-MNIST imge sınıflandırma veri kümesi (:numref:`sec_fashion_mnist`) ile çalışmaya devam edeceğiz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Model Parametrelerini İlkleme

Fashion-MNIST'in 10 sınıf içerdiğini ve her imgenin $28 \times 28 = 784$ gri tonlamalı piksel değerleri ızgarasından oluştuğunu hatırlayın. Yine şimdilik pikseller arasındaki uzamsal yapıyı göz ardı edeceğiz, bu nedenle bunu 784 girdi özniteliği ve 10 sınıf içeren basit bir sınıflandırma veri kümesi olarak düşünebiliriz. Başlarken, [**bir gizli katman ve 256 gizli birim içeren bir MLP uygulayacağız.**] Bu miktarların ikisini de hiper parametreler olarak kabul edebileceğimizi unutmayın. Tipik olarak, belleğin donanımda öyle tahsis edildiğinden ve adreslendiğinden hesaplama açısından verimli olma eğiliminde olan 2'nin katlarında katman genişliklerini seçiyoruz.

Yine, parametrelerimizi birkaç tensörle temsil edeceğiz. *Her katman* için, bir ağırlık matrisini ve bir ek girdi vektörünü izlememiz gerektiğini unutmayın. Her zaman olduğu gibi, bu parametrelere göre kaybın gradyanları için bellek ayırıyoruz.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## Etkinleştirme Fonksiyonu

Her şeyin nasıl çalıştığını bildiğimizden emin olmak için, [**ReLU aktivasyonunu**] yerleşik `relu` işlevini doğrudan çağırmak yerine maksimum işlevi kullanarak [**kendimiz uygulayacağız**].

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## Model

Uzamsal yapıyı göz ardı ettiğimiz için, her iki boyutlu imgeyi `num_inputs` uzunluğuna sahip düz bir vektör halinde  (`reshape`) yeniden şekillendiriyoruz. Son olarak, (**modelimizi**) sadece birkaç satır kodla (**uyguluyoruz**).

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Burada '@' matris carpimini temsil eder
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## Kayıp İşlevi

Sayısal kararlılığı sağlamak için ve softmaks işlevini zaten sıfırdan uygularken (:numref:`sec_softmax_scratch`), softmaks ve çapraz entropi kaybını hesaplamak için yüksek seviyeli API'lerden birleşik işlevi kullanıyoruz. Bu karmaşıklıkla ilgili önceki tartışmamızı :numref:`subsec_softmax-implementation-revisited` içinden hatırlayın. İlgili okuru, uygulama ayrıntıları hakkındaki bilgilerini derinleştirmek için kayıp işlevi kaynak kodunu incelemeye teşvik ediyoruz.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## Eğitim

Neyse ki, [**MLP'ler için eğitim döngüsü softmax bağlanımıyla tamamen aynıdır.**] Tekrar `d2l` paketini kullanarak, `train_ch3` fonksiyonunu çağırıyoruz (bkz. :numref:`sec_softmax_scratch`), dönem sayısını 10 ve öğrenme oranını 0.1 olarak ayarlıyoruz.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

Öğrenilen modeli değerlendirmek için onu [**bazı test verisine uyguluyoruz**].

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## Özet

* Manuel olarak yapıldığında bile basit bir MLP uygulamanın kolay olduğunu gördük.
* Bununla birlikte, çok sayıda katmanla, MLP'leri sıfırdan uygulamak yine de karmaşık olabilir (örneğin, modelimizin parametrelerini adlandırmak ve takip etmek).

## Alıştırmalar

1. Hiper parametre `num_hiddens` değerini değiştirin ve bu hiper parametrenin sonuçlarınızı nasıl etkilediğini görün. Diğerlerini sabit tutarak bu hiper parametrenin en iyi değerini belirleyiniz.
1. Sonuçları nasıl etkilediğini görmek için ek bir gizli katman eklemeyi deneyiniz.
1. Öğrenme oranını değiştirmek sonuçlarınızı nasıl değiştirir? Model mimarisini ve diğer hiper parametreleri (dönem sayısı dahil) sabitlersek, hangi öğrenme oranı size en iyi sonuçları verir?
1. Tüm hiper parametreleri (öğrenme oranı, dönem sayısı, gizli katman sayısı, katman başına gizli birim sayısı) birlikte optimize ederek elde edebileceğiniz en iyi sonuç nedir?
1. Birden fazla hiper parametre ile uğraşmanın neden çok daha zor olduğunu açıklayınız.
1. Birden fazla hiper parametre üzerinde bir arama yapılandırmak için düşünebileceğiniz en akıllı strateji nedir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/227)
:end_tab:
