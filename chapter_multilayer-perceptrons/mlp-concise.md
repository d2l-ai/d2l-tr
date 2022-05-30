# Çok Katmanlı Algılayıcıların Kısa Uygulaması
:label:`sec_mlp_concise`

Tahmin edebileceğiniz gibi, (**yüksek seviye API'lere güvenerek, MLP'leri daha da kısaca uygulayabiliriz.**)

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
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

## Model

Kısa softmaks bağlanım uygulamamızla karşılaştırıldığında (:numref:`sec_softmax_concise`), tek fark *iki* tam bağlı katman eklememizdir (önceden *bir* tane ekledik). İlki, (**256 gizli birim içeren ve ReLU etkinleştirme fonksiyonunu uygulayan**) [**gizli katmanımızdır**]. İkincisi, çıktı katmanımızdır.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

[**Eğitim döngüsü**], softmaks bağlanımını uyguladığımız zamanki ile tamamen aynıdır. Bu modülerlik, model mimarisiyle ilgili konuları dikey düşünmelerden ayırmamızı sağlar.

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Özet

* Yüksek seviye API'leri kullanarak MLP'leri çok daha kısaca uygulayabiliriz.
* Aynı sınıflandırma problemi için, bir MLP'nin uygulanması, etkinleştirme fonksiyonlarına sahip ek gizli katmanlar haricinde softmaks bağlanımının uygulanmasıyla aynıdır.

## Alıştırmalar

1. Farklı sayıda gizli katman eklemeyi deneyiniz (öğrenme oranını da değiştirebilirsiniz). Hangi ayar en iyi sonucu verir?
1. Farklı etkinleştirme işlevlerini deneyin. Hangisi en iyi çalışır?
1. Ağırlıkları ilkletmek için farklı tertipler deneyiniz. En iyi hangi yöntem işe yarar?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/262)
:end_tab:
