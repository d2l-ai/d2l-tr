# Ağda Ağ (NiN)
:label:`sec_nin`

LeNet, AlexNet ve VGG ortak bir tasarım deseni paylaşır: ekstrakt özellikleri bir dizi evrişim ve havuzlama katmanları aracılığıyla *mekansal* yapıyı istismar ve daha sonra tam bağlı katmanlar aracılığıyla temsil sonrası işlem sonrası. AlexNet ve VGG tarafından LeNet üzerindeki gelişmeler esas olarak bu daha sonraki ağların bu iki modülü nasıl genişlediğini ve derinleştirdiğini anlatıyor. Alternatif olarak, süreç içinde daha önce tam bağlı katmanları kullanmayı hayal edebiliriz. Bununla birlikte, yoğun katmanların dikkatsiz kullanımı, temsilin mekansal yapısını tamamen bırakabilir.
*ağda ağ* (* NiN*) blokları bir alternatif sunuyor.
Çok basit bir anlayışa dayalı olarak önerildi: her piksel için kanallarda ayrı ayrı bir MLP kullanmak :cite:`Lin.Chen.Yan.2013`.

## NiN Blokları

Konvolusyonel katmanların giriş ve çıkışlarının, örneğe, kanala, yüksekliğe ve genişliğe karşılık gelen eksenlere sahip dört boyutlu tensörlerden oluştuğunu hatırlayın. Ayrıca, tam bağlı katmanların giriş ve çıkışlarının tipik olarak örnek ve özelliğe karşılık gelen iki boyutlu tensörler olduğunu hatırlayın. NiN'in arkasındaki fikir, her piksel konumuna (her yükseklik ve genişlik için) tam bağlı bir katman uygulamaktır. Ağırlıkları her mekansal konum boyunca bağlarsak, bunu bir $1\times 1$ kıvrımsal katman (:numref:`sec_channels`'te açıklandığı gibi) veya her piksel konumuna bağımsız olarak hareket eden tam bağlı bir katman olarak düşünebiliriz. Bunu görmenin bir başka yolu da mekansal boyuttaki her elemanın (yükseklik ve genişlik) bir örneğe eşdeğer ve bir kanalın bir özelliğe eşdeğer olduğunu düşünmektir.

:numref:`fig_nin`, VGG ve NiN arasındaki ana yapısal farklılıkları ve bloklarını göstermektedir. NiN bloğu, bir kıvrımsal katmandan ve ardından ReLU aktivasyonlarıyla piksel başına tam bağlı katmanlar olarak hareket eden iki $1\times 1$ evrimsel katmandan oluşur. İlk katmanın evrişim penceresi şekli genellikle kullanıcı tarafından ayarlanır. Sonraki pencere şekilleri $1 \times 1$'ya sabitlenir.

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

## NiN Modeli

Orijinal NiN ağı, AlexNet'ten kısa bir süre sonra önerildi ve açıkça bazı ilham aldı. NiN, $11\times 11$, $5\times 5$ ve $3\times 3$ pencere şekilleri ile evrimsel katmanlar kullanır ve karşılık gelen çıkış kanalı sayıları AlexNet'teki ile aynıdır. Her NiN bloğu, 2'lik bir adım ve $3\times 3$'lık bir pencere şekli ile maksimum bir havuzlama katmanı izler.

NiN ve AlexNet arasındaki önemli bir fark, NiN'in tamamen bağlı katmanlardan kaçınmasıdır. Bunun yerine NiN, etiket sınıflarının sayısına eşit sayıda çıkış kanalı içeren bir NiN bloğu kullanır ve ardından *global* ortalama havuzlama katmanı izleyerek bir lojistik vektörü oluşturur. NiN'in tasarımının bir avantajı, gerekli model parametrelerinin sayısını önemli ölçüde azaltmasıdır. Bununla birlikte, pratikte, bu tasarım bazen artan model eğitim süresi gerektirir.

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # The global average pooling layer automatically sets the window shape
        # to the height and width of the input
        nn.GlobalAvgPool2D(),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output with a
    # shape of (batch size, 10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        tf.keras.layers.Flatten(),
        ])
```

Her bloğun çıkış şeklini görmek için bir veri örneği oluşturuyoruz.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## Eğitim

Moda-MNIST modeli eğitmek için daha önce olduğu gibi. NiN'in eğitimi AlexNet ve VGG için benzer.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## Özet

* NiN, konvolusyonel bir tabaka ve birden fazla $1\times 1$ evrimsel katmanlardan oluşan bloklar kullanır. Bu, piksel başına daha fazla doğrusal olmamasına izin vermek için kıvrımsal yığın içinde kullanılabilir.
* NiN, tam bağlı katmanları kaldırır ve kanal sayısını istenen çıkış sayısına indirdikten sonra (örneğin, Moda-MNIST için 10) küresel ortalama havuzlama ile değiştirir (örneğin, tüm konumlar üzerinden toplanır).
* Tam bağlı katmanların çıkarılması aşırı uyumu azaltır. NiN önemli ölçüde daha az parametreye sahiptir.
* NiN tasarımı, müteakip birçok CNN tasarımını etkiledi.

## Egzersizler

1. Sınıflandırma doğruluğunu artırmak için hiperparametreleri ayarlayın.
1. NiN bloğunda neden iki $1\times 1$ evrimsel katman var? Bunlardan birini çıkarın ve deneysel fenomenleri gözlemleyin ve analiz edin.
1. NiN için kaynak kullanımını hesaplayın.
    1. Parametrelerin sayısı nedir?
    1. Hesaplama miktarı nedir?
    1. Eğitim sırasında ihtiyaç duyulan bellek miktarı nedir?
    1. Tahmin sırasında gereken bellek miktarı nedir?
1. $384 \times 5 \times 5$ gösterimini bir adımda $10 \times 5 \times 5$ temsiline indirgeme ile ilgili olası sorunlar nelerdir?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/332)
:end_tab:
