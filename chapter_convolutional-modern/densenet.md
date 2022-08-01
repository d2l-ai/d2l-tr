# Yoğun Bağlı Ağlar (DenseNet)

ResNet, derin ağlardaki işlevlerin nasıl parametrize edileceği görüşünü önemli ölçüde değiştirdi. *DenseNet* (yoğun evrişimli ağ) bir dereceye kadar bunun :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` mantıksal uzantısıdır. Ona nasıl ulaşacağımızı anlamak için, matematikte küçük bir gezinti yapalım.

## ResNet'ten DenseNet'e

Fonksiyonlar için Taylor açılımını hatırlayın. $x = 0$ noktası için şu şekilde yazılabilir:

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

Önemli nokta, bir işlevi giderek daha yüksek dereceli terimlerine ayırmasıdır. Benzer bir davranış ile, ResNet fonksiyonları aşağıdaki gibi parçalar

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

Yani, ResNet $f$'i basit bir doğrusal terime ve daha karmaşık doğrusal olmayan bir terime ayırır. İki terimin ötesinde bilgi özümsemek (zorunlu olarak eklemek değil) istersek ne olur? Bir çözüm DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` oldu.

![Katmanlar arası bağlantılarda ResNet (sol) ve DenseNet (sağ) arasındaki temel fark: Toplama ve bitiştirme kullanımı.](../img/densenet-block.svg)
:label:`fig_densenet_block`

:numref:`fig_densenet_block` içinde gösterildiği gibi, ResNet ve DenseNet arasındaki temel fark, ikinci durumda çıktıların toplanmaktan ziyade *bitiştirilmesidir* ($[,]$ ile gösterilir). Sonuç olarak, giderek daha karmaşık bir işlev dizisini uyguladıktan sonra $\mathbf{x}$'ten değerlerine bir eşleme gerçekleştiriyoruz:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

Sonunda, tüm bu işlevler tekrar öznitelik sayısını azaltmak için MLP'de birleştirilir. Uygulama açısından bu oldukça basittir: Terimleri toplamak yerine, bunları bitiştiririz. DenseNet'in adı, değişkenler arasındaki bağımlılık grafiğinin oldukça yoğunlaştığı gerçeğinden kaynaklanmaktadır. Böyle bir zincirin son katmanı, önceki tüm katmanlara yoğun bir şekilde bağlanır. Yoğun bağlantılar :numref:`fig_densenet` içinde gösterilmiştir.

![DenseNet'teki yoğun bağlantılar.](../img/densenet.svg)
:label:`fig_densenet`

DenseNet oluşturan ana bileşenler *yoğun bloklar* ve *geçiş katmanları*dır. Birincisi, girdilerin ve çıktıların nasıl bitiştirildiğini tanımlarken, ikincisi kanal sayısını kontrol eder, böylece çok büyümezlerdir.

## [**Yoğun Bloklar**]

DenseNet, ResNet'in değiştirilmiş “toplu normalleştirme, etkinleştirme ve evrişim” yapısını kullanır (bkz. :numref:`sec_resnet` içindeki alıştırma). İlk olarak, bu evrişim blok yapısını uyguluyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

*Yoğun blok*, her biri aynı sayıda çıktı kanalı kullanan birden fazla evrişim bloğundan oluşur. Bununla birlikte, ileri yaymada, kanal boyutundaki her evrişim bloğunun girdi ve çıktısını bitiştiririz.

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Her bloğun girdi ve çıktısını kanal boyutunda birleştirin
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Her bloğun girdi ve çıktısını kanal boyutunda birleştirin
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

Aşağıdaki örnekte, 10 çıktı kanalından oluşan 2 evrişim bloğu içeren [**bir `DenseBlock` örneği tanımlıyoruz**]. 3 kanallı bir girdi kullanırken, $3+2\times 10=23$ kanallı bir çıktı alacağız. Evrişim blok kanallarının sayısı, girdi kanallarının sayısına göre çıktı kanallarının sayısındaki büyümeyi kontrol eder. Bu aynı zamanda *büyüme oranı* olarak da adlandırılır.

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## [**Geçiş Katmanları**]

Her yoğun blok kanal sayısını artıracağından, çok fazla sayıda eklemek aşırı karmaşık bir modele yol açacaktır. Modelin karmaşıklığını kontrol etmek için bir *geçiş katmanı* kullanılır. $1\times 1$ evrişimli katmanı kullanarak kanal sayısını azaltılır ve ortalama ortaklama tabakasının yüksekliğini ve genişliğini 2'lik bir uzun adımla yarıya indirilir ve modelin karmaşıklığını daha da azaltılır.

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

Önceki örnekte yoğun bloğun çıktısına 10 kanallı [**bir geçiş katmanı uygulayın**]. Bu, çıktı kanallarının sayısını 10'a düşürür ve yüksekliği ve genişliği yarıya indirir.

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## [**DenseNet Modeli**]

Şimdi, bir DenseNet modeli inşa edeceğiz. DenseNet, ilk olarak ResNet'te olduğu gibi aynı tek evrişimli katmanı ve maksimum ortaklama katmanını kullanır.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Daha sonra, ResNet'in kullandığı artık bloklardan oluşan dört modüle benzer, DenseNet de dört yoğun blok kullanır. ResNet'e benzer şekilde, her yoğun blokta kullanılan evrişimli katmanların sayısını da ayarlayabiliriz. Burada, :numref:`sec_resnet` içindeki ResNet-18 modeliyle uyumlu olarak katmanların sayısını 4'e ayarladık. Ayrıca, yoğun bloktaki evrişimli katmanlar için kanal sayısını (yani büyüme hızı) 32'ye ayarladık, böylece her yoğun bloğa 128 kanal eklenecektir.

ResNet'te, her modül arasındaki yükseklik ve genişlik, 2'lik uzun adımlı bir artık bloğu ile azaltılır. Burada, yükseklik ve genişliği yarıya indirip kanal sayısını yarılayarak geçiş katmanını kullanıyoruz.

```{.python .input}
# `num_channels`: mevcut kanal sayısı
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # Bu, önceki yoğun bloktaki çıktı kanallarının sayısıdır.
    num_channels += num_convs * growth_rate
    # Yoğun bloklar arasına kanal sayısını yarıya indiren bir geçiş katmanı 
    # eklenir
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: mevcut kanal sayısı
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # Bu, önceki yoğun bloktaki çıktı kanallarının sayısıdır.
    num_channels += num_convs * growth_rate
    # Yoğun bloklar arasına kanal sayısını yarıya indiren bir geçiş katmanı 
    # eklenir
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: mevcut kanal sayısı
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # Bu, önceki yoğun bloktaki çıktı kanallarının sayısıdır.
        num_channels += num_convs * growth_rate
        # Yoğun bloklar arasına kanal sayısını yarıya indiren bir geçiş katmanı eklenir
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

ResNet'e benzer şekilde, çıktıyı üretmek için global bir ortaklama katmanı ile tam bağlı bir katman bağlanır.

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## [**Eğitim**]

Burada daha derin bir ağ kullandığımızdan, bu bölümde, hesaplamaları basitleştirmek için girdi yüksekliğini ve genişliğini 224'ten 96'ya düşüreceğiz.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Özet

* DenseNet, girdilerin ve çıktıların birlikte toplandığı ResNet'in aksine, çapraz katman bağlantıları bağlanımda, kanal boyutundaki girdi ve çıktıları bitiştirir.
* DenseNet'i oluşturan ana bileşenler yoğun bloklar ve geçiş katmanlarıdır.
* Kanal sayısını tekrar küçülten geçiş katmanları ekleyerek ağı oluştururken boyutsallığı kontrol altında tutmamız gerekir.

## Alıştırmalar

1. Neden geçiş katmanında maksimum ortaklama yerine ortalama ortaklama kullanıyoruz?
1. DenseNet makalesinde belirtilen avantajlardan biri, model parametrelerinin ResNet'ten daha küçük olmasıdır. Bu neden burada geçerlidir?
1. DenseNet'in eleştirildiği bir sorun, yüksek bellek tüketimidir.
    1. Bu gerçekten doğru mudur? Gerçek GPU bellek tüketimini görmek için girdi şeklini $224\times 224$ olarak değiştirmeye çalışın.
    1. Bellek tüketimini azaltmanın alternatif bir yolunu düşünebiliyor musunuz? Çerçeveyi nasıl değiştirmeniz gerekecektir?
1. DenseNet makalesi :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` Tablo 1'de sunulan çeşitli DenseNet sürümlerini uygulayın.
1. DenseNet fikrini uygulayarak MLP tabanlı bir model tasarlayın. :numref:`sec_kaggle_house` içindeki konut fiyatı tahmini çalışmasına uygulayın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/331)
:end_tab:
