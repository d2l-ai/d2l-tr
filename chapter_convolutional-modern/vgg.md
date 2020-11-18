# Blokları Kullanan Ağlar (VGG)
:label:`sec_vgg`

AlexNet derin CNN'lerin iyi sonuçlar elde edebileceğine dair ampirik kanıtlar sunarken, yeni ağlar tasarlamada sonraki araştırmacılara rehberlik etmek için genel bir şablon sağlamadı. Aşağıdaki bölümlerde, derin ağları tasarlamak için yaygın olarak kullanılan çeşitli sezgisel kavramları tanıtacağız.

Bu alandaki ilerleme, mühendislerin transistörlerin yerleştirilmesinden mantıksal elemanlara mantık bloklarına geçtiği talaş tasarımında yansıtmaktadır. Benzer şekilde, sinir ağı mimarilerinin tasarımı, araştırmacıların bireysel nöronlar açısından düşünmekten bütün katmanlara ve şimdi bloklara, katmanların kalıplarını tekrarlayarak hareket etmeleriyle giderek daha soyut bir hale gelmişti.

Blokları kullanma fikri ilk olarak Oxford Üniversitesi'ndeki [Görsel Geometri Grubu](http://www.robots.ox.ac.uk/~vgg/) (VGG), kendi adını taşıyan *VGG* ağında ortaya çıkmıştır. Bu tekrarlanan yapıları, döngüler ve alt programlar kullanarak herhangi bir modern derin öğrenme çerçevesi ile kodda uygulamak kolaydır.

## VGG Blokları

Klasik CNN'lerin temel yapı taşı aşağıdakilerin bir dizisidir: (i) çözünürlüğü korumak için dolgulu bir kıvrımsal katman, (ii) ReLU gibi bir doğrusal olmayan, (iii) maksimum havuzlama katmanı gibi bir havuzlama katmanı. Bir VGG bloğu, uzamsal altörnekleme için bir maksimum havuzlama katmanı izleyen bir kıvrımsal katman dizisinden oluşur. Orijinal VGG kağıdında :cite:`Simonyan.Zisserman.2014`, yazarlar $3\times3$ çekirdeklerle (yükseklik ve genişlik tutarak) ve $2 \times 2$ maksimum 2 adımla havuzlama (her bloktan sonra çözünürlüğü yarıya indirerek) ile kıvrımlar kullandılar. Aşağıdaki kodda, bir VGG bloğu uygulamak için `vgg_block` adlı bir işlev tanımlıyoruz. İşlev, evrimsel katmanların sayısına karşılık gelen iki argüman alır `num_convs` ve çıkış kanalı sayısı `num_channels`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## VGG Ağı

AlexNet ve LeNet gibi, VGG Ağı iki kısma bölünebilir: Birincisi çoğunlukla konvolüsyonel ve havuzlama katmanlarından ve ikincisi tam bağlı katmanlardan oluşur. Bu tasvir edilmiştir :numref:`fig_vgg`.

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

Ağın kıvrımsal kısmı, :numref:`fig_vgg`'ten (`vgg_block` işlevinde de tanımlanmıştır) birkaç VGG bloğu arkaya bağlar. Aşağıdaki değişken `conv_arch`, her biri iki değer içeren bir dizinin (blok başına bir) listesinden oluşur: evrimsel katmanların sayısı ve çıkış kanallarının sayısı, tam olarak `vgg_block` işlevini çağırmak için gerekli argümanlardır. VGG ağının tam bağlı kısmı AlexNet'te kapsananla aynıdır.

Orijinal VGG ağı, ilk ikisinin her biri bir evrimsel tabakaya sahip olduğu ve ikincisi üçünün her biri iki konvolüsyonel katman içerdiği 5 evrimsel blok vardı. İlk blokta 64 çıkış kanalı vardır ve sonraki her blok, bu sayı 512'ye ulaşıncaya kadar çıkış kanalı sayısını iki katına çıkarır. Bu ağ 8 evrimsel katman ve 3 tam bağlı katman kullandığından, genellikle VGG-11 olarak adlandırılır.

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

Aşağıdaki kod VGG-11'i uygular. Bu, `conv_arch` üzerinde bir for-loop yürütme basit bir konudur.

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    # The convolutional part
    conv_blks=[]
    in_channels=1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

Daha sonra, her katmanın çıkış şeklini gözlemlemek için 224 yükseklik ve genişliğe sahip tek kanallı bir veri örneği oluşturacağız.

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

Gördüğünüz gibi, her blokta yükseklik ve genişliği yarıya indiriyoruz, nihayet ağın tam bağlı kısmı tarafından işleme için temsilleri düzleştirmeden önce 7'lik bir yüksekliğe ve genişliğe ulaşıyoruz.

## Eğitim

VGG-11 AlexNet'ten daha hesaplamalı olarak daha ağır olduğundan, daha az sayıda kanala sahip bir ağ oluşturuyoruz. Bu, Moda-MNIST üzerinde eğitim için fazlasıyla yeterli.

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
net = lambda: vgg(small_conv_arch)
```

Biraz daha büyük bir öğrenme hızı kullanmanın yanı sıra, model eğitim süreci :numref:`sec_alexnet`'teki AlexNet'e benzer.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## Özet

* VGG-11 yeniden kullanılabilir evrimsel blokları kullanarak bir ağ oluşturur. Farklı VGG modelleri, her bloktaki kıvrımsal katman ve çıkış kanallarının sayısındaki farklılıklarla tanımlanabilir.
* Blokların kullanımı, ağ tanımının çok kompakt temsillerine yol açar. Karmaşık ağların verimli tasarımını sağlar.
* Simonyan ve Ziserman, VGG makalelerinde çeşitli mimarilerle deneyler yaptılar. Özellikle, derin ve dar kıvrımların (yani $3 \times 3$) birkaç katmanının daha az geniş kıvrım katmanından daha etkili olduğunu buldular.

## Egzersizler

1. Katmanların boyutlarını yazdırırken 11 yerine sadece 8 sonuç gördük. Kalan 3 katmanlı bilgi nereye gitti?
1. AlexNet ile karşılaştırıldığında, VGG hesaplama açısından çok daha yavaştır ve ayrıca daha fazla GPU belleğine ihtiyaç duyar. Bunun nedenlerini analiz edin.
1. Moda-MNIST içindeki görüntülerin yüksekliğini ve genişliğini 224'ten 96'ya değiştirmeyi deneyin. Bunun deneyler üzerinde ne etkisi var?
1. VGG-16 veya VGG-19 gibi diğer yaygın modelleri oluşturmak için VGG kağıt :cite:`Simonyan.Zisserman.2014` Tablo 1'e bakın.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab: