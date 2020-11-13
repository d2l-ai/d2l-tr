# Paralel Birleştirici Ağlar (GoogLeNet)
:label:`sec_googlenet`

2014'te, *GoogLeNet* ImageNet Challenge'ı kazandı ve NiN'in güçlü yanlarını ve :cite:`Szegedy.Liu.Jia.ea.2015`'ün tekrarlanan bloklarının paradigmalarını birleştiren bir yapı önerdi. Kağıdın odak noktası, hangi büyüklükteki evrim çekirdeklerinin en iyi olduğu sorusunu ele almaktı. Sonuçta, önceki popüler ağlar $1 \times 1$ gibi küçük ve $11 \times 11$ kadar büyük seçimler kullandı. Bu yazıda bir öngörü, bazen çeşitli boyutlarda çekirdeklerin bir kombinasyonunu kullanmanın avantajlı olabileceğiydi. Bu bölümde, orijinal modelin biraz basitleştirilmiş bir versiyonunu sunan GoogLeNet'i tanıtacağız: eğitimi stabilize etmek için eklenen ancak artık daha iyi eğitim algoritmaları ile gereksiz olan birkaç geçici özelliği atlıyoruz.

## Inception Blokları

GoogLeNet'teki temel kıvrımsal bloğa, viral bir memeyi başlatan *Inception* (“Daha derine gitmemiz gerekiyor”) filminden bir alıntı nedeniyle adlandırılmış bir *Inception bloğu* denir.

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

:numref:`fig_inception`'te gösterildiği gibi, başlangıç bloğu dört paralel yoldan oluşur. İlk üç yol, farklı uzamsal boyutlardan bilgi ayıklamak için $1\times 1$, $3\times 3$ ve $5\times 5$ pencere boyutlarına sahip evrimsel katmanlar kullanır. Orta iki yol, kanalların sayısını azaltmak ve modelin karmaşıklığını azaltmak için giriş üzerinde bir $1\times 1$ evrişim gerçekleştirir. Dördüncü yol, $3\times 3$ en fazla havuzlama katmanı kullanır ve ardından kanal sayısını değiştirmek için $1\times 1$ kıvrımsal katman izler. Dört yol, giriş ve çıkışa aynı yükseklik ve genişlik vermek için uygun dolgu kullanır. Son olarak, her yol boyunca çıkışlar kanal boyutu boyunca birleştirilir ve bloğun çıktısını oluşturur. Inception bloğunun yaygın olarak ayarlanmış hiperparametreleri, katman başına çıkış kanallarının sayısıdır.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

Bu ağın neden bu kadar iyi çalıştığına dair sezgi kazanmak için filtrelerin kombinasyonunu göz önünde bulundurun. Görüntüyü çeşitli filtre boyutlarında keşfediyorlar. Bu, farklı boyutlardaki ayrıntıların farklı boyutlardaki filtrelerle verimli bir şekilde tanınabileceği anlamına gelir. Aynı zamanda, farklı filtreler için farklı miktarlarda parametre tahsis edebiliriz.

## GoogLeNet Modeli

:numref:`fig_inception_full`'te gösterildiği gibi, GoogLeNet tahminlerini oluşturmak için toplam 9 başlangıç bloğu ve küresel ortalama havuzdan oluşan bir yığın kullanır. Başlangıç blokları arasında maksimum havuzlama boyutsallığı azaltır. İlk modül AlexNet ve LeNet'e benzer. Blokların yığını VGG'den devralınır ve küresel ortalama havuzlama sonunda tam bağlı katman yığınını önler.

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

Artık GoogLeNet'i parça parça uygulayabiliriz. İlk modül 64 kanallı $7\times 7$ evrimsel bir katman kullanır.

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

İkinci modül iki evrimsel katman kullanır: Birincisi, 64 kanallı $1\times 1$ evrimsel tabaka, daha sonra kanal sayısını üçe katlayan bir $3\times 3$ evrimsel tabaka. Bu, Inception bloğundaki ikinci yola karşılık gelir.

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Üçüncü modül, iki komple Inception bloklarını seri olarak bağlar. İlk Inception bloğunun çıkış kanallarının sayısı $64+128+32+32=256$'dir ve dört yol arasındaki çıkış kanalı oranı $64:128:32:32=2:4:1:1$'dir. İkinci ve üçüncü yollar önce giriş kanallarının sayısını sırasıyla $96/192=1/2$ ve $16/192=1/12$'e düşürür ve daha sonra ikinci konvolüsyonel tabakayı bağlar. İkinci Inception bloğunun çıkış kanallarının sayısı $128+192+96+64=480$'e yükseltilir ve dört yol arasındaki çıkış kanalı oranı $128:192:96:64 = 4:6:3:2$'dur. İkinci ve üçüncü yollar önce giriş kanalı sayısını sırasıyla $128/256=1/2$ ve $32/256=1/8$'ya düşürür.

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Dördüncü modül daha karmaşıktır. Beş Inception bloğu seri olarak bağlar ve sırasıyla $192+208+48+64=512$, $160+224+64+64=512$, $128+256+64+64=512$, $112+288+64+64=528$ ve $256+320+128+128=832$ çıkış kanallarına sahiptir. Bu yollara atanan kanalların sayısı üçüncü modülde benzerdir: $3\times 3$ evrimsel tabaka ile ikinci yol, sadece $1\times 1$ evrimsel tabaka ile ilk yol izledi kanal sayısı, $5\times 5$ evrimsel tabaka ile üçüncü yol ve $3\times 3$ maksimum havuzlama katmanı ile dördüncü yol. İkinci ve üçüncü yollar önce orana göre kanal sayısını azaltacaktır. Bu oranlar farklı Inception bloklarında biraz farklıdır.

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Beşinci modül, $256+320+128+128=832$ ve $384+384+128+128=1024$ çıkış kanallarına sahip iki Inception bloğu içerir. Her yola atanan kanal sayısı, üçüncü ve dördüncü modüllerdeki kanallarla aynıdır, ancak belirli değerlerde farklılık gösterir. Beşinci bloğun çıkış katmanı tarafından takip edildiğine dikkat edilmelidir. Bu blok, her kanalın yüksekliğini ve genişliğini NiN'de olduğu gibi 1'e değiştirmek için küresel ortalama havuzlama katmanını kullanır. Son olarak, çıktıyı iki boyutlu bir diziye dönüştürüyoruz ve ardından çıkış sayısı etiket sınıflarının sayısı olan tam bağlı bir katman oluşturuyoruz.

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveMaxPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

GoogLeNet modeli hesaplama açısından karmaşıktır, bu nedenle VG'deki gibi kanal sayısını değiştirmek kolay değildir. Moda-MNIST üzerinde makul bir eğitim süresine sahip olmak için giriş yüksekliğini ve genişliğini 224'ten 96'ya düşürüyoruz. Bu, hesaplamayı basitleştirir. Çeşitli modüller arasındaki çıkış şeklindeki değişiklikler aşağıda gösterilmiştir.

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## Eğitim

Daha önce olduğu gibi modelimizi Moda-MNIST veri kümesini kullanarak eğitiyoruz. Eğitim prosedürünü çağırmadan önce $96 \times 96$ piksel çözünürlüğüne dönüştürüyoruz.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## Özet

* Inception bloğu, dört yolu olan bir alt ağa eşdeğerdir. Farklı pencere şekillerinin ve maksimum havuzlama katmanlarının evrimsel katmanları aracılığıyla bilgileri paralel olarak ayıklar. $1 \times 1$ kıvrım piksel başına bir düzeyde kanal boyutsallığını azaltır. Maksimum havuzlama çözünürlüğü azaltır.
* GoogLeNet, çok iyi tasarlanmış Inception bloklarını seri olarak diğer katmanlarla bağlar. Inception bloğunda atanan kanal sayısının oranı, ImageNet veri kümesi üzerinde çok sayıda deney yoluyla elde edilir.
* GoogLeNet ve başarılı sürümleri, ImageNet'teki en verimli modellerden biriydi ve daha düşük hesaplama karmaşıklığı ile benzer test doğruluğunu sağladı.

## Egzersizler

1. GoogLeNet'in birkaç yinelemesi vardır. Uygulamaya ve çalıştırmaya çalışın. Bazıları aşağıdakileri içerir:
    * :numref:`sec_batch_norm`'te daha sonra açıklandığı gibi bir toplu normalleştirme katmanı :cite:`Ioffe.Szegedy.2015` ekleyin.
    * :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` Inception bloğu için ayarlamalar yapın.
    * Model düzenlemesi için etiket yumuşatma kullanın :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
    * :numref:`sec_resnet`'te daha sonra açıklandığı gibi artık bağlantı :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`'e dahil edin.
1. GoogLeNet'in çalışması için minimum görüntü boyutu nedir?
1. AlexNet, VGG ve NiN model parametre boyutlarını GoogLeNet ile karşılaştırın. İkinci iki ağ mimarisi, model parametre boyutunu önemli ölçüde nasıl azaltır?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:
