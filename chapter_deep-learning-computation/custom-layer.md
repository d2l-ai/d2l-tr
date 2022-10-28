# Özelleştirilmiş Katmanlar

Derin öğrenmenin başarısının ardındaki faktörlerden biri, çok çeşitli görevlere uygun mimariler tasarlamak için yaratıcı yollarla oluşturulabilen çok çeşitli katmanların mevcut olmasıdır. Örneğin, araştırmacılar özellikle imgelerle baş etmek, metin, dizili veriler üzerinde döngü yapmak ve dinamik programlama yapmak için katmanlar icat ettiler. Er ya da geç, derin öğrenme çerçevesinde henüz var olmayan bir katmanla karşılaşacaksınız (veya onu icat edeceksiniz). Özel bir katman oluşturmanız gerekebilir. Bu bölümde size bunu nasıl yapacağınızı gösteriyoruz.

## (**Paramatresiz Katmanlar**)

Başlangıç olarak, kendi parametresi olmayan özelleştirilmiş bir katman oluşturalım. Bloğu tanıttığımızı bölümü hatırlarsanız, :numref:`sec_model_construction` içindeki, burası size tanıdık gelecektir. Aşağıdaki `CenteredLayer` sınıfı, girdiden ortalamayı çıkarır. Bunu inşa etmek için, temel katman sınıfından kalıtımla üretmemiz ve ileri yayma işlevini uygulamamız gerekir.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

Katmanımızın amaçlandığı gibi çalıştığını, ona biraz veri besleyerek doğrulayalım.

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

[**Artık katmanımızı daha karmaşık modeller oluşturmada bir bileşen olarak kullanabiliriz.**]

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

Ekstra bir makulluk kontrolü olarak, ağa rastgele veri gönderebilir ve ortalamanın gerçekte 0 olup olmadığına bakabiliriz. Kayan virgüllü sayılarla uğraştığımız için, nicemlemeden dolayı çok küçük sıfır olmayan bir sayı görebiliriz.

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## [**Parametreli Katmanlar**]

Artık basit katmanları nasıl tanımlayacağımızı bildiğimize göre, eğitim yoluyla ayarlanabilen parametrelerle katmanları tanımlamaya geçelim. Bazı temel idari işlevleri sağlayan parametreler oluşturmak için yerleşik işlevleri kullanabiliriz. Özellikle erişim, ilkleme, paylaşma, modeli kaydetme ve yükleme parametrelerini yönetirler. Bu şekilde, diğer faydaların yanı sıra, her özel katman için özel serileştirme (serialization) rutinleri yazmamız gerekmeyecek.

Şimdi tam bağlı katman sürümümüzü uygulayalım. Bu katmanın iki parametreye ihtiyaç duyduğunu hatırlayınız, biri ağırlığı ve diğeri ek girdiyi temsil etmek için. Bu uygulamada, varsayılan olarak ReLU etkinleştirmesini kullanıyoruz. Bu katman, sırasıyla girdilerin ve çıktıların sayısını gösteren `in_units` ve `units` girdi argümanlarının girilmesini gerektirir.

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow`
Daha sonra, `MyDense` sınıfını ilkliyoruz ve model parametrelerine erişiyoruz.
:end_tab:

:begin_tab:`pytorch`
Daha sonra, `MyLinear` sınıfını ilkliyoruz ve model parametrelerine erişiyoruz.
:end_tab:

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

[**Özel kesim katmanları kullanarak doğrudan ileri yayma hesaplamaları yapabiliriz.**]

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

(**Özelleştirilmiş kesim katmanlar kullanarak da modeller oluşturabiliriz.**) Bir kere ona sahip olduğumuzda, onu tıpkı yerleşik tam bağlı katman gibi kullanabiliriz.

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## Özet

* Temel katman sınıfı üzerinden özel kesim katmanlar tasarlayabiliriz. Bu, kütüphanedeki mevcut katmanlardan farklı davranan yeni esnek katmanlar tanımlamamıza olanak tanır.
* Tanımlandıktan sonra, özel kesim katmanlar keyfi bağlamlarda ve mimarilerde çağrılabilir.
* Katmanlar, yerleşik işlevler aracılığıyla yaratılabilen yerel parametrelere sahip olabilirler.


## Alıştırmalar

1. Bir girdi alan ve bir tensör indirgemesi hesaplayan bir katman tasarlayınız, yani $y_k = \sum_{i, j} W_{ijk} x_i x_j$ döndürsün.
1. Verilerin Fourier katsayılarının ilk baştaki yarısını döndüren bir katman tasarlayınız.


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/279)
:end_tab:
