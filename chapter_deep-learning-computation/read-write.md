# Dosya Okuma/Yazma

Şu ana kadar verilerin nasıl işleneceğini ve derin öğrenme modellerinin nasıl oluşturulacağını, eğitileceğini ve test edileceğini tartıştık. Bununla birlikte, bir noktada, öğrenilmiş modellerden yeterince mutlu olacağımızı, öyle ki sonuçları daha sonra çeşitli bağlamlarda kullanmak için saklamak isteyeceğimiz umuyoruz (belki de konuşlandırmada tahminlerde bulunmak için). Ek olarak, uzun bir eğitim sürecini çalıştırırken, sunucumuzun güç kablosuna takılırsak birkaç günlük hesaplamayı kaybetmememiz için en iyi mesleki uygulama, periyodik olarak ara sonuçları kaydetmektir (kontrol noktası belirleme). Bu nedenle, hem bireysel ağırlık vektörlerini hem de tüm modelleri nasıl yükleyeceğimizi ve depolayacağımızı öğrenmenin zamanı geldi. Bu bölüm her iki sorunu da irdelemektedir.

## (**Tensörleri Yükleme ve Kaydetme**)

Tek tensörler için, sırasıyla okumak ve yazmak için `load` ve `save` işlevlerini doğrudan çağırabiliriz. Her iki işleve de bir ad girmemiz gerekir ve `save` işlevi için kaydedilecek değişkeni de girdi olarak girmemiz gerekir.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save('x-file.npy', x)
```

Artık verileri kayıtlı dosyadan belleğe geri okuyabiliriz.

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

[**Tensörlerin bir listesini kaydedebilir ve bunları belleğe geri okuyabiliriz.**]

```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

[**Dizgilerden (string) tensörlere eşleşen bir sözlük bile yazabilir ve okuyabiliriz.**] Bu, bir modeldeki tüm ağırlıkları okumak veya yazmak istediğimizde kullanışlıdır.

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Model Parametrelerini Yükleme ve Kayıt Etme**]

Bireysel ağırlık vektörlerini (veya diğer tensörleri) kaydetmek faydalıdır, ancak tüm modeli kaydetmek (ve daha sonra yüklemek) istiyorsak bu çok bezdirici hale gelir. Sonuçta, her yere dağıtılmış yüzlerce parametre grubumuz olabilir. Bu nedenle derin öğrenme çerçevesi, tüm ağları yüklemek ve kaydetmek için yerleşik işlevsellikler sağlar. Farkında olunması gereken önemli bir ayrıntı, bunun tüm modeli değil, model *parametrelerini* kaydetmesidir. Örneğin, 3 katmanlı bir MLP'ye sahipsek, mimariyi ayrıca belirtmemiz gerekir. Bunun nedeni, modellerin kendilerinin keyfi kodlar içerebilmeleri, dolayısıyla doğal olarak serileştirilememeleridir. Bu nedenle, bir modeli eski haline getirmek için, kod içinde mimariyi oluşturmamız ve ardından parametreleri diskten yüklememiz gerekir. (**Tanıdık MLP'mizle başlayalım.**)

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

Daha sonra [**modelin parametrelerini**] `mlp.params` adıyla [**bir dosya olarak kaydediyoruz.**]

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

Modeli geri yüklemek için, orijinal MLP modelinin bir klonunu örnekliyoruz. Model parametrelerini rastgele ilklemek yerine, [**dosyada saklanan parametreleri doğrudan okuyoruz.**]

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

Her iki örnek de aynı model parametrelerine sahip olduğundan, aynı `X` girdisinin hesaplamalı sonucu aynı olmalıdır. Bunu doğrulayalım.

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## Özet

* `save` ve `load` fonksiyonları, tensör nesneler için dosya yazma/okuma gerçekleştirmek için kullanılabilir.
* Parametre sözlüğü aracılığıyla bir ağ için tüm parametre kümelerini kaydedebilir ve yükleyebiliriz.
* Mimarinin kaydedilmesi parametreler yerine kodda yapılmalıdır.

## Alıştırmalar

1. Eğitilmiş modelleri farklı bir cihaza konuşlandırmaya gerek olmasa bile, model parametrelerini saklamanın pratik faydaları nelerdir?
1. Bir ağın yalnızca belli bölümlerini farklı bir mimariye sahip bir ağa dahil edilecek şekilde yeniden kullanmak istediğimizi varsayalım. Yeni bir ağdaki eski bir ağdan ilk iki katmanı nasıl kullanmaya başlarsınız?
1. Ağ mimarisini ve parametrelerini kaydetmeye ne dersiniz? Mimariye ne gibi kısıtlamalar uygularsınız?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/327)
:end_tab:
