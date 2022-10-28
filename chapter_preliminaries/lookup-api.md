# Belgeler (Dökümantasyon)
:begin_tab:`mxnet`
Bu kitabın uzunluğundaki kısıtlamalar nedeniyle, her bir MXNet işlevini ve sınıfını tanıtamayız (ve muhtemelen bizim yapmamızı siz de istemezsiniz). API (Application Programming Interface - Uygulama Programlama Arayüzü) belgeleri ve ek öğreticiler (tutorial) ve örnekler, kitabın ötesinde pek çok belge sağlar. Bu bölümde size MXNet API'sini keşfetmeniz için biraz rehberlik sunuyoruz.
:end_tab:

:begin_tab:`pytorch`
Bu kitabın uzunluğundaki kısıtlamalar nedeniyle, her bir PyTorch işlevini ve sınıfını tanıtamayız (ve muhtemelen bizim yapmamızı siz de istemezsiniz). API (Application Programming Interface - Uygulama Programlama Arayüzü) belgeleri ve ek öğreticiler (tutorial) ve örnekler kitabın ötesinde pek çok belge sağlar. Bu bölümde size PyTorch API'sini keşfetmeniz için biraz rehberlik sunuyoruz.
:end_tab:

:begin_tab:`tensorflow`
Bu kitabın uzunluğundaki kısıtlamalar nedeniyle, her bir TensorFlow işlevini ve sınıfını tanıtamayız (ve muhtemelen bizim yapmamızı siz de istemezsiniz). API (Application Programming Interface - Uygulama Programlama Arayüzü) belgeleri ve ek öğreticiler (tutorial) ve örnekler kitabın ötesinde pek çok belge sağlar. Bu bölümde size TensorFlow API'sini keşfetmeniz için biraz rehberlik sunuyoruz.
:end_tab:

## Bir Modüldeki Tüm İşlevleri ve Sınıfları Bulma

Bir modülde hangi fonksiyonların ve sınıfların çağrılabileceğini bilmek için `dir` fonksiyonunu çağırırız. Örneğin, (**rastgele sayılar oluşturmak için modüldeki tüm özellikleri sorgulayabiliriz**):

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

Genel olarak, `__` (Python'daki özel nesneler) ile başlayan ve biten işlevleri veya tek bir `_` ile başlayan işlevleri (genellikle dahili işlevler) yok sayabiliriz. Kalan işlev veya özellik adlarına bağlı olarak bu modülün tekdüze dağılım (`uniform`), normal dağılım (`normal`) ve çok terimli dağılımdan (`multinomial`) örnekleme dahil, bu modülün rastgele sayılar oluşturmak için çeşitli yöntemler sunduğunu tahmin edebiliriz.

## Belli İşlevlerin ve Sınıfların Kullanımını Bulma

Belirli bir işlevin veya sınıfın nasıl kullanılacağına ilişkin daha özel talimatlar için `help` (yardım) işlevini çağırabiliriz. Örnek olarak, [**tensörlerin `ones` işlevi için kullanım talimatlarını**] inceleyelim. 

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

Dökümantasyondan, `ones` işlevinin belirtilen şekle sahip yeni bir tensör oluşturduğunu ve tüm öğeleri 1 değerine ayarladığını görebiliriz. Mümkün oldukça, yorumunuzu onaylamak için (**hızlı bir test**) yapmalısınız:

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

Jupyter not defterinde, belgeyi başka bir pencerede görüntülemek için `?` kullanabiliriz. Örneğin, `list?`, `help(list)` ile neredeyse aynı olan içerik üretecek ve onu yeni bir tarayıcı penceresinde görüntüleyecektir. Ek olarak, `list??` gibi iki soru işareti kullanırsak, işlevi uygulayan Python kodu da görüntülenecektir.


## Özet

* Resmi belgeler, bu kitabın dışında pek çok açıklama ve örnek sağlar.
* Jupyter not defterlerinde `dir` ve `help` işlevlerini veya `?` ve `??` işlevlerini çağırarak bir API'nin kullanımına ilişkin belgelere bakabiliriz.


## Alıştırmalar

1. Derin öğrenme çerçevesindeki herhangi bir işlev veya sınıf için belgelere (dökümantasyon) bakın. Belgeleri çerçevenin resmi web sitesinde de bulabilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/199)
:end_tab:
