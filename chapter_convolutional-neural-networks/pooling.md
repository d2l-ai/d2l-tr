# Havuzlama
:label:`sec_pooling`

Çoğu zaman, görüntüleri işledikçe, gizli temsillerimizin mekansal çözünürlüğünü yavaş yavaş azaltmak istiyoruz, böylece ağda ne kadar yüksek olursak, her gizli düğümün hassas olduğu alıcı alan (girişte) o kadar büyük olur.

Genellikle nihai görevimiz görüntü hakkında küresel bir soru sorar, örn. bir kedi içeriyor mu?* Bu nedenle tipik olarak son katmanımızın birimleri tüm girişe karşı hassas olmalıdır. Bilgiyi kademeli olarak toplayarak, daha kaba ve daha kaba haritalar üreterek, nihai olarak küresel bir gösterimi öğrenme amacını gerçekleştiririz ve evrimsel katmanların tüm avantajlarını işleme ara katmanlarında tutuyoruz.

Dahası, kenarlar gibi (:numref:`sec_conv_layer`'te tartışıldığı gibi) alt düzey özellikleri tespit ederken, genellikle temsillerimizin çeviri için değişmez olmasını isteriz. Örneğin, `X` görüntüsünü siyah beyaz arasında keskin bir çizimle alıp tüm görüntüyü bir pikselle sağa kaydırırsak, yani `Z[i, j] = X[i, j + 1]` yeni görüntünün çıkışı çok farklı olabilir. Kenar bir piksel ile kaydırılmış olacaktır. Gerçekte, nesneler neredeyse hiç aynı yerde meydana gelmez. Aslında, bir tripod ve sabit bir nesneyle bile, deklanşörün hareketi nedeniyle kameranın titreşimi her şeyi bir pikselle değiştirebilir (üst düzey kameralar bu sorunu gidermek için özel özelliklerle yüklenir).

Bu bölümde, evrimsel katmanların konuma duyarlılığını azaltmak ve mekansal olarak altörnekleme gösterimleri için ikili amaçlara hizmet eden *havuzlama katmanları* tanıtılmaktadır.

## Maksimum Havuzlama ve Ortalama Havuzlama

* Havuzlama operatörleri, kıvrımsal katmanlar gibi, sabit şekilli pencerenin (bazen *havuz penceresi olarak da bilinir) geçtiği her konum için tek bir çıkış hesaplayarak, adımına göre girdideki tüm bölgelere kaydırılan sabit şekilli bir pencereden oluşur. Bununla birlikte, evrimsel katmandaki giriş ve çekirdeklerin çapraz korelasyon hesaplamasının aksine, havuzlama katmanı hiçbir parametre içermez (*çekirdek* yoktur). Bunun yerine, havuzlama operatörleri deterministtir ve genellikle havuzlama penceresindeki öğelerin maksimum veya ortalama değerini hesaplar. Bu işlemler sırasıyla*maksimum havuz* (*kısaca havuz*) ve *ortalama havuz* olarak adlandırılır.

Her iki durumda da, çapraz korelasyon operatöründe olduğu gibi, havuzlama penceresinin giriş tensörünün sol üstünden başlayarak giriş tensörünün soldan sağa ve yukarıdan aşağıya doğru kayması olarak düşünebiliriz. Havuzlama penceresinin vurduğu her konumda, maksimum veya ortalama havuzlamanın kullanılmasına bağlı olarak, pencerede giriş alt tensörünün maksimum veya ortalama değerini hesaplar.

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling`'teki çıkış tensör 2 yüksekliğe ve 2 genişliğe sahiptir. Dört öğe, her havuzlama penceresindeki maksimum değerden türetilir:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Havuz penceresi şeklindeki $p \times q$ havuzlama katmanına $p \times q$ havuzlama katmanı denir. Havuzlama işlemi $p \times q$ havuzlama olarak adlandırılır.

Bize bu bölümün başında belirtilen nesne kenar algılama örneğine dönelim. Şimdi $2\times 2$ maksimum havuzlama için giriş olarak konvolusyonel tabakanın çıktısını kullanacağız. Konvolüsyonel katman girdisini `X` ve havuzlama katmanı çıktısını `Y` olarak ayarlayın. `X[i, j]` ve `X[i, j + 1]` değerlerinin farklı olup olmadığı veya `X[i, j + 1]` ve `X[i, j + 2]` farklı olup olmadığı, havuzlama katmanı her zaman `Y[i, j] = 1` çıktılarını verir. Yani, $2\times 2$ maksimum havuzlama katmanını kullanarak, evrimsel katman tarafından tanınan desenin yükseklik veya genişlik olarak birden fazla eleman hareket edip etmediğini hala tespit edebiliriz.

Aşağıdaki kodda, `pool2d` işlevinde havuzlama katmanının ileri yayılmasını uyguluyoruz. Bu işlev :numref:`sec_conv_layer`'teki `corr2d` işlevine benzer. Ancak, burada çekirdeğimiz yok, çıktıyı girdideki her bölgenin maksimum veya ortalaması olarak hesaplıyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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
#@tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

İki boyutlu maksimum havuzlama tabakasının çıktısını doğrulamak için :numref:`fig_pooling`'te giriş tensörünü `X`'i inşa edebiliriz.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Ayrıca, ortalama havuzlama katmanını deneriz.

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## Dolgu ve Stride

Konvolusyonel katmanlarda olduğu gibi, havuzlama katmanları da çıktı şeklini değiştirebilir. Ve daha önce olduğu gibi, girişi doldurarak ve adım adımını ayarlayarak istenen çıkış şeklini elde etmek için işlemi değiştirebiliriz. Derin öğrenme çerçevesinden yerleşik iki boyutlu maksimum havuzlama katmanı aracılığıyla havuzlama katmanlarında dolgu ve adımların kullanımını gösterebiliriz. İlk olarak bir giriş tensörü inşa ediyoruz `X` şekli dört boyuta sahip, burada örneklerin sayısı ve kanal sayısı her ikisi de 1.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Varsayılan olarak, çerçevenin yerleşik sınıfındaki örnekteki adım ve havuzlama penceresi aynı şekle sahiptir. Aşağıda, `(3, 3)` şeklindeki bir havuzlama penceresi kullanıyoruz, bu nedenle varsayılan olarak `(3, 3)`'lü bir adım şekli alıyoruz.

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

Ayamak ve dolgu manuel olarak belirtilebilir.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same',
                                   strides=2)
pool2d(X)
```

Tabii ki, keyfi bir dikdörtgen havuzlama penceresi belirleyebilir ve sırasıyla yükseklik ve genişlik için dolgu ve adım belirtebiliriz.

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='same',
                                   strides=(2, 3))
pool2d(X)
```

## Birden Çok Kanal

Çok kanallı giriş verilerini işlerken, havuzlama katmanı, girdileri bir kıvrımsal katmanda olduğu gibi kanallar üzerinden toplamak yerine her giriş kanalını ayrı ayrı havuzlar. Bu, havuzlama katmanının çıkış kanallarının sayısının giriş kanalı sayısıyla aynı olduğu anlamına gelir. Aşağıda, 2 kanallı bir giriş oluşturmak için kanal boyutundaki `X` ve `X + 1` tensörleri birleştiririz.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.stack([X, X+1], 0), (1, 2, 4, 4))
```

Gördüğümüz gibi, çıkış kanallarının sayısı havuzlamadan sonra hala 2'dir.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(3, padding='same', strides=2)
pool2d(X)
```

## Özet

* Havuz penceresinde giriş öğelerini alarak, maksimum havuzlama işlemi çıktı olarak maksimum değeri atar ve ortalama havuzlama işlemi ortalama değeri çıktı olarak atar.
* Bir havuzlama tabakasının en önemli avantajlarından biri, konvolusyonel tabakanın konumuna aşırı duyarlılığını hafifletmektir.
* Havuz katmanı için dolgu ve adım belirtebiliriz.
* Uzamsal boyutları (örn. genişlik ve yükseklik) azaltmak için 1'den büyük bir adımla birlikte maksimum havuzlama kullanılabilir.
* Havuzlama katmanının çıkış kanalı sayısı, giriş kanallarının sayısıyla aynıdır.

## Egzersizler

1. Bir evrişim tabakasının özel bir durumu olarak ortalama havuzlama uygulayabilir misiniz? Eğer öyleyse, yap.
1. Bir evrişim tabakasının özel bir durumu olarak maksimum havuzlama uygulayabilir misiniz? Eğer öyleyse, yap.
1. Havuz katmanının hesaplama maliyeti nedir? Havuz katmanına giriş boyutu $c\times h\times w$ olduğunu varsayalım, havuz penceresi $p_h\times p_w$ bir dolgu $(p_h, p_w)$ ve bir adım $(s_h, s_w)$ bir şekle sahiptir.
1. Neden maksimum havuzlama ve ortalama havuzlama farklı çalışmasını bekliyorsunuz?
1. Ayrı bir minimum havuzlama katmanına ihtiyacımız var mı? Başka bir işlemle değiştirebilir misin?
1. Ortalama ve maksimum havuzlama arasında düşünebileceğiniz başka bir işlem var mı (ipucu: softmax'i geri çağırma)? Neden bu kadar popüler olmasın?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
