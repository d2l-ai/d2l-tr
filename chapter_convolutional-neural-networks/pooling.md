# Ortaklama
:label:`sec_pooling`

Çoğu zaman, imgeleri işledikçe, gizli temsillerimizin konumsal çözünürlüğünü yavaş yavaş azaltmak istiyoruz, böylece ağda ne kadar yükseğe çıkarsak, her gizli düğümün hassas olduğu alım alanı (girdide) o kadar büyük olur.

Genellikle esas görevimiz bize imge hakkında küresel bir soru sormaktadır, örn. *bir kedi içeriyor mu?* Bu nedenle tipik olarak son katmanımızın birimleri tüm girdiye karşı hassas olmalıdır. Bilgiyi kademeli olarak toplayarak, daha kaba eşlemeler üreterek, en sonunda küresel bir gösterimi öğrenme amacını gerçekleştiriyoruz ve bunu yaparken evrişimli ara katmanlardaki işlemlerin tüm avantajlarını tutuyoruz.

Dahası, kenarlar gibi (:numref:`sec_conv_layer` içinde tartışıldığına benzer) alt seviye öznitelikleri tespit ederken, genellikle temsillerimizin yer değiştirmelerden etkilenmez olmasını isteriz. Örneğin, siyah beyaz arasında keskin gösterimli bir `X` imgesini alıp tüm imgeyi bir pikselle sağa kaydırırsak, yani `Z[i, j] = X[i, j + 1]`, yeni imgenin çıktısı çok farklı olabilir. Kenar bir piksel ile kaydırılmış olacaktır. Gerçekte, nesneler neredeyse hiç bir zaman aynı yerde olmaz. Aslında, bir tripod ve sabit bir nesneyle bile, deklanşörün hareketi nedeniyle kameranın titreşimi her şeyi bir piksel kaydırabilir (üst düzey kameralar bu sorunu gidermek için özel özelliklerle donatılmıştır).

Bu bölümde, evrişimli katmanların konuma duyarlılığını azaltmak ve gösterimleri uzaysal örnek seyreltmek gibi ikili amaçlara hizmet eden *ortaklama katmanları* tanıtılmaktadır.

## Maksimum Ortaklama ve Ortalama Ortaklama

*Ortaklama* işlemcileri, evrişimli katmanlar gibi, sabit şekilli pencerenin (bazen *ortaklama penceresi* olarak da bilinir) geçtiği her konum için tek bir çıktı hesaplayarak, uzun adımına göre girdideki tüm bölgelere kaydırılan sabit şekilli bir pencereden oluşur. Bununla birlikte, evrişimli katmandaki girdi ve çekirdeklerin çapraz korelasyon hesaplamasının aksine, ortaklama katmanı hiçbir parametre içermez (*çekirdek* yoktur). Bunun yerine, ortaklama uygulayıcıları gerekircidir (determinist) ve genellikle ortaklama penceresindeki öğelerin maksimum veya ortalama değerini hesaplar. Bu işlemler sırasıyla *maksimum ortaklama* (*kısaca ortaklama*) ve *ortalama ortaklama* olarak adlandırılır.

Her iki durumda da, çapraz korelasyon uygulayıcısında olduğu gibi, ortaklama penceresinin girdi tensörünün sol üstünden başlayarak girdi tensörünün soldan sağa ve yukarıdan aşağıya doğru kayması olarak düşünebiliriz. Ortaklama penceresinin vurduğu her konumda, maksimum veya ortalama ortaklamanın kullanılmasına bağlı olarak, pencerede girdi alt tensörünün maksimum veya ortalama değerini hesaplar.

![$2 \times 2$ şeklinde bir ortaklama penceresi ile maksimum ortaklama. Gölgeli kısımlar, ilk çıktı elemanı ve çıktı hesaplaması için kullanılan girdi tensör elemanlarıdır: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling` içindeki çıktı tensör 2'lik yüksekliğe ve 2'lik genişliğe sahiptir. Dört öğe, her ortaklama penceresindeki maksimum değerden türetilir:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

Ortaklama penceresi şeklindeki $p \times q$ ortaklama katmanına $p \times q$ ortaklama katmanı denir. Ortaklama işlemi $p \times q$ ortaklama olarak adlandırılır.

Bu bölümün başında belirtilen nesne kenarı algılama örneğine dönelim. Şimdi $2\times 2$ maksimum ortaklama için girdi olarak evrişimli tabakanın çıktısını kullanacağız. Evrişimli katman girdisini `X` ve ortaklama katmanı çıktısını `Y` olarak düzenleyelim. `X[i, j]` ve `X[i, j + 1]` veya `X[i, j + 1]` ve `X[i, j + 2]` değerleri farklı olsa da olmasa da, ortaklama katmanı her zaman `Y[i, j] = 1` çıktısını verir. Yani, $2\times 2$ maksimum ortaklama katmanını kullanarak, evrişimli katman tarafından tanınan desenin yükseklik veya genişlik olarak birden fazla eleman yine de hareket edip etmediğini tespit edebiliriz.

Aşağıdaki kodda, `pool2d` işlevinde (**ortaklama katmanının ileri yaymasını uyguluyoruz**). Bu işlev :numref:`sec_conv_layer` içindeki `corr2d` işlevine benzer. Ancak, burada çekirdeğimiz yok, çıktıyı girdideki her bölgenin maksimumu veya ortalaması olarak hesaplıyoruz.

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

[**İki boyutlu maksimum ortaklama tabakasının çıktısını doğrulamak için**] :numref:`fig_pooling` içinde girdi tensörü `X`'i inşa ediyoruz.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Ayrıca, (**ortalama ortaklama katmanıyla**) da deney yapalım.

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## [**Dolgu ve Uzun Adım**]

Evrişimli katmanlarda olduğu gibi, ortaklama katmanları da çıktının şeklini değiştirebilir. Ayrıca daha önce olduğu gibi, girdiyi dolgulayarak ve uzun adımı ayarlayarak istenen çıktı şeklini elde etmek için işlemi değiştirebiliriz. Derin öğrenme çerçevesinden yerleşik iki boyutlu maksimum ortaklama katmanı aracılığıyla ortaklama katmanlarında dolgu ve uzun adımların kullanımını gösterebiliriz. İlk olarak dört boyutlu şekle sahip bir `X` girdi tensörü inşa ediyoruz, burada örneklerin sayısı (iş boyutu) ve kanalların sayısının her ikisi de 1'dir.
:begin_tab:`tensorflow`
Dikkat edilecek bir husus tensorflow'un *kanallar son sıra* (son eksen) girdileri tercih ettiği ve ona göre optimize edildiğidir.
:end_tab:

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

Varsayılan olarak, (**çerçevenin yerleşik sınıfındaki örnekteki uzun adım ve ortaklama penceresi aynı şekle sahiptir.**) Aşağıda, `(3, 3)` şeklindeki bir ortaklama penceresi kullanıyoruz, bu nedenle varsayılan olarak `(3, 3)`'lük bir adım şekli alıyoruz.

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Ortaklama katmanında model parametresi olmadığı için parametre 
# ilkleme fonksiyonunu çağırmamız gerekmez
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

[**Uzun adım ve dolgu manuel olarak belirtilebilir.**]

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
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`mxnet`
Tabii ki, keyfi bir dikdörtgen ortaklama penceresi belirleyebilir ve sırasıyla yükseklik ve genişlik için dolguyu ve uzun adımı belirtebiliriz.
:end_tab:

:begin_tab:`pytorch`
Tabii ki, keyfi bir dikdörtgen ortaklama penceresi belirleyebilir ve sırasıyla yükseklik ve genişlik için dolguyu ve uzun adımı belirtebiliriz.
:end_tab:

:begin_tab:`tensorflow`
Tabii ki, keyfi bir dikdörtgen ortaklama penceresi belirleyebilir ve sırasıyla yükseklik ve genişlik için dolguyu ve uzun adımı belirtebiliriz.
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

## Çoklu Kanal

Çok kanallı girdi verilerini işlerken, ortaklama katmanı, girdileri bir evrişimli katmanda olduğu gibi kanallar üzerinden toplamak yerine [**her girdi kanalını ayrı ayrı ortaklar**]. Bu, ortaklama katmanının çıktı kanallarının sayısının girdi kanalı sayısıyla aynı olduğu anlamına gelir. Aşağıda, 2 kanallı bir girdi oluşturmak için kanal boyutundaki `X` ve `X + 1` tensörleri birleştiriyoruz.

:begin_tab:`tensorflow`
Bunun, kanallar son sıra sözdizimi nedeniyle TensorFlow için son boyut boyunca bir birleştirme gerektireceğini unutmayın.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

Gördüğümüz gibi, çıktı kanallarının sayısı ortaklamadan sonra hala 2'dir.

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
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

:begin_tab:`tensorflow`
Tensorflow ortaklama çıktısının ilk bakışta farklı göründüğünü, ancak sayısal olarak aynı sonuçların MXNet ve PyTorch'teki gibi temsil edildiğini unutmayın.
Fark, boyutlulukta yatmaktadır ve çıktıyı dikey olarak okumak, diğer uygulamalarla aynı çıktıyı verir.
:end_tab:

## Özet

* Ortaklama penceresinde girdi öğelerini alarak, maksimum ortaklama işlemi çıktı olarak maksimum değeri atar ve ortalama ortaklama işlemi ortalama değeri çıktı olarak atar.
* Bir ortaklama tabakasının en önemli avantajlarından biri, evrişimli tabakanın konumuna aşırı duyarlılığını hafifletmektir.
* Ortaklama katmanı için dolgu ve uzun adım belirtebiliriz.
* Uzamsal boyutları (örn. genişlik ve yükseklik) azaltmak için 1'den büyük bir uzun adımla birlikte maksimum ortaklama kullanılabilir.
* Ortaklama katmanının çıktı kanalı sayısı, girdi kanallarının sayısıyla aynıdır.

## Alıştırmalar

1. Bir evrişim tabakasının özel bir durumu olarak ortalama ortaklama uygulayabilir misiniz? Eğer öyleyse, yapınız.
1. Bir evrişim tabakasının özel bir durumu olarak maksimum ortaklama uygulayabilir misiniz? Eğer öyleyse, yapınız.
1. Ortaklama katmanının hesaplama maliyeti nedir? Ortaklama katmanına girdi boyutunun $c\times h\times w$ olduğunu, ortaklama penceresinin $p_h\times p_w$ bir şekle sahip, $(p_h, p_w)$ dolgulu ve $(s_h, s_w)$ uzun adımlı olduğunu varsayalım.
1. Neden maksimum ortaklama ile ortalama ortaklamanın farklı çalışmasını beklersiniz?
1. Ayrı bir minimum ortaklama katmanına ihtiyacımız var mıdır? Onu başka bir işlemle değiştirebilir misiniz?
1. Ortalama ve maksimum ortaklama arasında düşünebileceğiniz başka bir işlem var mıdır (İpucu: Softmaks'i anımsayın)? Neden o kadar popüler olmayacaktır?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/274)
:end_tab:
