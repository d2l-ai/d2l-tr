# Görüntüler için Konvolutions
:label:`sec_conv_layer`

Artık evrimsel katmanların teoride nasıl çalıştığını anladığımıza göre, pratikte nasıl çalıştıklarını görmeye hazırız. Görüntü verilerindeki yapıyı keşfetmek için etkili mimariler olarak evrimsel sinir ağlarının motivasyonunu temel alan örneklerimiz olarak görüntülere sadık kalıyoruz.

## Çapraz Korelasyon İşlemi

Kesin olarak konuşulan, evrimsel katmanların yanlış adlandırıldığını hatırlayın, çünkü ifade ettikleri işlemler çapraz korelasyon olarak daha doğru bir şekilde tanımlanmıştır. :numref:`sec_why-conv`'teki evrimsel tabakaların açıklamalarına dayanarak, böyle bir katmanda, bir giriş tensör ve bir çekirdek tensör çapraz korelasyon işlemi yoluyla bir çıkış tensörü üretmek için birleştirilir.

Şimdilik kanalları yok sayalım ve bunun iki boyutlu veri ve gizli temsillerle nasıl çalıştığını görelim. :numref:`fig_correlation`'te, giriş, 3 yüksekliği ve 3 genişliğindeki iki boyutlu bir tensördür. Tensörün şeklini $3 \times 3$ veya ($3$, $3$) olarak işaretliyoruz. Çekirdeğin yüksekliği ve genişliği her ikisi de 2'dir. *Çekirdek penceresinin* (veya *evrim penceresi*) şekli çekirdeğin yüksekliği ve genişliği ile verilir (burada $2 \times 2$).

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

İki boyutlu çapraz korelasyon işleminde, giriş tensörünün sol üst köşesinde konumlandırılmış evrim penceresi ile başlar ve hem soldan sağa hem de yukarıdan aşağıya doğru giriş tensörü boyunca kaydırırız. Evrişim penceresi belirli bir konuma kaydırıldığında, bu pencerede bulunan giriş subtensör ve çekirdek tensör eleman olarak çarpılır ve elde edilen tensör tek bir skaler değer oluşturarak toplanır. Bu sonuç, ilgili konumdaki çıkış tensörünün değerini verir. Burada, çıkış tensörünün yüksekliği 2 ve genişliği 2'dir ve dört eleman iki boyutlu çapraz korelasyon işleminden türetilmiştir:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Her eksen boyunca, çıkış boyutunun giriş boyutundan biraz daha küçük olduğunu unutmayın. Çekirdeğin genişliği ve yüksekliği birden fazla olduğundan, çekirdeğin tamamen görüntü içine sığdığı konumlar için çapraz korelasyonu düzgün bir şekilde hesaplayabiliriz, çıktı boyutu $n_h \times n_w$ eksi evrim çekirdeğinin boyutu $k_h \times k_w$ ile verilir.

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

Evrim çekirdeğini görüntü boyunca “kaydırmak” için yeterli alana ihtiyacımız olduğu için durum budur. Daha sonra, görüntüyü sınırının etrafında sıfırlarla doldurarak boyutun değişmeden nasıl tutulacağını göreceğiz, böylece çekirdeği kaydırmak için yeterli alan var. Daha sonra, `X` giriş tensör `X` ve bir çekirdek tensör `K` kabul eden ve bir çıkış tensör `Y` döndüren `corr2d` işlevinde bu işlemi uyguluyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

İki boyutlu çapraz korelasyon işleminin yukarıdaki uygulamasının çıktısını doğrulamak için :numref:`fig_correlation`'ten `X`'i ve çekirdek tensörünü `K`'yı inşa edebiliriz.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Konvolüsyonel Katmanlar

Bir kıvrımsal katman, girdi ve çekirdeği çapraz bağlar ve çıktı üretmek için bir skaler önyargı ekler. Bir kıvrımsal tabakanın iki parametresi çekirdek ve skaler yanlılıktır. Modelleri evrimsel katmanlara göre eğitirken, tam bağlantılı bir katmanda olduğu gibi, çekirdekleri genelde rastgele olarak başlatırız.

Yukarıda tanımlanan `corr2d` işlevine dayanan iki boyutlu bir evrimsel katman uygulamaya hazırız. `__init__` yapıcı işlevinde, iki model parametresi olarak `weight` ve `bias`'yi beyan ederiz. İleri yayılma işlevi `corr2d` işlevini çağırır ve önyargı ekler.

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

$h \times w$ evrişim veya $h \times w$ evrim çekirdeğinde, evrim çekirdeğinin yüksekliği ve genişliği sırasıyla $h$ ve $w$'dir. Ayrıca $h \times w$ evrişim çekirdeğine sahip bir evrimsel tabaka olarak sadece $h \times w$ evrimsel tabaka olarak atıfta bulunuyoruz.

## Görüntülerde Nesne Kenarı Algılama

Kıvrımsal bir katmanın basit bir uygulamasını ayrıştırmak için biraz zaman ayıralım: piksel değişiminin yerini bularak bir görüntüdeki nesnenin kenarını tespit etme. İlk olarak, $6\times 8$ piksellik bir “görüntü” oluşturuyoruz. Orta dört sütun siyah (0) ve geri kalanı beyazdır (1).

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

Daha sonra, 1 yüksekliğinde ve 2 genişliğinde bir çekirdek `K` inşa ediyoruz. Giriş ile çapraz korelasyon işlemini gerçekleştirdiğimizde, yatay olarak bitişik elemanlar aynıysa, çıkış 0'dır. Aksi takdirde, çıktı sıfır değildir.

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

`X` (girdimiz) ve `K` (çekirdeğimiz) argümanlarıyla çapraz korelasyon işlemini gerçekleştirmeye hazırız. Gördüğünüz gibi, kenar için beyazdan siyaha ve -1 için siyahtan beyaza kenar tespit ediyoruz. Diğer tüm çıkışlar 0 değerini alır.

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

Artık çekirdeği transpoze edilmiş görüntüye uygulayabiliriz. Beklendiği gibi, yok oluyor. Çekirdek `K` yalnızca dikey kenarları algılar.

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## Bir Çekirdek Öğrenme

Sonlu farklar `[1, -1]` ile bir kenar dedektörü tasarlamak, aradığımız şeyin tam olarak ne olduğunu biliyorsak düzgün olur. Ancak, daha büyük çekirdeklere baktığımızda ve ardışık kıvrımların katmanlarını göz önünde bulundurduğumuzda, her filtrenin manuel olarak ne yapması gerektiğini tam olarak belirtmek imkansız olabilir.

Şimdi `X`'ten `Y`'ü oluşturan çekirdeği yalnızca giriş-çıkış çiftlerine bakarak öğrenip öğrenemeyeceğimizi görelim. Önce bir kıvrımsal tabaka oluşturup çekirdeğini rastgele bir tensör olarak başlatırız. Daha sonra, her yinelemede, `Y`'ü evrimsel tabakanın çıktısıyla karşılaştırmak için kareli hatayı kullanacağız. Daha sonra çekirdeği güncellemek için degradeyi hesaplayabiliriz. Basitlik uğruna, aşağıda iki boyutlu evrimsel katmanlar için yerleşik sınıfı kullanıyoruz ve önyargıyı görmezden geliyoruz.

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

Hata 10 yineleme sonra küçük bir değere düştü unutmayın. Şimdi öğrendiğimiz çekirdek tensörüne bir göz atacağız.

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

Gerçekten de, öğrenilen çekirdek tensör, daha önce tanımladığımız çekirdek tensörüne `K`'e oldukça yakındır.

## Çapraz Korelasyon ve Konvolüsyon

Çapraz korelasyon ve evrim işlemleri arasındaki yazışmaların :numref:`sec_why-conv`'ten gözlem hatırlayın. Burada iki boyutlu evrimsel katmanları düşünmeye devam edelim. Bu tür katmanlar çapraz korelasyon yerine :eqref:`eq_2d-conv-discrete`'te tanımlandığı gibi katı evrim işlemlerini gerçekleştirirse ne olur? Sıkı *kıvrım* işleminin çıktısını elde etmek için, iki boyutlu çekirdek tensörünü hem yatay hem de dikey olarak çevirmemiz ve daha sonra giriş tensörüyle *çapraz korelasyon * işlemini gerçekleştirmemiz gerekir.

Çekirdekler derin öğrenmede verilerden öğrenildiğinden, bu tür katmanlar katı evrim işlemlerini veya çapraz korelasyon işlemlerini gerçekleştirse de, evrimsel katmanların çıktılarının etkilenmeden kalması dikkat çekicidir.

Bunu göstermek için, bir kıvrımsal katmanın *çapraz korelasyon* gerçekleştirdiğini ve çekirdeği :numref:`fig_correlation`'te öğrendiğini varsayalım, burada $\mathbf{K}$ matris olarak gösterilir. Bu katman yerine katı *kıvrım* gerçekleştirdiğinde, diğer koşulların değişmeden kaldığını varsayarsak, $\mathbf{K}'$ $\mathbf{K}'$ hem yatay hem de dikey olarak çevrildikten sonra $\mathbf{K}$ ile aynı olacaktır. Yani, evrimsel tabaka :numref:`fig_correlation` ve $\mathbf{K}'$'deki giriş için katı *konvolution* gerçekleştirdiğinde, :numref:`fig_correlation`'te aynı çıktı (giriş ve $\mathbf{K}$ çapraz korelasyon) elde edilecektir.

Derin öğrenme literatürüne sahip standart terminolojiye uygun olarak, çapraz korelasyon işlemine bir evrim olarak atıfta bulunmaya devam edeceğiz, sıkı bir şekilde konuşsak da, biraz farklı olsa da. Ayrıca, bir katman temsilini veya bir evrişim çekirdeğini temsil eden herhangi bir tensörün girişini (veya bileşenini) ifade etmek için *element* terimini kullanırız.

## Özellik Haritası ve Alıcı Alan

:numref:`subsec_why-conv-channels`'te açıklandığı gibi, :numref:`fig_correlation`'teki evrimsel katman çıktısı bazen *özellik haritası* olarak adlandırılır, çünkü uzamsal boyutlarda (örn. genişlik ve yükseklik) sonraki katmana öğrenilen temsiller (özellikler) olarak kabul edilebilir. CNN'lerde, bazı tabakanın herhangi bir elemanı $x$ için, *alıcı alanı*, ileri yayılma sırasında $x$'nın hesaplanmasını etkileyebilecek tüm elemanları (önceki katmanlardan) ifade eder. Alıcı alan girdinin gerçek boyutundan daha büyük olabileceğini unutmayın.

Alıcı alanı açıklamak için :numref:`fig_correlation`'ü kullanmaya devam edelim. $2 \times 2$ evrişim çekirdeği göz önüne alındığında, gölgeli çıkış elemanının alıcı alanı ($19$ değeri) girdinin gölgeli kısmındaki dört öğedir. Şimdi $2 \times 2$ çıkışını $\mathbf{Y}$ olarak gösterelim ve $\mathbf{Y}$ tek bir eleman çıkışı olarak $2 \times 2$ evrimsel tabaka ile daha derin bir CNN düşünelim $z$. Bu durumda, $\mathbf{Y}$'deki $z$'nın alıcı alanı $\mathbf{Y}$'nin dört unsurunu içerirken, girişteki alıcı alan dokuz giriş elemanını içerir. Böylece, bir özellik haritasındaki herhangi bir elemanın daha geniş bir alan üzerindeki giriş özelliklerini algılamak için daha büyük bir alıcı alana ihtiyacı olduğunda, daha derin bir ağ kurabiliriz.

## Özet

* İki boyutlu bir konvolüsyonel tabakanın çekirdek hesaplaması, iki boyutlu bir çapraz korelasyon işlemidir. En basit haliyle, bu iki boyutlu giriş verisi ve çekirdek üzerinde çapraz korelasyon işlemi gerçekleştirir ve sonra bir önyargı ekler.
* Görüntülerdeki kenarları tespit etmek için bir çekirdek tasarlayabiliriz.
* Çekirdeğin parametrelerini verilerden öğrenebiliriz.
* Verilerden öğrenilen çekirdekler ile, kıvrımsal katmanların çıktıları, bu tür katmanların gerçekleştirilen işlemlerinden bağımsız olarak (sıkı konvolüsyon veya çapraz korelasyon) etkilenmez.
* Bir özellik haritasındaki herhangi bir öğe, girişteki daha geniş özellikleri algılamak için daha büyük bir alıcı alana ihtiyaç duyduğunda, daha derin bir ağ düşünülebilir.

## Egzersizler

1. Bir görüntü oluşturmak `X` diyagonal kenarları ile.
    1. Çekirdek `K`'ü bu bölümde uygularsanız ne olur?
    1. `X`'ü transpoze ederseniz ne olur?
    1. `K`'ü transpoze ederseniz ne olur?
1. Oluşturduğumuz `Conv2D` sınıfının degradeyi otomatik olarak bulmaya çalıştığınızda ne tür bir hata mesajı görüyorsunuz?
1. Giriş ve çekirdek tensörlerini değiştirerek çapraz korelasyon işlemini matris çarpımı olarak nasıl temsil edersiniz?
1. Bazı çekirdekleri manuel olarak tasarlayın.
    1. İkinci türev için bir çekirdeğin şekli nedir?
    1. Bir integral için çekirdek nedir?
    1. Derece türevi elde etmek için bir çekirdeğin minimum boyutu nedir $d$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
