# Çoklu Giriş ve Çoklu Çıkış Kanalları
:label:`sec_channels`

Her görüntüyü (örneğin, renkli görüntüler kırmızı, yeşil ve mavi miktarını belirtmek için standart RGB kanallarına sahiptir) ve :numref:`subsec_why-conv-channels`'te birden fazla kanal için evrimsel katmanlar tarif etmişken, şimdiye kadar, tüm sayısal örneklerimizi sadece tek bir giriş ve tek bir çıkış kanalı. Bu, girişlerimizi, evrim çekirdeklerini ve çıktılarımızı iki boyutlu tensörler olarak düşünmemizi sağladı.

Karışıma kanal eklediğimizde, girişlerimiz ve gizli temsillerimiz üç boyutlu tensörler haline gelir. Örneğin, her RGB giriş görüntüsü $3\times h\times w$ şeklindedir. Bu eksene, 3'lük bir boyutta, *kanal* boyutu olarak atıfta bulunuyoruz. Bu bölümde, birden fazla giriş ve birden fazla çıkış kanalı içeren evrim çekirdeklerine daha derin bir bakış atacağız.

## Çoklu Giriş Kanalları

Giriş verileri birden çok kanal içerdiğinde, giriş verileriyle aynı sayıda giriş kanalı içeren bir evrim çekirdeği oluşturmamız gerekir, böylece giriş verisi ile çapraz korelasyon gerçekleştirebilir. Giriş verisi için kanal sayısının $c_i$ olduğunu varsayarsak, evrişim çekirdeğinin giriş kanallarının sayısının da $c_i$ olması gerekir. Eğer evrim çekirdeğimizin pencere şekli $k_h\times k_w$ ise, o zaman $c_i=1$ olduğunda, evrim çekirdeğimizi $k_h\times k_w$ şeklindeki iki boyutlu bir tensör olarak düşünebiliriz.

Ancak, $c_i>1$ olduğunda, *her* giriş kanalı için $k_h\times k_w$ şeklindeki bir tensör içeren bir çekirdeğe ihtiyacımız var. Bu $c_i$ tensörlerin birleştirilmesi, $c_i\times k_h\times k_w$ şeklindeki bir evrim çekirdeği verir. Giriş ve evrişim çekirdeğinin her biri $c_i$ kanallara sahip olduğundan, her kanal için giriş iki boyutlu tensör ve konvolüsyon çekirdeğinin iki boyutlu tensör üzerinde çapraz korelasyon işlemi gerçekleştirebiliriz, $c_i$ sonuçlarını birlikte ekleyerek (kanallar üzerinde toplanarak) bir iki verim için-boyutsal tensör. Bu, çok kanallı giriş ve çok girişli bir evrim çekirdeği arasındaki iki boyutlu çapraz korelasyonun sonucudur.

:numref:`fig_conv_multi_in`'te, iki giriş kanalı ile iki boyutlu çapraz korelasyon örneğini gösteriyoruz. Gölgeli kısımlar, çıkış hesaplaması için kullanılan giriş ve çekirdek tensör elemanlarının yanı sıra ilk çıkış öğesidir: $(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Cross-correlation computation with 2 input channels.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

Burada neler olduğunu gerçekten anladığımızdan emin olmak için birden fazla giriş kanalı ile çapraz korelasyon işlemlerini kendimiz uygulayabiliriz. Yaptığımız tek şey kanal başına bir çapraz korelasyon işlemi gerçekleştirmek ve ardından sonuçları eklemek olduğuna dikkat edin.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

Çapraz korelasyon işleminin çıktısını doğrulamak için :numref:`fig_conv_multi_in`'teki değerlere karşılık gelen `X` giriş tensörünü ve çekirdek tensörünü `K`'i inşa edebiliriz.

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Çoklu Çıkış Kanalları

Giriş kanallarının sayısı ne olursa olsun, şimdiye kadar her zaman bir çıkış kanalı ile sona erdi. Bununla birlikte, :numref:`subsec_why-conv-channels`'te tartıştığımız gibi, her katmanda birden fazla kanalın olması gerekli olduğu ortaya çıkıyor. En popüler sinir ağı mimarilerinde, sinir ağında daha yükseğe çıktıkça kanal boyutunu artırıyoruz, tipik olarak uzamsal çözünürlüğü daha büyük bir kanal derinliği* için ticaret yapmak için altörnekleme yapıyoruz. Sezgisel olarak, her kanalı bazı farklı özellik kümesine yanıt olarak düşünebilirsiniz. Gerçeklik, bu sezginin en saf yorumlarından biraz daha karmaşıktır, çünkü temsiller bağımsız olarak öğrenilmemiştir, ancak ortaklaşa yararlı olmak için optimize edilmiştir. Bu nedenle, tek bir kanalın bir kenar dedektörünü öğrenmesi değil, kanal alanındaki bazı yönlerin kenarları algılamaya karşılık gelmesi olabilir.

$c_i$ ve $c_o$ ile sırasıyla giriş ve çıkış kanallarının sayısını belirtin ve $k_h$ ve $k_w$'in çekirdeğin yüksekliği ve genişliği olmasına izin verin. Birden fazla kanal içeren bir çıkış elde etmek için, *her* çıkış kanalı için $c_i\times k_h\times k_w$ şeklindeki bir çekirdek tensör oluşturabiliriz. Onları çıkış kanalı boyutuna birleştiririz, böylece evrim çekirdeğinin şekli $c_o\times c_i\times k_h\times k_w$ olur. Çapraz korelasyon işlemlerinde, her çıkış kanalındaki sonuç, çıkış kanalına karşılık gelen evrişim çekirdeğinden hesaplanır ve giriş tensöründeki tüm kanallardan girdi alır.

Aşağıda gösterildiği gibi birden fazla kanalın çıktısını hesaplamak için bir çapraz korelasyon fonksiyonu uyguluyoruz.

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

`K` çekirdek tensörünü `K+1` (artı `K`'deki her eleman için bir tane) ve `K+2` ile birleştirerek 3 çıkış kanalı içeren bir evrim çekirdeği inşa ediyoruz.

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Aşağıda, `X` çekirdek tensör `K` ile giriş tensöründe çapraz korelasyon işlemleri gerçekleştiriyoruz. Şimdi çıkış 3 kanal içeriyor. İlk kanalın sonucu, önceki giriş tensör `X` ve çoklu giriş kanalı, tek çıkışlı kanal çekirdeğinin sonucu ile tutarlıdır.

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ Konvolüsyonel Katman

İlk başta, bir $1 \times 1$ evrimi, yani $k_h = k_w = 1$, çok mantıklı görünmüyor. Sonuçta, bir evrim bitişik pikselleri ilişkilendirir. Bir $1 \times 1$ evrim besbelli değil. Bununla birlikte, bazen karmaşık derin ağların tasarımlarına dahil olan popüler operasyonlardır. Bize aslında ne yaptığını biraz ayrıntılı olarak görelim.

Minimum pencere kullanıldığından, $1\times 1$ evrişim, daha büyük evrimsel katmanların yükseklik ve genişlik boyutlarındaki bitişik elemanlar arasındaki etkileşimlerden oluşan desenleri tanıma yeteneğini kaybeder. $1\times 1$ evriminin tek hesaplaması kanal boyutunda gerçekleşir.

:numref:`fig_conv_1x1`, 3 giriş kanalı ve 2 çıkış kanalı ile $1\times 1$ evrim çekirdeği kullanılarak çapraz korelasyon hesaplamalarını gösterir. Giriş ve çıkışların aynı yükseklik ve genişliğe sahip olduğunu unutmayın. Çıktıdaki her öğe, giriş görüntüsündeki *aynı konumda* öğelerinin doğrusal bir kombinasyonundan türetilir. $1\times 1$ kıvrımsal katmanın, $c_i$'ye karşılık gelen giriş değerlerini $c_o$ çıkış değerlerine dönüştürmek için her piksel konumunda uygulanan tam bağlı bir katman oluşturduğunu düşünebilirsiniz. Bu hala bir kıvrımsal katman olduğundan, ağırlıklar piksel konumu boyunca bağlanır. Böylece $1\times 1$ evrimsel tabaka $c_o\times c_i$ ağırlık gerektirir (artı önyargı).

![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Bunun pratikte işe yarayıp yaramadığını kontrol edelim: tam bağlı bir tabaka kullanarak bir $1 \times 1$ evrimini uyguluyoruz. Tek şey, matris çarpımından önce ve sonra veri şekline bazı ayarlamalar yapmamız gerektiğidir.

```{.python .input}
#@tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    Y = d2l.matmul(K, X)  # Matrix multiplication in the fully-connected layer
    return d2l.reshape(Y, (c_o, h, w))
```

$1\times 1$ evrişim gerçekleştirirken, yukarıdaki işlev daha önce uygulanan çapraz korelasyon fonksiyonu `corr2d_multi_in_out` ile eşdeğerdir. Bunu örnek verilerle kontrol edelim.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
#@tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Özet

* Konvolusyonel tabakanın model parametrelerini genişletmek için birden fazla kanal kullanılabilir.
* $1\times 1$ kıvrımsal katman, piksel başına uygulandığında, tam bağlı katmana eşdeğerdir.
* $1\times 1$ kıvrımsal katman genellikle ağ katmanları arasındaki kanal sayısını ayarlamak ve model karmaşıklığını kontrol etmek için kullanılır.

## Egzersizler

1. sırasıyla $k_1$ ve $k_2$ boyutunda iki evrim çekirdeği olduğunu varsayalım (aralarında hiçbir doğrusal olmayan).
    1. Operasyonun sonucunun tek bir evrişim ile ifade edilebileceğini kanıtlayın.
    1. Eşdeğer tek evrimin boyutsallığı nedir?
    1. Bu tersi doğru mu?
1. $c_i\times h\times w$ şeklinde bir giriş ve $c_o\times c_i\times k_h\times k_w$ şeklindeki bir evrim çekirdeği, $(p_h, p_w)$ dolgusu ve $(s_h, s_w)$ adımını varsayalım.
    1. İleri yayılma için hesaplama maliyeti (çarpma ve eklemeler) nedir?
    1. Hafıza ayak izi nedir?
    1. Geriye dönük hesaplama için bellek ayak izi nedir?
    1. Geri yayılma için hesaplama maliyeti nedir?
1. $c_i$ giriş kanallarının sayısını ve $c_o$ çıkış kanallarının sayısını iki katına çıkarırsak, hesaplamaların sayısı hangi faktörle artar? Dolgu ikiye katlarsak ne olur?
1. Bir evrişim çekirdeğinin yüksekliği ve genişliği $k_h=k_w=1$ ise, ileri yayılımın hesaplama karmaşıklığı nedir?
1. Değişkenler `Y1` ve `Y2` Bu bölümün son örnekte tam olarak aynı mı? Neden?
1. Evrim penceresi $1\times 1$ olmadığında matris çarpımı kullanarak kıvrımları nasıl uygularsınız?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:
