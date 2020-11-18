# Dolgu ve Uzun Adımlar
:label:`sec_padding`

:numref:`fig_correlation`'ün önceki örneğinde, girişimiz hem yükseklik hem de 3 genişliğine sahipti ve evrişim çekirdeğimiz 2'nin hem yüksekliği hem de genişliği vardı, bu da $2\times2$ boyutuyla bir çıkış gösterimi sağladı. :numref:`sec_conv_layer`'te genelleştirdiğimiz gibi, giriş şeklinin $n_h\times n_w$ olduğunu ve evrişim çekirdeğinin şeklinin $k_h\times k_w$ olduğunu varsayarsak, çıkış şekli $(n_h-k_h+1) \times (n_w-k_w+1)$ olacaktır. Bu nedenle, konvolusyonel tabakanın çıkış şekli, girdinin şekli ve konvolüsyon çekirdeğinin şekli ile belirlenir.

Birkaç durumda, çıktının boyutunu etkileyen dolgu ve çizgili kıvrımlar da dahil olmak üzere teknikleri dahil ediyoruz. Motivasyon olarak, çekirdeklerin genellikle $1$'den daha büyük genişlik ve yüksekliğe sahip olduğundan, birçok ardışık kıvrımları uyguladıktan sonra, girişimizden çok daha küçük olan çıkışlarla sarılma eğilimindeyiz. $240 \times 240$ piksel görüntüyle başlarsak, $10$ katmanları $5 \times 5$ kıvrımların $200 \times 200$ piksele indirir, görüntünün 30$\ %$ 'ını keser ve orijinal görüntünün sınırları hakkında ilginç bilgileri yok eder.
*Dolgu*, bu sorunu ele almak için en popüler araçtır.

Diğer durumlarda, boyutsallığı büyük ölçüde azaltmak isteyebiliriz, örneğin orijinal giriş çözünürlüğünün kullanışsız olduğunu görürsek.
*Çizgili kıvrımlar*, bu örneklerde yardımcı olabilecek popüler bir tekniktir.

## Dolgu

Yukarıda açıklandığı gibi, kıvrımsal katmanları uygularken zor bir sorun, resmimizin çevresi üzerindeki pikselleri kaybetme eğiliminde olmamızdır. Tipik olarak küçük çekirdekler kullandığımızdan, herhangi bir konvolüsyon için, yalnızca birkaç piksel kaybedebiliriz, ancak birçok ardışık kıvrımsal katman uyguladığımız için bu da eklenebilir. Bu soruna basit bir çözüm, giriş resmimizin sınırına ekstra dolgu pikselleri eklemek, böylece görüntünün etkili boyutunu arttırmaktır. Tipik olarak, ekstra piksellerin değerlerini sıfıra ayarlarız. :numref:`img_conv_pad`'te, $3 \times 3$ girişi doldurarak boyutunu $5 \times 5$'e yükseltiyoruz. Karşılık gelen çıktı daha sonra bir $4 \times 4$ matrisine yükselir. Gölgeli kısımlar, çıkış hesaplaması için kullanılan giriş ve çekirdek tensör elemanlarının yanı sıra ilk çıkış elemanlarıdır: $0\times0+0\times1+0\times2+0\times3=0$.

![Two-dimensional cross-correlation with padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

Genel olarak, toplam $p_h$ sıra dolgu (kabaca yarısı üstte ve altta yarısı) ve toplam $p_w$ sütun dolgu (kabaca yarısı solda ve sağda yarısı) eklersek, çıkış şekli

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$

Bu, çıktının yüksekliği ve genişliğinin sırasıyla $p_h$ ve $p_w$ artacağı anlamına gelir.

Birçok durumda, giriş ve çıkışa aynı yükseklik ve genişlikte vermek için $p_h=k_h-1$ ve $p_w=k_w-1$'i ayarlamak isteyeceğiz. Bu, ağ oluştururken her katmanın çıkış şeklini tahmin etmeyi kolaylaştıracaktır. $k_h$ burada garip olduğunu varsayarsak, yüksekliğin her iki tarafında $p_h/2$ satırları ped olacak. Eğer $k_h$ eşit ise, bir olasılık altta girişin üst ve $\lfloor p_h/2\rfloor$ satır üzerinde pad $\lceil p_h/2\rceil$ satır olmasıdır. Genişliğin her iki tarafını da aynı şekilde doldıracağız.

CNN'ler genellikle 1, 3, 5 veya 7 gibi tek yükseklik ve genişlik değerlerine sahip evrişim çekirdeklerini kullanır. Tek çekirdek boyutlarını seçmek, üstte ve altta aynı sayıda satır ve solda ve sağda aynı sayıda sütun ile doldururken uzamsal boyutsallığı koruyabilmemiz avantajına sahiptir.

Dahası, boyutsallığı hassas bir şekilde korumak için tek çekirdekler ve dolgu kullanmanın bu uygulaması büro faydası sağlar. Herhangi bir iki boyutlu tensör `X` için, çekirdeğin boyutu tek olduğunda ve her iki taraftaki dolgu satırları ve sütunlarının sayısı aynı olduğunda, girdiyle aynı yükseklik ve genişliğe sahip bir çıkış üreten, `Y[i, j]` çıkışının giriş ve evrim çekirdeğinin çapraz korelasyonu ile hesaplandığını biliyoruz pencere ile `X[i, j]` ortalanmış.

Aşağıdaki örnekte, yüksekliği ve genişliği 3 olan iki boyutlu bir kıvrımsal tabaka oluşturuyoruz ve her tarafa 1 piksel dolgu uyguluyoruz. Yükseklik ve genişliği 8 olan bir giriş göz önüne alındığında, çıktının yüksekliğinin ve genişliğinin de 8 olduğunu buluyoruz.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return tf.reshape(Y, Y.shape[1:3])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

Evrim çekirdeğinin yüksekliği ve genişliği farklı olduğunda, yükseklik ve genişlik için farklı dolgu numaraları ayarlayarak çıkış ve girişin aynı yükseklik ve genişliğe sahip olmasını sağlayabiliriz.

```{.python .input}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='valid')
comp_conv2d(conv2d, X).shape
```

## Adım

Çapraz korelasyonu hesaplarken, giriş tensörünün sol üst köşesindeki evrişim penceresi ile başlar ve ardından hem aşağı hem de sağa doğru tüm konumların üzerinden kaydırırız. Önceki örneklerde, varsayılan olarak bir öğeyi aynı anda kaydırırız. Ancak, bazen, ya hesaplama verimliliği için ya da altörneklemek istediğimiz için, penceremizi aynı anda birden fazla öğeyi hareket ettirerek ara konumları atlıyoruz.

Slayt başına geçilen satır ve sütun sayısını*stride* olarak adlandırılır. Şimdiye kadar, hem yükseklik hem de genişlik için 1'lik adımlar kullandık. Bazen, daha büyük bir adım kullanmak isteyebiliriz. :numref:`img_conv_stride`, yatay olarak 3 ve 2 adımlı iki boyutlu bir çapraz korelasyon işlemi gösterir. Gölgeli kısımlar çıkış elemanlarının yanı sıra çıkış hesaplaması için kullanılan giriş ve çekirdek tensör elemanlarıdır: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$. İlk sütunun ikinci elemanı çıkıldığında, evrim penceresinin üç sıra aşağı kaydığını görebiliriz. İlk satırın ikinci öğesi çıktılandığında, evrişim penceresi sağa iki sütun kaydırır. Evrişim penceresi girdide sağa iki sütun kaydırmaya devam ettiğinde, giriş öğesi pencereyi dolduramayacağı için çıkış yoktur (başka bir dolgu sütunu eklemediğimiz sürece).

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

Genel olarak, yükseklik için adım $s_h$ olduğunda ve genişlik için adım $s_w$ olduğunda, çıkış şekli

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

$p_h=k_h-1$ ve $p_w=k_w-1$'i ayarlarsak, çıkış şekli $\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$'e basitleştirilir. Bir adım daha ileri gitmek, eğer giriş yüksekliği ve genişliği yükseklik ve genişlik üzerindeki adımlarla bölünebilirse, çıkış şekli $(n_h/s_h) \times (n_w/s_w)$ olacaktır.

Aşağıda, hem yükseklik hem de genişlik üzerindeki adımları 2'ye ayarladık, böylece giriş yüksekliğini ve genişliğini yarıya indirdik.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

Sonra, biraz daha karmaşık bir örneğe bakacağız.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

Kısalık uğruna, giriş yüksekliğinin ve genişliğinin her iki tarafındaki dolgu sayısı sırasıyla $p_h$ ve $p_w$ olduğunda, dolgu $(p_h, p_w)$ diyoruz. Özellikle, $p_h = p_w = p$ olduğunda, dolgu $p$'dir. Yükseklik ve genişlik üzerindeki adımlar sırasıyla $s_h$ ve $s_w$ olduğunda, adım $(s_h, s_w)$ diyoruz. Özellikle, $s_h = s_w = s$ olduğunda, adım $s$'dir. Varsayılan olarak, dolgu 0 ve adım 1'dir. Uygulamada, nadiren homojen olmayan adımlar veya dolgu kullanıyoruz, yani genellikle $p_h = p_w$ ve $s_h = s_w$'ümüz var.

## Özet

* Dolgu, çıkışın yüksekliğini ve genişliğini artırabilir. Bu genellikle çıktıya giriş ile aynı yükseklik ve genişlik vermek için kullanılır.
* Ayatım, çıktının çözünürlüğünü azaltabilir, örneğin çıkışın yüksekliğini ve genişliğini, girişin yüksekliğinin ve genişliğinin yalnızca $1/n$'ya düşürür ($n$, $1$'ten büyük bir tamsayıdır).
* Dolgu ve adım, verilerin boyutsallığını etkin bir şekilde ayarlamak için kullanılabilir.

## Alıştırmalar

1. Bu bölümdeki son örnek için, deneysel sonuçla tutarlı olup olmadığını görmek için çıktı şeklini hesaplamak için matematik kullanın.
1. Bu bölümdeki deneylerde diğer dolgu ve adım kombinasyonlarını deneyin.
1. Ses sinyalleri için, 2'lik bir adım neye karşılık gelir?
1. 1'den büyük bir adımın hesaplamalı faydaları nelerdir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/272)
:end_tab:
