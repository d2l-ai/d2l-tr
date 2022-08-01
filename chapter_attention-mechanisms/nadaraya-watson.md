# Dikkat Ortaklama: Nadaraya-Watson Çekirdek Bağlanımı
:label:`sec_nadaraya-watson`

Artık :numref:`fig_qkv` çerçevesinde dikkat mekanizmalarının ana bileşenlerini biliyorsunuz. Yeniden özetlemek için sorgular (istemli işaretler) ve anahtarlar (istemsiz işaretler) arasındaki etkileşimler *dikkat ortaklama* ile sonuçlanır. Dikkat ortaklama, çıktıyı üretmek için seçici olarak değerleri (duyusal girdiler) bir araya getirir. Bu bölümde, dikkat mekanizmalarının pratikte nasıl çalıştığına dair üst düzey bir görünüm vermek için dikkat ortaklamasını daha ayrıntılı olarak anlatacağız. Özellikle, 1964 yılında önerilen Nadaraya-Watson çekirdek bağlanım modeli, makine öğrenmesini dikkat mekanizmaları ile göstermek için basit ama eksiksiz bir örnektir.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
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
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
tf.random.set_seed(seed=1322)
```

## [**Veri Kümesi Oluşturma**]

İşleri basit tutmak için, aşağıdaki regresyon problemini ele alalım: $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ girdi-çıktı çiftlerinin bir veri kümesi verildiğinde $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, $\hat{y} = f(x)$ yeni bir $x$ girdisinin çıktısını tahmin etmek için $\hat{y} = f(x)$ nasıl öğrenilir? 

Burada $\epsilon$ gürültü terimi ile aşağıdaki doğrusal olmayan fonksiyona göre yapay bir veri kümesi oluşturuyoruz: 

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

burada $\epsilon$ sıfır ortalama ve 0.5 standart sapma ile normal bir dağılıma uyar. Hem 50 eğitim örneği hem de 50 test örneği üretilir. Dikkat modelini daha sonra daha iyi görselleştirmek için, eğitim girdileri sıralanır.

```{.python .input}
n_train = 50  # Eğitim örneklerinin adedi
x_train = np.sort(d2l.rand(n_train) * 5)   # Eğitim girdileri
```

```{.python .input}
#@tab pytorch
n_train = 50  # Eğitim örneklerinin adedi
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Eğitim girdileri
```

```{.python .input}
#@tab tensorflow
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

```{.python .input}
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Eğitim çıktıları
x_test = d2l.arange(0, 5, 0.1)  # Test örnekleri
y_truth = f(x_test)  # Test örneklerinin adedi gerçek referans değeri
n_test = len(x_test)  # Test örneklerinin adedi
n_test
```

```{.python .input}
#@tab pytorch
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Eğitim çıktıları
x_test = d2l.arange(0, 5, 0.1)  # Test örnekleri
y_truth = f(x_test)  # Test örneklerinin adedi gerçek referans değeri
n_test = len(x_test)  # Test örneklerinin adedi
n_test
```

```{.python .input}
#@tab tensorflow
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # Eğitim çıktıları
x_test = d2l.arange(0, 5, 0.1)  # Test örnekleri
y_truth = f(x_test)  # Test örneklerinin adedi gerçek referans değeri
n_test = len(x_test)  # Test örneklerinin adedi
n_test
```

Aşağıdaki işlev, tüm eğitim örneklerini (dairelerle temsil edilmiş), gürültü terimi olmadan gerçek referans değer veri üretme işlevi 'f'yi ("Truth" ile etiketlenmiş) ve öğrenilen tahmin işlevini ("Pred" ile etiketlenmiş) çizer.

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Ortalama Ortaklama

Bu regresyon problemi için belki de dünyanın “en aptalca” tahmin edicisiyle başlıyoruz: Ortalama ortaklama kullanarak tüm eğitim çıktıları üzerinde ortalama, 

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

aşağıda çizilmiştir. Gördüğümüz gibi, bu tahminci gerçekten o kadar akıllı değil.

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```

## [**Parametrik Olmayan Dikkat Ortaklama**]

Açıkçası, ortalama ortaklama girdileri, $x_i$, atlar. Girdi yerlerine göre $y_i$ çıktılarını ağırlıklandırmak için Nadaraya :cite:`Nadaraya.1964` ve Watson :cite:`Watson.1964` tarafından daha iyi bir fikir önerildi: 

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-watson`

burada $K$ bir *çekirdek*tir. :eqref:`eq_nadaraya-watson` içindeki tahmin ediciye *Nadaraya-Watson çekirdek regresyonu* denir. Burada çekirdeklerin ayrıntılarına dalmayacağız. :numref:`fig_qkv` içindeki dikkat mekanizmalarının çerçevesini hatırlayın. Dikkat açısından bakıldığında, :eqref:`eq_nadaraya-watson` denklemini *dikkat ortaklama*nın daha genelleştirilmiş bir formunda yeniden yazabiliriz: 

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

burada $x$ sorgu ve $(x_i, y_i)$ anahtar-değer çiftidir. :eqref:`eq_attn-pooling` ve :eqref:`eq_avg-pooling` karşılaştırılırsa, buradaki dikkat havuzlama $y_i$ değerlerinin ağırlıklı bir ortalamasıdır. :eqref:`eq_attn-pooling` içindeki *dikkat ağırlığı* $\alpha(x, x_i)$, $x$ sorgu ve $\alpha$ ile modellenen anahtar $x_i$ arasındaki etkileşime dayalı olarak karşılık gelen $y_i$ değerine atanır. Herhangi bir sorgu için, tüm anahtar-değer çiftleri üzerindeki dikkat ağırlıkları geçerli bir olasılık dağılımıdır: Negatif değillerdir ve bire toplanırlar. 

Dikkat ortaklama sezgileri kazanmak için, sadece aşağıda tanımlanan bir *Gauss çekirdeği*ni düşünün 

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$

Gauss çekirdeğini :eqref:`eq_attn-pooling` ve :eqref:`eq_nadaraya-watson` denklemlerine koyarsak 

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian`

:eqref:`eq_nadaraya-watson-gaussian` içinde, verilen $x$ sorgusuna daha yakın olan bir $x_i$ anahtarı, anahtarın karşılık gelen $y_i$ değerine atanan *daha büyük bir dikkat ağırlığı* aracılığıyla *daha fazla dikkat* alacaktır.

Nadaraya-Watson çekirdek regresyonu parametrik olmayan bir modeldir; bu nedenle :eqref:`eq_nadaraya-watson-gaussian`, *parametrik olmayan dikkat ortaklama* örneğidir. Aşağıda, bu parametrik olmayan dikkat modeline dayanarak tahmini çiziyoruz. Tahmin edilen çizgi düzgün ve ortalama ortaklama tarafından üretilen gerçek referans değere daha yakındır.

```{.python .input}
# `X_repeat` şekli: (`n_test`, `n_train`), burada her satır aynı test 
# girdilerini içerir (yani aynı sorgular)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# `x_train`'in anahtarları içerdiğine dikkat edin. `attention_weights` şekli: 
# (`n_test`, `n_train`)'dir, burada her satır, her sorguya verilen değerler 
# (`y_train`) arasında atanacak dikkat ağırlıklarını içerir
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
# `y_hat`'nin her bir öğesi, ağırlıkların dikkat ağırlıkları olduğu 
# değerlerin ağırlıklı ortalamasıdır.
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# `X_repeat` şekli: (`n_test`, `n_train`), burada her satır aynı test 
# girdilerini içerir (yani aynı sorgular)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# `x_train`'in anahtarları içerdiğine dikkat edin. `attention_weights` şekli: 
# (`n_test`, `n_train`)'dir, burada her satır, her sorguya verilen değerler 
# (`y_train`) arasında atanacak dikkat ağırlıklarını içerir
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# `y_hat`'nin her bir öğesi, ağırlıkların dikkat ağırlıkları olduğu 
# değerlerin ağırlıklı ortalamasıdır.
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# `X_repeat` şekli: (`n_test`, `n_train`), burada her satır aynı test 
# girdilerini içerir (yani aynı sorgular)
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# `x_train`'in anahtarları içerdiğine dikkat edin. `attention_weights` şekli: 
# (`n_test`, `n_train`)'dir, burada her satır, her sorguya verilen değerler 
# (`y_train`) arasında atanacak dikkat ağırlıklarını içerir
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# `y_hat`'nin her bir öğesi, ağırlıkların dikkat ağırlıkları olduğu 
# değerlerin ağırlıklı ortalamasıdır.
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

Şimdi [**dikkat ağırlıkları**]na bir göz atalım. Burada test girdileri sorgulardır, eğitim girdileri ise anahtarlardır. Her iki girdi sıralandığından, sorgu-anahtar çifti ne kadar yakın olursa, dikkat ortaklamasında dikkat ağırlığı o kadar yüksek olur.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## **Parametrik Dikkat Ortaklama**

Parametrik olmayan Nadaraya-Watson çekirdek regresyonu *tutarlılık* avantajından yararlanır: Yeterli veri verildiğinde bu model en uygun çözüme yakınlaşır. Bununla birlikte, öğrenilebilir parametreleri dikkat ortaklamasına kolayca tümleştirebiliriz. 

Örnek olarak, :eqref:`eq_nadaraya-watson-gaussian` denkleminden biraz farklı olarak, aşağıdaki gibi $x$ sorgu ve $x_i$ anahtarı arasındaki uzaklık, öğrenilebilir bir parametre $w$ ile çarpılır: 

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian-para`

Bölümün geri kalanında, :eqref:`eq_nadaraya-watson-gaussian-para` denklemindeki dikkat ortaklama parametresini öğrenerek bu modeli eğiteceğiz. 

### Toplu Matris Çarpması
:label:`subsec_batch_dot`

Minigruplar için dikkati daha verimli bir şekilde hesaplamak için, derin öğrenme çerçeveleri tarafından sağlanan toplu matris çarpma yardımcı programlarından yararlanabiliriz. 

İlk minigrup $n$ matrisleri $\mathbf{X}_1, \ldots, \mathbf{X}_n$ şekil $a\times b$ içerdiğini ve ikinci minibatch $n$ matrisleri $b\times c$ şekilli $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ matrisleri içerdiğini varsayalım. Onların toplu matris çarpımı $n$ şekil $a\times c$ matrisleri $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ ile sonuçlanır. Bu nedenle, [**iki şekil tensör verilen ($n$, $a$, $b$) ve ($n$, $b$, $c$), toplu matris çarpma çıktılarının şekli ($n$, $a$, $c$)'dir.**]

İlk minigrubun $a\times b$ şeklinde $\mathbf{X}_1, \ldots, \mathbf{X}_n$ $n$ matrisi içerdiğini ve ikinci minigrubun $b\times c$ şeklinde $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ $n$ matrisi içerdiğini varsayalım. Onların toplu matris çarpımı, $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ $a\times c$ şeklinde $n$ matris ile sonuçlanır. Bu nedenle, [**($n$, $a$, $b$) ve ($n$, $b$, $c$) şeklinde iki tensör verildiğinde, toplu matris çarpım çıktılarının şekli ($n$, $a$, $c$)'dir.**]

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape
```

Dikkat mekanizmaları bağlamında, [**bir minigruptaki değerlerin ağırlıklı ortalamalarını hesaplamak için minigrup matris çarpımını kullanabiliriz.**]

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

```{.python .input}
#@tab tensorflow
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

### Modeli Tanımlama

Minigrup matris çarpımını kullanarak, aşağıda :eqref:`eq_nadaraya-watson-gaussian-para` denklemindeki [**parametrik dikkat ortaklama**]yı temel alan Nadaraya-Watson çekirdek regresyonunun parametrik versiyonunu tanımlıyoruz.

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # `queries` ve `attention_weights` çıktısının şekli: 
        # (sorgu sayısı, anahtar/değer çifti sayısı)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # `values` (değerler) şekli: (sorgu sayısı, anahtar/değer çifti sayısı)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # `queries` ve `attention_weights` çıktısının şekli: 
        # (sorgu sayısı, anahtar/değer çifti sayısı)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # `values` (değerler) şekli: (sorgu sayısı, anahtar/değer çifti sayısı)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

```{.python .input}
#@tab tensorflow
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))
        
    def call(self, queries, keys, values, **kwargs):
        # For training queries are `x_train`. Keys are distance of taining data for each point. Values are `y_train`.
        # Shape of the output `queries` and `attention_weights`: (no. of queries, no. of key-value pairs)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```

### Eğitim

Aşağıda, dikkat modelini eğitmek için eğitim veri kümesini anahtar ve değerlere dönüştürürüz. Parametrik dikkat ortaklamada, herhangi bir eğitim girdisi çıktısını tahmin etmek için kendisi dışındaki tüm eğitim örneklerinden anahtar-değer çiftlerini alır.

```{.python .input}
# `X_tile`'in şekli: (`n_train`, `n_train`), burada her sütun aynı
#  eğitim girdilerini içerir
X_tile = np.tile(x_train, (n_train, 1))
# `Y_tile`'in şekli: (`n_train`, `n_train`), burada her sütun aynı 
# eğitim çıktılarını içerir
Y_tile = np.tile(y_train, (n_train, 1))
# `keys`'in şekli: (`n_train`, `n_train` - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# `values`'in şekli: (`n_train`, `n_train` - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# `X_tile`'in şekli: (`n_train`, `n_train`), burada her sütun aynı
#  eğitim girdilerini içerir
X_tile = x_train.repeat((n_train, 1))
# `Y_tile`'in şekli: (`n_train`, `n_train`), burada her sütun aynı 
# eğitim çıktılarını içerir
Y_tile = y_train.repeat((n_train, 1))
# `keys`'in şekli: (`n_train`, `n_train` - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# `values`'in şekli: (`n_train`, `n_train` - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

```{.python .input}
#@tab tensorflow
# `X_tile`'in şekli: (`n_train`, `n_train`), burada her sütun aynı
#  eğitim girdilerini içerir
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# `Y_tile`'in şekli: (`n_train`, `n_train`), burada her sütun aynı 
# eğitim çıktılarını içerir
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# `keys`'in şekli: (`n_train`, `n_train` - 1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# `values`'in şekli: (`n_train`, `n_train` - 1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

Kare kayıp ve rasgele gradyan inişi kullanarak, [**parametrik dikkat modelini eğitiriz**].

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab tensorflow
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```

Parametrik dikkat modelini eğittikten sonra, [**tahminini**] çizebiliriz. Eğitim veri kümesini gürültüye oturtmaya çalışırken, tahmin edilen çizgi, daha önce çizilen parametrik olmayan karşılığından daha az pürüzsüzdür.

```{.python .input}
# `keys`'in şekli: (`n_test`, `n_train`), burada her sütun aynı eğitim 
# girdilerini (yani aynı anahtarları) içerir
keys = np.tile(x_train, (n_test, 1))
# `value`'in şekli: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# `keys`'in şekli: (`n_test`, `n_train`), burada her sütun aynı eğitim 
# girdilerini (yani aynı anahtarları) içerir
keys = x_train.repeat((n_test, 1))
# `value`'in şekli: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# `keys`'in şekli: (`n_test`, `n_train`), burada her sütun aynı eğitim 
# girdilerini (yani aynı anahtarları) içerir
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# `value`'in şekli: (`n_test`, `n_train`)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

Parametrik olmayan dikkat ortaklama ile karşılaştırıldığında, öğrenilebilir ve parametrik ayarda [**büyük dikkat ağırlıkları olan bölge daha keskinleşir**].

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(net.attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## Özet

* Nadaraya-Watson çekirdek regresyonu, makine öğrenmesinin dikkat mekanizmaları ile bir örneğidir.
* Nadaraya-Watson çekirdek regresyonunun dikkat ortaklaması, eğitim çıktılarının ağırlıklı bir ortalamasıdır. Dikkat açısından bakıldığında, dikkat ağırlığı, bir sorgunun işlevine ve değerle eşleştirilmiş anahtara dayanan bir değere atanır.
* Dikkat ortaklama parametrik olabilir de olmayabilir de.

## Alıştırmalar

1. Eğitim örneklerinin sayısını artırın. Parametrik olmayan Nadaraya-Watson çekirdek regresyonunu daha iyi öğrenebilir misin?
1. Parametrik dikkat ortaklama deneyinde öğrendiğimiz $w$'nin değeri nedir? Dikkat ağırlıklarını görselleştirirken ağırlıklı bölgeyi neden daha keskin hale getiriyor?
1. Daha iyi tahmin etmek için parametrik olmayan Nadaraya-Watson çekirdek regresyonuna nasıl hiper parametre ekleyebiliriz?
1. Bu bölümün çekirdek regresyonu için başka bir parametrik dikkat ortaklama tasarlayın. Bu yeni modeli eğitin ve dikkat ağırlıklarını görselleştirin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1599)
:end_tab:
