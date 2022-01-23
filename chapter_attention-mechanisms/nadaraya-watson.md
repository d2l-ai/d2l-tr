# Dikkat havuzlama: Nadaraya-Watson Çekirdek Regresyon
:label:`sec_nadaraya-watson`

Artık :numref:`fig_qkv` çerçevesinde dikkat mekanizmalarının ana bileşenlerini biliyorsunuz. Yeniden sermayelenmek için sorgular (istemli işaretler) ve anahtarlar (istemsiz işaretler) arasındaki etkileşimler*dikkat havuzu* ile sonuçlanır. Dikkat biriktirme, çıktıyı üretmek için seçici olarak değerleri (duyusal girişler) bir araya getirir. Bu bölümde, dikkat mekanizmalarının pratikte nasıl çalıştığına dair üst düzey bir görünüm vermek için dikkat havuzunu daha ayrıntılı olarak anlatacağız. Özellikle, 1964 yılında önerilen Nadaraya-Watson çekirdek regresyon modeli, dikkat mekanizmaları ile makine öğrenimini göstermek için basit ama eksiksiz bir örnektir.

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

## [**Veri Kümesi Oluşturulur**]

İşleri basit tutmak için, aşağıdaki regresyon sorununu ele alalım: $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ giriş-çıkış çiftlerinin bir veri kümesi verildiğinde $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, $\hat{y} = f(x)$ yeni giriş $x$ çıktısını tahmin etmek için $\hat{y} = f(x)$ nasıl öğrenilir? 

Burada $\epsilon$ gürültü terimi ile aşağıdaki doğrusal olmayan fonksiyona göre yapay bir veri kümesi oluşturuyoruz: 

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

burada $\epsilon$ sıfır ortalama ve standart sapma ile normal bir dağılıma uyar 0.5. Hem 50 eğitim örneği hem de 50 test örneği oluşturulur. Dikkat modelini daha sonra daha iyi görselleştirmek için, eğitim girdileri sıralanır.

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab tensorflow
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

```{.python .input}
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab pytorch
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab tensorflow
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

Aşağıdaki fonksiyon, tüm eğitim örneklerini (çevrelerle temsil edilir), `f`'ü gürültü terimi olmadan `f`'ü ve öğrenilen öngörü işlevini (“Pred” ile etiketlenmiş) çizer.

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Ortalama Havuz

Bu regresyon problemi için belki de dünyanın “en aptalca” tahmin edicisiyle başlıyoruz: ortalama havuzlama kullanarak tüm eğitim çıktıları üzerinde ortalama: 

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

hangi aşağıda çizilmiştir. Gördüğümüz gibi, bu tahminci gerçekten o kadar akıllı değil.

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

## [**Parametrik Olmayan Dikkat Havuzu**]

Açıkçası, ortalama havuzlama girişleri atlar $x_i$. Giriş yerlerine göre $y_i$ çıkışlarını tartmak için Nadaraya :cite:`Nadaraya.1964` ve Watson :cite:`Watson.1964` tarafından daha iyi bir fikir önerildi: 

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-watson`

burada $K$ bir *çekirdek*. :eqref:`eq_nadaraya-watson` içindeki tahmin ediciye *Nadaraya-Watson çekirdek regresyonu* denir. Burada çekirdeklerin ayrıntılarına dalmayacağız. :numref:`fig_qkv`'teki dikkat mekanizmalarının çerçevesini hatırlayın. Dikkat açısından bakıldığında, :eqref:`eq_nadaraya-watson`'i*dikkat havuzu* daha genelleştirilmiş bir formda yeniden yazabiliriz: 

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

burada $x$ sorgu ve $(x_i, y_i)$ anahtar-değer çiftidir. :eqref:`eq_attn-pooling` ve :eqref:`eq_avg-pooling`'ü karşılaştırarak, buradaki dikkat biriktirme $y_i$ değerlerinin ağırlıklı bir ortalamasıdır. :eqref:`eq_attn-pooling` içindeki $\alpha(x, x_i)$, $x$ sorgu ve $\alpha$ ile modellenen anahtar $x_i$ arasındaki etkileşime dayalı olarak $y_i$ karşılık gelen değere atanır. Herhangi bir sorgu için, tüm anahtar-değer çiftleri üzerindeki dikkat ağırlıkları geçerli bir olasılık dağılımıdır: negatif değildir ve bir adede kadar toplamlar. 

Dikkat biriktirme sezgileri kazanmak için, sadece bir *Gauss çekirdeği* 

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$

Gauss çekirdeğini :eqref:`eq_attn-pooling` ve :eqref:`eq_nadaraya-watson`'e takmak 

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian`

:eqref:`eq_nadaraya-watson-gaussian`'te, $x$ numaralı sorgunun $x$'e daha yakın olduğu bir anahtar $x_i$ alacak
*anahtarın karşılık gelen değerine atanan bir *daha büyük dikkat ağırlığı* aracılığıyla daha fazla dikkat* $y_i$.

Nadaraya-Watson çekirdek regresyonu parametrik olmayan bir modeldir; bu nedenle :eqref:`eq_nadaraya-watson-gaussian`, *parametrik olmayan dikkat havuzu* bir örnektir. Aşağıda, bu parametrik olmayan dikkat modeline dayanarak tahmini çiziyoruz. Tahmin edilen çizgi düzgün ve ortalama havuzlama tarafından üretilen yerden gerçeğe daha yakındır.

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the 
# same testing inputs (i.e., same queries)
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# Each element of `y_hat` is weighted average of values, where weights are attention weights
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

Şimdi [**dikkat ağırlıkları**] bir göz atalım. Burada test girdileri sorgulardır, eğitim girişleri ise anahtardır. Her iki giriş sıralandığından, sorgu anahtarı çifti ne kadar yakın olursa, dikkat ağırlığının dikkat havuzunda olduğunu görebiliriz.

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

## **Parametrik Dikkat Havuzu**

Parametrik olmayan Nadaraya-Watson çekirdek regresyonu*tutarlı* avantajından yararlanır: Yeterli veri verildiğinde bu model en uygun çözüme yakınlaşır. Bununla birlikte, öğrenilebilir parametreleri dikkat havuzuna kolayca entegre edebiliriz. 

Örnek olarak, :eqref:`eq_nadaraya-watson-gaussian`'ten biraz farklı, aşağıdaki gibi $x$ sorgu ve $x_i$ anahtarı arasındaki mesafe, öğrenilebilir bir parametre $w$ ile çarpılır: 

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian-para`

Bölümün geri kalanında, :eqref:`eq_nadaraya-watson-gaussian-para`'teki dikkat havuzunun parametresini öğrenerek bu modeli eğiteceğiz. 

### Toplu Matris Çarpması
:label:`subsec_batch_dot`

Mini partiler için dikkati daha verimli bir şekilde hesaplamak için, derin öğrenme çerçeveleri tarafından sağlanan toplu matris çarpma yardımcı programlarından yararlanabiliriz. 

İlk mini batch $n$ matrisleri $\mathbf{X}_1, \ldots, \mathbf{X}_n$ şekil $a\times b$ içerdiğini ve ikinci minibatch $n$ matrisleri $b\times c$ şekil $b\times c$ $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ matrisleri içerdiğini varsayalım. Onların toplu matris çarpımı $n$ şekil $a\times c$ matrisleri $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ ile sonuçlanır. Bu nedenle, [**iki şekil tensör verilen ($n$, $a$, $b$) ve ($n$, $b$, $c$), toplu matris çarpma çıktılarının şekli ($n$, $a$, $c$) .**]

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

Dikkat mekanizmaları bağlamında [**mini toplu işlemdeki değerlerin ağırlıklı ortalamalarını hesaplamak için minibatch matris çarpımını kullanabiliriz**]

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

Mini batch matris çarpımını kullanarak, aşağıda :eqref:`eq_nadaraya-watson-gaussian-para`'teki [**parametrik dikkat havuzu**] temel alan Nadaraya-Watson çekirdek regresyonunun parametrik versiyonunu tanımlıyoruz.

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
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
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
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

Aşağıda, dikkat modelini eğitmek için eğitim veri kümesini anahtar ve değerlere dönüştürmekteyiz. Parametrik dikkat havuzunda, herhangi bir eğitim girdisi çıktısını tahmin etmek için kendisi dışındaki tüm eğitim örneklerinden anahtar değer çiftlerini alır.

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

```{.python .input}
#@tab tensorflow
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

Kare kayıp ve stokastik degrade iniş kullanarak, [**parametrik dikkat modelini eğitir**].

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

Parametrik dikkat modelini eğittikten sonra, [**tahmini**] çizebiliriz. Eğitim veri kümesini gürültüye sığdırmaya çalışırken, öngörülen çizgi, daha önce çizilen parametrik olmayan meslektaşından daha az pürüzsüzdür.

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# Shape of `value`: (`n_test`, `n_train`)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

Parametrik olmayan dikkat biriktirme ile karşılaştırıldığında, öğrenilebilir ve parametrik ortamda [**büyük dikkat ağırlıkları olan bölge daha keskinleşiyor**].

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

* Nadaraya-Watson çekirdek regresyonu, dikkat mekanizmaları ile makine öğreniminin bir örneğidir.
* Nadaraya-Watson çekirdek regresyonunun dikkat birikimi, eğitim çıktılarının ağırlıklı bir ortalamasıdır. Dikkat açısından bakıldığında, dikkat ağırlığı, bir sorgunun işlevine ve değerle eşleştirilmiş anahtara dayanan bir değere atanır.
* Dikkat havuzlama parametrik olmayan veya parametrik olabilir.

## Egzersizler

1. Eğitim örneklerinin sayısını artırın. Parametrik olmayan Nadaraya-Watson çekirdek regresyonunu daha iyi öğrenebilir misin?
1. Parametrik dikkat havuzlama deneyinde öğrendiğimiz $w$'ün değeri nedir? Dikkat ağırlıklarını görselleştirirken ağırlıklı bölgeyi neden daha keskin hale getiriyor?
1. Daha iyi tahmin etmek için parametrik olmayan Nadaraya-Watson çekirdek regresyonuna nasıl hiperparametre ekleyebiliriz?
1. Bu bölümün çekirdek regresyonu için başka bir parametrik dikkat havuzu tasarlayın. Bu yeni modeli eğitin ve dikkat ağırlıklarını görselleştirin.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:
