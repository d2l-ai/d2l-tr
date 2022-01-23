# RMSProp
:label:`sec_rmsprop`

:numref:`sec_adagrad`'teki en önemli konulardan biri, öğrenme hızının önceden tanımlanmış bir programda $\mathcal{O}(t^{-\frac{1}{2}})$ etkin bir şekilde azalması. Bu genellikle dışbükey problemler için uygun olsa da, derin öğrenmede karşılaşılanlar gibi dışbükey olmayan olanlar için ideal olmayabilir. Yine de, Adagrad'ın koordinat açısından uyarlanması ön koşullu olarak son derece arzu edilir. 

:cite:`Tieleman.Hinton.2012`, RMSProp algoritmasını koordinata uyarlamalı öğrenme oranlarından oran zamanlamasını ayırmak için basit bir düzeltme olarak önerdi. Sorun, Adagrad'ın $\mathbf{g}_t$ degradesinin karelerini bir devlet vektörü $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ olarak biriktirmesi. Sonuç olarak $\mathbf{s}_t$ normalleştirme eksikliği nedeniyle bağlı olmadan büyümeye devam eder, algoritma yakınlaştıkça aslında doğrusal olarak. 

Bu sorunu çözmenin bir yolu $\mathbf{s}_t / t$'ü kullanmaktır. $\mathbf{g}_t$'nin makul dağılımları için bu birleşecektir. Ne yazık ki, prosedür değerlerin tam yörüngesini hatırladığından sınır davranışı önemli olmaya başlayana kadar çok uzun zaman alabilir. Alternatif olarak, momentum yönteminde kullandığımız gibi sızan bir ortalama kullanmaktır, yani $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ bazı parametre $\gamma > 0$ için $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$. RMSProp verimleri değişmeden tüm diğer parçaları tutmak. 

## Algoritma

Denklemleri ayrıntılı olarak yazalım. 

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

$\epsilon > 0$ sabiti, sıfır veya aşırı büyük adım boyutlarına bölünmediğimizden emin olmak için tipik olarak $10^{-6}$ olarak ayarlanır. Bu genişleme göz önüne alındığında, koordinat başına uygulanan ölçeklendirmeden bağımsız olarak $\eta$ öğrenme oranını kontrol etmekte serbestiz. Sızdıran ortalamalar açısından, momentum yöntemi durumunda daha önce uygulandığı gibi aynı akıl yürütmesini uygulayabiliriz. $\mathbf{s}_t$ verim tanımını genişletmek 

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Daha önce olduğu gibi :numref:`sec_momentum`'te $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$ kullanıyoruz. Bu nedenle ağırlıkların toplamı $\gamma^{-1}$'lik bir gözlemin yarılanma ömrü ile $1$'e normalleştirilir. $\gamma$ çeşitli seçenekler için geçmiş 40 zaman adımlarının ağırlıklarını görselleştirelim.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Çizilmelerden Uygulama

Daha önce olduğu gibi, RMSProp yörüngesini gözlemlemek için $f(\mathbf{x})=0.1x_1^2+2x_2^2$ kuadratik işlevini kullanıyoruz. :numref:`sec_adagrad`'te Adagrad'da 0.4 öğrenme hızıyla kullandığımızda, öğrenme oranı çok hızlı bir şekilde azaldığından, algoritmanın sonraki aşamalarında değişkenlerin çok yavaş hareket ettiğini hatırlayın. $\eta$ ayrı ayrı kontrol edildiğinden bu RMSProp ile gerçekleşmez.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Ardından, derin bir ağda kullanılacak RMSProp'u uyguluyoruz. Bu da eşit derecede basit.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Başlangıç öğrenme oranını 0,01 ve ağırlıklandırma terimini $\gamma$'e 0,9'a ayarladık. Yani, $\mathbf{s}$ kare degradenin geçmiş $1/(1-\gamma) = 10$ gözlemleri üzerinde ortalama toplamlar.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Özlü Uygulama

RMSProp oldukça popüler bir algoritma olduğundan `Trainer` örneğinde de mevcuttur. Tek yapmamız gereken `rmsprop` adlı bir algoritma kullanarak, $\gamma$'yi `gamma1` parametresine atamak.

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Özet

* RMSProp, her ikisi de katsayıları ölçeklendirmek için degradenin karesini kullandığı için Adagrad'a çok benzer.
* RMSProp, sızıntı ortalamasını ivme ile paylaşır. Ancak, RMSProp katsayılı ön koşulu ayarlamak için tekniği kullanır.
* Öğrenme oranının pratikte deneyci tarafından planlanması gerekir.
* $\gamma$ katsayısı, koordinat başına ölçeği ayarlarken geçmişin ne kadar süreceğini belirler.

## Egzersizler

1. $\gamma = 1$'ü ayarlarsak deneysel olarak ne olur? Neden?
1. $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$'ü en aza indirmek için optimizasyon sorununu döndürün. Yakınsamaya ne oluyor?
1. Moda-MNIST eğitimi gibi gerçek bir makine öğrenimi probleminde RMSProp'a ne olduğunu deneyin. Öğrenme oranını ayarlamak için farklı seçeneklerle denemeler yapın.
1. Optimizasyon ilerledikçe $\gamma$'ü ayarlamak ister misiniz? RMSProp buna ne kadar duyarlıdır?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
