# Adadelta
:label:`sec_adadelta`

Adadelta, AdaGrad'ın başka bir çeşididir (:numref:`sec_adagrad`). Temel fark, öğrenme oranının koordinatlara uyarlanabilir olduğu miktarı azaltmasıdır. Dahası, geleneksel olarak değişim miktarını gelecekteki değişim için kalibrasyon olarak kullandığı için öğrenme oranına sahip olmamak olarak adlandırılır. Algoritma :cite:`Zeiler.2012`'te önerildi. Şimdiye kadar önceki algoritmaların tartışılması göz önüne alındığında oldukça basittir.  

## Algoritma

Özetle, Adadelta, degradenin ikinci anının sızıntılı ortalamasını depolamak için $\mathbf{s}_t$ ve modeldeki parametrelerin değişiminin ikinci anının sızıntılı ortalamasını depolamak için iki durum değişkeni kullanır. Yazarların orijinal gösterimini ve adlandırmalarını diğer yayınlarla ve uygulamalarla uyumluluk için kullandığımızı unutmayın (momentum, Adagrad, RMSProp ve Adadelta, Adagrad, RMSProp ve Adadelta'da aynı amaca hizmet eden bir parametreyi belirtmek için farklı Yunan değişkenleri kullanmasının başka bir gerçek nedeni yoktur).  

İşte Adadelta'nın teknik detayları. Parametre du jour $\rho$ olduğu göz önüne alındığında, :numref:`sec_rmsprop` benzer şekilde aşağıdaki sızdıran güncellemeleri elde ediyoruz: 

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop`'ün farkı, yeniden ölçeklendirilmiş degrade $\mathbf{g}_t'$, yani, 

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Peki yeniden ölçeklendirilmiş degrade $\mathbf{g}_t'$ nedir? Bunu aşağıdaki gibi hesaplayabiliriz: 

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

burada $\Delta \mathbf{x}_{t-1}$, kareli yeniden ölçeklendirilmiş degradelerin sızan ortalamasıdır $\mathbf{g}_t'$. $\Delta \mathbf{x}_{0}$'yi $0$ olarak başlatıyoruz ve her adımda $\mathbf{g}_t'$ ile güncelliyoruz, yani 

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

ve $\epsilon$ ($10^{-5}$ gibi küçük bir değer) sayısal kararlılığı korumak için eklenir. 

## Uygulama

Adadelta, $\mathbf{s}_t$ ve $\Delta\mathbf{x}_t$ değişken için her değişken için iki durum değişkenini korumalıdır. Bu, aşağıdaki uygulamayı verir.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

$\rho = 0.9$ seçilmesi, her parametre güncelleştirmesi için 10 yarı ömür süresine ulaşır. Bu oldukça iyi çalışma eğilimindedir. Aşağıdaki davranışları alırız.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Kısa bir uygulama için `Trainer` sınıfından `adadelta` algoritmasını kullanıyoruz. Bu, çok daha kompakt bir çağırma için aşağıdaki tek astarı verir.

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Özet

* Adadelta'nın öğrenme oranı parametresi yoktur. Bunun yerine, öğrenme oranını uyarlamak için parametrelerin kendisindeki değişim oranını kullanır. 
* Adadelta, degradenin ikinci anlarını ve parametrelerdeki değişikliği depolamak için iki durum değişkeni gerektirir. 
* Adadelta, uygun istatistiklerin çalışan bir tahmini tutmak için sızdıran ortalamaları kullanır. 

## Egzersizler

1. $\rho$ değerini ayarlayın. Ne oluyor?
1. $\mathbf{g}_t'$ kullanılmadan algoritmanın nasıl uygulanacağını gösterin. Bu neden iyi bir fikir olabilir ki?
1. Adadelta gerçekten öğrenme oranı ücretsiz mi? Adadelta'yı kıran optimizasyon problemlerini bulabilir misin?
1. Yakınsama davranışlarını tartışmak için Adadelta'yı Adagrad ve RMS prop ile karşılaştırın.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab: