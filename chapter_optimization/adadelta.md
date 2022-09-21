# Adadelta
:label:`sec_adadelta`

Adadelta, AdaGrad'ın başka bir çeşididir (:numref:`sec_adagrad`). Temel fark, öğrenme oranının koordinatlara uyarlanabilir olduğu miktarı azaltmasıdır. Ayrıca, gelecekteki değişim için kalibrasyon olarak değişiklik miktarını kullandığından, geleneksel olarak bir öğrenme oranına sahip olmamak olarak adlandırılır. Algoritma :cite:`Zeiler.2012` çalışmasında önerildi. Şimdiye kadar önceki algoritmaların tartışılması göz önüne alındığında oldukça basittir.  

## Algoritma

Özetle, Adadelta, gradyanın ikinci momentinin sızıntılı ortalamasını depolamak için $\mathbf{s}_t$ ve modeldeki parametrelerin değişiminin ikinci momentinin sızıntılı ortalamasını depolamak için $\Delta\mathbf{x}_t$ diye iki durum değişkeni kullanır. Yazarların orijinal gösterimini ve adlandırmalarını diğer yayınlarla ve uygulamalarla uyumluluk için kullandığımızı unutmayın (momentum, Adagrad, RMSProp ve Adadelta, Adagrad, RMSProp ve Adadelta'da aynı amaca hizmet eden bir parametreyi belirtmek için farklı Yunan değişkenleri kullanılmasının başka bir gerçek nedeni yoktur).  

İşte Adadelta'nın teknik detayları verelim. Servis edilen parametre $\rho$ olduğu göz önüne alındığında, :numref:`sec_rmsprop`'a benzer şekilde aşağıdaki sızdıran güncellemeleri elde ediyoruz: 

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop` ile arasındaki fark, güncellemeleri yeniden ölçeklendirilmiş $\mathbf{g}_t'$ gradyanı ile gerçekleştirmemizdir, yani

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Peki yeniden ölçeklendirilmiş gradyan $\mathbf{g}_t'$ nedir? Bunu aşağıdaki gibi hesaplayabiliriz: 

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

burada $\Delta \mathbf{x}_{t-1}$ karesi yeniden ölçeklendirilmiş gradyanların sızdıran ortalaması $\mathbf{g}_t'$ dir. $\Delta \mathbf{x}_{0}$'ı $0$ olacak şekilde ilkleriz ve her adımda $\mathbf{g}_t'$ ile güncelleriz, yani,
$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

ve $\epsilon$ ($10^{-5}$ gibi küçük bir değerdir) sayısal kararlılığı korumak için eklenir. 

## Uygulama

Adadelta, her değişken için iki durum değişkenini, $\mathbf{s}_t$ ve $\Delta\mathbf{x}_t$, korumalıdır. Bu, aşağıdaki uygulamayı verir.

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
        # [:] aracılığıyla yerinde güncellemeler 
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
            # [:] aracılığıyla yerinde güncellemeler 
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

$\rho = 0.9$ seçilmesi, her parametre güncelleştirmesi için 10'luk yarı ömür süresine ulaşır. Bu oldukça iyi çalışma eğilimindedir. Aşağıdaki davranışları görürüz.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

Kısa bir uygulama için `Trainer` sınıfından `adadelta` algoritmasını kullanıyoruz. Bu, çok daha sıkılmıştır bir çağırma için aşağıdaki tek-satırlık kodu verir.

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
* Adadelta, gradyanın ikinci momentleri ve parametrelerdeki değişikliği depolamak için iki durum değişkenine ihtiyaç duyar. 
* Adadelta, uygun istatistiklerin işleyen bir tahminini tutmak için sızdıran ortalamaları kullanır. 

## Alıştırmalar

1. $\rho$ değerini ayarlayın. Ne olur?
1. $\mathbf{g}_t'$ kullanmadan algoritmanın nasıl uygulanacağını gösterin. Bu neden iyi bir fikir olabilir?
1. Adadelta gerçekten oranını bedavaya mı öğreniyor? Adadelta'yı bozan optimizasyon problemlerini bulabilir misin?
1. Yakınsama davranışlarını tartışmak için Adadelta'yı Adagrad ve RMSprop ile karşılaştırın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1077)
:end_tab:
