# Adam
:label:`sec_adam`

Bu bölüme kadar uzanan tartışmalarda etkili optimizasyon için bir dizi teknikle karşılaştık. Onları burada ayrıntılı olarak özetleyelim: 

* :numref:`sec_sgd` içinde optimizasyon problemlerini çözerken, örneğin bol veriye olan doğal dayanıklılığı nedeniyle gradyan inişinden daha etkili olduğunu gördük. 
* :numref:`sec_minibatch_sgd` içinde bir minigrup içinde daha büyük gözlem kümeleri kullanarak vektörleştirmeden kaynaklanan önemli ek verimlilik sağladığını gördük. Bu, verimli çoklu makine, çoklu GPU ve genel paralel işleme için anahtardır. 
* :numref:`sec_momentum` yakınsamayı hızlandırmak için geçmiş gradyanların tarihçesini bir araya getirmeye yönelik bir mekanizma ekledi.
* :numref:`sec_adagrad` hesaplama açısından verimli bir ön koşul sağlamak için koordinat başına ölçekleme kullanıldı. 
* :numref:`sec_rmsprop` bir öğrenme hızı ayarlaması ile koordinat başına ölçeklendirmeyi ayrıştırdı. 

Adam :cite:`Kingma.Ba.2014`, tüm bu teknikleri tek bir verimli öğrenme algoritmasına birleştirir. Beklendiği gibi, bu, derin öğrenmede kullanılacak daha sağlam ve etkili optimizasyon algoritmalarından biri olarak oldukça popüler hale gelen bir algoritmadır. Yine de sorunları yok değildir. Özellikle, :cite:`Reddi.Kale.Kumar.2019`, Adam'ın zayıf varyans kontrolü nedeniyle ıraksayabileceği durumlar olduğunu göstermektedir. Bir takip çalışması :cite:`Zaheer.Reddi.Sachan.ea.2018` Adam için bu sorunları gideren Yogi denilen bir düzeltme önerdi. Bunun hakkında daha fazlasını birazdan vereceğiz. Şimdilik Adam algoritmasını gözden geçirelim.  

## Algoritma

Adam'ın temel bileşenlerinden biri, hem momentumu hem de gradyanın ikinci momentini tahmin etmek için üstel ağırlıklı hareketli ortalamaları (sızdıran ortalama olarak da bilinir) kullanmasıdır. Yani, durum değişkenleri kullanır 

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Burada $\beta_1$ ve $\beta_2$ negatif olmayan ağırlıklandırma parametreleridir. Onlar için yaygın seçenekler $\beta_1 = 0.9$ ve $\beta_2 = 0.999$'dur. Yani, varyans tahmini momentum teriminden *çok daha yavaş* hareket eder. $\mathbf{v}_0 = \mathbf{s}_0 = 0$ şeklinde ilklersek, başlangıçta daha küçük değerlere karşı önemli miktarda taraflı olduğumuzu unutmayın. Bu, $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$'nin terimleri yeniden normalleştirmesi gerçeğini kullanarak ele alınabilir. Buna göre normalleştirilmiş durum değişkenleri şu şekilde verilir: 

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ ve } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Uygun tahminlerle donanmış olarak şimdi güncelleme denklemlerini yazabiliriz. İlk olarak, RMSProp'a çok benzer bir şekilde gradyanı yeniden ölçeklendirerek aşağıdaki ifadeyi elde ederiz: 

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

RMSProp'un aksine güncellememiz gradyanın kendisi yerine $\hat{\mathbf{v}}_t$ momentumunu kullanır. Ayrıca, yeniden ölçeklendirme $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$ yerine $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ kullanarak gerçekleştiği için hafif bir kozmetik fark vardır. Sonraki, pratikte tartışmasız olarak biraz daha iyi çalışır, dolayısıyla RMSProp'tan bir sapmadır. Sayısal kararlılık ve aslına uygunluk arasında iyi bir denge için tipik olarak $\epsilon = 10^{-6}$'yı seçeriz. 

Şimdi güncellemeleri hesaplamak için tüm parçalara sahibiz. Bu biraz hayal kırıcıdır ve formun basit bir güncellemesi vardır: 

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Adam'ın tasarımı gözden geçirildiğinde ilham kaynağı açıktır. Momentum ve ölçek durum değişkenlerinde açıkça görülebilir. Oldukça tuhaf tanımları bizi terimleri yansızlaştırmaya zorlar (bu biraz farklı bir ilkleme ve güncelleme koşuluyla düzeltilebilir). İkincisi, her iki terim kombinasyonu RMSProp göz önüne alındığında oldukça basittir. Son olarak, açık öğrenme oranı $\eta$, yakınsama sorunlarını çözmek için adım uzunluğunu kontrol etmemizi sağlar.  

## Uygulama 

Adam'ı sıfırdan uygulama çok da göz korkutucu değil. Kolaylık sağlamak için `hyperparams` sözlüğünde $t$ zaman adımı sayacını saklıyoruz. Bunun ötesinde her şey basittir.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Modelini eğitmek için Adam'ı kullanmaya hazırız. $\eta = 0.01$ öğrenim oranını kullanıyoruz.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

`adam` Gluon `trainer` optimizasyon kütüphanesinin bir parçası olarak sağlanan algoritmalardan biri olduğundan daha kısa bir uygulama barizdir. Bu nedenle, Gluon'daki bir uygulama için yalnızca yapılandırma parametrelerini geçmemiz gerekiyor.

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Adam'ın sorunlarından biri, $\mathbf{s}_t$'teki ikinci moment tahmini patladığında dışbükey ayarlarda bile yakınsamayabilmesidir. Bir düzeltme olarak :cite:`Zaheer.Reddi.Sachan.ea.2018` $\mathbf{s}_t$ için arıtılmış bir güncelleme (ve ilkleme) önerdi. Neler olup bittiğini anlamak için, Adam güncellemesini aşağıdaki gibi yeniden yazalım: 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

$\mathbf{g}_t^2$ yüksek varyansa sahip olduğunda veya güncellemeler seyrek olduğunda, $\mathbf{s}_t$ geçmiş değerleri çok çabuk unutabilir. Bunun için olası bir düzeltme $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$'yı $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$ ile değiştirmektir. Artık güncellemenin büyüklüğü sapma miktarına bağlı değil. Bu Yogi güncellemelerini verir: 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Yazarlar ayrıca momentumu sadece ilk noktasal tahminden ziyade daha büyük bir ilk toplu iş ile ilklemeyi tavsiye ediyor. Tartışmada önemli olmadıkları ve bu yakınsama olmasa bile oldukça iyi kaldığı için ayrıntıları atlıyoruz.

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Özet

* Adam, birçok optimizasyon algoritmasının özelliklerini oldukça sağlam bir güncelleme kuralı haline getirir. 
* RMSProp temelinde oluşturulan Adam ayrıca minigrup rasgele gradyan üzerinde EWMA kullanır.
* Adam, momentumu ve ikinci bir momenti tahmin ederken yavaş başlatmayı ayarlamak için ek girdi düzeltmesini kullanır.
* Önemli varyansa sahip gradyanlar için yakınsama sorunlarıyla karşılaşabiliriz. Bunlar, daha büyük minigruplar kullanılarak veya $\mathbf{s}_t$ için geliştirilmiş bir tahmine geçilerek değiştirilebilir. Yogi böyle bir alternatif sunuyor. 

## Alıştırmalar

1. Öğrenme oranını ayarlayın ve deneysel sonuçları gözlemleyip analiz edin.
1. Eğer momentum ve ikinci moment güncellemeleri, ek girdi düzeltme gerektirmeyecek şekilde yeniden yazabilir misiniz?
1. Neden yakınsadığımızda $\eta$ öğrenme oranını düşürmeniz gerekiyor?
1. Adam'ın ıraksadığında ve Yogi'nin yakınsadığı bir durum oluşturmaya mı çalışın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1079)
:end_tab:
