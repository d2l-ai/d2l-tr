# Adem
:label:`sec_adam`

Bu bölüme kadar uzanan tartışmalarda etkili optimizasyon için bir dizi teknikle karşılaştık. Onları burada ayrıntılı olarak özetleyelim: 

* :numref:`sec_sgd`'ün optimizasyon problemlerini çözerken, örneğin yedekli verilere olan doğal dayanıklılığı nedeniyle Degrade Descent'ten daha etkili olduğunu gördük. 
* :numref:`sec_minibatch_sgd`'ün bir minibatch içinde daha büyük gözlem setleri kullanarak vektörleştirmeden kaynaklanan önemli ek verimlilik sağladığını gördük. Bu, verimli çoklu makine, çoklu GPU ve genel paralel işleme için anahtardır. 
* :numref:`sec_momentum` yakınsamayı hızlandırmak için geçmiş degradelerin geçmişini bir araya getirmeye yönelik bir mekanizma eklendi.
* :numref:`sec_adagrad` hesaplama açısından verimli bir ön koşul sağlamak için koordinat başına ölçekleme kullanılır. 
* :numref:`sec_rmsprop` bir öğrenme hızı ayarlamasından koordinat başına ölçeklendirme ayrıştırıldı. 

Adam :cite:`Kingma.Ba.2014`, tüm bu teknikleri tek bir verimli öğrenme algoritmasına birleştirir. Beklendiği gibi, bu, derin öğrenmede kullanılacak daha sağlam ve etkili optimizasyon algoritmalarından biri olarak oldukça popüler hale gelen bir algoritmadır. Yine de sorun olmadan değil. Özellikle, :cite:`Reddi.Kale.Kumar.2019`, Adam'ın zayıf varyans kontrolü nedeniyle ayrılabileceği durumlar olduğunu göstermektedir. Bir takip çalışmasında :cite:`Zaheer.Reddi.Sachan.ea.2018` Adam için bir düzeltme önerdi, Bu sorunları giderir Yogi denilen. Bunun hakkında daha sonra. Şimdilik Adam algoritmasını gözden geçirelim.  

## Algoritma

Adam'ın temel bileşenlerinden biri, hem momentumu hem de degradenin ikinci momentunu tahmin etmek için üstel ağırlıklı hareketli ortalamaları (sızdıran ortalama olarak da bilinir) kullanmasıdır. Yani, durum değişkenleri kullanır 

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Burada $\beta_1$ ve $\beta_2$ negatif olmayan ağırlıklandırma parametreleridir. Onlar için ortak seçenekler $\beta_1 = 0.9$ ve $\beta_2 = 0.999$'dir. Yani, varyans tahmini ivme teriminden çok daha yavaş* hareket eder. $\mathbf{v}_0 = \mathbf{s}_0 = 0$'yı başlatırsak başlangıçta daha küçük değerlere karşı önemli miktarda önyargılıyız olduğunu unutmayın. Bu, $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$'ün terimleri yeniden normalleştirmesi gerçeğini kullanarak ele alınabilir. Buna göre normalleştirilmiş durum değişkenleri  

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Uygun tahminlerle donanmış olarak şimdi güncelleme denklemlerini yazabiliriz. İlk olarak, elde etmek için RMSProp çok benzer bir şekilde degrade yeniden ölçeklendirmek 

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

RMSProp'un aksine güncellememiz degradenin kendisi yerine $\hat{\mathbf{v}}_t$ momentum kullanır. Ayrıca, yeniden ölçeklendirme $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$ yerine $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ kullanılarak gerçekleştiği için hafif bir kozmetik fark vardır. Eski, pratikte tartışmasız olarak biraz daha iyi çalışır, dolayısıyla RMSProp'tan sapma. Tipik olarak biz almak $\epsilon = 10^{-6}$ sayısal istikrar ve sadakat arasında iyi bir ticaret için.  

Şimdi güncellemeleri hesaplamak için tüm parçalara sahibiz. Bu biraz antilimaktik ve formun basit bir güncellemesi var 

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Adem'in tasarımının gözden geçirilmesi, ilham kaynağı açıktır. Momentum ve ölçek durum değişkenlerinde açıkça görülebilir. Oldukça tuhaf tanımları bizi terimleri debias etmeye zorlar (bu biraz farklı bir başlatma ve güncelleme koşuluyla düzeltilebilir). İkincisi, her iki terim kombinasyonu RMSProp göz önüne alındığında oldukça basittir. Son olarak, açık öğrenme oranı $\eta$ yakınsama sorunlarını çözmek için adım uzunluğunu kontrol etmemizi sağlar.  

## Uygulama 

Adam'ı sıfırdan uygulama çok da korkutucu değil. Kolaylık sağlamak için `hyperparams` sözlüğünde saat adım sayacını $t$ saklıyoruz. Bunun ötesinde her şey basittir.

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

`adam` Gluon `trainer` optimizasyon kütüphanesinin bir parçası olarak sağlanan algoritmalardan biri olduğundan daha özlü bir uygulama basittir. Bu nedenle sadece Gluon'da bir uygulama için yapılandırma parametrelerini geçmemiz gerekiyor.

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

Adam'ın sorunlarından biri, $\mathbf{s}_t$'teki ikinci an tahmini patladığında dışbükey ayarlarda bile yakınsamayabilir. Bir düzeltme olarak :cite:`Zaheer.Reddi.Sachan.ea.2018` rafine bir güncelleme önerdi (ve başlatma) için $\mathbf{s}_t$. Neler olup bittiğini anlamak için, Adam güncellemesini aşağıdaki gibi yeniden yazalım: 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

$\mathbf{g}_t^2$ yüksek varyansa veya güncellemeler seyrek olduğunda, $\mathbf{s}_t$ geçmiş değerleri çok hızlı unutabilir. Bunun için olası bir düzeltme $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$'yı $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$ tarafından değiştirmektir. Şimdi güncellemenin büyüklüğü artık sapma miktarına bağlı değil. Bu Yogi güncellemelerini verir 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

Yazarlar ayrıca, başlangıç noktasal tahmininden ziyade daha büyük bir başlangıç partide ivmeyi başlatmayı öneriyorlar. Tartışmaya maddi olmadıkları için ayrıntıları atlıyoruz ve bu yakınsama olmadan bile oldukça iyi kalır.

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
* RMSProp temelinde oluşturulan Adam ayrıca mini batch stokastik degrade üzerinde EWMA kullanır.
* Adam, momentum ve ikinci anı tahmin ederken yavaş bir başlangıç için uyum sağlamak için önyargı düzeltmesini kullanır. 
* Önemli farklılıkları olan degradeler için yakınsama ile ilgili sorunlarla karşılaşabiliriz. Daha büyük minibüsler kullanılarak veya $\mathbf{s}_t$ için geliştirilmiş bir tahmine geçerek değiştirilebilirler. Yogi böyle bir alternatif sunuyor. 

## Egzersizler

1. Öğrenme hızını ayarlayın ve deneysel sonuçları gözlemleyip analiz edin.
1. Eğer momentum ve ikinci an güncellemeleri, önyargı düzeltme gerektirmeyecek şekilde yeniden yazabilir misiniz?
1. Neden birbirimize yaklaştığımızda $\eta$ öğrenme oranını düşürmeniz gerekiyor?
1. Adam'ın ayrıştığı ve Yogi'nin birleştiği bir dava oluşturmaya mı çalışacaksın?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
