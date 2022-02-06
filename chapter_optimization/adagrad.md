# Adagrad
:label:`sec_adagrad`

Seyrek olarak ortaya çıkan özniteliklerle öğrenme problemlerini göz önünde bulundurarak başlayalım. 

## Seyrek Öznitelikler ve Öğrenme Oranları

Bir dil modelini eğittiğimizi hayal edin. İyi bir doğruluk oranı elde etmek için genellikle eğitime devam ettiğimizde, çoğunlukla $\mathcal{O}(t^{-\frac{1}{2}})$ veya daha yavaş bir hızda öğrenme oranını düşürmek isteriz. Şimdi seyrek öznitelikler üzerinde bir model eğitimi düşünün, yani, sadece ender olarak ortaya çıkan öznitelikler. Bu doğal dil için yaygındır, örn. *ön şartlandırma* kelimesini görmemiz *öğrenme* kelimesini görmemizden çok daha az olasıdır. Bununla birlikte, hesaplamalı reklamcılık ve kişiselleştirilmiş işbirlikçi filtreleme gibi diğer alanlarda da yaygındır. Sonuçta, sadece az sayıda insan için ilgi çeken birçok şey vardır. 

Sık görülen özniteliklerle ilişkili parametreler yalnızca bu öznitelikler ortaya çıktığında anlamlı güncellemeler alır. Azalan bir öğrenme oranı göz önüne alındığında, yaygın özniteliklerin parametrelerinin optimal değerlerine oldukça hızlı bir şekilde yakınlaştığı bir duruma düşebiliriz, oysa seyrek öznitelikler için en uygun değerler belirlenmeden önce onları yeterince sık gözlemlemekten kıt durumdayız. Başka bir deyişle, öğrenme oranı ya sık görülen öznitelikler için çok yavaş ya da seyrek olanlar için çok hızlı bir şekilde azalır. 

Bu sorunu çözmenin olası bir yolu, belirli bir özniteliği kaç kez gördüğümüzü saymak ve bunu öğrenme oranlarını ayarlamak için bir taksimetre gibi kullanmak olabilir. Yani, $\eta = \frac{\eta_0}{\sqrt{t + c}}$ formunun bir öğrenme oranını seçmek yerine $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$ kullanabiliriz. Burada $s(i, t)$ $t$ zamana kadar gözlemlediğimiz öznitelik $i$ için sıfır olmayanların sayısını sayar. Bunun aslında anlamlı bir ek yük olmadan uygulanması oldukça kolaydır. Bununla birlikte, oldukça seyrekliğe sahip olmadığımızda, bunun yerine sadece gradyanların genellikle çok küçük ve nadiren büyük olduğu veriye sahip olduğumuzda başarısız olur. Ne de olsa, gözlemlenen bir şeyin bir özellik olarak nitelendirmek veya nitelendirmemek arasındaki çizginin nerede çekileceği belirsizdir. 

Adagrad, :cite:`Duchi.Hazan.Singer.2011` tarafından, oldukça kaba olan $s(i, t)$ sayacını daha önce gözlemlenen gradyanların karelerinin bir toplamı ile değiştirerek bu sorunu giderir. Özellikle, öğrenme oranını ayarlamak için bir araç olarak $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$'yı kullanır. Bunun iki yararı vardır: Birincisi, artık bir gradyanın ne zaman yeterince büyük olduğuna karar vermemiz gerekmiyor. İkincisi, gradyanların büyüklüğü ile otomatik olarak ölçeklenir. Rutin olarak büyük gradyanlara karşılık gelen koordinatlar önemli ölçüde küçültülürken, küçük gradyanlara sahip diğerleri çok daha nazik bir muamele görür. Uygulamada bu, hesaplamalı reklamcılık ve ilgili problemler için çok etkili bir optimizasyon yöntemine yol açar. Ancak bu, Adagrad'da ön şartlandırma bağlamında en iyi anlaşılan ek faydalardan bazılarını gizler. 

## Ön Şartlandırma

Dışbükey optimizasyon problemleri algoritmaların özelliklerini analiz etmek için iyidir. Sonuçta, dışbükey olmayan sorunların çoğunda anlamlı teorik garantiler elde etmek zordur, ancak *sezgisition* ve *anlayış* genellikle devrilir. $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$'ü en aza indirme sorununa bakalım. 

:numref:`sec_momentum`'te gördüğümüz gibi, her koordinatın ayrı ayrı çözülebileceği çok basitleştirilmiş bir soruna varmak için bu sorunu özdekompozisyon $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ açısından yeniden yazmak mümkündür: 

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Burada $\mathbf{x} = \mathbf{U} \mathbf{x}$ ve dolayısıyla $\mathbf{c} = \mathbf{U} \mathbf{c}$ kullandık. Değiştirilen sorun, minimizer $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ ve minimum değer $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$ olarak vardır. $\boldsymbol{\Lambda}$, $\mathbf{Q}$'in özdeğerlerini içeren diyagonal bir matris olduğundan bu işlem çok daha kolaydır. 

$\mathbf{c}$'i biraz rahatsız edersek, $f$'nın minimizöründe sadece hafif değişiklikler bulmayı umuyoruz. Ne yazık ki durum böyle değil. $\mathbf{c}$'deki hafif değişiklikler $\bar{\mathbf{c}}$'da eşit derecede hafif değişikliklere yol açsa da, $f$ (ve sırasıyla $\bar{f}$) minimize edici durum böyle değildir. Özdeğerler $\boldsymbol{\Lambda}_i$ büyük olduğunda $\bar{x}_i$'te ve minimum $\bar{f}$'de sadece küçük değişiklikler göreceğiz. Tersine, $\bar{x}_i$'teki küçük $\boldsymbol{\Lambda}_i$ değişiklikleri dramatik olabilir. En büyük ve en küçük özdeğer arasındaki oran, bir optimizasyon sorununun koşul numarası olarak adlandırılır. 

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Durum numarası $\kappa$ büyükse, optimizasyon sorununu doğru bir şekilde çözmek zordur. Büyük bir dinamik değer aralığını doğru şekilde elde etmemize dikkat etmeliyiz. Analizlerimiz bariz, biraz naif bir soruya yol açıyor: Sorunu, tüm özdeğerler $1$ olacak şekilde alanı bozarak sorunu basitçe “düzeltemez miyiz”. Teorik olarak bu oldukça kolaydır: sorunu $\mathbf{x}$'dan $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$'te bir taneye yeniden ölçeklendirmek için $\mathbf{Q}$'nin özdeğerlerine ve özvektörlerine ihtiyacımız var. Yeni koordinat sisteminde $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ $\|\mathbf{z}\|^2$ basitleştirilebilir. Ne yazık ki, bu oldukça pratik bir öneri. Özdeğerler ve özvektörleri hesaplama genel olarak gerçek problemi çözmekten çok daha pahalı* pahalıdır. 

Özdeğerlerin hesaplanması tam olarak pahalı olsa da, onları tahmin etmek ve hatta biraz hesaplama yapmak zaten hiçbir şey yapmamaktan çok daha iyi olabilir. Özellikle, $\mathbf{Q}$'ün diyagonal girişlerini kullanabilir ve buna göre yeniden ölçekleyebiliriz. Bu, özdeğerlerin hesaplanmasından çok daha ucuzdur. 

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Bu durumda $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ ve özellikle $\tilde{\mathbf{Q}}_{ii} = 1$ tümü için $i$ var. Çoğu durumda bu durum durum numarasını önemli ölçüde basitleştirir. Örneğin, daha önce tartıştığımız vakalar, sorun eksen hizalandığı için eldeki sorunu tamamen ortadan kaldıracaktır. 

Ne yazık ki başka bir sorunla karşı karşıyayız: Derin öğrenmede genellikle objektif fonksiyonun ikinci türevine bile erişimimiz yok: $\mathbf{x} \in \mathbb{R}^d$ için bile bir minibatch üzerinde ikinci türev $\mathcal{O}(d^2)$ alan ve hesaplamak için çalışma gerektirebilir, böylece pratik olarak imkansız hale. Adagrad'ın ustaca fikri Hessian'ın zor diyagonal diyagonal için hem hesaplaması nispeten ucuz hem de etkili olan bir vekil kullanmaktır. 

Bunun neden çalıştığını görmek için $\bar{f}(\bar{\mathbf{x}})$'e bakalım. Elimizde bu var. 

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

burada $\bar{\mathbf{x}}_0$, $\bar{f}$'in en aza indiricisidir. Bu nedenle degradenin büyüklüğü hem $\boldsymbol{\Lambda}$'ye hem de optimaliteye olan mesafeye bağlıdır. $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ değişmeseydi, gereken tek şey bu olurdu. Sonuçta, bu durumda $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ degradesinin büyüklüğü yeterlidir. AdaGrad bir stokastik degrade iniş algoritması olduğundan, optimum düzeyde bile sıfır olmayan varyansı olan degradeleri göreceğiz. Sonuç olarak, degradelerin varyansını Hessian ölçeği için ucuz bir vekil olarak güvenle kullanabiliriz. Kapsamlı bir analiz, bu bölümün kapsamının dışındadır (birkaç sayfa olacaktır). Ayrıntılar için okuyucuyu :cite:`Duchi.Hazan.Singer.2011`'e yönlendiriyoruz. 

## Algoritma

Tartışmayı yukarıdan resmileştirelim. Geçmiş degrade varyansı aşağıdaki gibi biriktirmek için $\mathbf{s}_t$ değişkenini kullanıyoruz. 

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Burada işlem koordinat akıllıca uygulanır. Yani, $\mathbf{v}^2$ girdileri var $v_i^2$. Aynı şekilde $\frac{1}{\sqrt{v}}$ girdileri vardır $\frac{1}{\sqrt{v_i}}$ ve $\mathbf{u} \cdot \mathbf{v}$ girdileri vardır $u_i v_i$. Daha önce olduğu gibi $\eta$ öğrenme oranı ve $\epsilon$ $0$ ile bölmememizi sağlayan bir katkı sabitidir. Son olarak, $\mathbf{s}_0 = \mathbf{0}$'ü başlatırız. 

Tıpkı momentum durumunda olduğu gibi, koordinat başına bireysel bir öğrenme oranına izin vermek için yardımcı bir değişkeni takip etmeliyiz. Bu, ana maliyet tipik olarak $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ ve türevini hesaplamak olduğu için, SGD'ye göre Adagrad'ın maliyetini önemli ölçüde artırmaz. 

$\mathbf{s}_t$'te kare degradelerin biriktirilmesinin $\mathbf{s}_t$'in esas olarak doğrusal hızda büyüdüğü anlamına geldiğini unutmayın (gradyanlar başlangıçta azaldığından, pratikte doğrusal olandan biraz daha yavaş). Bu, koordinat bazında ayarlanmış olsa da $\mathcal{O}(t^{-\frac{1}{2}})$ öğrenme hızına yol açar. Dışbükey problemler için bu mükemmel bir şekilde yeterlidir. Derin öğrenmede, öğrenme oranını daha yavaş düşürmek isteyebiliriz. Bu, sonraki bölümlerde tartışacağımız bir dizi Adagrad varyantına yol açtı. Şimdilik, kuadratik dışbükey bir problemde nasıl davrandığını görelim. Daha önce olduğu gibi aynı sorunu kullanıyoruz: 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Adagrad'ı daha önce aynı öğrenme oranını kullanarak uygulayacağız, yani $\eta = 0.4$. Gördüğümüz gibi bağımsız değişkenin yinelemeli yörüngesi daha pürüzsüzdür. Bununla birlikte, $\boldsymbol{s}_t$'ün kümülatif etkisi nedeniyle, öğrenme hızı sürekli olarak bozulur, bu nedenle bağımsız değişken yinelemenin sonraki aşamalarında çok fazla hareket etmez.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Öğrenme oranını $2$'e yükselttikçe çok daha iyi davranışlar görüyoruz. Bu durum, öğrenme oranındaki düşüşün gürültüsüz durumda bile oldukça agresif olabileceğini gösteriyor ve parametrelerin uygun şekilde yakınlaşmasını sağlamalıyız.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Sıfırdan Uygulama

Tıpkı momentum yöntemi gibi, Adagrad'ın parametrelerle aynı şekle sahip bir durum değişkenini koruması gerekiyor.

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

:numref:`sec_minibatch_sgd`'teki deney ile karşılaştırıldığında, modeli eğitmek için daha büyük bir öğrenme hızı kullanıyoruz.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Özlü Uygulama

`adagrad` algoritmasının `Trainer` örneğini kullanarak, Gluon'daki Adagrad algoritmasını çağırabiliriz.

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Özet

* Adagrad, koordinat bazında öğrenme oranını dinamik olarak düşürür.
* Degradenin büyüklüğünü, ilerlemenin ne kadar hızlı bir şekilde elde edildiğini ayarlamak için bir araç olarak kullanır - büyük degradelere sahip koordinatlar daha küçük bir öğrenme oranı ile telafi edilir.
* Tam ikinci türevi hesaplamak, bellek ve hesaplama kısıtlamaları nedeniyle derin öğrenme problemlerinde genellikle mümkün değildir. Degrade yararlı bir proxy olabilir.
* Optimizasyon problemi oldukça düzensiz bir yapıya sahipse Adagrad bozulmayı azaltmaya yardımcı olabilir.
* Adagrad, seyrek görülen terimler için öğrenme oranının daha yavaş azalması gereken seyrek özellikler için özellikle etkilidir.
* Derin öğrenme problemleri üzerine Adagrad bazen öğrenme oranlarını düşürmede çok agresif olabilir. :numref:`sec_adam` bağlamında bunu hafifletmek için stratejileri tartışacağız.

## Alıştırmalar

1. Bir ortogonal matris $\mathbf{U}$ ve bir vektör $\mathbf{c}$ için aşağıdaki tutar kanıtlayın: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Bu neden, değişkenlerin dikdörtgen değişiminden sonra pertürbasyonların büyüklüğünün değişmediği anlamına geliyor?
1. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ için Adagrad'ı deneyin ve ayrıca objektif fonksiyon için 45 derece, yani $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ ile döndürüldü. Farklı davranıyor mu?
1. Kanıtlayın [Gerschgorin çember teoremi](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) bir matrisin $\lambda_i$ özdeğerleri $\mathbf{M}$ $\mathbf{M}$ en az bir seçim için $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ tatmin olduğunu belirtir.
1. Gerschgorin teoremi, çapraz olarak önceden koşullandırılmış matrisin $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$'ün özdeğerleri hakkında bize ne anlatıyor?
1. Moda MNIST'e uygulandığında :numref:`sec_lenet` gibi uygun bir derin ağ için Adagrad'ı deneyin.
1. Öğrenme hızında daha az saldırgan bir bozulma elde etmek için Adagrad'ı nasıl değiştirmeniz gerekir?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
