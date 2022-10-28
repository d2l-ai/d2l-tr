# Adagrad
:label:`sec_adagrad`

Seyrek olarak ortaya çıkan özniteliklerle öğrenme problemlerini göz önünde bulundurarak başlayalım. 

## Seyrek Öznitelikler ve Öğrenme Oranları

Bir dil modelini eğittiğimizi hayal edin. İyi bir doğruluk oranı elde etmek için genellikle eğitime devam ettiğimizde, çoğunlukla $\mathcal{O}(t^{-\frac{1}{2}})$ veya daha yavaş bir hızda öğrenme oranını düşürmek isteriz. Şimdi seyrek öznitelikler üzerinde bir model eğitimi düşünün, yani, sadece ender olarak ortaya çıkan öznitelikler. Bu doğal dil için yaygındır, örn. *ön şartlandırma* kelimesini görmemiz *öğrenme* kelimesini görmemizden çok daha az olasıdır. Bununla birlikte, hesaplamalı reklamcılık ve kişiselleştirilmiş işbirlikçi filtreleme gibi diğer alanlarda da yaygındır. Sonuçta, sadece az sayıda insan için ilgi çeken birçok şey vardır. 

Sık görülen özniteliklerle ilişkili parametreler yalnızca bu öznitelikler ortaya çıktığında anlamlı güncellemeler alır. Azalan bir öğrenme oranı göz önüne alındığında, yaygın özniteliklerin parametrelerinin optimal değerlerine oldukça hızlı bir şekilde yakınsadığı bir duruma düşebiliriz, oysa seyrek öznitelikler için en uygun değerler belirlenmeden önce onları yeterince sık gözlemlemede yetersiz kalırız. Başka bir deyişle, öğrenme oranı ya sık görülen öznitelikler için çok yavaş ya da seyrek olanlar için çok hızlı azalır. 

Bu sorunu çözmenin olası bir yolu, belirli bir özniteliği kaç kez gördüğümüzü saymak ve bunu öğrenme oranlarını ayarlamak için bir taksimetre gibi kullanmak olabilir. Yani, $\eta = \frac{\eta_0}{\sqrt{t + c}}$ formunda bir öğrenme oranı seçmek yerine $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$ kullanabiliriz. Burada $s(i, t)$ $t$ zamana kadar gözlemlediğimiz öznitelik $i$ için sıfır olmayanların sayısını sayar. Aslında bunun anlamlı bir ek yük olmadan uygulanması oldukça kolaydır. Bununla birlikte, oldukça seyrekliğe sahip olmadığımızda, bunun yerine sadece gradyanların genellikle çok küçük ve nadiren büyük olduğu veriye sahip olduğumuzda başarısız olur. Ne de olsa, gözlemlenen bir şeyin bir öznitelik olarak nitelendirmek veya nitelendirmemek arasındaki çizginin nerede çekileceği belirsizdir. 

Adagrad, :cite:`Duchi.Hazan.Singer.2011` tarafından, oldukça kaba olan $s(i, t)$ sayacını daha önce gözlemlenen gradyanların karelerinin toplamı ile değiştirerek bu sorunu giderir. Özellikle, öğrenme oranını ayarlamak için bir araç olarak $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$'yı kullanır. Bunun iki yararı vardır: Birincisi, artık bir gradyanın ne zaman yeterince büyük olduğuna karar vermemiz gerekmiyor. İkincisi, gradyanların büyüklüğü ile otomatik olarak ölçeklenir. Rutin olarak büyük gradyanlara karşılık gelen koordinatlar önemli ölçüde küçültülürken, küçük gradyanlara sahip diğerleri çok daha nazik bir muamele görür. Uygulamada bu, hesaplamalı reklamcılık ve ilgili problemler için çok etkili bir optimizasyon yöntemine yol açar. Ancak bu, en iyi ön koşullandırma bağlamında anlaşılan Adagrad'ın doğasında bulunan bazı ek faydaları gizler.

## Ön Şartlandırma

Dışbükey optimizasyon problemleri algoritmaların özelliklerini analiz etmek için iyidir. Sonuçta, dışbükey olmayan sorunların çoğunda anlamlı teorik garantiler elde etmek zordur, ancak *sezgi* ve *kavrama* genellikle buraya da taşınır. $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$'yi en aza indirme sorununa bakalım. 

:numref:`sec_momentum` içinde gördüğümüz gibi, her koordinatın ayrı ayrı çözülebileceği çok basitleştirilmiş bir soruna varmak için bu sorunu özayrışma $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ açısından yeniden yazmak mümkündür: 

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Burada $\mathbf{x} = \mathbf{U} \mathbf{x}$ ve dolayısıyla $\mathbf{c} = \mathbf{U} \mathbf{c}$'yi kullandık. Değiştirilen problemde, onun küçültücüsü $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ ve minimum değeri $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$ olarak bulunur. $\boldsymbol{\Lambda}$, $\mathbf{Q}$'nun özdeğerlerini içeren köşegen bir matris olduğundan bu işlem çok daha kolaydır. 

$\mathbf{c}$'yi biraz dürtersek, $f$ küçültücüsünde yalnızca küçük değişiklikler bulmayı umarız. Ne yazık ki durum böyle değil. $\mathbf{c}$'deki küçük değişiklikler $\bar{\mathbf{c}}$'de eşit derecede küçük değişikliklere yol açarken, $f$'nin (ve $\bar{f}$'nin  sırasıyla) küçültücüsü için durum böyle değildir. Özdeğerler $\boldsymbol{\Lambda}_i$ büyük olduğunda $\bar{x}_i$'te ve minimum $\bar{f}$'de sadece küçük değişiklikler göreceğiz. Tersine, $\bar{x}_i$'teki küçük $\boldsymbol{\Lambda}_i$ değişiklikleri dramatik olabilir. En büyük ve en küçük özdeğer arasındaki oran, bir optimizasyon probleminin sağlamlık sayısı (condition number) olarak adlandırılır. 

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Sağlamlık sayısı $\kappa$ büyükse, optimizasyon problemini doğru bir şekilde çözmek zordur. Geniş bir dinamik değer aralığını doğru bir şekilde elde etme konusunda dikkatli olduğumuzdan emin olmamız gerekir. Analizlerimiz bariz, ama biraz naif bir soruya yol açıyor: Problemi, tüm özdeğerler $1$ olacak şekilde uzayı bozarak sorunu basitçe "düzeltemez miyiz?" Teorik olarak bu oldukça kolaydır: Problemi $\mathbf{x}$'den $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$'te bir taneye yeniden ölçeklendirmek için $\mathbf{Q}$'nin özdeğerlerine ve özvektörlerine ihtiyacımız var. Yeni koordinat sisteminde $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$, $\|\mathbf{z}\|^2$'ye basitleştirilebilir. Ne yazık ki, bu oldukça pratik olmayan bir öneri. Özdeğerleri ve özvektörleri hesaplama genel olarak gerçek problemi çözmekten *çok daha* pahalıdır. 

Özdeğerlerin tam hesaplanması pahalı olsa da, onları tahmin etmek ve hatta biraz yaklaşık hesaplama yapmak, hiçbir şey yapmamaktan çok daha iyi olabilir. Özellikle, $\mathbf{Q}$'nun köşegen girdilerini kullanabilir ve buna göre yeniden ölçekleyebiliriz. Bu, özdeğerlerin hesaplanmasından *çok daha* ucuzdur. 

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Bu durumda $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ ve özellikle tüm $i$ için $\tilde{\mathbf{Q}}_{ii} = 1$ olur. Çoğu durumda bu sağlamlık sayısını önemli ölçüde basitleştirir. Örneğin, daha önce tartıştığımız vakalarda, problem eksen hizalandığından eldeki sorunu tamamen ortadan kaldıracaktır. 

Ne yazık ki başka bir sorunla karşı karşıyayız: Derin öğrenmede genellikle amaç fonksiyonun ikinci türevine bile erişimimiz yok: $\mathbf{x} \in \mathbb{R}^d$ için bile bir minigrup üzerinde ikinci türev $\mathcal{O}(d^2)$'lik alan ve hesaplama için çalışma gerektirebilir, bu yüzden pratikte imkansız hale gelir. Adagrad'ın dahiyane fikri, Hessian'ın bu anlaşılması zor köşegeni için hem hesaplanması nispeten ucuz hem de etkili olan bir vekil kullanmasıdır - gradyanın kendisinin büyüklüğü.

Bunun neden çalıştığını görmek için $\bar{f}(\bar{\mathbf{x}})$'e bakalım. Elimizde bu var: 

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

Burada $\bar{\mathbf{x}}_0$, $\bar{f}$'nin küçültücüsüdür. Bu nedenle gradyanın büyüklüğü hem $\boldsymbol{\Lambda}$'ya hem de eniyi değere olan mesafeye bağlıdır. $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ değişmeseydi, gereken tek şey bu olurdu. Sonuçta, bu durumda $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ gradyanın büyüklüğü yeterlidir. AdaGrad bir rasgele gradyan inişi algoritması olduğundan, eniyi düzeyde bile sıfır olmayan varyansı olan gradyanları göreceğiz. Sonuç olarak, gradyanların varyansını Hessian ölçeği için ucuz bir vekil olarak güvenle kullanabiliriz. Kapsamlı bir analiz, bu bölümün kapsamının dışındadır (olsaydı birkaç sayfa olacaktı). Ayrıntılar için okuyucuyu :cite:`Duchi.Hazan.Singer.2011`'e yönlendiriyoruz. 

## Algoritma

Yukarıdaki tartışmayı formüle dökelim. Geçmiş gradyan varyansı aşağıdaki gibi biriktirmek için $\mathbf{s}_t$ değişkenini kullanıyoruz. 

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Burada işlem koordinat yönlü olarak akıllıca uygulanır. Yani, $\mathbf{v}^2$'nin $v_i^2$ girdileri vardır. Aynı şekilde $\frac{1}{\sqrt{v}}$'nin $\frac{1}{\sqrt{v_i}}$  girdileri ve $\mathbf{u} \cdot \mathbf{v}$'nin $u_i v_i$ girdileri vardır. Daha önce olduğu gibi $\eta$ öğrenme oranıdır ve $\epsilon$ $0$ ile bölmememizi sağlayan bir katkı sabitidir. Son olarak, $\mathbf{s}_0 = \mathbf{0}$'ı ilkleriz. 

Tıpkı momentum durumunda olduğu gibi, koordinat başına bireysel bir öğrenme oranına izin vermek için yardımcı bir değişkeni takip etmeliyiz. Bu, ana maliyet tipik olarak $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ ve türevini hesaplamak olduğundan, SGD'ye göre Adagrad'ın maliyetini önemli ölçüde artırmaz. 

$\mathbf{s}_t$'te kare gradyanların biriktirilmesinin $\mathbf{s}_t$'in esas olarak doğrusal hızda büyüdüğü anlamına geldiğini unutmayın (gradyanlar başlangıçta azaldığından, pratikte doğrusal olandan biraz daha yavaş). Bu, koordinat tabanında ayarlanmış olsa da $\mathcal{O}(t^{-\frac{1}{2}})$ öğrenme hızına yol açar. Dışbükey problemler için bu mükemmel bir şekilde yeterlidir. Derin öğrenmede, öğrenme oranını daha yavaş düşürmek isteyebiliriz. Bu, sonraki bölümlerde tartışacağımız bir dizi Adagrad varyantına yol açtı. Şimdilik, ikinci deredecen polinom dışbükey bir problemde nasıl davrandığını görelim. Daha önce olduğu gibi aynı problemi kullanıyoruz: 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Adagrad'ı daha önceki aynı öğrenme oranını kullanarak uygulayacağız, yani $\eta = 0.4$. Gördüğümüz gibi bağımsız değişkenin yinelemeli yörüngesi daha pürüzsüzdür. Bununla birlikte, $\boldsymbol{s}_t$'nin biriktirici etkisi nedeniyle, öğrenme oranı sürekli olarak söner, bu nedenle bağımsız değişken yinelemenin sonraki aşamalarında çok fazla hareket etmez.

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

Öğrenme oranını $2$'ye yükselttikçe çok daha iyi davranışlar görüyoruz. Bu durum, öğrenme oranındaki düşüşün gürültüsüz durumda bile oldukça saldırgan olabileceğini gösteriyor ve parametrelerin uygun şekilde yakınsamasını sağlamalıyız.

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

:numref:`sec_minibatch_sgd` içindeki deney ile karşılaştırıldığında, modeli eğitmek için daha büyük bir öğrenme hızı kullanıyoruz.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Kısa Uygulama

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

* Adagrad, koordinat tabanında öğrenme oranını dinamik olarak düşürür.
* Gradyanın büyüklüğünü, ilerlemenin ne kadar hızlı bir şekilde elde edildiğini ayarlamak için bir araç olarak kullanır - büyük gradyanlara sahip koordinatlar daha küçük bir öğrenme oranı ile telafi edilir.
* İkinci türevi tam hesaplamak, bellek ve hesaplama kısıtlamaları nedeniyle derin öğrenme problemlerinde genellikle mümkün değildir. Gradyan yararlı bir vekil olabilir.
* Optimizasyon problemi oldukça düzensiz bir yapıya sahipse Adagrad bozulmayı azaltmaya yardımcı olabilir.
* Adagrad, özellikle seyrek olarak ortaya çıkan terimler için öğrenme oranının daha yavaş azalması gereken seyrek öznitelikler için etkilidir.
* Derin öğrenme problemleri üzerinde Adagrad bazen öğrenme oranlarını düşürmede çok saldırgan olabilir. :numref:`sec_adam` bağlamında bunu hafifletmek için stratejileri tartışacağız.

## Alıştırmalar

1. Bir dik matris $\mathbf{U}$ ve bir vektör $\mathbf{c}$ için şunu kanıtlayın: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Bu neden, değişkenlerin dik değişiminden sonra dürtmelerin büyüklüğünün değişmediği anlamına geliyor?
1. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ ve ayrıca amaç fonksiyonunun 45 derece, yani $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ ile döndürüldüğü durum için Adagrad'ı deneyin. Farklı davranıyor mu?
1. $\mathbf{M}$ matrisinin $\lambda_i$ özdeğerlerinin $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$'yi en az bir $j$ seçeneği için [Gerschgorin çember teoremini](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) sağladığını kanıtlayın. 
1. Gerschgorin teoremi, çapraz olarak önceden koşullandırılmış matrisin $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$'nin özdeğerleri hakkında bize ne anlatıyor?
1. Fashion MNIST'e uygulandığında :numref:`sec_lenet` gibi uygun bir derin ağ için Adagrad'ı deneyin.
1. Öğrenme oranında daha az saldırgan bir sönme elde etmek için Adagrad'ı nasıl değiştirmeniz gerekir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1073)
:end_tab:
