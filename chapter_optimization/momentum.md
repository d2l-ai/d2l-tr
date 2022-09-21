# Momentum
:label:`sec_momentum`

:numref:`sec_sgd` içinde, rasgele gradyan inişini gerçekleştirirken, yani, gradyanın yalnızca gürültülü bir sürümünün mevcut olduğu optimizasyonu gerçekleştirirken neler olduğunu inceledik. Özellikle, gürültülü gradyanlar için gürültü karşısında öğrenme oranını seçmek söz konusu olduğunda aşırı temkinli olmamız gerektiğini fark ettik. Eğer çok hızlı azaltırsak yakınsama durur. Eğer çok hoşgörülü olursak, gürültü bizi eniyi cevaptan uzaklaştırmaya devam ettiğinden, yeterince iyi bir çözüme yaklaşmayı başaramayız. 

## Temel Bilgiler

Bu bölümde, özellikle uygulamada yaygın olan belirli optimizasyon problemleri türleri için daha etkili optimizasyon algoritmaları araştıracağız. 

### Sızdıran Ortalamalar

Önceki bölümde, hesaplamayı hızlandırmak için bir araç olarak minigrup SGD tartışmamızı gördünüz. Ayrıca, gradyanların ortalama varyans miktarını azalttığı güzel yan etkisi de vardı. Minigrup rasgele gradyan inişi şu şekilde hesaplanabilir: 

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Notasyonu basit tutmak için, $t-1$ zamanında güncellenen ağırlıkları kullanarak $i$ örneğinin rasgele gradyan inişi olarak $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ kullandık. Bir minigrup üzerindeki gradyanların ortalamasının ötesinde bile varyans azaltmanın etkisinden faydalanabilsek güzel olurdu. Bu görevi yerine getirmek için bir seçenek gradyan hesaplamayı “sızıntılı ortalama” ile değiştirmektir: 

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

bazı $\beta \in (0, 1)$ için geçerlidir. Bu, anlık gradyanı birden çok *geçmiş* gradyan üzerinden ortalama alınmış bir gradyan ile etkili bir şekilde değiştirir. $\mathbf{v}$ *momentum* olarak adlandırılır. Amaç fonksiyonu yokuştan aşağı yuvarlanan ağır bir topun geçmiş kuvvetler üzerinde integral alması benzer şekilde geçmiş gradyanları biriktirir. Daha ayrıntılı olarak neler olup bittiğini görmek için $\mathbf{v}_t$'i özyinelemeli olarak aşağıdaki gibi açalım:

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

Büyük $\beta$, uzun menzilli ortalamaya karşılık gelirken, küçük $\beta$ ise gradyan yöntemine göre yalnızca hafif bir düzeltme anlamına gelir. Yeni gradyan değişimi artık belirli bir örnekte en dik iniş yönünü değil, geçmiş gradyanların ağırlıklı ortalamasını işaret ediyor. Bu, üzerinde gradyanları hesaplama maliyeti olmadan bir toplu iş üzerinde ortalamanın faydalarının çoğunu gerçekleştirmemize olanak tanır. Bu ortalama yordamını daha sonra daha ayrıntılı olarak tekrar gözden geçireceğiz. 

Yukarıdaki akıl yürütme, şimdi momentumlu gradyanlar gibi *hızlandırılmış* gradyan yöntemleri olarak bilinen şeyin temelini oluşturmuştur. Optimizasyon probleminin kötü koşullu olduğu durumlarda (yani ilerlemenin diğerlerinden çok daha yavaş olduğu, dar bir kanyona benzeyen bazı yönlerin olduğu yerlerde) çok daha etkili olmanın ek avantajından yararlanırlar. Ayrıca, daha kararlı iniş yönleri elde etmek için sonraki gradyanlar üzerinde ortalama yapmamıza izin verirlar. Gerçekten de, gürültüsüz dışbükey problemler için bile ivmenin yönü, momentumun neden çalıştığı ve neden bu kadar iyi çalıştığının temel nedenlerinden biridir. 

Beklendiği gibi, etkinliği nedeniyle ivme derin öğrenme ve ötesinde optimizasyonda iyi çalışılmış bir konudur. Örneğin, ayrıntılı bir analiz ve etkileşimli animasyon için güzel [açıklayıcı makale](https://distill.pub/2017/momentum/)'ye :cite:`Goh.2017` bakın. Bu :cite:`Polyak.1964` tarafından önerilmiştir. :cite:`Nesterov.2018` dışbükey optimizasyon bağlamında ayrıntılı bir teorik tartışma sunar. Derin öğrenmede momentumun uzun zamandır faydalı olduğu bilinmektedir. Ayrıntılar için örneğin :cite:`Sutskever.Martens.Dahl.ea.2013` çalışmasındaki tartışmalarına bakın. 

### Kötü Koşullu Bir Problem

Momentum yönteminin geometrik özelliklerini daha iyi anlamak için, önemli ölçüde daha az hoş bir amaç fonksiyonu ile de olsa, gradyan inişini tekrar gözden geçiriyoruz. :numref:`sec_gd` içinde $f(\mathbf{x}) = x_1^2 + 2 x_2^2$'i, yani orta miktarda çarpık elipsoid bir amaç kullandığımızı hatırlayın. Bu işlevi $x_1$ yönünde uzatarak daha da bozuyoruz 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Daha önce olduğu gibi $f$ $(0, 0)$'da minimum seviyeye sahiptir. Bu fonksiyon $x_1$ yönünde *çok* düzdür. Bu yeni işlevde daha önce olduğu gibi gradyan inişi gerçekleştirdiğimizde ne olacağını görelim. Öğrenme oranını $0.4$ seçiyoruz.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Yapı gereği, $x_2$ yönündeki gradyan *çok daha* yüksek ve yatay $x_1$ yönünden çok daha hızlı değişiyor. Böylece iki istenmeyen seçenek arasında sıkışmış olduk: Küçük bir öğrenme oranı seçersek, çözümün $x_2$ yönünde ıraksamamasını sağlarız, ancak $x_1$ yönünde yavaş yakınsama ile eyerleniriz. Tersine, büyük bir öğrenme oranı ile $x_1$ yönünde hızla ilerliyoruz ancak $x_2$'te ıraksıyoruz. Aşağıdaki örnek, $0.4$'ten $0.6$'ya kadar öğrenme hızında hafif bir artıştan sonra bile neler olduğunu göstermektedir. $x_1$ yönündeki yakınsama gelişir ancak genel çözüm kalitesi çok daha kötüdür.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### Momentum Yöntemi

Momentum yöntemi, yukarıda açıklanan gradyan inişi problemini çözmemizi sağlar. Yukarıdaki optimizasyon izine baktığımızda geçmişteki gradyanların ortalama işe yarayacağını düşünebiliriz. Sonuçta, $x_1$ yönünde bu, iyi hizalanmış gradyanları bir araya getirecek ve böylece her adımda kapladığımız mesafeyi artıracaktır. Tersine, gradyanların salınım yaptığı $x_2$ yönünde, bir toplam gradyan birbirini iptal eden salınımlar nedeniyle adım boyutunu azaltacaktır. $\mathbf{g}_t$ gradyan yerine $\mathbf{v}_t$ kullanarak aşağıdaki güncelleştirme denklemlerini verir: 

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

$\beta = 0$ için düzenli gradyan inişini kurtardığımızı unutmayın. Matematiksel özellikleri derinlemesine incelemeden önce, algoritmanın pratikte nasıl davrandığına hızlı bir göz atalım.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Gördüğümüz gibi, daha önce kullandığımız aynı öğrenme oranıyla bile, momentum hala iyi bir şekilde yakınsıyor. Momentum parametresini azalttığımızda neler olacağını görelim. $\beta = 0.25$'e kadar yarıya indirmek, neredeyse hiç yakınsamayan bir yörüngeye yol açar. Bununla birlikte, momentumsuz halinden çok daha iyidir (çözüm ıraksadığı zaman).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Rasgele gradyan inişi, özellikle minigrup rasgele gradyan inişi ile momentumu birleştirebileceğinizi unutmayın. Tek değişiklik, bu durumda $\mathbf{g}_{t, t-1}$ gradyanlarını $\mathbf{g}_t$ ile değiştirmemizdir. Son olarak, kolaylık sağlamak için $\mathbf{v}_0 = 0$'ı $t=0$'da ilklettik. Sızdıran ortalamaların güncellemelere ne yaptığına bakalım. 

### Etkili Örneklem Ağırlığı

Hatırlayalım $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. Limitte terimler $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$'ye toplanır. Başka bir deyişle, gradyan inişte veya rasgele gradyan inişte $\eta$ boyutunda bir adım atmak yerine, $\frac{\eta}{1-\beta}$ boyutunda bir adım atıyoruz ve aynı zamanda potansiyel olarak çok daha iyi davranan bir iniş yönüyle uğraşıyoruz. Bunlar ikisi bir arada faydalardır. $\beta$'yı farklı seçenekler için ağırlıklandırmanın nasıl davrandığını göstermek için aşağıdaki diyagramı göz önünde bulundurun.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Pratik Deneyler

Momentumun pratikte nasıl çalıştığını görelim, yani, uygun bir iyileştirici bağlamında kullanıldığında. Bunun için biraz daha ölçeklenebilir bir uygulamaya ihtiyacımız var. 

### Sıfırdan Uygulama

(Minigrup) rasgele gradyan inişi ile karşılaştırıldığında, momentum yönteminin bir dizi yardımcı değişken muhafaza etmesi gerekir, mesela hız. Gradyanlar (ve en iyileştirme probleminin değişkenleri) ile aynı şekle sahiptir. Aşağıdaki uygulamada bu değişkenleri `states` olarak adlandırıyoruz.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Bunun pratikte nasıl çalıştığını görelim.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Momentum hiper parametresi `momentum`'u 0.9'a yükselttiğimizde, bu, $\frac{1}{1 - 0.9} = 10$ gibi önemli ölçüde daha büyük bir etkin örneklem boyutu anlamına gelir. Sorunları kontrol altında tutmak için öğrenme oranını azar azar $0.01$'a indiriyoruz.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

Öğrenme oranını azaltmak, pürüzsüz olmayan optimizasyon problemleriyle ilgili her türlü sorunu daha da giderir. $0.005$ olarak ayarlamak iyi yakınsama özellikleri sağlar.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Özlü Uygulama

Standart `sgd` çözücüde momentum zaten yerleşik olduğundan Gluon'da yapılacak çok az şey var. Eşleşen parametrelerin ayarlanması çok benzer bir yörünge oluşturur.

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Kuramsal Analiz

Şimdiye kadar $f(x) = 0.1 x_1^2 + 2 x_2^2$'nin 2D örneği oldukça yapılandırılmış görünüyordu. Şimdi bunun, en azından dışbükey ikinci dereceden amaç fonksiyonlarının en aza indirilmesi durumunda karşılaşabileceği problem türlerinin açık bir temsilcisi olduğunu göreceğiz. 

### İkinci Dereceden Dışbükey Fonksiyonlar

Aşağıdaki işlevi düşünün: 

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Bu genel bir ikinci dereceden fonksiyondur. Pozitif kesin matrisler $\mathbf{Q} \succ 0$ için, yani, pozitif özdeğerlere sahip matrisler için, bu minimum değer $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$ ile $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ değerinde bir en küçük değer bulucuya sahiptir. Bu nedenle $h$'yi yeniden yazabiliriz 

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

Gradyan $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ ile verilir. Yani, $\mathbf{x}$ ile $\mathbf{Q}$ ile çarpılan en küçük değer bulucu arasındaki mesafe ile verilir. Sonuç olarak da momentum $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ terimlerinin doğrusal bir kombinasyonudur. 

$\mathbf{Q}$ pozitif kesin olduğundan $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ üzerinden bir dikey (çevirme) matrisi $\mathbf{O}$ ve  köşegen matrisi $\boldsymbol{\Lambda}$ için pozitif özdeğerlerine ayrıştırılabilir. Bu, çok basitleştirilmiş bir ifade elde etmek için $\mathbf{x}$'den $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$'ye değişken değişikliği yapmamıza olanak tanır: 

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Burada $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. $\mathbf{O}$ sadece bir dikey matris olduğundan bu durum gradyanları anlamlı bir şekilde bozmaz. $\mathbf{z}$  ile ifade edilirse gradyan inişi aşağıdaki ifadeye dönüşür: 

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Bu ifadedeki önemli gerçek gradyan inişinin farklı özuzaylar arasında *karışmadığıdır*. Yani, $\mathbf{Q}$'nun özsistemi açısından ifade edildiğinde optimizasyon problemi koordinat-yönlü bir şekilde ilerlemektedir. Bu aynı zamanda momentum için de geçerlidir. 

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Bunu yaparken ayrıca şu teoremi kanıtladık: Dışbükey bir ikinci derece fonksiyon için ivmeli ve ivmesiz gradyan inişi, ikinci dereceden matrisin özvektörleri yönünde koordinat yönlü optimizasyonuna ayrışır. 

### Skaler Fonksiyonlar

Yukarıdaki sonuç göz önüne alındığında, $f(x) = \frac{\lambda}{2} x^2$ işlevini en aza indirdiğimizde neler olduğunu görelim. Gradyan inişi için 

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

$|1 - \eta \lambda| < 1$ olduğunda bu optimizasyon üstel bir oranda yakınsar çünkü $t$ adımdan sonra $x_t = (1 - \eta \lambda)^t x_0$ olur. Bu, $\eta \lambda = 1$'e kadar $\eta$ öğrenme oranını artırdıkça yakınsama oranının başlangıçta nasıl arttığını gösterir. Bunun ötesinde ıraksar ve $\eta \lambda > 2$ için optimizasyon problemi ıraksıyor.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Momentum durumunda yakınsaklığı analiz etmek için güncelleme denklemlerini iki skaler açısından yeniden yazarak başlıyoruz: Biri $x$ ve diğeri momentum $v$ için. Bu şuna yol açar: 

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$
 
$\mathbf{R}$'ı, yakınsama davranışını yöneten $2 \times 2$'yi göstermek için kullandık. $t$ adımdan sonra ilk tercih $[v_0, x_0]$, $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$ olur. Bu nedenle, yakınsama hızını belirlemek $\mathbf{R}$'nin özdeğerlerine kalmıştır. Harika bir animasyon için :cite:`Goh.2017` [Distill post](https://distill.pub/2017/momentum/)una ve ayrıntılı analiz için :cite:`Flammarion.Bach.2015` bölümüne bakın. $0 < \eta \lambda < 2 + 2 \beta$ olduğunda momentumun yakınsadığını gösterilebilir. Bu, gradyan inişi için $0 < \eta \lambda < 2$ ile karşılaştırıldığında daha geniş bir uygulanabilir parametre aralığıdır. Ayrıca, genel olarak $\beta$'nin büyük değerlerinin arzu edildiğini de göstermektedir. Daha fazla ayrıntı, makul miktarda teknik detay gerektirir ve ilgilenen okuyucunun orijinal yayınlara başvurmasını öneririz. 

## Özet

* Momentum, geçmiş gradyanlara göre sızan bir ortalama ile gradyanların yerini alır. Bu, yakınsamayı önemli ölçüde hızlandırır.
* Hem gürültüsüz gradyan inişi hem de (gürültülü) rasgele gradyan inişi için arzu edilir.
* Momentum, rasgele gradyan inişi için ortaya çıkma olasılığı çok daha yüksek olan optimizasyon sürecinin durdurulmasını önler.
* Geçmiş verilerin katlanarak azaltılması nedeniyle etkin gradyan sayısı $\frac{1}{1-\beta}$ ile verilir.
* Dışbükey ikinci derece problemler durumunda bu ayrıntılı olarak açıkça analiz edilebilir.
* Uygulama oldukça basittir ancak ek bir durum vektörü (momentum $\mathbf{v}$) saklamamızı gerektirir.

## Alıştırmalar

1. Momentum hiper parametrelerinin ve öğrenme oranlarının diğer kombinasyonlarını kullanın ve farklı deneysel sonuçları gözlemleyip analiz edin.
1. Birden fazla özdeğeriniz olduğu, yani $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, örn. $\lambda_i = 2^{-i}$ gibi bir ikinci derece polinom problem için GD ve momentum deneyin. $x$ değerlerinin ilk $x_i = 1$ için nasıl azaldığını çizdirin.
1. $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$ için minimum değeri ve küçültücüyü türetin.
1. Biz momentumlu rasgele gradyan inişi gerçekleştirdiğinizde ne değişir? Momentumlu minigrup rasgele gradyan inişi kullandığımızda ne olur? Ya parametrelerle deney yaparsak ne olur?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1071)
:end_tab:
