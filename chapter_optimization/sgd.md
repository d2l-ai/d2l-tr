# Stokastik Degrade İniş
:label:`sec_sgd`

Daha önceki bölümlerde, eğitim prosedürümüzde stokastik degrade iniş kullanmaya devam ettik, ancak, neden çalıştığını açıklamadan. Üzerine biraz ışık tutmak için, :numref:`sec_gd`'te degrade inişinin temel prensiplerini tanımladık. Bu bölümde, tartışmak için devam
*daha ayrıntılı olarak stokastik degrade iniş*.

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

## Stokastik Degrade Güncellemeleri

Derin öğrenmede, objektif işlev genellikle eğitim veri kümelerindeki her örnek için kayıp fonksiyonlarının ortalamasıdır. $n$ örneklerinden oluşan bir eğitim veri kümesi göz önüne alındığında, $f_i(\mathbf{x})$'in $\mathbf{x}$'nin parametre vektörü olduğu endeks $i$'nın eğitim örneğine göre kayıp fonksiyonu olduğunu varsayıyoruz. Sonra objektif fonksiyona varıyoruz 

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$'teki objektif fonksiyonun degrade olarak hesaplanır 

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Degrade iniş kullanılıyorsa, her bağımsız değişken yineleme için hesaplama maliyeti $\mathcal{O}(n)$'tir ve $n$ ile doğrusal olarak büyür. Bu nedenle, eğitim veri kümesi daha büyük olduğunda, her yineleme için degrade iniş maliyeti daha yüksek olacaktır. 

Stokastik degrade iniş (SGD), her yinelemede hesaplama maliyetini düşürür. Stokastik degrade iniş her yinelemesinde, rastgele veri örnekleri için $i\in\{1,\ldots, n\}$ dizini eşit olarak örnekleriz ve $\mathbf{x}$'yı güncelleştirmek için $\nabla f_i(\mathbf{x})$ degradeyi hesaplarız: 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

burada $\eta$ öğrenme oranıdır. Her yineleme için hesaplama maliyetinin $\mathcal{O}(n)$'ten degrade inişinin $\mathcal{O}(n)$'ten $\mathcal{O}(1)$ sabitine düştüğünü görebiliriz. Dahası, stokastik degrade $\nabla f_i(\mathbf{x})$'ün $\nabla f(\mathbf{x})$'nın tam degradenin tarafsız bir tahmini olduğunu vurgulamak istiyoruz, çünkü 

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Bu, ortalama olarak stokastik degradenin degradenin iyi bir tahmini olduğu anlamına gelir. 

Şimdi, stokastik degrade inişi simüle etmek için degradeye 0 ortalamalı rastgele gürültü ve 1 varyansı ekleyerek degrade iniş ile karşılaştıracağız.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Gördüğümüz gibi, stokastik degrade inişindeki değişkenlerin yörüngesi, :numref:`sec_gd`'te degrade inişinde gözlemlediğimizden çok daha gürültülü. Bunun nedeni, degradenin stokastik doğasından kaynaklanmaktadır. Yani, asgari seviyeye yaklaştığımızda bile, $\eta \nabla f_i(\mathbf{x})$ aracılığıyla anlık gradyan tarafından enjekte edilen belirsizliğe hala tabiyiz. 50 adımdan sonra bile kalite hala o kadar iyi değil. Daha da kötüsü, ek adımlardan sonra iyileşmeyecektir (bunu onaylamak için daha fazla sayıda adım denemenizi öneririz). Bu bize tek alternatif bırakır: öğrenme oranını değiştirmek $\eta$. Ancak, bunu çok küçük seçersek, başlangıçta anlamlı bir ilerleme kaydetmeyeceğiz. Öte yandan, eğer çok büyük alırsak, yukarıda görüldüğü gibi iyi bir çözüm elde edemeyiz. Bu çelişkili hedefleri çözmenin tek yolu, optimizasyon ilerledikçe öğrenme oranını*dinamik* azaltmaktır. 

Bu aynı zamanda `sgd` adım işlevine `lr` öğrenme hızı işlevinin eklenmesinin de sebebidir. Yukarıdaki örnekte, ilişkili `lr` işlevini sabit olarak ayarladığımız için, öğrenme hızı zamanlaması için herhangi bir işlevsellik uykuda yatmaktadır. 

## Dinamik Öğrenme Hızı

$\eta$'in zamana bağlı öğrenme hızı $\eta(t)$ ile değiştirilmesi, optimizasyon algoritmasının yakınsamasını kontrol etmenin karmaşıklığını arttırır. Özellikle, $\eta$'in ne kadar hızlı çürümesi gerektiğini bulmamız gerekiyor. Çok hızlıysa, erken optimize etmeyi bırakacağız. Eğer çok yavaş azaltırsak, optimizasyon için çok fazla zaman harcıyoruz. Aşağıdakiler, $\eta$'in zamanla ayarlanmasında kullanılan birkaç temel stratejidir (daha sonra daha gelişmiş stratejileri tartışacağız): 

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{polynomial decay}
\end{aligned}
$$

İlk *parça sabit* senaryosunda, örneğin optimizasyondaki ilerleme durduğunda öğrenme oranını düşürüyoruz. Bu, derin ağları eğitmek için ortak bir stratejidir. Alternatif olarak çok daha agresif bir *üstel çürüme ile azaltabiliriz. Ne yazık ki bu genellikle algoritma birleşmeden önce erken durdurmaya yol açar. Popüler bir seçim, $\alpha = 0.5$ ile*polinom çürümesi*. Dışbükey optimizasyon durumunda, bu oranın iyi davrandığını gösteren bir dizi kanıt vardır. 

Üstel çürümenin pratikte neye benzediğini görelim.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Beklendiği gibi, parametrelerdeki varyans önemli ölçüde azaltılır. Bununla birlikte, bu, $\mathbf{x} = (0, 0)$'ün optimal çözümüne yakınsamamanın pahasına gelir. Hatta sonra 1000 yineleme adımlar biz çok uzakta optimal çözümden hala vardır. Gerçekten de, algoritma hiç yakınsama başarısız. Öte yandan, öğrenme hızının adım sayısının ters karekökü ile bozunduğu polinom çürümesi kullanırsak yakınsama sadece 50 adımdan sonra daha iyi olur.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Öğrenme oranının nasıl ayarlanacağı konusunda çok daha fazla seçenek var. Örneğin, küçük bir hızla başlayabilir, sonra hızlıca yükseltebilir ve daha yavaş da olsa tekrar azaltabiliriz. Hatta daha küçük ve daha büyük öğrenme oranları arasında geçiş yapabiliriz. Bu tür programların çok çeşitli var. Şimdilik kapsamlı bir teorik analizin mümkün olduğu öğrenme oranı programlarına odaklanalım, yani dışbükey bir ortamda öğrenme hızları üzerinde. Genel dışbükey olmayan problemler için anlamlı yakınsama garantileri elde etmek çok zordur, çünkü genel olarak doğrusal olmayan dışbükey problemlerin en aza indirilmesi NP zordur. Bir anket için örneğin, Tibshirani 2015'in mükemmel [ders notları](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) bakın. 

## Konveks Amaçlar için Yakınsaklık Analizi

Dışbükey objektif fonksiyonlar için stokastik degrade iniş aşağıdaki yakınsama analizi isteğe bağlıdır ve öncelikle sorun hakkında daha fazla sezgi iletmek için hizmet vermektedir. Kendimizi en basit kanıtlardan biriyle sınırlıyoruz :cite:`Nesterov.Vial.2000`. Önemli ölçüde daha gelişmiş ispat teknikleri mevcuttur, örn. objektif fonksiyon özellikle iyi davranıldığında. 

$f(\boldsymbol{\xi}, \mathbf{x})$'ün nesnel işlevinin $\boldsymbol{\xi}$ tümü için $\mathbf{x}$'da dışbükey olduğunu varsayalım. Daha somut olarak, stokastik degrade iniş güncellemesini göz önünde bulunduruyoruz: 

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

burada $f(\boldsymbol{\xi}_t, \mathbf{x})$, $t$ ve $\mathbf{x}$ adımında bazı dağılımlardan çizilen eğitim örneğine göre objektif fonksiyondur ve $\mathbf{x}$ model parametresidir. Tarafından gösterin 

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

beklenen risk ve $R^*$ ile ilgili olarak en az $\mathbf{x}$. Son izin $\mathbf{x}^*$ minimizer (biz $\mathbf{x}$ tanımlandığı etki alanı içinde var olduğunu varsayalım). Bu durumda $t$ zamanında $\mathbf{x}_t$ ve risk minimizer $\mathbf{x}^*$ arasındaki mesafeyi izleyebilir ve zamanla iyileşip iyileşmediğini görebiliriz: 

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

$L_2$ stokastik degrade $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$'in $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ normunun bazı sabit $L$ ile sınırlandırıldığını varsayıyoruz, dolayısıyla 

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

Çoğunlukla $\mathbf{x}_t$ ve $\mathbf{x}^*$ arasındaki mesafenin beklentide* ile nasıl değiştiğiyle ilgileniyoruz. Aslında, herhangi bir belirli adım dizisi için, karşılaştığımız $\boldsymbol{\xi}_t$ hangisine bağlı olarak mesafe artabilir. Bu yüzden nokta ürününü bağlamamız gerekiyor. Herhangi bir dışbükey fonksiyon $f$ için $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$'nin $\mathbf{x}$ ve $\mathbf{y}$ tümü için, dışbükeylik ile 

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

:eqref:`eq_sgd-L` ve :eqref:`eq_sgd-f-xi-xstar`'in her iki eşitsizliği :eqref:`eq_sgd-xt+1-xstar`'e takarak :eqref:`eq_sgd-xt+1-xstar`'te parametreler arasındaki mesafeye bağlı olarak $t+1$ aşağıdaki gibi: 

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Bu, mevcut kayıp ile optimal kayıp arasındaki fark $\eta_t L^2/2$'ten ağır bastığı sürece ilerleme kaydetmemiz anlamına gelir. Bu farkın sıfıra yakınsama bağlı olduğu için $\eta_t$ öğrenme oranının da *kaybolması gerektiği izlenir. 

Sonraki :eqref:`eqref_sgd-xt-diff` üzerinde beklentileri almak. Bu verim 

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Son adım $t \in \{1, \ldots, T\}$ için eşitsizlikler üzerinde toplamayı içerir. Toplam teleskoplar ve alt vadeli bırakarak bu yana elde 

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

$\mathbf{x}_1$'ün verildiğini ve böylece beklentinin düştüğünü sömürdüğümüzü unutmayın. Son tanımla 

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Beri 

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

Jensen'in eşitsizliği ile ($i=t$, :eqref:`eq_jensens-inequality` içinde $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ ayar) ve $R$ dışbükeyliği ile $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, bu nedenle 

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Bu eşitsizlik içine takmak :eqref:`eq_sgd-x1-xstar` sınırı verir 

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

burada $r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$, parametrelerin ilk seçimi ile nihai sonuç arasındaki mesafeye bağlıdır. Kısacası, yakınsama hızı, stokastik degradenin normunun nasıl sınırlandığına ($L$) ve başlangıç parametre değerinin optimaliteden ne kadar uzakta olduğuna bağlıdır ($r$). Bağlı $\mathbf{x}_T$ yerine $\bar{\mathbf{x}}$ açısından olduğunu unutmayın. $\bar{\mathbf{x}}$'nin optimizasyon yolunun yumuşatılmış bir sürümü olduğu için bu durum böyledir. $r, L$ ve $T$ bilindiğinde $\eta = r/(L \sqrt{T})$ öğrenme oranını alabiliriz. Bu, üst sınır $rL/\sqrt{T}$ olarak verir. Yani, $\mathcal{O}(1/\sqrt{T})$ oranıyla optimal çözüme yakınlaşıyoruz. 

## Stokastik Gradyanlar ve Sonlu Örnekler

Bu stokastik degrade iniş söz konusu olduğunda Şimdiye kadar biraz hızlı ve gevşek oynadık. $x_i$ örneklerini çizdiğimizi, tipik olarak $y_i$ etiketleriyle $y_i$ bazı dağıtımlardan $p(x, y)$ ve bunu model parametrelerini bir şekilde güncellemek için kullandığımızı belirttik. Özellikle, sonlu bir örnek boyutu için, $\delta_{x_i}$ ve $\delta_{y_i}$ bazı fonksiyonlar için ayrık dağılımın $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ ve $\delta_{y_i}$ üzerinde stokastik degrade iniş gerçekleştirmemize izin verdiğini savunduk. 

Ancak, gerçekten yaptığımız şey bu değildi. Mevcut bölümdeki oyuncak örneklerinde, aksi takdirde stokastik olmayan bir degradeye gürültü ekledik, yani $(x_i, y_i)$ çiftleri varmış gibi davrandık. Bunun burada haklı olduğu ortaya çıkıyor (ayrıntılı bir tartışma için alıştırmalara bakın). Daha rahatsız edici olan, önceki tüm tartışmalarda bunu açıkça yapmadığımızdır. Bunun yerine tüm örneklerde yineledik, *tam bir defa*. Bunun neden tercih edildiğini görmek için, yani biz örnekleme olduğunu $n$ ayrı dağılımdan gözlemler* değiştirile*. Rastgele $i$ öğesi seçme olasılığı $1/n$'dır. Böylece seçmek*en azından* bir kez 

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Benzer bir akıl yürütme, bir numunenin (yani eğitim örneği) seçme olasılığının tam olarak bir kez verildiğini göstermektedir. 

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Bu, değişimsiz* örneklemesine göreli* oranla artan bir varyans ve veri verimliliğinin azalmasına yol açar. Bu nedenle, pratikte ikincisini gerçekleştiriyoruz (ve bu kitap boyunca varsayılan seçimdir). Eğitim veri kümesindeki tekrarlanan son not*farklı* rastgele sırayla geçiyor. 

## Özet

* Dışbükey problemler için, geniş bir öğrenme oranları seçeneği için stokastik degrade iniş optimal çözüme yakınsama olacağını kanıtlayabiliriz.
* Derin öğrenme için bu genellikle böyle değildir. Bununla birlikte, dışbükey problemlerin analizi, optimizasyona nasıl yaklaşılacağına dair yararlı bilgiler verir, yani öğrenme hızını aşamalı olarak azaltma, çok hızlı olmasa da.
* Öğrenme oranı çok küçük veya çok büyük olduğunda sorunlar ortaya çıkar. Pratikte uygun bir öğrenme oranı genellikle sadece birden fazla deneyden sonra bulunur.
* Eğitim veri kümesinde daha fazla örnek olduğunda, degrade iniş için her yineleme hesaplamak daha pahalıya mal olur, bu nedenle bu durumlarda stokastik degrade iniş tercih edilir.
* Stokastik degrade iniş için optimizasyon garantileri genel olarak dışbükey olmayan durumlarda mevcut değildir, çünkü kontrol gerektiren yerel minimum sayısı da üstel olabilir.

## Egzersizler

1. Stokastik degrade iniş ve farklı sayıda yineleme ile farklı öğrenme oranı programları ile deney yapın. Özellikle, yineleme sayısının bir fonksiyonu olarak optimal çözümden $(0, 0)$'ten uzaklığı çizin.
1. $f(x_1, x_2) = x_1^2 + 2 x_2^2$ işlev için degradeye normal gürültü ekleyerek $\mathbf{x}$'in normal dağılımdan çekildiği $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ kayıp fonksiyonunun en aza indirilmesine eşdeğer olduğunu kanıtlayın.
1. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$'ten örnek aldığınızda stokastik degrade iniş yakınsamasını karşılaştırın ve yerine geçmeden örnek aldığınızda.
1. Bazı degrade (veya onunla ilişkili bir koordinat) tüm diğer degradelerden tutarlı bir şekilde daha büyükse stokastik degrade iniş çözücüsünü nasıl değiştirirsiniz?
1. Bunu varsayalım $f(x) = x^2 (1 + \sin x)$. $f$'te kaç tane yerel minima var? $f$'ü en aza indirgemek için tüm yerel minimamı değerlendirmek için ihtiyaç duyacak şekilde değiştirebilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
