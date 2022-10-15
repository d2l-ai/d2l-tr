# Rasgele Gradyan İnişi
:label:`sec_sgd`

Daha önceki bölümlerde, eğitim prosedürümüzde rasgele gradyan inişi kullanmaya devam ettik, ancak, neden çalıştığını açıklamadık. Üzerine biraz ışık tutmak için, :numref:`sec_gd` içinde gradyan inişinin temel prensiplerini tanımladık. Bu bölümde, *rasgele gradyan iniş*ini daha ayrıntılı olarak tartışmaya devam ediyoruz.

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

## Rasgele Gradyan Güncellemeleri

Derin öğrenmede, amaç işlevi genellikle eğitim veri kümelerindeki her örnek için kayıp fonksiyonlarının ortalamasıdır. $n$ örnekten oluşan bir eğitim veri kümesi verildiğinde, $f_i(\mathbf{x})$'in  $i.$ dizindeki eğitim örneğine göre kayıp işlevi olduğunu varsayarız, burada $\mathbf{x}$ parametre vektörüdür. Sonra şu amaç fonksiyonuna varırız

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$'teki amaç fonksiyonunun gradyanı böyle hesaplanır:

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Gradyan inişi kullanılıyorsa, her bağımsız değişken yineleme için hesaplama maliyeti $\mathcal{O}(n)$'dir ve $n$ ile doğrusal olarak büyür. Bu nedenle, eğitim veri kümesi daha büyük olduğunda, her yineleme için gradyan iniş maliyeti daha yüksek olacaktır. 

Rasgele gradyan iniş (SGD), her yinelemede hesaplama maliyetini düşürür. Rasgele gradyan inişin her yinelemesinde, rastgele veri örnekleri için $i\in\{1,\ldots, n\}$ dizinden eşit olarak örneklemleriz ve $\mathbf{x}$'i güncelleştirmek için $\nabla f_i(\mathbf{x})$ gradyanını hesaplarız: 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

burada $\eta$ öğrenme oranıdır. Her yineleme için gradyan inişinin hesaplama maliyetinin $\mathcal{O}(n)$'den $\mathcal{O}(1)$ sabitine düştüğünü görebiliriz. Dahası, rasgele gradyan $\nabla f_i(\mathbf{x})$'in $\nabla f(\mathbf{x})$'in tam gradyanının tarafsız bir tahmini olduğunu vurgulamak istiyoruz, çünkü 

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Bu, ortalama olarak rasgele gradyanın, gradyanın iyi bir tahmini olduğu anlamına gelir. 

Şimdi, bir rasgele gradyan inişini benzetmek için gradyana ortalaması 0 ve varyansı 1 olan rastgele gürültü ekleyerek gradyan inişiyle karşılaştıracağız.

```{.python .input}
#@tab all
def f(x1, x2):  # Amaç fonksiyonu
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Amaç fonksiyonun gradyanı
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Gürültülü gradyan benzetimi
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Gürültülü gradyan benzetimi
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
lr = constant_lr  # Sabit öğrenme oranı
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Gördüğümüz gibi, rasgele gradyan inişindeki değişkenlerin yörüngesi, :numref:`sec_gd` içinde gradyan inişinde gözlemlediğimizden çok daha gürültülüdür. Bunun nedeni, gradyanın rasgele doğasından kaynaklanmaktadır. Yani, en düşük seviyeye yaklaştığımızda bile, $\eta \nabla f_i(\mathbf{x})$ aracılığıyla anlık gradyan tarafından aşılanan belirsizliğe hala tabiyiz. 50 adımdan sonra bile kalite hala o kadar iyi değil. Daha da kötüsü, ek adımlardan sonra iyileşmeyecektir (bunu onaylamak için daha fazla sayıda adım denemenizi öneririz). Bu bize tek alternatif bırakır: Öğrenme oranı  $\eta$'yı değiştirmek. Ancak, bunu çok küçük seçersek, ilkin bir ilerleme kaydetmeyeceğiz. Öte yandan, eğer çok büyük alırsak, yukarıda görüldüğü gibi iyi bir çözüm elde edemeyiz. Bu çelişkili hedefleri çözmenin tek yolu, optimizasyon ilerledikçe öğrenme oranını *dinamik* azaltmaktır. 

Bu aynı zamanda `sgd` adım işlevine `lr` öğrenme hızı işlevinin eklenmesinin de sebebidir. Yukarıdaki örnekte, ilişkili 'lr' işlevini sabit olarak ayarladığımız için öğrenme hızı planlaması için herhangi bir işlevsellik uykuda kalmaktadır.

## Dinamik Öğrenme Oranı

$\eta$'nın zamana bağlı öğrenme hızı $\eta(t)$ ile değiştirilmesi, optimizasyon algoritmasının yakınsamasını kontrol etmenin karmaşıklığını arttırır. Özellikle, $\eta$'nın ne kadar hızlı sönmesi gerektiğini bulmamız gerekiyor. Çok hızlıysa, eniyilemeyi erken bırakacağız. Eğer çok yavaş azaltırsak, optimizasyon için çok fazla zaman harcarız. Aşağıdakiler, $\eta$'nın zamanla ayarlanmasında kullanılan birkaç temel stratejidir (daha sonra daha gelişmiş stratejileri tartışacağız): 

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ eğer } t_i \leq t \leq t_{i+1}  && \text{parçalı sabit} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{üstel sönüm} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{polinomsal sönüm}
\end{aligned}
$$

İlk *parçalı sabit* senaryosunda, öğrenme oranını düşürüyoruz, mesela optimizasyondaki ilerleme durduğunda. Bu, derin ağları eğitmek için yaygın bir stratejidir. Alternatif olarak çok daha saldırgan bir *üstel sönüm* ile azaltabiliriz. Ne yazık ki bu genellikle algoritma yakınsamadan önce erken durmaya yol açar. Popüler bir seçim, $\alpha = 0.5$ ile *polinom sönüm*dür. Dışbükey optimizasyon durumunda, bu oranın iyi davrandığını gösteren bir dizi kanıt vardır. 

Üstel sönmenin pratikte neye benzediğini görelim.

```{.python .input}
#@tab all
def exponential_lr():
    # Bu fonksiyonun dışında tanımlanan ve içinde güncellenen global değişken
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Beklendiği gibi, parametrelerdeki varyans önemli ölçüde azaltılır. Bununla birlikte, bu, $\mathbf{x} = (0, 0)$ eniyi çözümüne yakınsamama pahasına gelir. Hatta sonra 1000 yineleme adımında sonra bile eniyi çözümden hala çok uzaktayız. Gerçekten de, algoritma yakınsamada tam anlamıyla başarısız. Öte yandan, öğrenme oranının adım sayısının ters karekökü ile söndüğü polinom sönmesi kullanırsak yakınsama sadece 50 adımdan sonra daha iyi olur.

```{.python .input}
#@tab all
def polynomial_lr():
    # Bu fonksiyonun dışında tanımlanan ve içinde güncellenen global değişken
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Öğrenme oranının nasıl ayarlanacağı konusunda çok daha fazla seçenek var. Örneğin, küçük bir hızla başlayabilir, sonra hızlıca yükseltebilir ve daha yavaş da olsa tekrar azaltabiliriz. Hatta daha küçük ve daha büyük öğrenme oranları arasında geçiş yapabiliriz. Bu tür ayarlamaların çok çeşidi vardır. Şimdilik kapsamlı bir teorik analizin mümkün olduğu öğrenme oranı ayarlamalarına odaklanalım, yani dışbükey bir ortamda öğrenme oranları üzerine. Genel dışbükey olmayan problemler için anlamlı yakınsama garantileri elde etmek çok zordur, çünkü genel olarak doğrusal olmayan dışbükey problemlerin en aza indirilmesi NP zorludur. Bir araştırma için örneğin, Tibshirani'nin 2015'teki mükemmel [ders notları](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf)na bakınız. 

## Dışbükey Amaç İşlevleri için Yakınsaklık Analizi

Dışbükey amaç fonksiyonları için rasgele gradyan inişinin aşağıdaki yakınsama analizi isteğe bağlıdır ve öncelikle problem hakkında daha fazla sezgi iletmek için hizmet vermektedir. Kendimizi en basit kanıtlardan biriyle sınırlıyoruz :cite:`Nesterov.Vial.2000`. Önemli ölçüde daha gelişmiş ispat teknikleri mevcuttur, örn. amaç fonksiyonu özellikle iyi-huyluysa. 

$f(\boldsymbol{\xi}, \mathbf{x})$'in amaç işlevinin $\boldsymbol{\xi}$'nin tümü için $\mathbf{x}$'de dışbükey olduğunu varsayalım. Daha somut olarak, rasgele gradyan iniş güncellemesini aklımızda bulunduruyoruz: 

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

burada $f(\boldsymbol{\xi}_t, \mathbf{x})$, bir dağılımdan $t$ adımında çekilen $\boldsymbol{\xi}_t$ eğitim örneğine göre amaç fonksiyonudur ve $\mathbf{x}$ model parametresidir. Aşağıda 

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

beklenen riski ifade eder ve $R^*$, $\mathbf{x}$'e göre en düşük değerdir. Son olarak $\mathbf{x}^*$  küçültücü (minimizer) olsun ($\mathbf{x}$'in tanımlandığı etki alanında var olduğunu varsayıyoruz). Bu durumda $t$ zamanında $\mathbf{x}_t$ ve risk küçültücü $\mathbf{x}^*$ arasındaki mesafeyi izleyebilir ve zamanla iyileşip iyileşmediğini görebiliriz: 

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Rasgele gradyan $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$'in $L_2$ normunun bir $L$ sabiti ile sınırlandığını varsayıyoruz, dolayısıyla 

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

Çoğunlukla $\mathbf{x}_t$ ve $\mathbf{x}^*$ arasındaki mesafenin *beklenti*de nasıl değiştiğiyle ilgileniyoruz. Aslında, herhangi bir belirli adım dizisi için, karşılaştığımız herhangi $\boldsymbol{\xi}_t$'ye bağlı olarak mesafe artabilir. Bu yüzden nokta çarpımını sınırlamamız gerekiyor. Herhangi bir dışbükey fonksiyon $f$ için $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$'nin tüm $\mathbf{x}$ ve $\mathbf{y}$ için, dışbükeylik ile şuna varırız:

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

:eqref:`eq_sgd-L` ve :eqref:`eq_sgd-f-xi-xstar` içindeki her iki eşitsizliği :eqref:`eq_sgd-xt+1-xstar` denklemine yerleştirerek parametreler arasındaki mesafeye $t+1$ zamanında aşağıdaki gibi sınır koyarız: 

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Bu, mevcut kayıp ile optimal kayıp arasındaki fark $\eta_t L^2/2$'ten ağır bastığı sürece ilerleme kaydettiğimiz anlamına gelir. Bu fark sıfıra yakınsamaya bağlı olduğundan, $\eta_t$ öğrenme oranının da *yok olması* gerekir. 

Sonrasında :eqref:`eqref_sgd-xt-diff` üzerinden beklentileri alırız. Şu sonuca varırız: 

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Son adım $t \in \{1, \ldots, T\}$ için eşitsizlikler üzerinde toplamayı içerir. Toplam içeriye daralır ve düşük terimi düşürürsek şunu elde ederiz: 

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

$\mathbf{x}_1$'in verildiğini ve böylece beklentinin düştüğünü unutmayın. Son tanımımız 

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Çünkü 

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

Jensen'in eşitsizliği ile ($i=t$, :eqref:`eq_jensens-inequality` içinde $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ ayarlarız) ve $R$ dışbükeyliği ile $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, şuna ulaşırız:

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Bunu :eqref:`eq_sgd-x1-xstar` eşitsizliğinin içine koymak aşağıdaki sınırı verir:

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

burada $r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$, parametrelerin ilk seçimi ile nihai sonuç arasındaki mesafeye bağlıdır. Kısacası, yakınsama hızı, rasgele gradyanın normunun nasıl sınırlandığına ($L$) ve ilk parametre ($r$) değerinin eniyi değerden ne kadar uzakta olduğuna bağlıdır. Sınırın $\mathbf{x}_T$ yerine $\bar{\mathbf{x}}$ cinsinden olduğuna dikkat edin. $\bar{\mathbf{x}}$'in optimizasyon yolunun yumuşatılmış bir sürümü olduğu için bu durum böyledir. $r, L$ ve $T$ bilindiğinde öğrenme oranını $\eta = r/(L \sqrt{T})$ alabiliriz. Bu, üst sınırı $rL/\sqrt{T}$ olarak verir. Yani, $\mathcal{O}(1/\sqrt{T})$ oranıyla eniyi çözüme yakınsıyoruz. 

## Rasgele Gradyanlar ve Sonlu Örnekler

Şimdiye kadar, rasgele gradyan inişi hakkında konuşurken biraz hızlı ve gevşek hareket ettik. $x_i$ örneklerini, tipik olarak $y_i$ etiketleriyle bazı $p(x, y)$ dağılımlardan çektiğimizi ve bunu model parametrelerini bir şekilde güncellemek için kullandığımızı belirttik. Özellikle, sonlu bir örnek boyutu için $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ ayrık dağılımın olduğunu savunduk. Bazı $\delta_{x_i}$ ve $\delta_{y_i}$ fonksiyonları için üzerinde rasgele gradyan inişini gerçekleştirmemizi sağlar. 

Ancak, gerçekten yaptığımız şey bu değildi. Mevcut bölümdeki basit örneklerde, aksi takdirde rasgele olmayan bir gradyana gürültü ekledik, yani $(x_i, y_i)$ çiftleri varmış gibi davrandık. Bunun burada haklı olduğu ortaya çıkıyor (ayrıntılı bir tartışma için alıştırmalara bakın). Daha rahatsız edici olan, önceki tüm tartışmalarda bunu açıkça yapmadığımızdır. Bunun neden tercih edilebilir olduğunu görmek için, tersini düşünün, yani ayrık dağılımdan *değiştirmeli*  $n$ gözlemleri örnekliyoruz. Rastgele $i$ elemanını seçme olasılığı $1/n$'dır. Böylece onu *en azından* bir kez seçeriz:

$$P(\mathrm{seç~} i) = 1 - P(\mathrm{yoksay~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Benzer bir akıl yürütme, bir numunenin (yani eğitim örneği) *tam bir kez* seçme olasılığının şöyle verildiğini göstermektedir: 

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Bu, *değiştirmesiz* örneklemeye göreceli oranla artan bir varyansa ve azalan veri verimliliğine yol açar. Bu nedenle, pratikte ikincisini gerçekleştiriyoruz (ve bu kitap boyunca varsayılan seçimdir). Son olarak eğitim veri kümesindeki tekrarlanan geçişler ondan *farklı* rastgele sırayla geçer. 

## Özet

* Dışbükey problemler için, geniş bir öğrenme oranları seçeneği için rasgele gradyan inişinin eniyi çözüme yakınsayacağını kanıtlayabiliriz.
* Derin öğrenme için bu genellikle böyle değildir. Bununla birlikte, dışbükey problemlerin analizi, optimizasyona nasıl yaklaşılacağına dair, yani öğrenme oranını çok hızlı olmasa da aşamalı olarak azaltmak için, bize yararlı bilgiler verir.
* Öğrenme oranı çok küçük veya çok büyük olduğunda sorunlar ortaya çıkar. Pratikte uygun bir öğrenme oranı genellikle sadece birden fazla deneyden sonra bulunur.
* Eğitim veri kümesinde daha fazla örnek olduğunda, gradyan inişi için her yinelemede hesaplamak daha pahalıya mal olur, bu nedenle bu durumlarda rasgele gradyan inişi tercih edilir.
* Rasgele gradyan inişi için optimizasyon garantileri genel olarak dışbükey olmayan durumlarda mevcut değildir, çünkü kontrol gerektiren yerel minimum sayısı da üstel olabilir.

## Alıştırmalar

1. Rasgele gradyan inişini farklı sayıda yineleme ile farklı öğrenme oranı düzenleri ile deneyin. Özellikle, yineleme sayısının bir fonksiyonu olarak optimal çözüm $(0, 0)$'dan uzaklığı çizin.
1. $f(x_1, x_2) = x_1^2 + 2 x_2^2$ işlevi için gradyana normal gürültü eklemenin bir kayıp işlevini $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ en aza indirmeye eşdeğer olduğunu kanıtlayın, burada $\mathbf{x}$ normal dağılımdan çekilmektedir.
1. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$'ten değiştirmeli ve değiştirmesiz örnek aldığınızda rasgele gradyan inişinin yakınsamasını karşılaştırın.
1. Bir gradyan (veya onunla ilişkili bir koordinat) diğer tüm gradyanlardan tutarlı bir şekilde daha büyükse, rasgele gradyan inişi çözücüsünü nasıl değiştirirsiniz?
1. Bunu varsayalım: $f(x) = x^2 (1 + \sin x)$. $f$'te kaç tane yerel minimum vardır? $f$', en aza indirgemek için tüm yerel minimumları değerlendirmek zorunda kalacak şekilde değiştirebilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1067)
:end_tab:
