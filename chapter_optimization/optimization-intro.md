# Optimizasyon ve Derin Öğrenme

Bu bölümde, optimizasyon ve derin öğrenme arasındaki ilişkiyi ve derin öğrenmede optimizasyonu kullanmanın zorluklarını tartışacağız. Derin öğrenme problemi için, genellikle önce *kayıp fonksiyonu* tanımlarız. Kayıp işlevini aldığımızda kayıpları en aza indirmek için bir optimizasyon algoritması kullanabiliriz. Optimizasyonda, bir kayıp fonksiyonu genellikle optimizasyon sorununun*objektif fonksiyonu* olarak adlandırılır. Gelenek ve kongreye göre çoğu optimizasyon algoritması, *minimizasyon* ile ilgilidir. Eğer bir hedefi en üst düzeye çıkarmamız gerekirse basit bir çözüm vardır: sadece hedefin üzerindeki işareti çevirin. 

## Optimizasyonun Hedefi

Optimizasyon derin öğrenme için kayıp işlevini en aza indirmenin bir yolunu sağlasa da, özünde optimizasyon ve derin öğrenmenin amaçları temelde farklıdır. Birincisi öncelikle bir hedefi en aza indirmekle ilgiliyken, ikincisi, sınırlı miktarda veri verildiğinde uygun bir model bulmakla ilgilidir. :numref:`sec_model_selection`'te, bu iki hedef arasındaki farkı ayrıntılı olarak tartıştık. Örneğin, eğitim hatası ve genelleme hatası genellikle farklılık gösterir: optimizasyon algoritmasının objektif işlevi genellikle eğitim veri kümesine dayalı bir kayıp fonksiyonu olduğundan, optimizasyonun amacı eğitim hatasını azaltmaktır. Bununla birlikte, derin öğrenmenin amacı (veya daha geniş bir şekilde istatistiksel çıkarım) genelleme hatasını azaltmaktır. İkincisini başarmak için, eğitim hatasını azaltmak için optimizasyon algoritmasını kullanmanın yanı sıra aşırı uydurma işlemine dikkat etmeliyiz.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Yukarıda belirtilen farklı hedefleri göstermek için, ampirik riski ve riski ele alalım. :numref:`subsec_empirical-risk-and-risk`'te açıklandığı gibi, ampirik risk, eğitim veri kümesinde ortalama bir kayıptır ve risk veri nüfusunun tamamında beklenen kayıptır. Aşağıda iki fonksiyon tanımlıyoruz: risk fonksiyonu `f` ve ampirik risk fonksiyonu `g`. Sadece sınırlı miktarda eğitim verisi olduğunu varsayalım. Sonuç olarak, burada `g` `f`'den daha az pürüzsüzdür.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Aşağıdaki grafik, bir eğitim veri kümesinde ampirik riskin minimum riskinin minimum riskten farklı bir konumda olabileceğini göstermektedir (genelleme hatası).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Derin Öğrenmede Optimizasyon Zorlukları

Bu bölümde, bir modelin genelleme hatası yerine objektif işlevi en aza indirmede optimizasyon algoritmalarının performansına özellikle odaklanacağız. :numref:`sec_linear_regression`'te optimizasyon problemlerinde analitik çözümler ve sayısal çözümler arasında ayrım yaptık. Derin öğrenmede, çoğu objektif fonksiyonlar karmaşıktır ve analitik çözümleri yoktur. Bunun yerine, sayısal optimizasyon algoritmaları kullanmalıyız. Bu bölümdeki optimizasyon algoritmaları tüm bu kategoriye girer. 

Derin öğrenme optimizasyonunda birçok zorluk vardır. En üzücü olanlardan bazıları yerel minimum, eyer noktaları ve kaybolan degradelerdir. Onlara bir göz atalım. 

### Yerel Minima

$f(x)$ herhangi bir objektif işlev için $f(x)$ değerinin $x$ değerinin $x$ civarındaki diğer noktalardaki $f(x)$ değerlerinden daha küçükse, $f(x)$ yerel minimum olabilir. $x$ değerindeki $f(x)$ değeri, tüm etki alanı üzerinde nesnel işlevin minimum değeriyse, $f(x)$ genel minimum değerdir. 

Örneğin, fonksiyon göz önüne alındığında 

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

Bu fonksiyonun yerel minimum ve küresel asgari değerlerini yaklaşık olarak değerlendirebiliriz.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

Derin öğrenme modellerinin objektif işlevi genellikle birçok yerel optima sahiptir. Bir optimizasyon probleminin sayısal çözümü yerel optimum seviyeye yakın olduğunda, nihai yineleme ile elde edilen sayısal çözüm, objektif fonksiyonun çözümlerinin degradesinin yaklaştığı veya sıfır olduğu için, *küresel* yerine yalnızca objektif işlevi en aza indirebilir. Parametreyi yalnızca bir dereceye kadar gürültü yerel minimum seviyeden çıkarabilir. Aslında bu, minibatch üzerindeki degradelerin doğal varyasyonunun parametreleri yerel minima yerinden çıkarabildiği minibatch stokastik degrade inişinin yararlı özelliklerinden biridir. 

### Eyer Noktaları

Yerel minimmanın yanı sıra eyer noktaları degradelerin kaybolmasının bir başka nedenidir. *eyer noktası*, bir işlevin tüm degradelerinin kaybolduğu ancak ne global ne de yerel minimum olmayan herhangi bir konumdur. $f(x) = x^3$ işlevini düşünün. Onun birinci ve ikinci türevi $x=0$ için kaybolur. Optimizasyon en az olmasa bile, bu noktada durabilir.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Aşağıdaki örnekte gösterildiği gibi, daha yüksek boyutlardaki eyer noktaları daha da sinsidir. $f(x, y) = x^2 - y^2$ işlevini düşünün. Bu onun eyer noktası vardır $(0, 0)$. Bu, $y$'e göre maksimum ve $x$ ile ilgili minimum değerdir. Dahası, bu matematiksel mülkün adını aldığı bir eyer gibi görünüyor*.

```{.python .input}
#@tab all
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Bir fonksiyonun girişinin $k$ boyutlu bir vektör olduğunu ve çıkışının bir skaler olduğunu varsayıyoruz, bu nedenle Hessian matrisinin $k$ özdeğerlerine sahip olacağını varsayıyoruz ([online appendix on eigendecompositions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html)'e bakın). Fonksiyonun çözümü yerel minimum, yerel maksimum veya işlev degradesinin sıfır olduğu bir konumda eyer noktası olabilir: 

* Sıfır-degrade konumundaki fonksiyonun Hessian matrisinin özdeğerleri pozitif olduğunda, işlev için yerel minimum değerlere sahibiz.
* Sıfır-degrade konumundaki fonksiyonun Hessian matrisinin özdeğerleri negatif olduğunda, işlev için yerel maksimum değerlere sahibiz.
* Sıfır-degrade konumundaki fonksiyonun Hessian matrisinin özdeğerleri negatif ve pozitif olduğunda, fonksiyon için bir eyer noktamız vardır.

Yüksek boyutlu problemler için özdeğerlerin en azından*bazı* negatif olma olasılığı oldukça yüksektir. Bu eyer noktaları yerel minima daha olasıdır yapar. Dışbükeyliği tanıtırken bir sonraki bölümde bu durumun bazı istisnalarını tartışacağız. Kısacası, dışbükey fonksiyonlar Hessian'ın özdeğerlerinin asla negatif olmadığı yerlerdir. Ne yazık ki, yine de, çoğu derin öğrenme problemi bu kategoriye girmiyor. Yine de optimizasyon algoritmaları incelemek için harika bir araçtır. 

### Ufuk Degradeleri

Muhtemelen karşılaşılması gereken en sinsi sorun kaybolan degradedir. :numref:`subsec_activation-functions`'te yaygın olarak kullanılan aktivasyon fonksiyonlarımızı ve türevlerini hatırlayın. Örneğin, $f(x) = \tanh(x)$ işlevini en aza indirmek istediğimizi ve $x = 4$'da başladığımızı varsayalım. Gördüğümüz gibi, $f$'in gradyan sıfır yakın. Daha spesifik olarak, $f'(x) = 1 - \tanh^2(x)$ ve böylece $f'(4) = 0.0013$. Sonuç olarak, ilerleme kaydetmeden önce optimizasyon uzun süre sıkışacak. Bu, derin öğrenme modellerinin, ReLU aktivasyon işlevinin tanıtılmasından önce oldukça zor olmasının nedenlerinden biri olduğu ortaya çıkıyor.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Gördüğümüz gibi, derin öğrenme için optimizasyon zorluklarla doludur. Neyse ki iyi performans ve yeni başlayanlar için bile kullanımı kolay algoritmalar sağlam bir dizi var. Ayrıca, ** en iyi çözümü bulmak gerçekten gerekli değildir. Yerel optima veya hatta yaklaşık çözümler hala çok faydalıdır. 

## Özet

* Eğitim hatasının en aza indirilmesi, genelleme hatasını en aza indirmek için en iyi parametre kümesini bulduğumuzu garanti etmez.
* Optimizasyon sorunları birçok yerel minima olabilir.
* Sorunun daha fazla eyer noktası olabilir, genellikle sorunlar dışbükey değildir.
* Ufuk degradeler en iyi duruma getirmenin durmasına neden olabilir. Genellikle sorunun yeniden parameterizasyonu yardımcı olur. Parametrelerin iyi başlatılması da faydalı olabilir.

## Egzersizler

1. Gizli katmandaki $d$ boyutlarının tek bir gizli katmanına ve tek bir çıktıya sahip basit bir MLP düşünün. Herhangi bir yerel minimum için en az $d olduğunu gösterin! $ aynı davranır eşdeğer çözümler.
1. Biz simetrik rastgele matris olduğunu varsayalım $\mathbf{M}$ girişleri $M_{ij} = M_{ji}$ her bazı olasılık dağılımından çizilir $p_{ij}$. Ayrıca, $p_{ij}(x) = p_{ij}(-x)$'nin, yani dağılımın simetrik olduğunu varsayalım (ayrıntılar için bkz. :cite:`Wigner.1958`).
    1. Özdeğerler üzerindeki dağılımın da simetrik olduğunu kanıtlayın. Yani, herhangi bir özvektör için $\mathbf{v}$, ilişkili özdeğer $\lambda$'ün $P(\lambda > 0) = P(\lambda < 0)$'i karşılama olasılığı.
    1. Yukarıdaki*değil* neden $P(\lambda > 0) = 0.5$ anlamına geliyor?
1. Derin öğrenme optimizasyonunda yer alan diğer zorlukları düşünebilirsiniz?
1. Bir (gerçek) topu (gerçek) bir eyer üzerinde dengelemek istediğinizi varsayalım.
    1. Neden bu kadar zor?
    1. Eğer optimizasyon algoritmaları için de bu etkiyi kullanabilir miyim?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
