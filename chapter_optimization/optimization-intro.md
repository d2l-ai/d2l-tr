# Eniyileme ve Derin Öğrenme

Bu bölümde, eniyileme ve derin öğrenme arasındaki ilişkiyi ve derin öğrenmede optimizasyonu kullanmanın zorluklarını tartışacağız. Derin öğrenme problemi için, genellikle önce *kayıp fonksiyonu* tanımlarız. Kayıp işlevini aldığımızda kayıpları en aza indirmek için bir optimizasyon algoritması kullanabiliriz. Optimizasyonda, bir kayıp fonksiyonu genellikle optimizasyon sorununun *amaç fonksiyonu* olarak adlandırılır. Geleneksel ve alışılmış olarak çoğu optimizasyon algoritması, *minimizasyon (en aza indirme)* ile ilgilidir. Eğer bir hedefi en üst düzeye çıkarmamız (maksimize etmemiz) gerekirse basit bir çözüm vardır: Sadece amaç işlevindeki işareti tersine çevirin. 

## Eniyilemenin Hedefi

Optimizasyon derin öğrenme için kayıp işlevini en aza indirmenin bir yolunu sağlasa da, özünde optimizasyonun ve derin öğrenmenin amaçları temelde farklıdır. Birincisi öncelikle bir amaç işlevini en aza indirmekle ilgiliyken, ikincisi, sınırlı miktarda veri verildiğinde uygun bir model bulmakla ilgilidir. :numref:`sec_model_selection` içinde, bu iki hedef arasındaki farkı ayrıntılı olarak tartıştık. Örneğin, eğitim hatası ve genelleme hatası genellikle farklılık gösterir: Optimizasyon algoritmasının amaç işlevi genellikle eğitim veri kümesine dayalı bir kayıp fonksiyonu olduğundan, optimizasyonun amacı eğitim hatasını azaltmaktır. Bununla birlikte, derin öğrenmenin amacı (veya daha geniş bir şekilde istatistiksel çıkarımın) genelleme hatasını azaltmaktır. İkincisini başarmak için, eğitim hatasını azaltmada optimizasyon algoritmasını kullanmanın yanı sıra aşırı öğrenme işlemine de dikkat etmeliyiz.

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

Yukarıda belirtilen farklı hedefleri göstermek için, riski ve deneysel riski ele alalım. :numref:`subsec_empirical-risk-and-risk` içinde açıklandığı gibi, deneysel risk, eğitim veri kümesinde ortalama bir kayıptır ve risk veri nüfusunun tamamında beklenen kayıptır. Aşağıda iki fonksiyon tanımlıyoruz: Risk fonksiyonu `f` ve deneysel risk fonksiyonu `g`. Sadece sınırlı miktarda eğitim verisi olduğunu varsayalım. Sonuç olarak, burada `g` `f`'den daha az pürüzsüzdür.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Aşağıdaki grafik, bir eğitim veri kümesinde deneysel riskin minimumunun, minimum riskten farklı bir konumda olabileceğini göstermektedir (genelleme hatası).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min \ndeneysel risk', (1.0, -1.2), (0.5, -1.1))
annotate('min risk', (1.1, -1.05), (0.95, -0.5))
```

## Derin Öğrenmede Eniyileme Zorlukları

Bu bölümde, özellikle bir modelin genelleme hatası yerine amaç işlevini en aza indirmede optimizasyon algoritmalarının başarımına odaklanacağız. :numref:`sec_linear_regression` içinde optimizasyon problemlerinde analitik çözümler ve sayısal çözümler arasında ayrım yaptık. Derin öğrenmede, çoğu amaç fonksiyonu karmaşıktır ve analitik çözümleri yoktur. Bunun yerine, sayısal optimizasyon algoritmaları kullanmalıyız. Bu bölümdeki optimizasyon algoritmalarının tümü bu kategoriye girer. 

Derin öğrenme optimizasyonunda birçok zorluk vardır. En eziyetli olanlardan bazıları yerel minimum, eyer noktaları ve kaybolan gradyanlardır. Onlara bir göz atalım. 

### Yerel En Düşüklükler

Herhangi bir $f(x)$ amaç işlevi için $f(x)$ değerinin $x$'teki değeri $x$ civarındaki diğer noktalardaki $f(x)$ değerlerinden daha küçükse, $f(x)$ yerel minimum olabilir. $x$ değerindeki $f(x)$ değeri, tüm etki alanı üzerinde amaç işlevin minimum değeriyse, $f(x)$ genel minimum değerdir. 

Örneğin, aşağıdaki fonksiyon göz önüne alındığında 

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

bu fonksiyonun yerel minimum ve küresel minimum değerlerini yaklaşık olarak değerlendirebiliriz.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('yerel minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('kuresel minimum', (1.1, -0.95), (0.6, 0.8))
```

Derin öğrenme modellerinin amaç işlevi genellikle birçok yerel eniyi değere sahiptir. Bir optimizasyon probleminin sayısal çözümü yerel eniyi seviyesine yakın olduğunda, nihai yineleme ile elde edilen sayısal çözüm, amaç fonksiyonun çözümlerinin gradyanının sıfıra yaklaştığı veya olduğu için, objektif işlevi *küresel* yerine yalnızca *yerel* en aza indirebilir. Parametreyi yalnızca bir dereceye kadar gürültü yerel minimum seviyeden çıkarabilir. Aslında bu, minigrup üzerindeki gradyanların doğal çeşitliliğin parametreleri yerel en düşük yerinden çıkarabildiği minigrup rasgele gradyan inişinin yararlı özelliklerinden biridir. 

### Eyer Noktaları

Yerel minimumun yanı sıra eyer noktaları gradyanların kaybolmasının bir başka nedenidir. *Eyer noktası*, bir işlevin tüm gradyanların kaybolduğu ancak ne küresel ne de yerel minimum olmayan herhangi bir konumdur. $f(x) = x^3$ işlevini düşünün. Onun birinci ve ikinci türevi $x=0$ için kaybolur. Eniyileme minimum olmasa bile, bu noktada durabilir.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('eyer noktasi', (0, -0.2), (-0.52, -5.0))
```

Aşağıdaki örnekte gösterildiği gibi, daha yüksek boyutlardaki eyer noktaları daha da gizli tehlikedir. $f(x, y) = x^2 - y^2$ işlevini düşünün. $(0, 0)$ onun eyer noktası vardır. Bu, $y$'ye göre maksimum ve $x$'e göre minimum değerdir. Dahası, bu, matematiksel özelliğin adını aldığı bir eyer gibi *görünür*.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
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

Bir fonksiyonun girdisinin $k$ boyutlu bir vektör olduğunu ve çıktısının bir skaler olduğunu varsayıyoruz, bu nedenle Hessian matrisinin $k$ tane özdeğere sahip olacağını varsayıyoruz ([özayrışımlar üzerine çevrimiçi ek](https://tr.d2l.ai/chapter_appendix-mathematics-for-deep-learning/eigendecomposition.html)'e bakın). Fonksiyonun çözümü yerel minimum, yerel maksimum veya işlev gradyanının sıfır olduğu bir konumda eyer noktası olabilir: 

* Sıfır-gradyan konumundaki fonksiyonun Hessian matrisinin özdeğerleri pozitif olduğunda, işlev için yerel minimum değerlere sahibiz.
* Sıfır-gradyan konumundaki fonksiyonun Hessian matrisinin özdeğerleri negatif olduğunda, işlev için yerel maksimum değerlere sahibiz.
* Sıfır-gradyan konumundaki fonksiyonun Hessian matrisinin özdeğerleri negatif ve pozitif olduğunda, fonksiyon için bir eyer noktamız vardır.

Yüksek boyutlu problemler için özdeğerlerin en azından *bazılarının* negatif olma olasılığı oldukça yüksektir. Bu eyer noktalarını yerel minimumlardan daha olasıdır yapar. Dışbükeyliği tanıtırken bir sonraki bölümde bu durumun bazı istisnalarını tartışacağız. Kısacası, dışbükey fonksiyonlar Hessian'ın özdeğerlerinin asla negatif olmadığı yerlerdir. Ne yazık ki, yine de, çoğu derin öğrenme problemi bu kategoriye girmiyor. Yine de optimizasyon algoritmaları incelemek için harika bir araçtır. 

### Kaybolan Gradyanlar

Muhtemelen aşılması gereken en sinsi sorun kaybolan gradyanlardır. :numref:`subsec_activation-functions` içinde yaygın olarak kullanılan etkinleştirme fonksiyonlarımızı ve türevlerini hatırlayın. Örneğin, $f(x) = \tanh(x)$ işlevini en aza indirmek istediğimizi ve $x = 4$'te başladığımızı varsayalım. Gördüğümüz gibi, $f$'nin gradyanı sıfıra yakındır. Daha özel durum olarak, $f'(x) = 1 - \tanh^2(x)$'dir ve o yüzden $f'(4) = 0.0013$'tür. Sonuç olarak, ilerleme kaydetmeden önce optimizasyon uzun süre takılıp kalacak. Bunun, derin öğrenme modellerinin, ReLU etkinleştirme işlevinin tanıtılmasından önce oldukça zor olmasının nedenlerinden biri olduğu ortaya çıkıyor.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('kaybolan gradyan', (4, 1), (2, 0.0))
```

Gördüğümüz gibi, derin öğrenme için optimizasyon zorluklarla doludur. Neyse ki iyi performans ve yeni başlayanlar için bile kullanımı kolay bir dizi gürbüz algoritma var. Ayrıca, en iyi çözümü bulmak gerçekten gerekli değildir. Yerel en iyi veya hatta yaklaşık çözümler hala çok faydalıdır. 

## Özet

* Eğitim hatasının en aza indirilmesi, genelleme hatasını en aza indirmek için en iyi parametre kümesini bulduğumuzu garanti etmez.
* Optimizasyon sorunlarının birçok yerel minimumu olabilir.
* Problemlerin daha fazla eyer noktası olabilir, genellikle problemler dışbükey değildir.
* Kaybolan gradyanlar eniyilemenin durmasına neden olabilir. Genellikle sorunun yeniden parametrelendirilmesi yardımcı olur. Parametrelerin iyi ilklenmesi de faydalı olabilir.

## Alıştırmalar

1. Gizli katmanında $d$ boyutlu tek bir gizli katmanına ve tek bir çıktıya sahip basit bir MLP düşünün. Herhangi bir yerel minimum için en az $d!$ tane aynı davranan eşdeğer çözüm olduğunu gösterin.
1. Simetrik bir rastgele $\mathbf{M}$ matrisimiz olduğunu varsayalım, burada $M_{ij} = M_{ji}$ girdilerinin her biri $p_{ij}$ olasılık dağılımından çekilir. Ayrıca, $p_{ij}(x) = p_{ij}(-x)$'nin, yani dağılımın simetrik olduğunu varsayalım (ayrıntılar için bkz. :cite:`Wigner.1958`).
    1. Özdeğerler üzerindeki dağılımın da simetrik olduğunu kanıtlayın. Yani, herhangi bir özvektör $\mathbf{v}$ için, ilişkili özdeğer $\lambda$'nin $P(\lambda > 0) = P(\lambda < 0)$'i karşılama olasılığı.
    1. Yukarıdaki ifade neden $P(\lambda > 0) = 0.5$ anlamına *gelmez*?
1. Derin öğrenme optimizasyonunda yer alan hangi diğer zorlukları düşünebilirsiniz?
1. (Gerçek) bir topu (gerçek) bir eyer üzerinde dengelemek istediğinizi varsayalım.
    1. Neden bu kadar zordur?
    1. Optimizasyon algoritmaları için de bu etkiyi kullanabilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/489)
:end_tab:
