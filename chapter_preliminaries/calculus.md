# Hesaplama (Kalkülüs)
:label:`sec_calculus`

Bir çokgenin alanını bulmak, eski Yunanlıların bir çokgeni üçgenlere böldüğü ve alanlarını topladığı en az 2.500 yıl öncesine kadar gizemli kalmıştı.
Bir daire gibi kavisli şekillerin alanını bulmak için, eski Yunanlılar bu şekillerin içine çokgenler yerleştirdiler.
:numref:`fig_circle_area`'de gösterildiği gibi, eşit uzunlukta daha fazla kenarı olan çizili bir çokgen daireye bayağı yaklaşır. Bu işlem *tüketme yöntemi* olarak da bilinir.

![Tüketme yöntemiyle bir dairenin alanını bulun.](../img/polygon-circle.svg)
:label:`fig_circle_area`

Aslında, tüketme yöntemi *integral hesabının* (şurada açıklanacaktır :numref:`sec_integral_calculus`) kaynaklandığı yerdir.
2.000 yıldan fazla bir müddetten sonra, diğer kalkülüs alanı, *diferansiyel (türevsel) kalkülüs* icat edildi.
Diferansiyel kalkülüsün en kritik uygulamaları arasındaki optimizasyon problemleri bir şeyin nasıl *en iyi* şekilde yapılacağına kafa yorar.
:numref:`subsec_norms_and_objectives`'de tartışıldığı gibi, bu tür sorunlar derin öğrenmede her yerde bulunur.

Derin öğrenmede, modelleri daha fazla veri gördükçe daha iyi ve daha iyi olmaları için arka arkaya güncelleyerek *eğitiyoruz*.
Genellikle, daha iyi olmak, "modelimiz ne kadar *kötü*?" sorusuna cevap veren bir skor olan *kayıp (yitim) fonksiyonunu* en aza indirmek anlamına gelir.
Bu soru göründüğünden daha zekicedir.
Sonuçta, gerçekten önemsediğimiz, daha önce hiç görmediğimiz veriler üzerinde iyi performans gösteren bir model üretmektir.
Ancak modeli yalnızca gerçekten görebildiğimiz verilere uydurabiliriz.
Böylece modellerin uydurulması görevini iki temel kaygıya ayırabiliriz: 
(i) *Eniyileme*: Modellerimizi gözlemlenen verilere uydurma süreci;
(ii) *Genelleme*: Geçerliliği onları eğitmek için kullanılan kesin veri örnekleri kümesinin ötesine geçen modellerin nasıl üretileceğinde bize rehberlik eden matematiksel ilkelerin ve uygulayıcılarının bilgeliği.

Daha sonraki bölümlerde optimizasyon problemlerini ve yöntemlerini anlamanıza yardımcı olmak için burada, derin öğrenmede yaygın olarak kullanılan diferansiyel hesaplama hakkında bir tutam bilgi veriyoruz.

## Türev ve Türev Alma

Hemen hemen tüm derin öğrenme optimizasyon algoritmalarında önemli bir adım olan türevlerin hesaplanmasını ele alarak başlıyoruz.
Derin öğrenmede, tipik olarak modelimizin parametrelerine göre türevi alınabilen kayıp fonksiyonlarını seçeriz.
Basitçe ifade etmek gerekirse, bu, her parametre için, o parametreyi sonsuz derecede küçük bir miktarda *arttırırsak* veya *azaltırsak* kaybın ne kadar hızlı artacağını veya azalacağını belirleyebileceğimiz anlamına gelir.

Girdi ve çıktıların her ikisi de skaler olan $f: \mathbb {R} \rightarrow \mathbb{R}$ fonksiyonumuz olduğunu varsayalım.
[**$f$'nin *türevi* şöyle tanımlanır**]:

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$**)
:eqlabel:`eq_derivative`

eğer bu limit varsa.
$f'(a)$ varsa, $f$'nin $a$'da *türevlenebilir* olduğu söylenir.
$f$, bir aralığın her sayısında türevlenebilirse, o zaman bu fonksiyon bu aralıkta türevlenebilir.
$f'(x)$'in :eqref:`eq_derivative`'deki türevini $f(x)$'in $x$'e göre *anlık* değişim oranı olarak yorumlayabiliriz.
Sözde anlık değişim oranı, $x$ cinsinden $h$ $0$'a yaklaşırken değişimini temel alır.

Türevleri açıklamayı için bir örnekle deneyelim.
(**$u = f(x) = 3x^2-4x$ tanımlayın.**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x = 1$ diye ayarlayıp $h$ değerinin $0$ değerine yaklaşmasına izin verince $\frac{f(x+h) - f(x)}{h}$**] :eqref:`eq_derivative`'in sayısal sonucu (**$2$'ye yaklaşır.**)
Bu deney matematiksel bir kanıt olmasa da, daha sonra $u'$ türevinin $x=1$ olduğunda $2$ olduğunu göreceğiz.

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerik limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

Kendimizi türevler için birkaç eşdeğer gösterimle tanıştıralım.
$y = f(x)$ verildiğinde, $x$ ve $y$ sırasıyla $f$ işlevinin bağımsız ve bağımlı değişkenleridir. Aşağıdaki ifadeler eşdeğerdir:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

burada $\frac{d}{dx}$ ve $D$ sembolleri *türev alma* işlevini gösteren *türev alma* operatörleridir.
Yaygın işlevlerin türevini alma için aşağıdaki kuralları kullanabiliriz:

* $DC = 0$ ($C$ bir sabit),
* $Dx^n = nx^{n-1}$ (*üs kuralı*, $n$ bir gerçel sayı),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

Yukarıdaki yaygın işlevler gibi birkaç basit işlevden oluşan bir işlevin türevini alırken için aşağıdaki kurallar bizim için kullanışlı olabilir.
$f$ ve $g$ işlevlerinin ikisinin de türevlenebilir ve $C$'nin sabit olduğunu varsayalım, elimizde *sabit çarpım kuralı*,

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*toplam kuralı*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*çarpım kuralı*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

ve *bölme kuralı* vardır.

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

Şimdi $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$'ı bulmak için yukarıdaki kurallardan birkaçını uygulayabiliriz.
Bu nedenle, $x = 1$ atadığımız da, $u' = 2$ değerine sahibiz: Sayısal sonucun $2$'ye yaklaştığı, bu bölümdeki önceki denememiz tarafından desteklenmektedir.
Bu türev aynı zamanda $u = f(x)$ eğrisine $x = 1$'deki teğet doğrusunun eğimidir.

[**Türevlerin bu tür yorumunu görselleştirmek için**] Python'da popüler bir [**çizim kütüphanesi olan `matplotlib`'i**] kullanacağız.
`matplotlib` tarafından üretilen şekillerin özelliklerini yapılandırmak için birkaç işlev tanımlamamız gerekir.
Aşağıdaki `use_svg_display` işlevi, daha keskin görüntülü svg şekilleri çıktısı almak için `matplotlib` paketini özelleştirir.
`#@save` yorumunun, aşağıdaki işlev, sınıf veya ifadelerin `d2l` paketine kaydedildiği ve böylece daha sonra yeniden tanımlanmadan doğrudan çağrılabilecekleri (örneğin, `d2l.use_svg_display()`) özel bir terim olduğuna dikkat edin.

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Jupyter içinde şekli göstermek için svg formatı kullan"""
    backend_inline.set_matplotlib_formats('svg')
```

Şekil boyutlarını belirtmek için `set_figsize` fonksiyonunu tanımlarız. Burada doğrudan `d2l.plt`'yi kullandığımıza dikkat edin, çünkü içe aktarma komutu, `from matplotlib import pyplot as plt`, önsöz bölümündeki `d2l` paketine kaydedilmek üzere işaretlenmişti.

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Şekil ebatını matplotlib için ayarla"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

Aşağıdaki `set_axes` işlevi, `matplotlib` tarafından üretilen şekillerin eksenlerinin özelliklerini ayarlar.

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Eksenleri matplotlib için ayarla."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Şekil biçimlendirmeleri için bu üç işlevle, kitap boyunca birçok eğriyi görselleştirmemiz gerekeceğinden, çoklu eğrileri kısa ve öz olarak çizmek için `plot` işlevini tanımlıyoruz.

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Veri noktalarını çiz."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # `X` (tensor veya liste) 1 eksenli ise True değeri döndür
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Şimdi [**$u = f(x)$ fonksiyonunu ve $y = 2x - 3$ teğet doğrusunu $x=1$'de**] çizebiliriz, burada $2$ katsayısı teğet doğrusunun eğimidir.

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Teget dogrusu (x=1)'])
```

## Kısmi Türevler

Şimdiye kadar sadece tek değişkenli fonksiyonların türevinin alınması ile uğraştık.
Derin öğrenmede, işlevler genellikle *birçok* değişkene bağlıdır.
Bu nedenle, türev alma fikirlerini bu *çok değişkenli* fonksiyonlara genişletmemiz gerekiyor.

$y = f(x_1, x_2, \ldots, x_n)$,  $n$ değişkenli bir fonksiyon olsun. $y$'nin $i.$ parametresi $x_i$'ye göre *kısmi türevi* şöyledir:

$$\frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

$\frac{\partial y}{\partial x_i}$'ı hesaplarken, $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$'ı sabitler olarak kabul eder ve $y$'nin $x_i$'ye göre türevini hesaplayabiliriz.
Kısmi türevlerin gösterimi için aşağıdakiler eşdeğerdir:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$


## Gradyanlar (Eğimler)
:label:`subsec_calculus-grad`

Fonksiyonun *gradyan* vektörünü elde etmek için çok değişkenli bir fonksiyonun tüm değişkenlerine göre kısmi türevlerini art arda bitiştirebiliriz.
$f : \mathbb{R}^n \rightarrow \mathbb{R}$ işlevinin girdisinin $n$ boyutlu bir vektör, $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ olduğunu varsayalım ve çıktı bir skalerdir. $\mathbf{x}$'e göre $f(\mathbf{x})$ fonksiyonunun gradyanı, $n$ tane kısmi türevli bir vektördür:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

belirsizlik olmadığında, $\nabla_{\mathbf{x}} f(\mathbf{x})$ genellikle $\nabla f(\mathbf{x})$ ile değiştirilir.

$\mathbf{x}$ bir $n$ boyutlu vektör olsun, aşağıdaki kurallar genellikle çok değişkenli fonksiyonların türevini alırken kullanılır:

* Her $\mathbf{A} \in \mathbb{R}^{m \times n}$ için, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
* Her $\mathbf{A} \in \mathbb{R}^{n \times m}$ için, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
* Her $\mathbf{A} \in \mathbb{R}^{n \times n}$ için, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Benzer şekilde, herhangi bir $\mathbf{X}$ matrisi için, $\nabla_{\mathbf{X}} \|\mathbf {X} \|_F^2 = 2\mathbf{X}$ olur. Daha sonra göreceğimiz gibi, gradyanlar derin öğrenmede optimizasyon algoritmaları tasarlamak için kullanışlıdır.

## Zincir kuralı

Bununla birlikte, bu tür gradyanları bulmak zor olabilir.
Bunun nedeni, derin öğrenmedeki çok değişkenli işlevlerin genellikle *bileşik* olmasıdır, bu nedenle bu işlevlerin türevleri için yukarıda belirtilen kuralların hiçbirini uygulamayabiliriz.
Neyse ki, *zincir kuralı* bileşik işlevlerin türevlerini almamızı sağlar.

Önce tek değişkenli fonksiyonları ele alalım.
$y = f(u)$ ve $u=g(x)$ işlevlerinin her ikisinin de türevlenebilir olduğunu varsayalım, bu durumda zincir kuralı şunu belirtir:

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

Şimdi dikkatimizi, fonksiyonların keyfi sayıda değişkene sahip olduğu daha genel bir senaryoya çevirelim.
$y$ türevlenebilir fonksiyonunun $u_1, u_2, \ldots, u_m$ değişkenlerine sahip olduğunu varsayalım, burada her türevlenebilir fonksiyon $u_i$, $x_1, x_2, \ldots, x_n$ değişkenlerine sahiptir.
$y$ değerinin $x_1, x_2, \ldots, x_n$'nin bir işlevi olduğuna dikkat edin.
Sonra bütün $i = 1, 2, \ldots, n$ için, zincir kuralı şunu gösterir:

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

## Özet


* Diferansiyel kalkülüs ve integral kalkülüs, ilki derin öğrenmede her yerde bulunan optimizasyon problemlerine uygulanabilen iki analiz dalıdır.
* Bir türev, bir fonksiyonun değişkenine göre anlık değişim hızı olarak yorumlanabilir. Aynı zamanda fonksiyonun eğrisine teğet doğrusunun eğimidir.
* Gradyan, bileşenleri çok değişkenli bir fonksiyonun tüm değişkenlerine göre kısmi türevleri olan bir vektördür.
* Zincir kuralı, bileşik fonksiyonların türevlerini almamızı sağlar.

## Alıştırmalar

1. $y = f(x) = x^3 - \frac{1}{x}$ fonksiyonunu ve $x = 1$ olduğunda teğet doğrusunu çizin.
1. $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ fonksiyonunun gradyanını bulun.
1. $f(\mathbf{x}) = \|\mathbf{x}\|_2$ fonksiyonunun gradyanı nedir?
1. $u = f(x, y, z)$ ve $x = x(a, b)$, $y = y(a, b)$ ve $z = z(a, b)$ olduğu durum için zincir kuralını yazabilir misiniz? 

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/197)
:end_tab:
