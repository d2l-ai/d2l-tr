# Hesaplama (Kalkülüs)
:label:`sec_calculus`

Bir çokgenin alanını bulmak, eski Yunanlıların bir çokgeni üçgenlere böldüğü ve alanlarını topladığı en az 2.500 yıl öncesine kadar gizemli kalmıştı.
Bir daire gibi kavisli şekillerin alanını bulmak için, eski Yunanlılar bu şekillerin içine çokgenler yerleştirdiler.
:numref:`fig_circle_area`'da gösterildiği gibi, eşit uzunlukta daha fazla kenarı olan çizili bir çokgen daireye daha iyi yaklaşır. Bu işlem *tükenme yöntemi* olarak da bilinir.

![Tükenme yöntemiyle bir dairenin alanını bulun.](../img/polygon_circle.svg)
:label:`fig_circle_area`

Aslında, tükenme yöntemi *integral hesabının* (şurada açıklanacaktır :numref:`sec_integral_calculus`) kaynaklandığı yerdir.
2.000 yıldan fazla bir süre sonra, diğer kalkülüs alanı, *diferansiyel (türevsel) kalkülüs* icat edildi.
Diferansiyel kalkülüsün en kritik uygulamaları arasındaki optimizasyon problemleri bir şeyin nasıl *en iyi* şekilde yapılacağını düşünür.
:numref:`subsec_norms_and_objectives`'te tartışıldığı gibi, bu tür sorunlar derin öğrenmede her yerde bulunur.

Derin öğrenmede, modelleri daha fazla veri gördükçe daha iyi ve daha iyi olmaları için arka arkaya güncelleyerek *eğitiyoruz*.
Genellikle, daha iyi olmak, "modelimiz ne kadar *kötü*?" sorusuna cevap veren bir skor olan *kayıp (yitim) fonksiyonunu* en aza indirmek anlamına gelir.
Bu soru göründüğünden daha zekicedir.
Sonuçta, gerçekten önemsediğimiz, daha önce hiç görmediğimiz veriler üzerinde iyi performans gösteren bir model üretmektir.
Ancak modeli yalnızca gerçekten görebildiğimiz verilere uydurabiliriz.
Böylece modellerin uydurulması görevini iki temel kaygıya ayırabiliriz: i) *optimizasyon*: modellerimizi gözlemlenen verilere uydurma süreci;
ii) *genelleme*: geçerliliği onları eğitmek için kullanılan kesin veri örnekleri kümesinin ötesine geçen modellerin nasıl üretileceğinde bize rehberlik eden matematiksel ilkelerin ve uygulayıcılarının bilgeliği.

Daha sonraki bölümlerde optimizasyon problemlerini ve yöntemlerini anlamanıza yardımcı olmak için, burada derin öğrenmede yaygın olarak kullanılan diferansiyel matematik hakkında çok kısa bir kapsül bilgi veriyoruz.

## Türev ve Türev Alma

Hemen hemen tüm derin öğrenme optimizasyon algoritmalarında önemli bir adım olan türevlerin hesaplanmasını ele alarak başlıyoruz.
Derin öğrenmede, tipik olarak modelimizin parametrelerine göre türevi alınabilen yitim fonksiyonlarını seçeriz.
Basitçe ifade etmek gerekirse, bu, her parametre için, o parametreyi sonsuz derecede küçük bir miktarda *arttırırsak* veya *azaltırsak* kaybın ne kadar hızlı artacağını veya azalacağını belirleyebileceğimiz anlamına gelir.

Girdi ve çıktıların her ikisi de skaler olan $f: \mathbb {R} \rightarrow \mathbb{R}$ fonksiyonumuz olduğunu varsayalım.
$f$'in *türevi* şöyle tanımlanır:

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$
:eqlabel:`eq_derivative`

eğer bu limit varsa.
$f'(a)$ varsa, $f$'in $a$'da *türevlenebilir* olduğu söylenir.
$f$, bir aralığın her sayısında türevlenebilirse, o zaman bu fonksiyon bu aralıkta türevlenebilir.
$f'(x)$'in :eqref:`eq_derivative`'deki türevini $f(x)$'in $x$'e göre *anlık* değişim oranı olarak yorumlayabiliriz.
Sözde anlık değişim oranı, $x$ cinsinden $h$ $0$'a yaklaşırken değişimini temel alır.

Türevleri açıklamayı için bir örnekle deneyelim.
$u = f(x) = 3x^2-4x$ tanımlayın.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

$x = 1$ diye ayarlayıp $h$ değerinin $0$ değerine yaklaşmasına izin verince $\frac{f(x+h) - f(x)}{h}$ :eqref:`eq_derivative`'in sayısal sonucu $2$'ye yaklaşır .
Bu deney matematiksel bir kanıt olmasa da, daha sonra $u'$ türevinin $x=1$ olduğunda $2$ olduğunu göreceğiz.

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

Kendimizi türevler için birkaç eşdeğer gösterimle tanıştıralım.
$y = f(x)$ verildiğinde, $x$ ve $y$ sırasıyla $f$ işlevinin bağımsız değişkeni ve bağımlı değişkenleridir. Aşağıdaki ifadeler eşdeğerdir:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

burada $\frac{d}{dx}$ ve $D$ sembolleri *türev alma* işlevini gösteren *türev alma* operatörleridir.
Yaygın işlevlerin türevini alma için aşağıdaki kuralları kullanabiliriz:

* $DC = 0$ ($C$ bir sabit),
* $Dx^n = nx^{n-1}$ (*üs kuralı*, $n$ bir gerçel sayı),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

Yukarıdaki yaygın işlevler gibi birkaç basit işlevlerden oluşan bir işlevin türevini alırken için aşağıdaki kurallar bizim için kullanışlı olabilir.
$f$ ve $g$ işlevlerinin ikisinin de türevlenebilir ve $C$'nin sabit olduğunu varsayalım, elimizde *sabit çarpım kuralı*,

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*toplam kuralı*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*çarpım kuralı*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

ve *bölme kuralı* var.

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

Şimdi $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$'ı bulmak için yukarıdaki kurallardan birkaçını uygulayabiliriz .
Bu nedenle, $x = 1$ atadığımız da, $u '= 2$ değerine sahibiz: Sayısal sonucun $2$'ye yaklaştığı bu bölümdeki önceki denememiz tarafından desteklenmektedir.
Bu türev aynı zamanda $u = f(x)$ eğrisine $x = 1$'deki teğet doğrusunun eğimidir.

To visualize such an interpretation of derivatives, we will use `matplotlib`, a popular plotting library in Python.
To configure properties of the figures produced by `matplotlib`, we need to define a few functions.
In the following, the `use_svg_display` function specifies the `matplotlib` package to output the svg figures for sharper images.

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

We define the `set_figsize` function to specify the figure sizes. Note that here we directly use `d2l.plt` since the import statement `from matplotlib import pyplot as plt` has been marked for being saved in the `d2l` package in the preface.

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

The following `set_axes` function sets properties of axes of figures produced by `matplotlib`.

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
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

With these three functions for figure configurations, we define the `plot` function to plot multiple curves succinctly since we will need to visualize many curves throughout the book.

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data instances."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
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

Now we can plot the function $u = f(x)$ and its tangent line $y = 2x - 3$ at $x=1$, where the coefficient $2$ is the slope of the tangent line.

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Partial Derivatives

So far we have dealt with the differentiation of functions of just one variable.
In deep learning, functions often depend on *many* variables.
Thus, we need to extend the ideas of differentiation to these *multivariate* functions.


Let $y = f(x_1, x_2, \ldots, x_n)$ be a function with $n$ variables. The *partial derivative* of $y$ with respect to its $i^\mathrm{th}$  parameter $x_i$ is

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


To calculate $\frac{\partial y}{\partial x_i}$, we can simply treat $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ as constants and calculate the derivative of $y$ with respect to $x_i$.
For notation of partial derivatives, the following are equivalent:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$


## Gradients

We can concatenate partial derivatives of a multivariate function with respect to all its variables to obtain the *gradient* vector of the function.
Suppose that the input of function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is an $n$-dimensional vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ and the output is a scalar. The gradient of the function $f(\mathbf{x})$ with respect to $\mathbf{x}$ is a vector of $n$ partial derivatives:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

where $\nabla_{\mathbf{x}} f(\mathbf{x})$ is often replaced by $\nabla f(\mathbf{x})$ when there is no ambiguity.

Let $\mathbf{x}$ be an $n$-dimensional vector, the following rules are often used when differentiating multivariate functions:

* For all $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
* For all  $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
* For all  $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Similarly, for any matrix $\mathbf{X}$, we have $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$. As we will see later, gradients are useful for designing optimization algorithms in deep learning.


## Chain Rule

However, such gradients can be hard to find.
This is because multivariate functions in deep learning are often *composite*, so we may not apply any of the aforementioned rules to differentiate these functions.
Fortunately, the *chain rule* enables us to differentiate composite functions.

Let us first consider functions of a single variable.
Suppose that functions $y=f(u)$ and $u=g(x)$ are both differentiable, then the chain rule states that

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

Now let us turn our attention to a more general scenario where functions have an arbitrary number of variables.
Suppose that the differentiable function $y$ has variables $u_1, u_2, \ldots, u_m$, where each differentiable function $u_i$ has variables $x_1, x_2, \ldots, x_n$.
Note that $y$ is a function of $x_1, x_2, \ldots, x_n$.
Then the chain rule gives

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

for any $i = 1, 2, \ldots, n$.



## Summary


* Differential calculus and integral calculus are two branches of calculus, where the former can be applied to the ubiquitous optimization problems in deep learning.
* A derivative can be interpreted as the instantaneous rate of change of a function with respect to its variable. It is also the slope of the tangent line to the curve of the function.
* A gradient is a vector whose components are the partial derivatives of a multivariate function with respect to all its variables.
* The chain rule enables us to differentiate composite functions.



## Exercises

1. Plot the function $y = f(x) = x^3 - \frac{1}{x}$ and its tangent line when $x = 1$.
1. Find the gradient of the function $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. What is the gradient of the function $f(\mathbf{x}) = \|\mathbf{x}\|_2$?
1. Can you write out the chain rule for the case where $u = f(x, y, z)$ and $x = x(a, b)$, $y = y(a, b)$, and $z = z(a, b)$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
