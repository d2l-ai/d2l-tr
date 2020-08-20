# Tek Değişkenli Kalkülüs
:label:`sec_single_variable_calculus`

:numref:`sec_calculus`'de, diferansiyel (türevsel) hesabın (kalkülüsün) temel elemanlarını gördük. Bu bölüm, analizin temellerine ve bunu makine öğrenmesi bağlamında nasıl anlayıp uygulayabileceğimize daha derin bir dalış yapıyor.

## Diferansiyel Kalkülüs
Diferansiyel hesap, temelde fonksiyonların küçük değişiklikler altında nasıl davrandığının incelenmesidir. Bunun neden derin öğrenmenin özü olduğunu görmek için bir örnek ele alalım.

Kolaylık olması açısından ağırlıkların tek bir $\mathbf{w} = (w_1, \ldots, w_n)$ vektörüne birleştirildiği derin bir sinir ağımız olduğunu varsayalım. Bir eğitim veri kümesi verildiğinde, bu veri kümesindeki sinir ağımızın $\mathcal{L}(\mathbf{w})$ olarak yazacağımız kaybını dikkate alıyoruz.

Bu işlev olağanüstü derecede karmaşıktır ve belirli mimarinin tüm olası modellerinin performansını bu veri kümesinde şifreler, dolayısıyla $\mathbf{w}$ hangi ağırlık kümesinin kaybı en aza indireceğini söylemek neredeyse imkansızdır. Bu nedenle, pratikte genellikle ağırlıklarımızı *rastgele* ilkleyerek başlarız ve ardından yinelemeli olarak kaybı olabildiğince hızlı düşüren yönde küçük adımlar atarız.

O zaman soru, yüzeyde daha kolay olmayan bir şeye dönüşür: ağırlıkları olabildiğince çabuk düşüren yönü nasıl buluruz? Bunu derinlemesine incelemek için, önce tek bir ağırlığa sahip durumu inceleyelim: Tek bir $x$ gerçel değeri için $L(\mathbf{w}) = L(x)$.

Şimdi $x$ alalım ve onu küçük bir miktar $x+\epsilon$ olarak değiştirdiğimizde ne olacağını anlamaya çalışalım. Somut olmak istiyorsanız, $\epsilon = 0.0000001$ gibi bir sayı düşünün. Neler olduğunu görselleştirmemize yardımcı olmak için, $[0, 3]$ üzerinden $f(x) = \sin(x^x)$ gibi bir örnek fonksiyonun grafiğini çizelim.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

Bu büyük ölçekte, işlevin davranışı basit değildir. Ancak, aralığımızı $[1.75,2.25]$ gibi daha küçük bir değere düşürürsek, grafiğin çok daha basit hale geldiğini görürüz.

```{.python .input}
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

Bunu aşırıya götürürsek, küçücük bir bölüme yakınlaştırırsak, davranış çok daha basit hale gelir: Bu sadece düz bir çizgidir.

```{.python .input}
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

Bu, tek değişkenli analizin temel gözlemidir: Tanıdık fonksiyonların davranışı, yeterince küçük bir aralıktaki bir doğru ile modellenebilir. Bu, çoğu fonksiyon için, fonksiyonun $x$ değerini biraz kaydırdığımızda, $f(x)$ çıktısının da biraz kayacağını beklemenin mantıklı olduğu anlamına gelir. Yanıtlamamız gereken tek soru, "Girdideki değişime kıyasla çıktıdaki değişim ne kadar büyük? Yarısı kadar büyük mü? İki kat büyük mü?"

Bu nedenle, bir fonksiyonun çıktısındaki değişim oranını, fonksiyonun girdisindeki küçük bir değişiklik için dikkate alabiliriz. Bunu kurallı olarak yazabiliriz:

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

Bu, kod içinde oynamaya başlamak için zaten yeterli. Örneğin, $L(x) = x^{2} + 1701(x-4)^3$ olduğunu bildiğimizi varsayalım, o zaman bu değerin $x = 4$ noktasında ne kadar büyük olduğunu aşağıdaki gibi görebiliriz.

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

Şimdi, eğer dikkatli olursak, bu sayının çıktısının şüpheli bir şekilde $8$'e yakın olduğunu fark edeceğiz. Aslında, $\epsilon$'u düşürürsek, değerin giderek $8$'e yaklaştığını göreceğiz. Böylece, doğru bir şekilde, aradığımız değerin (girdideki bir değişikliğin çıktıyı değiştirdiği derece) $x = 4$ noktasında $8$ olması gerektiği sonucuna varabiliriz. Bir matematikçinin bu gerçeği kodlama şekli aşağıdadır:

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

Biraz tarihsel bir bilgi olarak: Sinir ağı araştırmalarının ilk birkaç on yılında, bilim adamları bu algoritmayı (*sonlu farklar yöntemi*) küçük bir oynama altında bir kayıp fonksiyonunun nasıl değiştiğini değerlendirmek için kullandılar; sadece ağırlıkları değiştirin ve kayıp nasıl değişir izleyin. Bu, sayısal olarak verimsizdir ve bir değişkendeki tek bir değişikliğin kaybı nasıl etkilediğini görmek için kayıp fonksiyonunun iki değerlendirmesini gerektirir. Bunu birkaç bin parametreyle bile yapmaya çalışsaydık, tüm veri kümesi üzerinde ağın birkaç bin değerlendirme yapmasını gerektirecekti! :cite:`Rumelhart.Hinton.Williams.ea.1988`de sunulan *geri yayma algoritması* ile ağırlıkların herhangi bir değişikliğinin, veri kümesi üzerindeki ağın tek bir tahminiyle aynı hesaplama süresinde kaybı nasıl değiştireceğini hesaplamak için bir yol sağladığı 1986 yılına kadar çözülemedi.

Örneğimizdeki bu $8$ değeri, $x$'in farklı değerleri için farklıdır, bu nedenle, $x$'in bir işlevi olarak tanımlamak mantıklıdır. Daha resmi olarak, bu değere bağlı değişim oranına *türev* denir ve şöyle yazılır:

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

Türev için farklı metinler farklı gösterimler kullanacaktır. Örneğin, aşağıdaki tüm gösterimler aynı şeyi gösterir:

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

Çoğu yazar tek bir gösterim seçer ve ona sadık kalır, ancak bu bile garanti edilmez. Bunların hepsine aşina olmak en iyisidir. Karmaşık bir ifadenin türevini almak istemiyorsak, bu metin boyunca $\frac{df}{dx}$ gösterimini kullanacağız, bu durumda aşağıdaki gibi ifadeler yazmak için $\frac{d}{dx}f$ kullanacağız 

$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

Çoğu zaman, $x$ değerinde küçük bir değişiklik yaptığımızda bir fonksiyonun nasıl değiştiğini görmek için türev tanımını :eqref:`eq_der_def` yeniden açmak sezgisel olarak kullanışlıdır:

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

Son denklem açıkça belirtilmeye değer. Bize, herhangi bir işlevi alırsanız ve girdiyi küçük bir miktar değiştirirseniz, çıktının türevin ölçeklendirdiği küçük miktarda değişeceğini söyler.

Bu şekilde türevi, girdideki bir değişiklikten çıktıda ne kadar büyük değişiklik elde ettiğimizi söyleyen ölçeklendirme faktörü olarak anlayabiliriz.

## Kalkülüs Kuralları
:label:`sec_derivative_table`

Şimdi, açık bir fonksiyonun türevinin nasıl hesaplanacağını anlama görevine dönüyoruz. Kalkülüsün tam bir biçimsel incelemesi, her şeyi ilk ilkelerden türetecektir. Burada bu cazibeye kapılmayacağız, bunun yerine karşılaşılan genel kuralların anlaşılmasını sağlayacağız.

### Yaygın Türevler
:numref:`sec_calculus`da görüldüğü gibi, türevleri hesaplarken hesaplamayı birkaç temel işleve indirgemek için çoğu zaman bir dizi kural kullanılabilir. Referans kolaylığı için burada tekrar ediyoruz.

* **Sabitlerin türevi.** $\frac{d}{dx}c = 0$.
* **Doğrusal fonksiyonların türevi.** $\frac{d}{dx}(ax) = a$.
* **Kuvvet kuralı.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Üstellerin türevi.** $\frac{d}{dx}e^x = e^x$.
* **Logaritmanın türevi.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### Türev Kuralları
Her türevin ayrı ayrı hesaplanması ve bir tabloda depolanması gerekirse, diferansiyel kalkülüs neredeyse imkansız olurdu. Yukarıdaki türevleri genelleştirebilmemiz ve $f(x) = \log\left(1+(x-1)^{10}\right)$ türevini bulmak gibi daha karmaşık türevleri hesaplayabilmemiz matematiğin bir armağanıdır. :numref:`sec_calculus`da bahsedildiği gibi, bunu yapmanın anahtarı, fonksiyonları aldığımızda ve bunları çeşitli şekillerde birleştirdiğimizde, özellikle toplamlar, çarpımlar ve bileşimler, olanları kodlamaktır.

* **Toplam kuralı.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$
* **Çarpım kuralı.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Zincir kuralı.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

Bu kuralları anlamak için :eqref:`eq_small_change`'i nasıl kullanabileceğimize bir bakalım. Toplam kuralı için aşağıdaki akıl yürütme zincirini düşünün:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

Bu sonucu $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$ gerçeğiyle karşılaştırdığımızda istenildiği gibi $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ olduğunu görürüz. Buradaki sezgi şudur: $x$ girdisini değiştirdiğimizde, $g$ ve $h$ çıktının değişmesine $\frac{dg}{dx}(x)$$ ve $\frac{dh}{dx}(x)$ ile birlikte katkıda bulunur.

Çarpım daha inceliklidir ve bu ifadelerle nasıl çalışılacağı konusunda yeni bir gözlem gerektirecektir. Önceden olduğu gibi :eqref:`eq_small_change`'i kullanarak başlayacağız :

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$

Bu, yukarıda yapılan hesaplamaya benzer ve aslında görürüz ki cevabımız ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) $\epsilon$'un yanında duruyor, ancak $\epsilon^{2}$ terimininin boyutu ile ilgili bir durum var. $\epsilon^2$'nin gücü $\epsilon^1$'in gücünden daha yüksek olduğu için buna *yüksek dereceli bir terim* diyeceğiz. Daha sonraki bir bölümde bazen bunların kaydını tutmak isteyeceğimizi göreceğiz, ancak şimdilik $\epsilon = 0.0000001$ ise $\epsilon^{2} = 0.0000000000001$'nın çok daha küçük olduğunu gözlemleyeceğiz. $\epsilon \rightarrow 0$ gider iken, daha yüksek dereceli terimleri güvenle göz ardı edebiliriz. Bu ek bölümünde genel bir kural olarak, iki terimin daha yüksek mertebeden terimlere kadar eşit olduğunu belirtmek için "$\approx$" kullanacağız. Ancak, daha resmi olmak istiyorsak, fark oranını inceleyebiliriz.

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

and see that as we send $\epsilon \rightarrow 0$, the right hand term goes to zero as well.

Finally, with the chain rule, we can again progress as before using :eqref:`eq_small_change` and see that

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

where in the second line we view the function $g$ as having its input ($h(x)$) shifted by the tiny quantity $\epsilon \frac{dh}{dx}(x)$.

These rule provide us with a flexible set of tools to compute essentially any expression desired.  For instance,

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

Where each line has used the following rules:

1. The chain rule and derivative of logarithm.
2. The sum rule.
3. The derivative of constants, chain rule, and power rule.
4. The sum rule, derivative of linear functions, derivative of constants.

Two things should be clear after doing this example:

1. Any function we can write down using sums, products, constants, powers, exponentials, and logarithms can have its derivate computed mechanically by following these rules.
2. Having a human follow these rules can be tedious and error prone!

Thankfully, these two facts together hint towards a way forward: this is a perfect candidate for mechanization!  Indeed backpropagation, which we will revisit later in this section, is exactly that.

### Linear Approximation
When working with derivatives, it is often useful to geometrically interpret the approximation used above.  In particular, note that the equation 

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

approximates the value of $f$ by a line which passes through the point $(x, f(x))$ and has slope $\frac{df}{dx}(x)$.  In this way we say that the derivative gives a linear approximation to the function $f$, as illustrated below:

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### Higher Order Derivatives

Let us now do something that may on the surface seem strange.  Take a function $f$ and compute the derivative $\frac{df}{dx}$.  This gives us the rate of change of $f$ at any point.

However, the derivative, $\frac{df}{dx}$, can be viewed as a function itself, so nothing stops us from computing the derivative of $\frac{df}{dx}$ to get $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$.  We will call this the second derivative of $f$.  This function is the rate of change of the rate of change of $f$, or in other words, how the rate of change is changing. We may apply the derivative any number of times to obtain what is called the $n$-th derivative. To keep the notation clean, we will denote the $n$-th derivative as 

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Let us try to understand *why* this is a useful notion.  Below, we visualize $f^{(2)}(x)$, $f^{(1)}(x)$, and $f(x)$.  

First, consider the case that the second derivative $f^{(2)}(x)$ is a positive constant.  This means that the slope of the first derivative is positive.  As a result, the first derivative $f^{(1)}(x)$ may start out negative, becomes zero at a point, and then becomes positive in the end. This tells us the slope of our original function $f$ and therefore, the function $f$ itself decreases, flattens out, then increases.  In other words, the function $f$ curves up, and has a single minimum as is shown in :numref:`fig_positive-second`.

![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
:label:`fig_positive-second`


Second, if the second derivative is a negative constant, that means that the first derivative is decreasing.  This implies the first derivative may start out positive, becomes zero at a point, and then becomes negative. Hence, the function $f$ itself increases, flattens out, then decreases.  In other words, the function $f$ curves down, and has a single maximum as is shown in :numref:`fig_negative-second`.

![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
:label:`fig_negative-second`


Third, if the second derivative is a always zero, then the first derivative will never change---it is constant!  This means that $f$ increases (or decreases) at a fixed rate, and $f$ is itself a straight line  as is shown in :numref:`fig_zero-second`.

![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

To summarize, the second derivative can be interpreted as describing the way that the function $f$ curves.  A positive second derivative leads to a upwards curve, while a negative second derivative means that $f$ curves downwards, and a zero second derivative means that $f$ does not curve at all.

Let us take this one step further. Consider the function $g(x) = ax^{2}+ bx + c$.  We can then compute that

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

If we have some original function $f(x)$ in mind, we may compute the first two derivatives and find the values for $a, b$, and $c$ that make them match this computation.  Similarly to the previous section where we saw that the first derivative gave the best approximation with a straight line, this construction provides the best approximation by a quadratic.  Let us visualize this for $f(x) = \sin(x)$.

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

We will extend this idea to the idea of a *Taylor series* in the next section. 

### Taylor Series


The *Taylor series* provides a method to approximate the function $f(x)$ if we are given values for the first $n$ derivatives at a point $x_0$, i.e., $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$. The idea will be to find a degree $n$ polynomial that matches all the given derivatives at $x_0$.

We saw the case of $n=2$ in the previous section and a little algebra shows this is

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

As we can see above, the denominator of $2$ is there to cancel out the $2$ we get when we take two derivatives of $x^2$, while the other terms are all zero.  Same logic applies for the first derivative and the value itself.

If we push the logic further to $n=3$, we will conclude that

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

where the $6 = 3 \times 2 = 3!$ comes from the constant we get in front if we take three derivatives of $x^3$.


Furthermore, we can get a degree $n$ polynomial by 

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

where the notation 

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


Indeed, $P_n(x)$ can be viewed as the best $n$-th degree polynomial approximation to our function $f(x)$.

While we are not going to dive all the way into the error of the above approximations, it is worth mentioning the infinite limit. In this case, for well behaved functions (known as real analytic functions) like $\cos(x)$ or $e^{x}$, we can write out the infinite number of terms and approximate the exactly same function

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

Take $f(x) = e^{x}$ as am example. Since $e^{x}$ is its own derivative, we know that $f^{(n)}(x) = e^{x}$. Therefore, $e^{x}$ can be reconstructed by taking the Taylor series at $x_0 = 0$, i.e.,

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

Let us see how this works in code and observe how increasing the degree of the Taylor approximation brings us closer to the desired function $e^x$.

```{.python .input}
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

Taylor series have two primary applications:

1. *Theoretical applications*: Often when we try to understand a too complex function, using Taylor series enables we turn it into a polynomial that we can work with directly.

2. *Numerical applications*: Some functions like $e^{x}$ or $\cos(x)$ are  difficult for machines to compute.  They can store tables of values at a fixed precision (and this is often done), but it still leaves open questions like "What is the 1000-th digit of $\cos(1)$?"  Taylor series are often helpful to answer such questions.  


## Summary

* Derivatives can be used to express how functions change when we change the input by a small amount.
* Elementary derivatives can be combined using derivative rules to create arbitrarily complex derivatives.
* Derivatives can be iterated to get second or higher order derivatives.  Each increase in order provides more fine grained information on the behavior of the function.
* Using information in the derivatives of a single data point, we can approximate well behaved functions by polynomials obtained from the Taylor series.


## Exercises

1. What is the derivative of $x^3-4x+1$?
2. What is the derivative of $\log(\frac{1}{x})$?
3. True or False: If $f'(x) = 0$ then $f$ has a maximum or minimum at $x$?
4. Where is the minimum of $f(x) = x\log(x)$ for $x\ge0$ (where we assume that $f$ takes the limiting value of $0$ at $f(0)$)?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:
