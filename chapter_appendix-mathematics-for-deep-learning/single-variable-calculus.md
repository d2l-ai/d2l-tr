# Tek Değişkenli Hesap
:label:`sec_single_variable_calculus`

:numref:`sec_calculus` içinde, türevsel (diferansiyel) hesabın (kalkülüsün) temel elemanlarını gördük. Bu bölüm, analizin temellerine ve bunu makine öğrenmesi bağlamında nasıl anlayıp uygulayabileceğimize daha derin bir dalış yapıyor.

## Türevsel Hesap
Türevsel hesap, temelde fonksiyonların küçük değişiklikler altında nasıl davrandığının incelenmesidir. Bunun neden derin öğrenmenin özü olduğunu görmek için bir örnek ele alalım.

Kolaylık olması açısından ağırlıkların tek bir $\mathbf{w} = (w_1, \ldots, w_n)$ vektörüne bitiştirildiği derin bir sinir ağımız olduğunu varsayalım. Bir eğitim veri kümesi verildiğinde, bu veri kümesindeki sinir ağımızın $\mathcal{L}(\mathbf{w})$ olarak yazacağımız kaybını dikkate alıyoruz.

Bu işlev olağanüstü derecede karmaşıktır ve belirli mimarinin tüm olası modellerinin performansını bu veri kümesinde şifreler, dolayısıyla hangi $\mathbf{w}$ ağırlık kümesinin kaybı en aza indireceğini söylemek neredeyse imkansızdır. Bu nedenle, pratikte genellikle ağırlıklarımızı *rastgele* ilkleyerek başlarız ve ardından yinelemeli olarak kaybı olabildiğince hızlı düşüren yönde küçük adımlar atarız.

O zaman soru, yüzeyde daha kolay olmayan bir şeye dönüşür: Ağırlıkları olabildiğince çabuk düşüren yönü nasıl buluruz? Bunu derinlemesine incelemek için önce tek bir ağırlığa sahip durumu inceleyelim: Tek bir $x$ gerçel değeri için $L(\mathbf{w}) = L(x)$.

Şimdi $x$'i alalım ve onu küçük bir miktar $x+\epsilon$ olarak değiştirdiğimizde ne olacağını anlamaya çalışalım. Somut olmak istiyorsanız, $\epsilon = 0.0000001$ gibi bir sayı düşünün. Neler olduğunu görselleştirmemize yardımcı olmak için, $[0, 3]$ üzerinden $f(x) = \sin(x^x)$ gibi bir örnek fonksiyonun grafiğini çizelim.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Normal bir aralıktaki bir fonksiyon çiz
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
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Pi'yi tanımla

# Normal bir aralıktaki bir fonksiyon çiz
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # # Pi'yi tanımla

# Normal bir aralıktaki bir fonksiyon çiz
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

Bu büyük ölçekte, işlevin davranışı basit değildir. Ancak, aralığımızı $[1.75,2.25]$ gibi daha küçük bir değere düşürürsek, grafiğin çok daha basit hale geldiğini görürüz.

```{.python .input}
# Aynı işlevi küçük bir aralıkta çizin
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Aynı işlevi küçük bir aralıkta çizin
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Aynı işlevi küçük bir aralıkta çizin
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

Bunu aşırıya götürürsek, küçücük bir bölüme yakınlaştırırsak, davranış çok daha basit hale gelir: Bu sadece düz bir çizgidir.

```{.python .input}
# Aynı işlevi küçük bir aralıkta çizin
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Aynı işlevi küçük bir aralıkta çizin
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Aynı işlevi küçük bir aralıkta çizin
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
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
# Fonksiyonumuzu tanımlayalım
def L(x):
    return x**2 + 1701*(x-4)**3

# Birkaç epsilon için farkı epsilon'a bölerek yazdırın
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

Şimdi, eğer dikkatli olursak, bu sayının çıktısının şüpheli bir şekilde $8$'e yakın olduğunu fark edeceğiz. Aslında, $\epsilon$'u düşürürsek, değerin giderek $8$'e yaklaştığını göreceğiz. Böylece, doğru bir şekilde, aradığımız değerin (girdideki bir değişikliğin çıktıyı değiştirdiği derece) $x = 4$ noktasında $8$ olması gerektiği sonucuna varabiliriz. Bir matematikçinin bu gerçeği kodlama şekli aşağıdadır:

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

Biraz tarihsel bir bilgi olarak: Sinir ağı araştırmalarının ilk birkaç on yılında, bilim adamları bu algoritmayı (*sonlu farklar yöntemi*) küçük bir oynama altında bir kayıp fonksiyonunun nasıl değiştiğini değerlendirmek için kullandılar; sadece ağırlıkları değiştirin ve kayıp nasıl değişir izleyin. Bu, sayısal olarak verimsizdir ve bir değişkendeki tek bir değişikliğin kaybı nasıl etkilediğini görmek için kayıp fonksiyonunun iki değerlendirmesini gerektirir. Bunu birkaç bin parametreyle bile yapmaya çalışsaydık, tüm veri kümesi üzerinde ağın birkaç bin değerlendirme yapmasını gerektirecekti! :cite:`Rumelhart.Hinton.Williams.ea.1988` çalışmasında sunulan *geri yayma algoritması* ile ağırlıkların herhangi bir değişikliğinin, veri kümesi üzerindeki ağın tek bir tahminiyle aynı hesaplama süresinde kaybı nasıl değiştireceğini hesaplamak için bir yol sağladığı 1986 yılına kadar çözülemedi.

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

Bu şekilde türevi, girdideki bir değişiklikten çıktıda ne kadar büyük değişiklik elde ettiğimizi söyleyen ölçeklendirme çarpanı olarak anlayabiliriz.

## Kalkülüs Kuralları
:label:`sec_derivative_table`

Şimdi, açık bir fonksiyonun türevinin nasıl hesaplanacağını anlama görevine dönüyoruz. Kalkülüsün tam bir biçimsel incelemesi, her şeyi ilk ilkelerden türetecektir. Burada bu cazibeye kapılmayacağız, bunun yerine karşılaşılan genel kuralların anlaşılmasını sağlayacağız.

### Yaygın Türevler
:numref:`sec_calculus` içinde görüldüğü gibi, türevleri hesaplarken hesaplamayı birkaç temel işleve indirgemek için çoğu zaman bir dizi kural kullanılabilir. Referans kolaylığı için burada tekrar ediyoruz.

* **Sabitlerin türevi.** $\frac{d}{dx}c = 0$.
* **Doğrusal fonksiyonların türevi.** $\frac{d}{dx}(ax) = a$.
* **Kuvvet kuralı.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Üstellerin türevi.** $\frac{d}{dx}e^x = e^x$.
* **Logaritmanın türevi.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### Türev Kuralları
Her türevin ayrı ayrı hesaplanması ve bir tabloda depolanması gerekirse, türevsel hesap neredeyse imkansız olurdu. Yukarıdaki türevleri genelleştirebilmemiz ve $f(x) = \log\left(1+(x-1)^{10}\right)$ türevini bulmak gibi daha karmaşık türevleri hesaplayabilmemiz matematiğin bir armağanıdır. :numref:`sec_calculus` içinde bahsedildiği gibi, bunu yapmanın anahtarı, fonksiyonları aldığımızda ve bunları çeşitli şekillerde birleştirdiğimizde, özellikle toplamlar, çarpımlar ve bileşimler, olanları kodlamaktır.

* **Toplam kuralı.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$
* **Çarpım kuralı.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Zincir kuralı.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

Bu kuralları anlamak için :eqref:`eq_small_change` ifadesini nasıl kullanabileceğimize bir bakalım. Toplam kuralı için aşağıdaki akıl yürütme zincirini düşünün:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

Bu sonucu $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$ gerçeğiyle karşılaştırdığımızda istenildiği gibi $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ olduğunu görürüz. Buradaki sezgi şudur: $x$ girdisini değiştirdiğimizde, $g$ ve $h$ çıktının değişmesine $\frac{dg}{dx}(x)$ ve $\frac{dh}{dx}(x)$ ile birlikte katkıda bulunur.

Çarpım daha inceliklidir ve bu ifadelerle nasıl çalışılacağı konusunda yeni bir gözlem gerektirecektir. Önceden olduğu gibi :eqref:`eq_small_change` ifadesini kullanarak başlayacağız 
:

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

$\epsilon \rightarrow 0$ giderken, sağdaki terimin de sıfıra gittiğini görürüz.

Son olarak, zincir kuralı ile, :eqref:`eq_small_change` ifadesini kullanmadan önceki gibi tekrar ilerleyebiliriz ve görürüz ki

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

Burada ikinci satırda $g$ fonksiyonunun girdisinin ($h(x)$) minik bir miktar, $\epsilon\frac{dh}{dx} (x)$ kadar, kaydırıldığını görüyoruz.

Bu kurallar, esasen istenen herhangi bir ifadeyi hesaplamak için bize bir dizi esnek araç sağlar. Örneğin,

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

Her satırda sırasıyla aşağıdaki kurallar kullanmıştır:

1. Zincir kuralı ve logaritmanın türevi.
2. Toplam kuralı.
3. Sabitlerin türevi, zincir kuralı ve kuvvet kuralı.
4. Toplam kuralı, doğrusal fonksiyonların türevi, sabitlerin türevi.

Bu örneği yaptıktan sonra iki şey netleşmiş olmalıdır:

1. Toplamları, çarpımları, sabitleri, üsleri, üstelleri ve logaritmaları kullanarak yazabileceğimiz herhangi bir fonksiyonun türevi bu kuralları takip ederek mekanik olarak hesaplanabilir.
2. Bir insanın bu kuralları takip etmesi yorucu ve hataya açık olabilir!

Neyse ki, bu iki gerçek birlikte ileriye doğru bir yol gösteriyor: Bu, mekanikleştirme için mükemmel bir aday! Aslında bu bölümde daha sonra tekrar ele alacağımız geri yayma tam olarak da budur. 

### Doğrusal Yaklaşıklama
Türevlerle çalışırken, yukarıda kullanılan yaklaşıklamayı geometrik olarak yorumlamak genellikle yararlıdır. Özellikle, aşağıdaki denklemin

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

$f$ değerine $(x, f(x))$ noktasından geçen ve $\frac{df}{dx}(x)$ eğimine sahip bir çizgi ile yaklaştığına dikkat edin. Bu şekilde, türevin, aşağıda gösterildiği gibi $f$ fonksiyonuna doğrusal bir yaklaşıklama verdiğini söylüyoruz:

```{.python .input}
# Sinüsü hesapla
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Biraz doğrusal yaklaşım hesaplayın. d(sin(x)) / dx = cos(x) kullanın.
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Sinüsü hesapla
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Biraz doğrusal yaklaşım hesaplayın. d(sin(x)) / dx = cos(x) kullanın.
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Sinüsü hesapla
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Biraz doğrusal yaklaşım hesaplayın. d(sin(x)) / dx = cos(x) kullanın.
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### Yüksek Dereceli Türevler

Şimdi yüzeyde garip görünebilecek bir şey yapalım. Bir $f$ fonksiyonunu alın ve $\frac{df}{dx}$ türevini hesaplayın. Bu bize herhangi bir noktada $f$'nin değişim oranını verir.

Bununla birlikte, $\frac{df}{dx}$ türevi, bir işlev olarak görülebilir, bu nedenle hiçbir şey bizi $\frac{df}{dx}$ türevini, $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$, hesaplamaktan alıkoyamaz. Buna $f$'nin ikinci türevi diyeceğiz. Bu fonksiyon, $f$'nin değişim oranının değişim oranıdır veya başka bir deyişle, değişim oranının nasıl değiştiğidir. $n.$ türev denen türevi elde etmek için herhangi bir sayıda art arda türev uygulayabiliriz. Gösterimi temiz tutmak için, $n.$ türevi şu şekilde göstereceğiz:

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Bunun *neden* yararlı bir fikir olduğunu anlamaya çalışalım. Aşağıda, $f^{(2)}(x)$, $f^{(1)}(x)$ ve $f(x)$'i görselleştiriyoruz.

İlk olarak, ikinci türevin $f^{(2)}(x)$ pozitif bir sabit olduğu durumu düşünün. Bu, birinci türevin eğiminin pozitif olduğu anlamına gelir. Sonuç olarak, birinci türev, $f^{(1)}(x)$, negatif olarak başlayabilir, bir noktada sıfır olur ve sonra sonunda pozitif olur. Bu bize esas fonksiyonumuz $f$'nin eğimini anlatır; dolayısıyla $f$ fonksiyonunun kendisi azalır, düzleşir, sonra artar. Başka bir deyişle, $f$ işlevi yukarı doğru eğrilir ve :numref:`fig_positive-second` şeklinde gösterildiği gibi tek bir minimuma sahiptir.

![İkinci türevin pozitif bir sabit olduğunu varsayarsak, artmakta olan ilk türev, fonksiyonun kendisinin bir minimuma sahip olduğu anlamına gelir.](../img/posSecDer.svg)
:label:`fig_positive-second`

İkincisi, eğer ikinci türev negatif bir sabitse, bu, birinci türevin azaldığı anlamına gelir. Bu, ilk türevin pozitif başlayabileceği, bir noktada sıfır olacağı ve sonra negatif olacağı anlamına gelir. Dolayısıyla, $f$ işlevinin kendisi artar, düzleşir ve sonra azalır. Başka bir deyişle, $f$ işlevi aşağı doğru eğilir ve :numref:`fig_negative-second` şeklinde gösterildiği gibi tek bir maksimuma sahiptir.

![İkinci türevin negatif bir sabit olduğunu varsayarsak, azalmakta olan ilk türev, fonksiyonun kendisinin bir maksimuma sahip olduğu anlamına gelir.](../img/negSecDer.svg)
:label:`fig_negative-second`

Üçüncüsü, eğer ikinci türev her zaman sıfırsa, o zaman ilk türev asla değişmeyecektir---sabittir! Bu, $f$'nin sabit bir oranda arttığı (veya azaldığı) anlamına gelir ve $f$'nin kendisi de :numref:`fig_zero-second` içinde gösterildiği gibi düz bir doğrudur.

![İkinci türevin sıfır olduğunu varsayarsak, ilk türev sabittir, bu da fonksiyonun kendisinin düz bir doğru olduğu anlamına gelir.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

Özetlemek gerekirse, ikinci türev, $f$ fonksiyonunun eğrilerinin şeklinin açıklanması olarak yorumlanabilir. Pozitif bir ikinci türev yukarı doğru bir eğriye yol açarken, negatif bir ikinci türev $f$'nin aşağı doğru eğrildiği ve sıfır ikinci türev ise $f$'nin hiç eğrilmediği anlamına gelir.

Bunu bir adım daha ileri götürelim. $g(x) = ax^{2}+ bx + c$ işlevini düşünün. Birlikte şunu hesaplayabiliriz

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

Aklımızda belirli bir orijinal fonksiyon $f(x)$ varsa, ilk iki türevi hesaplayabilir ve $a,b$ ve $c$ değerlerini bu hesaplamayla eşleşecek şekilde bulabiliriz. İlk türevin düz bir doğru ile en iyi yaklaşıklamayı verdiğini gördüğümüz önceki bölüme benzer şekilde, bu yapı bir ikinci dereceden en iyi yaklaşımı sağlar. Bunu $f(x) = \sin(x)$ için görselleştirelim.

```{.python .input}
# Sinüsü hesapla.
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Biraz ikinci dereceden yaklaşım hesaplayın. d(sin(x)) / dx = cos(x) kullanın.
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Sinüsü hesapla.
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Biraz ikinci dereceden yaklaşım hesaplayın. d(sin(x)) / dx = cos(x) kullanın.
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Sinüsü hesapla.
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Biraz ikinci dereceden yaklaşım hesaplayın. d(sin(x)) / dx = cos(x) kullanın.
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

Bu fikri bir sonraki bölümde bir *Taylor serisi* fikrine genişleteceğiz.

### Taylor Serisi


*Taylor serisi*, bir $x_0$ noktası için ilk $n$ türevlerinin değerleri, $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$ verilirse, $f(x)$ fonksiyonuna yaklaşıklamak için bir yöntem sağlar. Buradaki fikir, $x_0$'da verilen tüm türevlerle eşleşen bir $n$ dereceli polinom bulmak olacaktır.

Önceki bölümde $n = 2$ durumunu gördük ve biraz cebir bunu gösterecektir:

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

Yukarıda gördüğümüz gibi, $2$ paydası, $x^2$'nin türevini iki defa aldığımızda elde ettiğimiz $2$'yi iptal etmek için oradadır, diğer terimlerin hepsi sıfırdır. Aynı mantık, birinci türev ve değerin kendisi için de geçerlidir.

Mantığı $n = 3$ değerine uygularsak, şu sonuca varacağız:

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$


burada $6 = 3 \times 2 = 3!$, $x^3$'ün üçüncü türevini alırsak önümüze gelen sabitten gelir.

Dahası, bir $n$ dereceli polinom elde edebiliriz.

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

buradaki eşdeğer gösterim aşağıdadır.

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Aslında, $P_n(x)$, $f(x)$ fonksiyonumuza yaklaşan en iyi $n$-dereceli polinom olarak görülebilir.

Yukarıdaki tahminlerin hatasına tam olarak dalmayacak olsak da, sonsuz limitten bahsetmeye değer. Bu durumda, $\cos(x)$ veya $e^{x}$ gibi iyi huylu işlevler (gerçel analitik işlevler olarak bilinir) için, sonsuz sayıda terim yazabilir ve tam olarak aynı işlevi yaklaşık olarak tahmin edebiliriz

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

Örnek olarak $f(x) = e^{x}$'i alalım. $e^{x}$ kendisinin türevi olduğundan, $f^{(n)}(x) = e^{x}$ olduğunu biliyoruz. Bu nedenle, $e^{x}$, Taylor serisi $x_0 = 0$ alınarak yeniden yapılandırılabilir, yani,

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

Bunun kodda nasıl çalıştığını görelim ve Taylor yaklaşımının derecesini artırmanın bizi istenen $e^x$ fonksiyonuna nasıl yaklaştırdığını görelim.

```{.python .input}
# Üstel işlevi hesaplayın
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Birkaç Taylor serisi yaklaşımını hesaplayın
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Üstel işlevi hesaplayın
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Birkaç Taylor serisi yaklaşımını hesaplayın
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Üstel işlevi hesaplayın
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Birkaç Taylor serisi yaklaşımını hesaplayın
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

Taylor serisinin iki ana uygulaması vardır:

1. *Teorik uygulamalar*: Genellikle çok karmaşık bir fonksiyonu anlamaya çalıştığımızda, Taylor serisini kullanmak onu doğrudan çalışabileceğimiz bir polinom haline dönüştürebilmemizi sağlar.

2. *Sayısal (numerik) uygulamalar*: $e^{x}$ veya $\cos(x)$ gibi bazı işlevlerin hesaplanması makineler için zordur. Değer tablolarını sabit bir hassasiyette depolayabilirler (ve bu genellikle yapılır), ancak yine de "$\cos(1)$'in 1000'inci basamağı nedir?" gibi açık sorular kalır. Taylor serileri, bu tür soruları cevaplamak için genellikle yardımcı olur.

## Özet

* Türevler, girdiyi küçük bir miktar değiştirdiğimizde fonksiyonların nasıl değişeceğini ifade etmek için kullanılabilir.
* Temel türevler, karmaşık türevleri bulmak için türev kuralları kullanılarak birleştirilebilir.
* Türevler, ikinci veya daha yüksek dereceden türevleri elde etmek için yinelenebilir. Derecedeki her artış, işlevin davranışı hakkında daha ayrıntılı bilgi sağlar.
* Tek bir veri örneğinin türevlerindeki bilgileri kullanarak, Taylor serisinden elde edilen polinomlarla iyi huylu fonksiyonları yaklaşık elde edebiliriz.


## Alıştırmalar

1. $x^3-4x+1$'in türevi nedir?
2. $\log(\frac{1}{x})$'in türevi nedir?
3. Doğru veya yanlış: $f'(x) = 0$ ise, $f$ işlevi $x$'te maksimum veya minimuma sahip midir?
4. $x\ge0$ için $f(x) = x\log(x)$ minimumu nerededir (burada $f$'nin $f(0)$'da $0$'ın limit değerini aldığını varsayıyoruz)?


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1088)
:end_tab:


:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1089)
:end_tab:
