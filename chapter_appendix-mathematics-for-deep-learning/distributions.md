# Dağılımlar
:label:`sec_distributions`

Artık hem kesikli hem de sürekli ortamda olasılıkla nasıl çalışılacağını öğrendiğimize göre, şimdi karşılaşılan yaygın dağılımlardan bazılarını öğrenelim. Makine öğrenmesi alanına bağlı olarak, bunlardan çok daha fazlasına aşina olmamız gerekebilir; derin öğrenmenin bazı alanları için potansiyel olarak hiç kullanılmıyorlar. Ancak bu, aşina olunması gereken iyi bir temel listedir. Önce bazı ortak kütüphaneleri içeri aktaralım.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Pi'yi tanımla
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Pi'yi tanımla
```

## Bernoulli

Bu, genellikle karşılaşılan en basit rastgele değişkendir. Bu rastgele değişken, $p$ olasılıkla $1$ ve $1-p$ olasılıkla $0$ gelen bir yazı tura atmayı kodlar. Bu dağılımla rastgele bir değişkenimiz $X$ varsa, şunu yazacağız:

$$
X \sim \mathrm{Bernoulli}(p).
$$

Birikimli dağılım fonksiyonu şöyledir:

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

Olasılık kütle fonksiyonu aşağıda çizilmiştir.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Şimdi, :eqref:`eq_bernoulli_cdf` birikimli dağılım fonksiyonunu çizelim.

```{.python .input}
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

Eğer $X \sim \mathrm{Bernoulli}(p)$ ise, o zaman:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

Bir Bernoulli rastgele değişkeninden keyfi şekilli bir diziyi aşağıdaki gibi örnekleyebiliriz.

```{.python .input}
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## Ayrık Tekdüze Dağılım

Bir sonraki yaygın karşılaşılan rastgele değişken, ayrık bir tekdüzedir. Buradaki tartışmamız için, $\{1, 2, \ldots, n\}$ tam sayılarında desteklendiğini varsayacağız, ancak herhangi bir tüm değerler kümesi serbestçe seçilebilir. Bu bağlamda *tekdüze* kelimesinin anlamı, olabilir her değerin eşit derecede olası olmasıdır. $i \in \{1, 2, 3, \ldots, n\}$ değerinin olasılığı $p_i = \frac{1}{n}$'dir. Bu dağılımla $X$ rastgele değişkenini şu şekilde göstereceğiz:

$$
X \sim U(n).
$$

Birikimli dağılım fonksiyonunu böyledir:

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \text{ öyleki } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

İlk olarak olasılık kütle fonksiyonunu çizelim.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Şimdi, :eqref:`eq_discrete_uniform_cdf` birikimli dağılım fonksiyonunu çizelim.

```{.python .input}
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Eğer $X \sim U(n)$ ise, o zaman:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

Aşağıdaki gibi, ayrık bir tekdüze rastgele değişkenden keyfi şekilli bir diziyi örnekleyebiliriz.

```{.python .input}
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## Sürekli Tekdüze Dağılım

Şimdi, sürekli tekdüze dağılımı tartışalım. Bu rastgele değişkenin arkasındaki fikir şudur: Ayrık tekdüze dağılımdaki $n$'yi arttırırsak ve bunu $[a, b]$ aralığına sığacak şekilde ölçeklendirirsek; sadece $[a, b]$ aralığında, hepsi eşit olasılıkla, keyfi bir değer seçen sürekli bir rastgele değişkene yaklaşacağız. Bu dağılımı şu şekilde göstereceğiz

$$
X \sim U(a, b).
$$

Olasılık yoğunluk fonksiyonu şöyledir: 

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

Birikimli dağılım fonksiyonunu şöyledir:

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

Önce, :eqref:`eq_cont_uniform_pdf` olasılık yoğunluk dağılımı fonksiyonunu çizelim.

```{.python .input}
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```


Şimdi, :eqref:`eq_cont_uniform_cdf` birikimli dağılım fonksiyonunu çizelim.

```{.python .input}
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Eğer $X \sim U(a, b)$ ise, o zaman:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

Aşağıdaki gibi tekdüze bir rastgele değişkenden keyfi şekilli bir diziyi örnekleyebiliriz. Bunun $U(0,1)$'den varsayılan örnekler olduğuna dikkat edin, bu nedenle farklı bir aralık istiyorsak, onu ölçeklendirmemiz gerekir.

```{.python .input}
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## Binom (İki Terimli) Dağılım

İşleri biraz daha karmaşık hale getirelim ve *iki terimli (binom)* rastgele değişkeni inceleyelim. Bu rastgele değişken, her birinin başarılı olma olasılığı $p$ olan $n$ bağımsız deneyler dizisi gerçekleştirmekten ve kaç tane başarı görmeyi beklediğimizi sormaktan kaynaklanır.

Bunu matematiksel olarak ifade edelim. Her deney, başarıyı kodlamak için $1$ ve başarısızlığı kodlamak için $0$ kullanacağımız bağımsız bir rastgele değişken $X_i$'dir. Her biri $p$ olasılığı ile başarılı olan bağımsız bir yazı tura olduğu için, $X_i \sim \mathrm{Bernoulli}(p)$ diyebiliriz. Sonra, iki terimli rastgele değişken aşağıdaki gibidir:

$$
X = \sum_{i=1}^n X_i.
$$

Bu durumda, böyle yazacağız:

$$
X \sim \mathrm{Binomial}(n, p).
$$

Birikimli dağılım işlevini elde etmek için, tam olarak $k$ başarı elde etmenin gerçekleşebileceği hepsi $ p^k(1-p)^{nk}$ olasılıklı $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ yolu olduğunu fark etmemiz gerekir. Böylece birikimli dağılım işlevi bu olur:

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ öyleki } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

İlk olarak olasılık kütle fonksiyonu çizelim.

```{.python .input}
n, p = 10, 0.2

# Binom katsayısını hesapla
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Binom katsayısını hesapla
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Binom katsayısını hesapla
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Şimdi, :eqref:`eq_binomial_cdf` birikimli dağılım fonksiyonunu çizelim.

```{.python .input}
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Eğer $X \sim \mathrm{Binomial}(n, p)$ ise, o zaman:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

Bu, $n$ Bernoulli rastgele değişkenlerinin toplamı üzerindeki beklenen değerin doğrusallığından ve bağımsız rastgele değişkenlerin toplamının varyansının varyansların toplamı olduğu gerçeğinden kaynaklanır. Bu aşağıdaki gibi örneklenebilir.

```{.python .input}
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## Poisson Dağılımı

Şimdi bir düşünce deneyi yapalım. Bir otobüs durağında duruyoruz ve önümüzdeki dakika içinde kaç otobüsün geleceğini bilmek istiyoruz. $X^{(1)} \sim \mathrm{Bernoulli}(p)$'i ele alarak başlayalım, bu basitçe bir otobüsün bir dakikalık pencerede varma olasılığıdır. Bir şehir merkezinden uzaktaki otobüs durakları için bu oldukça iyi bir yaklaşıklama olabilir. Bir dakikada birden fazla otobüs göremeyebiliriz.

Ancak, yoğun bir bölgedeysek, iki otobüsün gelmesi mümkün veya hatta muhtemeldir. Bunu, rastgele değişkenimizi ilk 30 saniye veya ikinci 30 saniye için ikiye bölerek modelleyebiliriz. Bu durumda bunu yazabiliriz:

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

burada $X^{(2)}$ tüm toplamdır ve $X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$. Toplam dağılım bu durumda $X^{(2)} \sim \mathrm{Binomial}(2, p/2)$ olur.

Neden burada duralım? O dakikayı $n$ parçaya ayırmaya devam edelim. Yukarısıyla aynı mantıkla, bunu görüyoruz:

$$X^{(n)} \sim \mathrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

Bu rastgele değişkenleri düşünün. Önceki bölümden şunu biliyoruz :eqref:`eq_eq_poisson_approx`, ortalama $\mu_{X^{(n)}} = n(p/n) = p$ ve varyans $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$'dir. $n \rightarrow \infty$ alırsak, bu sayıların ortalama $\mu_{X^{(\infty)}} = p$ ve varyans $\sigma_{X^{(\infty)}}^2 = p$ şeklinde sabitlendiğini görebiliriz. Bu, bu sonsuz alt bölüm limitinde tanımlayabileceğimiz bazı rastgele değişkenlerin *olabileceğini* gösterir.

Bu çok fazla sürpriz olmamalı, çünkü gerçek dünyada sadece otobüslerin geliş sayısını sayabiliyoruz, ancak matematiksel modelimizin iyi tanımlandığını görmek güzel. Bu tartışma, *nadir olaylar yasası* olarak kurallı yapılabilir.

Bu akıl yürütmeyi dikkatlice takip ederek aşağıdaki modele ulaşabiliriz. $\{0,1,2, \ldots \}$ değerlerini aşağıdaki olasılıkla alan rastgele bir değişken ise $X \sim \mathrm{Poisson}(\lambda)$ diyeceğiz.

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass` 

$\lambda > 0$ değeri *oran* (veya *şekil* parametresi) olarak bilinir ve bir zaman biriminde beklediğimiz ortalama varış sayısını belirtir.

Birikimli dağılım işlevini elde etmek için bu olasılık kütle işlevini toplayabiliriz.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ öyleki } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

İlk olarak :eqref:`eq_poisson_mass` olasılık kütle fonksiyonu çizelim.

```{.python .input}
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Şimdi, :eqref:`eq_poisson_cdf` birikimli dağılım fonksiyonunu çizelim.

```{.python .input}
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Yukarıda gördüğümüz gibi, ortalamalar ve varyanslar özellikle nettir. Eğer $X \sim \mathrm{Poisson}(\lambda)$ ise, o zaman:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

Bu aşağıdaki gibi örneklenebilir.

```{.python .input}
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gauss Dağılımı

Şimdi farklı ama ilişkili bir deney yapalım. Tekrar $n$ bağımsız $\mathrm{Bernoulli}(p)$ ölçümlerini, $X_i$'yi, gerçekleştirdiğimizi varsayalım. Bunların toplamının dağılımı $X^{(n)} \sim \mathrm{Binomial}(n, p)$ şeklindedir. Burada $n$ arttıkça ve $p$ azaldıkça bir limit almak yerine, $p$'yi düzeltelim ve sonra $n \rightarrow \infty$ diyelim. Bu durumda $\mu_{X^{(n)}} = np \rightarrow \infty$ ve $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$ olur, bu nedenle bu limitin iyi tanımlanması gerektiğini düşünmek için hiçbir neden yoktur.

Ancak, tüm umudumuzu kaybetmeyelim! Onları tanımlayarak ortalama ve varyansın iyi davranmasını sağlayalım:

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

Burada ortalama sıfır ve varyans bir olduğu görülebilir ve bunun nedenle bazı sınırlayıcı dağılıma yakınsayacağına inanmak mantıklıdır. Bu dağılımların neye benzediğini çizersek, işe yarayacağına daha da ikna olacağız.

```{.python .input}
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

Unutulmaması gereken bir şey: Poisson durumu ile karşılaştırıldığında, şimdi standart sapmaya bölüyoruz, bu da olası sonuçları gittikçe daha küçük alanlara sıkıştırdığımız anlamına gelir. Bu, limitimizin artık ayrık olmayacağının, bilakis sürekli olacağının bir göstergesidir.

Burada oluşan şeyin türetilmesi bu kitabın kapsamı dışındadır, ancak *merkezi limit teoremi*, $n \rightarrow \infty$ iken bunun Gauss Dağılımını (veya bazen normal dağılımı) vereceğini belirtir. Daha açık bir şekilde, herhangi bir $a, b$ için:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

burada rastgele bir değişkenin normalde verilen ortalama $\mu$ ve varyans $\sigma^2$ ile dağıldığını söyleriz, $X \sim \mathcal{N} (\mu, \sigma^2)$ ise, $X$ yoğunluğu şöyledir:

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

Önce, :eqref:`eq_gaussian_pdf` olasılık yoğunluk dağılım fonksiyonunu çizelim.

```{.python .input}
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

Şimdi, birikimli dağılım fonksiyonunu çizelim. Bu ek bölümün kapsamı dışındadır, ancak Gauss b.d.f.'nun daha temel işlevlerden tanımlı kapalı-şekil formülü yoktur. Bu integrali sayısal olarak hesaplamanın bir yolunu sağlayan `erf`'i kullanacağız.

```{.python .input}
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'b.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'b.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'b.d.f.')
```

Meraklı okuyucular bu terimlerin bazılarını tanıyacaktır. Aslında, bu integralla :numref:`sec_integral_calculus` içinde karşılaştık. Aslında, $p_X(x)$'nin toplamda bir birim alana sahip olduğunu ve dolayısıyla geçerli bir yoğunluk olduğunu görmek için tam olarak bu hesaplamaya ihtiyacımız var.

Bozuk para atmalarla çalışma seçimimiz, hesaplamaları kısalttı, ancak bu seçimle ilgili hiçbir şey temel (zorunlu) değildi. Gerçekten de, bağımsız aynı şekilde dağılmış rastgele değişkenlerden, $X_i$'den, oluşan herhangi bir koleksiyon alırsak ve aşağıdaki gibi hesaplarsak:

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

O zaman:

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

yaklaşık olarak Gauss olacak. Çalışması için ek gereksinimler vardır, en yaygın olarak da $E[X^4] < \infty$, ancak işin felsefesi açıktır.

Merkezi limit teoremi, Gauss'un olasılık, istatistik ve makine öğrenmesi için temel olmasının nedenidir. Ölçtüğümüz bir şeyin birçok küçük bağımsız katkının toplamı olduğunu söyleyebildiğimizde, ölçülen şeyin Gauss'a yakın olacağını varsayabiliriz.

Gauss'ların daha birçok büyüleyici özelliği var ve burada bir tanesini daha tartışmak istiyoruz. Gauss, *maksimum entropi dağılımı* olarak bilinen şeydir. Entropiye daha derinlemesine :numref:`sec_information_theory` içinde gireceğiz, ancak bu noktada bilmemiz gereken tek şey bunun bir rastgelelik ölçüsü olduğudur. Titiz bir matematiksel anlamda, Gauss'u sabit ortalama ve varyanslı rastgele değişkenin *en* rastgele seçimi olarak düşünebiliriz. Bu nedenle, rastgele değişkenimizin herhangi bir ortalama ve varyansa sahip olduğunu bilirsek, Gauss bir anlamda yapabileceğimiz en muhafazakar dağılım seçimidir.

Bölümü kapatırken, $X \sim \mathcal{N}(\mu, \sigma^2)$ ise şunu hatırlayalım:

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

Aşağıda gösterildiği gibi Gauss (veya standart normal) dağılımından örneklem alabiliriz.

```{.python .input}
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## Üstel Ailesi
:label:`subsec_exponential_family`

Yukarıda listelenen tüm dağılımlar için ortak bir özellik, hepsinin ait olduğu *üstel aile* olarak bilinmesidir. Üstel aile, yoğunluğu aşağıdaki biçimde ifade edilebilen bir dizi dağılımdır:

$$p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \mathrm{exp} \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

Bu tanım biraz incelikli olabileceğinden, onu yakından inceleyelim.

İlk olarak, $h(\mathbf{x})$, *altta yatan ölçü* veya *temel ölçü* olarak bilinir. Bu, üstel ağırlığımızla değiştirdiğimiz orijinal bir ölçü seçimi olarak görülebilir.

İkinci olarak, *doğal parametreler* veya *kanonik parametreler* olarak adlandırılan $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$ vektörüne sahibiz. Bunlar, temel ölçünün nasıl değiştirileceğini tanımlar. Doğal parametreler, $\mathbf{x}= (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ değişkeninin bazı $T(\cdot)$ fonksiyonlarına karşı bu parametrelerin nokta çarpımını ve üstünü alarak yeni ölçüye girerler. $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ vektörüne *$\boldsymbol{\eta}$ için yeterli istatistikler* denir. Bu ad, $T(\mathbf{x})$ tarafından temsil edilen bilgiler olasılık yoğunluğunu hesaplamak için yeterli olduğundan ve $\mathbf{x}$'in örnekleminden başka hiçbir bilgi gerekmediğinden kullanılır.

Üçüncü olarak, elimizde yukarıdaki :eqref:`eq_exp_pdf` dağılımının intergralinin bir olmasını sağlayan, birikim fonksiyonu olarak adlandırılan $A(\boldsymbol{\eta})$ var, yani:

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \mathrm{exp}
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

Somut olmak için Gauss'u ele alalım. $\mathbf{x}$ öğesinin tek değişkenli bir değişken olduğunu varsayarsak, yoğunluğunun şu olduğunu gördük:

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \mathrm{exp} 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \mathrm{exp} \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

Bu, üstel ailenin tanımıyla eşleşir:

* *Temel ölçü*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *Doğal parametreler*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *Yeterli istatistikler*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, and
* *Birikim işlevi*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

Yukarıdaki terimlerin her birinin kesin seçiminin biraz keyfi olduğunu belirtmekte fayda var. Gerçekten de önemli olan özellik, dağılımın tam formunun kendisi değil, bu formda ifade edilebilmesidir.

:numref:`subsec_softmax_and_derivatives` içinde bahsettiğimiz gibi, yaygın olarak kullanılan bir teknik, $\mathbf{y}$ nihai çıktısının üstel bir aile dağılımını takip ettiğini varsaymaktır. Üstel aile, makine öğrenmesinde sıkça karşılaşılan yaygın ve güçlü bir dağılım ailesidir.

## Özet
* Bernoulli rastgele değişkenleri, evet/hayır sonucu olan olayları modellemek için kullanılabilir.
* Ayrık tekdüze dağılım modeli, bir küme sınırlı olasılıktan seçim yapar.
* Sürekli tekdüze dağılımlar bir aralıktan seçim yapar.
* Binom dağılımları bir dizi Bernoulli rasgele değişkeni modeller ve başarıların sayısını sayar.
* Poisson rastgele değişkenleri, nadir olayların oluşunu modeller.
* Gauss rastgele değişkenleri, çok sayıda bağımsız rastgele değişkenin toplam sonucunu modeller.
* Yukarıdaki tüm dağılımlar üstel aileye aittir.

## Alıştırmalar
1. İki bağımsız iki terimli rastgele değişkenin, $X, Y \sim \mathrm{Binomial}(16, 1/2)$ arasındaki $X-Y$ farkı olan rastgele bir değişkenin standart sapması nedir?
2. Poisson rastgele değişkenini, $X \sim \mathrm{Poisson}(\lambda)$, alırsak ve $(X - \lambda)/\sqrt{\lambda}$'i $\lambda \rightarrow \infty$ olarak kabul edersek, bunun yaklaşık olarak Gauss olduğunu gösterebiliriz. Bu neden anlamlıdır?
3. $n$ elemanda tanımlı iki tane ayrık tekdüze rasgele değişkenin toplamı için olasılık kütle fonksiyonu nedir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1099)
:end_tab:
