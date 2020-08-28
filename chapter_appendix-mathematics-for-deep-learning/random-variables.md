# Rastgele Değişkenler
:label:`sec_random_variables`

:numref:`sec_prob`'de, bizim durumumuzda ya sonlu olası değerler kümesini ya da tamsayıları alan rastgele değişkenlere atıfta bulunan ayrık rastgele değişkenlerle nasıl çalışılacağının temellerini gördük. Bu bölümde, herhangi bir gerçel değeri alabilen rastgele değişkenler olan *sürekli rastgele değişkenler* teorisini geliştiriyoruz.

## Sürekli Rastgele Değişkenler

Sürekli rastgele değişkenler, ayrık rastgele değişkenlerden önemli ölçüde daha incelikli bir konudur. Yapılması gereken adil bir benzetme, buradaki teknik sıçramanın sayı listeleri toplama ve işlevlerin integralini alma arasındaki sıçramayla karşılaştırılabilir olmasıdır. Bu nedenle, teoriyi geliştirmek için biraz zaman ayırmamız gerekecek.

### Ayrık Değerden Sürekli Değere

Sürekli rastgele değişkenlerle çalışırken karşılaşılan ek teknik zorlukları anlamak için düşüncesel bir deney yapalım. Dart tahtasına bir dart fırlattığımızı ve tahtanın ortasından tam olarak $2 \text{cm}$ uzağa saplanma olasılığını bilmek istediğimizi varsayalım.

Başlangıç ​​olarak, tek basamaklı bir doğrulukla, yani $0 \text{cm}$, $1 \text{cm}$, $2 \text{cm}$ gibi bölmeler kullanarak ölçtüğümüzü hayal ediyoruz. Dart tahtasına diyelim ki $100$ dart atıyoruz ve eğer bunlardan $20$'si $2 \text{cm}$ bölmesine düşerse, attığımız dartların $\% 20$'sinin tahtada merkezden $2 \text{cm}$ uzağa saplandığı sonucuna varıyoruz.

Ancak daha yakından baktığımızda, bu sorumuzla örtüşmüyor! Tam eşitlik istiyorduk, oysa bu bölmeler diyelim ki $1,5 \text{cm}$ ile $2,5 \text{cm}$ arasındaki her şeyi tutuyor.

Kesintisiz, daha ileriye devam edelim. Daha da keskin bir şekilde ölçüyoruz, diyelim ki $1,9 \text{cm}$, $2,0 \text{cm}$, $2,1 \text{cm}$ ve şimdi, belki de $100$ darttan $3$'ünün $2.0 \text{cm}$ hattında tahtaya saplandığını görüyoruz. Böylece olasılığın $\% 3$ olduğu sonucuna vardık.

Ancak bu hiçbir şeyi çözmez! Sorunu bir basamak daha aşağıya ittik. Biraz soyutlayalım. İlk $k$ hanesinin $2,00000 \ldots$ ile eşleşme olasılığını bildiğimizi ve ilk $k + 1$ hanesi için eşleşme olasılığını bilmek istediğimizi düşünün. ${k + 1}.$ basamağının aslında $\{0, 1, 2, \ ldots, 9 \}$ kümesinden rastgele bir seçim olduğunu varsaymak oldukça mantıklıdır. En azından, mikrometre mertebesindeki değeri merkezden uzaklaşarak $7$ veya $3$'e karşılık gelmeye tercih etmeye zorlayacak fiziksel olarak anlamlı bir süreç düşünemiyoruz.

Bunun anlamı, özünde ihtiyaç duyduğumuz her ek doğruluk basamağının eşleştirme olasılığını $10$'luk bir faktörle azaltması gerektiğidir. Ya da başka bir deyişle, bunu beklerdik:

$$
P(\text{distance is}\; 2.00\ldots, \;\text{to}\; k \;\text{digits} ) \approx p\cdot10^{-k}.
$$

Değer $p$ esasen ilk birkaç basamakta olanları kodlar ve $10^{-k}$ gerisini halleder.

Ondalık sayıdan sonra konumu $k = 4$ basamağa kadar doğru bildiğimize dikkat edin. Bu, değerin $[(1.99995,2.00005]$ aralığında olduğunu bildiğimiz anlamına gelir, bu da $2.00005-1.99995 = 10^{-4}$ uzunluğunda bir aralıktır. Dolayısıyla, bu aralığın uzunluğunu $\epsilon$ olarak adlandırırsak, diyebiliriz ki

$$
P(\text{distance is in an}\; \epsilon\text{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

Bunu son bir adım daha ileri götürelim. Bunca zamandır $2$ noktasını düşünüyorduk, ama diğer noktaları asla düşünmedik. Orada temelde hiçbir şey farklı değildir, ancak $p$ değeri muhtemelen farklı olacaktır. En azından bir dart atıcısının, $20 \text{cm}$ yerine $2 \text{cm}$ gibi merkeze daha yakın bir noktayı vurma olasılığının daha yüksek olduğunu umuyoruz. Bu nedenle, $p$ değeri sabit değildir, bunun yerine $x$ noktasına bağlı olmalıdır. Bu da bize şunu beklememiz gerektiğini söylüyor:

$$P(\text{distance is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

Aslında, :eqref:`eq_pdf_deriv` tam olarak *olasılık yoğunluk fonksiyonunu* tanımlar. Bu, bir noktayı başka yakın bir noktaya göre vurma olasılığını kodlayan bir $p(x)$ fonksiyonudur. Böyle bir fonksiyonun neye benzeyebileceğini görselleştirelim.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

İşlev değerinin büyük olduğu konumlar, rastgele değeri bulma olasılığımızın daha yüksek olduğu bölgeleri gösterir. Düşük kısımlar, rastgele değeri bulamaya yatkın olmadığımız alanlardır.

### Olasılık Yoğunluk Fonksiyonları

Şimdi bunu daha ayrıntılı inceleyelim. Bir rastgele değişken $X$ için olasılık yoğunluk fonksiyonunun sezgisel olarak ne olduğunu daha önce görmüştük, yani yoğunluk fonksiyonu bir $p(x)$ fonksiyonudur.

$$P(X \; \text{is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

Peki bu $p(x)$'nin özellikleri için ne anlama geliyor?

Birincisi, olasılıklar asla negatif değildir, dolayısıyla $p(x)\ge 0$ olmasını bekleriz.

İkinci olarak, $\mathbb{R}$'yi $\epsilon$ genişliğinde sonsuz sayıda dilime böldüğümüzü varsayalım, diyelim ki $(\epsilon\cdot i, \epsilon \cdot (i+1)]$ gibi. Bunların her biri için, :eqref:`eq_pdf_def`'den biliyoruz, olasılık yaklaşık olarak

$$
P(X \; \text{is in an}\; \epsilon\text{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

bu yüzden hepsi üzerinden toplanabilmeli

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

Bu, :numref:`sec_integral_calculus`da tartışılan bir integral yaklaşımından başka bir şey değildir, dolayısıyla şunu söyleyebiliriz

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

$P(X\in\mathbb{R}) = 1$ olduğunu biliyoruz, çünkü rasgele değişkenin *herhangi bir* sayı alması gerektiğinden, herhangi bir yoğunluk için şu sonuca varabiliriz:

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

Aslında, bu konuyu daha ayrıntılı olarak incelersek, herhangi $a$ ve $b$ için şunu görürüz:

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

Bunu, daha önce olduğu gibi aynı ayrık yaklaşıklama yöntemlerini kullanarak kodda yaklaşıklaştırabiliriz. Bu durumda mavi bölgeye düşme olasılığını tahmin edebiliriz.

```{.python .input}
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

Bu iki özelliğin, olası olasılık yoğunluk fonksiyonlarının (veya yaygın olarak karşılaşılan kısaltma için *o.y.f. (p.d.f)*'ler) uzayını tastamam tanımladığı ortaya çıkar. Negatif olmayan fonksiyonlar $p(x) \ge 0$ için

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

Bu işlevi, rastgele değişkenimizin belirli bir aralıkta olma olasılığını elde etmek için integral alarak yorumluyoruz:

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

:numref:`sec_distributions`'da bir dizi yaygın dağılımı göreceğiz, ancak soyut oalrak çalışmaya devam edelim.

### Birikimli (Kümülatif) Dağılım Fonksiyonları

Önceki bölümde, o.y.f kavramını gördük. Uygulamada, bu sürekli rastgele değişkenleri tartışmak için yaygın olarak karşılaşılan bir yöntemdir, ancak önemli bir görünmez tuzak vardır: o.y.f.'nin değerlerinin kendileri olasılıklar değil, olasılıkları elde etmek için integralini almamız gereken bir fonksiyondur. $1/10$'dan daha uzun bir aralık için $10$'dan fazla olmadığı sürece, yoğunluğun $10$'dan büyük olmasının yanlış bir tarafı yoktur. Bu sezgiye aykırı olabilir, bu nedenle insanlar genellikle *birikimli dağılım işlevi* veya *bir olasılık olan* b.d.f. açısından düşünürler.

Özellikle :eqref:`eq_pdf_int_int` kullanarak b.d.f'yi tanımlarız. $p(x)$ yoğunluğuna sahip rastgele bir değişken $X$ için
 
$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

Birkaç özelliği inceleyelim.

* $F(x) \rightarrow 0$ iken $x\rightarrow -\infty$.
* $F(x) \rightarrow 1$ iken $x\rightarrow \infty$.
* $F(x)$ azalmaz ($y > x \implies F(y) \ge F(x)$).
* $X$ sürekli bir rasgele değişkense $F(x)$ süreklidir (sıçrama yoktur).

Dördüncü maddede, $X$ ayrık olsaydı bunun doğru olmayacağına dikkat edin, mesela $0$ ve $1$ değerlerini $1/2$ olasılıkla alalım. Bu durumda

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

Bu örnekte, bdf ile çalışmanın faydalarından birini, aynı çerçevede sürekli veya ayrık rastgele değişkenlerle veya ötesi ikisinin karışımlarıyla başa çıkma becerisini görüyoruz (Yazı tura atın: tura gerlirse zar atın, yazı gelirse bir dart atışının dart tahtasının merkezinden mesafesini ölçün).

### Ortalama

Rastgele değişkenler, $X$, ile uğraştığımızı varsayalım. Dağılımın kendisini yorumlamak zor olabilir. Bir rastgele değişkenin davranışını kısaca özetleyebilmek genellikle yararlıdır. Rastgele bir değişkenin davranışını özetlememize yardımcı olan sayılara *özet istatistikleri* denir. En sık karşılaşılanlar *ortalama*, *varyans (değişinti)* ve *standart sapmadır*.

*Ortalama*, rastgele bir değişkenin ortalama değerini kodlar. $p_i$ olasılıklarıyla $x_i$ değerlerini alan ayrık bir rastgele değişkenimiz, $X$, varsa, ortalama, ağırlıklı ortalama ile verilir: Rastgele değişken değerlerinin bu değerleri alma olasılığıyla çarpımlarının toplamıdır:

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

Ortalamayı yorumlamamız gereken anlam (dikkatli olarak), bize rastgele değişkenin nerede bulunma eğiliminde olduğunu söylemesidir.

Bu bölümde inceleyeceğimiz minimalist bir örnek olarak, $a-2$ değerini $p$, $a + 2$ değerini $p$ ve $a$ değerini $1-2p$ olasılıkla alan rastgele değişken olarak $X$'i alalım. :eqref:`eq_exp_def`'yi kullanarak, olası herhangi bir $a$ ve $p$ seçimi için ortalama hesaplayabiliriz:

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

Böylece ortalamanın $a$ olduğunu görüyoruz. Rastgele değişkenimizi ortaladığımız konum $a$ olduğundan, bu sezgimizle eşleşir.

Yararlı oldukları için birkaç özelliği özetleyelim.

* Herhangi bir rastgele değişken $X$ ve $a$ ve $b$ sayıları için, $\mu_{aX + b} = a\mu_X + b$ var.
* İki rastgele değişkenimiz varsa $X$ ve $Y$, $\mu_{X + Y} = \mu_X + \mu_Y$ olur.

Ortalamalar, rastgele bir değişkenin ortalama davranışını anlamak için yararlıdır, ancak ortalama, tam bir sezgisel anlayışa sahip olmak için bile yeterli değildir. Satış başına $10\$ \pm 1\$$ kar etmek, aynı ortalama değere sahip olmasına rağmen satış başına $10\$ \pm 15\$$ kar etmekten çok farklıdır. İkincisi çok daha büyük bir dalgalanma derecesine sahiptir ve bu nedenle çok daha büyük bir riski temsil eder. Bu nedenle, rastgele bir değişkenin davranışını anlamak için en az bir ölçüye daha ihtiyacımız olacak: Bir rasgele değişkenin ne kadar geniş dalgalandığına dair bir ölçü.

### Varyanslar

Bu bizi rastgele bir değişkenin *varyansını* düşünmeye götürür. Bu, rastgele bir değişkenin ortalamadan ne kadar saptığının nicel bir ölçüsüdür. $X - \mu_X $ifadesini düşünün. Bu, rastgele değişkenin ortalamasından sapmasıdır. Bu değer pozitif ya da negatif olabilir, bu yüzden onu pozitif yapmak için bir şeyler yapmalıyız ki böylece sapmanın büyüklüğünü ölçebilelim.

Denenecek makul bir şey, $\left|X-\mu_X\right|$'a bakmaktır ve bu gerçekten de *ortalama mutlak sapma* olarak adlandırılan faydalı bir miktara yol açar, ancak diğer matematik ve istatistik alanlarıyla olan bağlantılarından dolayı, insanlar genellikle farklı bir çözüm kullanır.

Özellikle $(X-\mu_X)^2$'ye bakarlar. Bu miktarın tipik boyutuna ortalamasını alarak bakarsak, varyansa ulaşırız:

$$\sigma_X^2 = \mathrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

:eqref:`eq_var_def`'deki son eşitlik ortadaki tanımı genişleterek ve beklentinin özelliklerini uygulayarak devam eder.

$X$'in $p$ olasılıkla $a-2$ değerini, $p$ olasılıkla $a + 2$ ve $1-2p$ olasılıkla $a$ değerini alan rastgele değişken olduğu örneğimize bakalım. Bu durumda $\mu_X = a$'dır, dolayısıyla hesaplamamız gereken tek şey $E\left[X^2\right]$'dir. Bu kolaylıkla yapılabilir:

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)p = a^2 + 8p.
$$

Böylece görürüz ki :eqref:`eq_var_def` tanımıyla varyansımız:

$$
\sigma_X^2 = \mathrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

Bu sonuç yine mantıklıdır. $p$ en büyük  $1/2$ olabilir ve bu da yazı tura ile $a-2$ veya $a + 2$ seçmeye karşılık gelir. Bunun $4$ olması, hem $a-2$ hem de $a + 2$'nin ortalamadan $2$ birim uzakta ve $2^2 = 4$ olduğu gerçeğine karşılık gelir. Spektrumun (izgenin) diğer ucunda, eğer $p = 0$ ise, bu rasgele değişken her zaman $0$ değerini alır ve bu nedenle hiçbir varyansı yoktur.

Aşağıda varyansın birkaç özelliğini listeleyeceğiz:

* Herhangi bir rastgele değişken için $X$, $\mathrm{Var}(X) \ ge 0$, ancak ve ancak $X$ bir sabitse $\mathrm {Var}(X) = 0 $'dır.
* Herhangi bir rastgele değişken $X$ ve $a$ ve $b$ sayıları için, $\mathrm{Var}(aX + b) = a^2 \mathrm{Var}(X)$'dır.
* İki *bağımsız* rastgele değişkenimiz varsa, $X$ ve $Y$, $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$'dir.

Bu değerleri yorumlarken biraz tutukluk olabilir. Özellikle, bu hesaplama yoluyla birimleri takip edersek ne olacağını hayal edelim. Web sayfasındaki bir ürüne atanan yıldız derecelendirmesiyle çalıştığımızı varsayalım. Daha sonra $a$, $a-2$ ve $a + 2$ değerlerinin tümü yıldız birimleriyle ölçülür. Benzer şekilde, ortalama, $\mu_X$, daha sonra yıldızlarla da ölçülür (ağırlıklı ortalama). Bununla birlikte, varyansa ulaşırsak, hemen bir sorunla karşılaşırız, bu da *yıldız kare* birimleri cinsinden $(X-\mu_X)^2$'ye bakmak istediğimizdendir. Bu, varyansın kendisinin orijinal ölçümlerle karşılaştırılamayacağı anlamına gelir. Bunu yorumlanabilir hale getirmek için orijinal birimlerimize dönmemiz gerekecek.

### Standart Sapmalar

Bu özet istatistik, karekök alınarak varyanstan her zaman çıkarılabilir! Böylece *standart sapmayı* tanımlıyoruz:

$$
\sigma_X = \sqrt{\mathrm{Var}(X)}.
$$

Örneğimizde bu, standart sapmanın $\sigma_X = 2\sqrt{2p}$ olduğu anlamına gelir. İnceleme örneğimiz için yıldız birimleriyle uğraşıyorsak, $\sigma_X$ yine yıldız birimindedir.

Varyans için sahip olduğumuz özellikler, standart sapma için yeniden ifade edilebilir.

* Herhangi bir rastgele değişken $X$ için , $\sigma_{X} \ge 0$'dır.
* Herhangi bir rastgele değişken $X$ ve $a$ ve $b$ sayıları için, $\sigma_{aX+b} = |a|\sigma_{X}$'dır.
* İki *bağımsız* rastgele değişkenimiz, $ X$ ve $Y$, varsa, $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$ olur.

Şu anda şunu sormak doğaldır, "Eğer standart sapma orijinal rasgele değişkenimizin birimlerindeyse, bu rasgele değişkenle ilgili olarak çizebileceğimiz bir şeyi temsil eder mi?" Cevap yankılanan bir evettir! Aslında ortalamanın bize rastgele değişkenimizin tipik konumunu söylediğine benzer şekilde, standart sapma o rastgele değişkenin tipik varyasyon aralığını verir. Bunu, Chebyshev eşitsizliği olarak bilinen şeyle sıkı hale getirebiliriz:

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

Veya sözlü olarak ifade etmek gerekirse, $\alpha = 10$ durumunda, herhangi bir rasgele değişkenden alınan örneklerin $\%99$'u, ortalamadan $10$ standart sapmalık aralığa düşer. Bu, standart özet istatistiklerimize anında bir anlam sağlar.

Bu ifadenin ne kadar ince olduğunu görmek için, $X$'in $p$ olasılıkla $a-2$, $p$ olasılıkla $a+2$ ve $1-2p$ olasılıkla $a$ değerini alan rastgele değişken olduğu, mevcut örneğimize tekrar bakalım. $. Ortalamanın $a$ ve standart sapmanın $2\sqrt{2p}$ olduğunu gördük. Bu demektir ki, Chebyshev'in eşitsizliğini, :eqref:`eq_chebyshev`, $\alpha = 2$ ile alırsak, ifadenin şöyle olduğunu görürüz:

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

Bu, zamanın $\%75$'inde, bu rastgele değişkenin herhangi bir $p$ değeri için bu aralığa denk geleceği anlamına gelir. Şimdi, $p \rightarrow 0$ olarak, bu aralığın da tek bir $a$ noktasına yakınsadığına dikkat edin. Ancak rasgele değişkenimizin yalnızca $a-2, a$ ve $a + 2$ değerlerini aldığını biliyoruz, bu nedenle sonunda $a-2$ ve $a + 2$ aralığının dışında kalacağından emin olabiliriz! Soru, bunun hangi $p$'de olduğu. Bu yüzden bunu çözmek istiyoruz: Hangi $p$'de $a + 4 \sqrt{2p} = a + 2$ yapar, ki bu $p = 1/8$ olduğunda çözülür, bu da dağılımdan $1/4$'ten fazla örneklemin aralığın dışında kalmayacağı iddiamızı ihlal etmeden gerçekleştirebilecek *tam olarak* ilk $p$'dir ($1/8$ sola ve $1/8$ sağa).

Bunu görselleştirelim. Üç değeri alma olasılığını olasılıkla orantılı yüksekliği olan üç dikey çubuk olarak göstereceğiz. Aralık ortada yatay bir çizgi olarak çizilecektir. İlk grafik, aralığın güvenli bir şekilde tüm noktaları içerdiği $p > 1/8$ için ne olduğunu gösterir.

```{.python .input}
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

İkinci görsel, $p = 1/8$'de aralığın tam olarak iki noktaya dokunduğunu gösterir. Bu, eşitsizliğin doğru tutulurken daha küçük bir aralık alınamayacağı için eşitsizliğin *keskin* olduğunu gösterir.

```{.python .input}
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

Üçüncüsü, $p < 1/8$ için aralığın yalnızca merkezi içerdiğini gösterir. Bu, eşitsizliği geçersiz kılmaz, çünkü yalnızca olasılığın $1/4$'ten fazlasının aralığın dışında kalmamasını sağlamamız gerekiyor, yani $p < 1/8$ olduğunda, iki nokta $a-2$ ve $a + 2$ yok sayılabilir edilebilir.

```{.python .input}
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

### Means and Variances in the Continuum

This has all been in terms of discrete random variables, but the case of continuous random variables is similar.  To intuitively understand how this works, imagine that we split the real number line into intervals of length $\epsilon$ given by $(\epsilon i, \epsilon (i+1)]$.  Once we do this, our continuous random variable has been made discrete and we can use :eqref:`eq_exp_def` say that

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

where $p_X$ is the density of $X$.  This is an approximation to the integral of $xp_X(x)$, so we can conclude that

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

Similarly, using :eqref:`eq_var_def` the variance can be written as

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

Everything stated above about the mean, the variance, and the standard deviation still applies in this case.  For instance, if we consider the random variable with density

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \text{otherwise}.
\end{cases}
$$

we can compute

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

and

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

As a warning, let us examine one more example, known as the *Cauchy distribution*.  This is the distribution with p.d.f. given by

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

This function looks innocent, and indeed consulting a table of integrals will show it has area one under it, and thus it defines a continuous random variable.

To see what goes astray, let us try to compute the variance of this.  This would involve using :eqref:`eq_var_def` computing

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

The function on the inside looks like this:

```{.python .input}
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

This function clearly has infinite area under it since it is essentially the constant one with a small dip near zero, and indeed we could show that

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

This means it does not have a well-defined finite variance.

However, looking deeper shows an even more disturbing result.  Let us try to compute the mean using :eqref:`eq_exp_def`.  Using the change of variables formula, we see

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

The integral inside is the definition of the logarithm, so this is in essence $\log(\infty) = \infty$, so there is no well-defined average value either!

Machine learning scientists define their models so that we most often do not need to deal with these issues, and will in the vast majority of cases deal with random variables with well-defined means and variances.  However, every so often random variables with *heavy tails* (that is those random variables where the probabilities of getting large values are large enough to make things like the mean or variance undefined) are helpful in modeling physical systems, thus it is worth knowing that they exist.

### Joint Density Functions

The above work all assumes we are working with a single real valued random variable.  But what if we are dealing with two or more potentially highly correlated random variables?  This circumstance is the norm in machine learning: imagine random variables like $R_{i, j}$ which encode the red value of the pixel at the $(i, j)$ coordinate in an image, or $P_t$ which is a random variable given by a stock price at time $t$.  Nearby pixels tend to have similar color, and nearby times tend to have similar prices.  We cannot treat them as separate random variables, and expect to create a successful model (we will see in :numref:`sec_naive_bayes` a model that under-performs due to such an assumption).  We need to develop the mathematical language to handle these correlated continuous random variables.

Thankfully, with the multiple integrals in :numref:`sec_integral_calculus` we can develop such a language.  Suppose that we have, for simplicity, two random variables $X, Y$ which can be correlated.  Then, similar to the case of a single variable, we can ask the question:

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ).
$$

Similar reasoning to the single variable case shows that this should be approximately

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

for some function $p(x, y)$.  This is referred to as the joint density of $X$ and $Y$.  Similar properties are true for this as we saw in the single variable case. Namely:

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

In this way, we can deal with multiple, potentially correlated random variables.  If we wish to work with more than two random variables, we can extend the multivariate density to as many coordinates as desired by considering $p(\mathbf{x}) = p(x_1, \ldots, x_n)$.  The same properties of being non-negative, and having total integral of one still hold.

### Marginal Distributions
When dealing with multiple variables, we oftentimes want to be able to ignore the relationships and ask, "how is this one variable distributed?"  Such a distribution is called a *marginal distribution*.

To be concrete, let us suppose that we have two random variables $X, Y$ with joint density given by $p _ {X, Y}(x, y)$.  We will be using the subscript to indicate what random variables the density is for.  The question of finding the marginal distribution is taking this function, and using it to find $p _ X(x)$.

As with most things, it is best to return to the intuitive picture to figure out what should be true.  Recall that the density is the function $p _ X$ so that

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

There is no mention of $Y$, but if all we are given is $p _{X, Y}$, we need to include $Y$ somehow. We can first observe that this is the same as

$$
P(X \in [x, x+\epsilon] \text{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

Our density does not directly tell us about what happens in this case, we need to split into small intervals in $y$ as well, so we can write this as

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \text{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![By summing along the columns of our array of probabilities, we are able to obtain the marginal distribution for just the random variable represented along the $x$-axis.](../img/Marginal.svg)
:label:`fig_marginal`

This tells us to add up the value of the density along a series of squares in a line as is shown in :numref:`fig_marginal`.  Indeed, after canceling one factor of epsilon from both sides, and recognizing the sum on the right is the integral over $y$, we can conclude that

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

Thus we see

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

This tells us that to get a marginal distribution, we integrate over the variables we do not care about.  This process is often referred to as *integrating out* or *marginalized out* the unneeded variables.

### Covariance

When dealing with multiple random variables, there is one additional summary statistic which is helpful to know: the *covariance*.  This measures the degree that two random variable fluctuate together.

Suppose that we have two random variables $X$ and $Y$, to begin with, let us suppose they are discrete, taking on values $(x_i, y_j)$ with probability $p_{ij}$.  In this case, the covariance is defined as

$$\sigma_{XY} = \mathrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

To think about this intuitively: consider the following pair of random variables.  Suppose that $X$ takes the values $1$ and $3$, and $Y$ takes the values $-1$ and $3$.  Suppose that we have the following probabilities

$$
\begin{aligned}
P(X = 1 \; \text{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \text{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

where $p$ is a parameter in $[0,1]$ we get to pick.  Notice that if $p=1$ then they are both always their minimum or maximum values simultaneously, and if $p=0$ they are guaranteed to take their flipped values simultaneously (one is large when the other is small and vice versa).  If $p=1/2$, then the four possibilities are all equally likely, and neither should be related.  Let us compute the covariance.  First, note $\mu_X = 2$ and $\mu_Y = 1$, so we may compute using :eqref:`eq_cov_def`:

$$
\begin{aligned}
\mathrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

When $p=1$ (the case where the are both maximally positive or negative at the same time) has a covariance of $2$. When $p=0$ (the case where they are flipped) the covariance is $-2$.  Finally, when $p=1/2$ (the case where they are unrelated), the covariance is $0$.  Thus we see that the covariance measures how these two random variables are related.

A quick note on the covariance is that it only measures these linear relationships.  More complex relationships like $X = Y^2$ where $Y$ is randomly chosen from $\{-2, -1, 0, 1, 2\}$ with equal probability can be missed.  Indeed a quick computation shows that these random variables have covariance zero, despite one being a deterministic function of the other.

For continuous random variables, much the same story holds.  At this point, we are pretty comfortable with doing the transition between discrete and continuous, so we will provide the continuous analogue of :eqref:`eq_cov_def` without any derivation.

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

For visualization, let us take a look at a collection of random variables with tunable covariance.

```{.python .input}
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

Let us see some properties of covariances:

* For any random variable $X$, $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$.
* For any random variables $X, Y$ and numbers $a$ and $b$, $\mathrm{Cov}(aX+b, Y) = \mathrm{Cov}(X, aY+b) = a\mathrm{Cov}(X, Y)$.
* If $X$ and $Y$ are independent then $\mathrm{Cov}(X, Y) = 0$.

In addition, we can use the covariance to expand a relationship we saw before.  Recall that is $X$ and $Y$ are two independent random variables then

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y).
$$

With knowledge of covariances, we can expand this relationship.  Indeed, some algebra can show that in general,

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y).
$$

This allows us to generalize the variance summation rule for correlated random variables.

### Correlation

As we did in the case of means and variances, let us now consider units.  If $X$ is measured in one unit (say inches), and $Y$ is measured in another (say dollars), the covariance is measured in the product of these two units $\text{inches} \times \text{dollars}$.  These units can be hard to interpret.  What we will often want in this case is a unit-less measurement of relatedness.  Indeed, often we do not care about exact quantitative correlation, but rather ask if the correlation is in the same direction, and how strong the relationship is.

To see what makes sense, let us perform a thought experiment.  Suppose that we convert our random variables in inches and dollars to be in inches and cents.  In this case the random variable $Y$ is multiplied by $100$.  If we work through the definition, this means that $\mathrm{Cov}(X, Y)$ will be multiplied by $100$.  Thus we see that in this case a change of units change the covariance by a factor of $100$.  Thus, to find our unit-invariant measure of correlation, we will need to divide by something else that also gets scaled by $100$.  Indeed we have a clear candidate, the standard deviation!  Indeed if we define the *correlation coefficient* to be

$$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

we see that this is a unit-less value.  A little mathematics can show that this number is between $-1$ and $1$ with $1$ meaning maximally positively correlated, whereas $-1$ means maximally negatively correlated.

Returning to our explicit discrete example above, we can see that $\sigma_X = 1$ and $\sigma_Y = 2$, so we can compute the correlation between the two random variables using :eqref:`eq_cor_def` to see that

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

This now ranges between $-1$ and $1$ with the expected behavior of $1$ meaning most correlated, and $-1$ meaning minimally correlated.

As another example, consider $X$ as any random variable, and $Y=aX+b$ as any linear deterministic function of $X$.  Then, one can compute that

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX+b) = a\mathrm{Cov}(X, X) = a\mathrm{Var}(X),$$

and thus by :eqref:`eq_cor_def` that

$$
\rho(X, Y) = \frac{a\mathrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \mathrm{sign}(a).
$$

Thus we see that the correlation is $+1$ for any $a > 0$, and $-1$ for any $a < 0$ illustrating that correlation measures the degree and directionality the two random variables are related, not the scale that the variation takes.

Let us again plot a collection of random variables with tunable correlation.

```{.python .input}
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

Let us list a few properties of the correlation below.

* For any random variable $X$, $\rho(X, X) = 1$.
* For any random variables $X, Y$ and numbers $a$ and $b$, $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$.
* If $X$ and $Y$ are independent with non-zero variance then $\rho(X, Y) = 0$.

As a final note, you may feel like some of these formulae are familiar.  Indeed, if we expand everything out assuming that $\mu_X = \mu_Y = 0$, we see that this is

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

This looks like a sum of a product of terms divided by the square root of sums of terms.  This is exactly the formula for the cosine of the angle between two vectors $\mathbf{v}, \mathbf{w}$ with the different coordinates weighted by $p_{ij}$:

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

Indeed if we think of norms as being related to standard deviations, and correlations as being cosines of angles, much of the intuition we have from geometry can be applied to thinking about random variables.

## Summary
* Continuous random variables are random variables that can take on a continuum of values.  They have some technical difficulties that make them more challenging to work with compared to discrete random variables.
* The probability density function allows us to work with continuous random variables by giving a function where the area under the curve on some interval gives the probability of finding a sample point in that interval.
* The cumulative distribution function is the probability of observing the random variable to be less than a given threshold.  It can provide a useful alternate viewpoint which unifies discrete and continuous variables.
* The mean is the average value of a random variable.
* The variance is the expected square of the difference between the random variable and its mean.
* The standard deviation is the square root of the variance.  It can be thought of as measuring the range of values the random variable may take.
* Chebyshev's inequality allows us to make this intuition rigorous by giving an explicit interval that contains the random variable most of the time.
* Joint densities allow us to work with correlated random variables.  We may marginalize joint densities by integrating over unwanted random variables to get the distribution of the desired random variable.
* The covariance and correlation coefficient provide a way to measure any linear relationship between two correlated random variables.

## Exercises
1. Suppose that we have the random variable with density given by $p(x) = \frac{1}{x^2}$ for $x \ge 1$ and $p(x) = 0$ otherwise.  What is $P(X > 2)$?
2. The Laplace distribution is a random variable whose density is given by $p(x = \frac{1}{2}e^{-|x|}$.  What is the mean and the standard deviation of this function?  As a hint, $\int_0^\infty xe^{-x} \; dx = 1$ and $\int_0^\infty x^2e^{-x} \; dx = 2$.
3. I walk up to you on the street and say "I have a random variable with mean $1$, standard deviation $2$, and I observed $25\%$ of my samples taking a value larger than $9$."  Do you believe me?  Why or why not?
4. Suppose that you have two random variables $X, Y$, with joint density given by $p_{XY}(x, y) = 4xy$ for $x, y \in [0,1]$ and $p_{XY}(x, y) = 0$ otherwise.  What is the covariance of $X$ and $Y$?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:
