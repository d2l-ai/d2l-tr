# Maksimum (Azami) Olabilirlik
:label:`sec_maximum_likelihood`

Makine öğrenmesinde en sık karşılaşılan düşünce yöntemlerinden biri, maksimum olabilirlik bakış açısıdır. Bu, bilinmeyen parametrelere sahip olasılıklı bir modelle çalışırken, verileri en yüksek olasılığa sahip kılan parametrelerin en olası parametreler olduğu kavramıdır.

## Maksimum Olabilirlik İlkesi

Bunun, üzerinde düşünmeye yardımcı olabilecek Bayesçi bir yorumu vardır. $\boldsymbol{\theta}$ parametrelerine ve $X$ veri örneklerine sahip bir modelimiz olduğunu varsayalım. Somutluk açısından, $\boldsymbol{\theta}$'nın, bir bozuk paranın ters çevrildiğinde tura gelme olasılığını temsil eden tek bir değer ve $X$'in bağımsız bir bozuk para çevirme dizisi olduğunu hayal edebiliriz. Bu örneğe daha sonra derinlemesine bakacağız.

Modelimizin parametreleri için en olası değeri bulmak istiyorsak, bu, bunu bulmak istediğimiz anlamına gelir:

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

Bayes kuralı gereği yukarıdaki ifade ile aşağıdaki gibi yazılabilir:

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

Verileri oluşturmanın parametreden bağımsız bir olasılığı olan $P(X)$ ifadesi, $\boldsymbol{\theta}$'ya hiç bağlı değildir ve bu nedenle en iyi $\boldsymbol{\theta}$ seçeneği değiştirilmeden atılabilir. Benzer şekilde, şimdi hangi parametre kümesinin diğerlerinden daha iyi olduğuna dair önceden bir varsayımımız olmadığını farzedebiliriz, bu yüzden $P(\boldsymbol{\theta})$'nın da $\boldsymbol{\theta}$'ya bağlı olmadığını beyan edebiliriz! Bu, örneğin, yazı tura atma örneğimizde, tura gelme olasılığının önceden adil olup olmadığına dair herhangi bir inanca sahip olmadan $[0,1]$ arasında herhangi bir değer olabileceği durumlarda anlamlıdır  (genellikle *bilgisiz önsel* olarak anılır). Böylece, Bayes kuralı uygulamamızın, en iyi $\boldsymbol{\theta}$ seçimimizin $\boldsymbol{\theta}$ için maksimum olasılık tahmini olduğunu gösterdiğini görüyoruz:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Ortak bir terminoloji için, ($P(X \mid \boldsymbol{\theta})$) parametreleri verilen verilerin olasılığı *olabilirlik* olarak adlandırılır.

### Somut Bir Örnek

Bunun nasıl çalıştığını somut bir örnekle görelim. Tura atma olasılığını temsil eden tek bir $\theta$ parametremiz olduğunu varsayalım. O zaman yazı atma olasılığı $1-\theta$ olur ve bu nedenle, gözlemlenen verimiz $X$, $n_T$ tura ve $n_Y$ yazıdan oluşan bir diziyse, bağımsız olasılıkları çarparak görürüz ki:

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

13 tane madeni parayı atarsak ve $n_T = 9$ ve $n_Y = 4$ olan "TTTYTYYTTTTTY" dizisini alırsak, bunun şu olduğunu görürüz:

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Bu örnekle ilgili güzel bir şey, cevabın nasıl geleceğini bilmemizdir. Gerçekten de, sözlü olarak, "13 para attım ve 9 tura geldi, tura gelmesi olasılığı için en iyi tahminimiz nedir?", herkes doğru bir şekilde $9/13$ olarak tahmin edecektir. Bu maksimum olabilirlik yönteminin bize vereceği şey, bu sayıyı ilk ilkelerden çok daha karmaşık durumlara genelleyecek bir şekilde elde etmenin bir yoludur.

Örneğimiz için, $P(X \mid \theta)$ grafiği aşağıdaki gibidir:

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

Bunun maksimum değeri, beklentisi $9/13 \approx 0.7\ldots$'a yakın bir yerde. Bunun tam olarak orada olup olmadığını görmek için kalkülüse dönebiliriz. Maksimumda fonksiyonun gradyanının düz olduğuna dikkat edin. Böylece, türevin sıfır olduğu yerde $\theta$ değerlerini bularak ve en yüksek olasılığı veren yerde maksimum olabilirlik tahminini :eqref:`eq_max_like` bulabiliriz. Hesaplayalım:

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Bunun üç çözümü vardır: $0$, $1$ ve $9/13$. İlk ikisi açıkça minimumdur, dizimize $0$ olasılık atadıkları için maksimum değildirler. Nihai değer, dizimize sıfır olasılık *atamaz* ve bu nedenle, maksimum olasılık tahmini $\hat \theta = 9/13$ olmalıdır.

## Sayısal Optimizasyon (Eniyileme) ve Negatif Logaritmik-Olabilirliği

Önceki örnek güzel, ama ya milyarlarca parametremiz ve veri örneğimiz varsa?

İlk olarak, tüm veri örneklerinin bağımsız olduğunu varsayarsak, pratik olarak olabilirliliğin kendisini artık pek çok olasılığın bir çarpımı olarak değerlendiremeyeceğimize dikkat edin. Aslında, her olasılık $[0,1]$ arasındadır, diyelim ki tipik olarak yaklaşık $1/2$ değerindedir ve $(1/2)^{1000000000}$ çarpımı makine hassasiyetinin çok altındadır. Bununla doğrudan çalışamayız.

Ancak, logaritmanın çarpımları toplamlara dönüştürdüğünü hatırlayın, bu durumda

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

Bu sayı, $32$-bitlik kayan virgüllü sayı tek duyarlılığına bile mükemmel şekilde uyuyor. Bu nedenle, *log-olabilirliği* göz önünde bulundurmalıyız.

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

$x \mapsto \log(x)$ işlevi arttığından, olabilirliliği en üst düzeye çıkarmak, log-olabilirliliği en üst düzeye çıkarmakla aynı şeydir. Nitekim :numref:`sec_naive_bayes` içinde, naif Bayes sınıflandırıcısının belirli bir örneğiyle çalışırken bu mantığın uygulandığını göreceğiz.

Genellikle kaybı en aza indirmek istediğimiz kayıp işlevleriyle çalışırız. *Negatif logaritmik-olabilirlik* olan $-\log(P(X \mid \boldsymbol{\theta}))$'ı alarak maksimum olabilirliliği bir kaybın en aza indirilmesine çevirebiliriz.

Bunu örnekle görselleştirmek için, yazı tura atma problemini önceden düşünün ve kapalı form çözümünü bilmiyormuşuz gibi davranın. Bunu hesaplayabiliriz

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Bu, koda yazılabilir ve milyarlarca bozuk para atmak için bile serbestçe optimize edilebilir.

```{.python .input}
# Verilerimizi ayarlayın
n_H = 8675309
n_T = 25624

# Parametrelerimizi ilkle
theta = np.array(0.5)
theta.attach_grad()

# Gradyan inişi gerçekleştir
lr = 0.00000000001
for iter in range(10):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Çıktıyı kontrol et
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Verilerimizi ayarlayın
n_H = 8675309
n_T = 25624

# Parametrelerimizi ilkle
theta = torch.tensor(0.5, requires_grad=True)

# Gradyan inişi gerçekleştir
lr = 0.00000000001
for iter in range(10):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Çıktıyı kontrol et
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Verilerimizi ayarlayın
n_H = 8675309
n_T = 25624

# Parametrelerimizi ilkle
theta = tf.Variable(tf.constant(0.5))

# Gradyan inişi gerçekleştir
lr = 0.00000000001
for iter in range(10):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Çıktıyı kontrol et
theta, n_H / (n_H + n_T)
```

İnsanların negatif logaritma olasılıklarını kullanmayı sevmesinin tek nedeni sayısal kolaylık değildir. Tercih edilmesinin birkaç nedeni daha var.

Logaritmik-olabilirliliğini düşünmemizin ikinci nedeni, kalkülüs kurallarının basitleştirilmiş uygulamasıdır. Yukarıda tartışıldığı gibi, bağımsızlık varsayımları nedeniyle, makine öğrenmesinde karşılaştığımız çoğu olasılık, bireysel olasılıkların çarpımıdır.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Bu demektir ki, bir türevi hesaplamak için çarpım kuralını doğrudan uygularsak,

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Bu, $(n-1)$ toplama ile birlikte $n(n-1)$ çarpımı gerektirir, bu nedenle işlem zamanı girdilerin ikinci dereceden polinomuyla orantılıdır! Yeterli akıllılık terimleri gruplayarak bunu doğrusal zamana indirgeyecektir, ancak biraz düşünmeyi gerektirir. Negatif logaritmik-olabilirlikte ise, bunun yerine

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

o da böyle buna dönüşür

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Bu sadece $n$ bölme ve $n-1$ toplam gerektirir ve dolayısıyla girdilerle doğrusal zamanlıdır.

Negatif logaritmik-olabilirliği göz önünde bulundurmanın üçüncü ve son nedeni, bilgi teorisi ile olan ilişkidir ve bunu :numref:`sec_information_theory` içinde ayrıntılı olarak tartışacağız. Bu, rastgele bir değişkendeki bilginin veya rasgeleliğin derecesini ölçmek için bir yol sağlayan titiz bir matematiksel teoridir. Bu alanda çalışmanın temel konusu entropidir.

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

Bir kaynağın rasgeleliğini ölçer. Bunun ortalama $-\log$ olasılığından başka bir şey olmadığına dikkat edin ve bu nedenle, negatif logaritmik-olabilirliğimizi alıp veri örneklerinin sayısına bölersek, çapraz-entropi (cross-entropy) olarak bilinen göreceli bir entropi elde ederiz. Tek başına bu teorik yorum, model performansını ölçmenin bir yolu olarak, veri kümesi üzerinden ortalama negatif logaritmik-olabilirliliği rapor etmeyi motive etmek için yeterince zorlayıcı olacaktır.

## Sürekli Değişkenler için Maksimum Olabilirlik

Şimdiye kadar yaptığımız her şey, ayrık rastgele değişkenlerle çalıştığımızı varsayıyor; ancak ya sürekli olanlarla çalışmak istersek?

Kısaca özet, olasılığın tüm örneklerini olasılık yoğunluğu ile değiştirmemiz dışında hiçbir şeyin değişmemesidir. Yoğunlukları küçük harfli $p$ ile yazdığımızı hatırlarsak, bu, örneğin şimdi şunu söylediğimiz anlamına gelir:

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

Soru, "Bu neden geçerli?" haline gelir. Sonuçta, yoğunlukları tanıtmamızın nedeni, belirli sonuçların kendilerinin elde edilme olasılıklarının sıfır olmasıydı ve dolayısıyla herhangi bir parametre kümesi için verilerimizi üretme olasılığımızın sıfır olmaz mı?

Aslında soru budur ve neden yoğunluklara geçebileceğimizi anlamak, epsilonlara ne olduğunu izlemeye yönelik bir alıştırmadır.

Önce hedefimizi yeniden tanımlayalım. Sürekli rastgele değişkenler için artık tam olarak doğru değeri elde etme olasılığını hesaplamak istemediğimizi, bunun yerine $\epsilon$ aralığında eşleştirme yapmak istediğimizi varsayalım. Basit olması için, verilerimizin aynı şekilde dağıtılmış rastgele değişkenler $X_1, \ldots, X_N$'nin tekrarlanan gözlemleri,  $x_1, \ldots, x_N$, olduğunu varsayıyoruz. Daha önce gördüğümüz gibi, bu şu şekilde yazılabilir:

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Böylece, bunun negatif logaritmasını alırsak bunu elde ederiz:

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Bu ifadeyi incelersek, $\epsilon$'un olduğu tek yer $-N\log (\epsilon)$ toplamsal sabitidir. Bu, $\boldsymbol{\theta}$ parametrelerine hiç bağlı değildir, dolayısıyla en uygun $\boldsymbol{\theta}$ seçimi, $\epsilon$ seçimimize bağlı değildir! Dört veya dört yüz basamaklı da talep edersek, en iyi $\boldsymbol{\theta}$ seçimi aynı kalır, böylece epsilon'u serbestçe dışarıda bırakıp optimize etmek istediğimiz şey bu olur:

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Böylelikle, olasılıkları olasılık yoğunlukları ile değiştirerek, maksimum olabilirlik bakış açısının sürekli rastgele değişkenlerle, kesikli olanlar kadar kolay bir şekilde çalışabileceğini görüyoruz.

## Özet
* Maksimum olabilirlik ilkesi bize, belirli bir veri kümesi için en uygun modelin en yüksek olasılıkla verileri üreten model olduğunu söyler.
* Genellikle insanlar çeşitli nedenlerde dolayı negatif logaritmik-olabilirlik ile çalışırlar: Sayısal kararlılık, çarpımların toplamlara dönüştürülmesi (ve bunun sonucunda gradyan hesaplamalarının basitleştirilmesi) ve bilgi teorisine yönelik teorik bağlar.
* Ayrık ortamda motive etmek en basiti olsa da, veri noktalarına atanan olasılık yoğunluğunu en üst düzeye çıkararak sürekli ortamda da serbestçe genelleştirilebilir.

## Alıştırmalar
1. Rastgele bir değişkenin bir $\alpha$ değeri için $\frac{1}{\alpha}e^{-\alpha x}$ yoğunluğuna sahip olduğunu bildiğinizi varsayalım. Rastgele değişkenden $3$ sayısını tek gözlem olarak elde ediyorsunuz. $\alpha$ için maksimum olabilirlik tahmini nedir?
2. Ortalama değeri bilinmeyen ancak varyansı $1$ olan bir Gauss'tan alınmış $\{x_i\}_{i=1}^N$ örnekten oluşan bir veri kümeniz olduğunu varsayalım. Ortalama için maksimum olabilirlik tahmini nedir?


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1097)
:end_tab:
