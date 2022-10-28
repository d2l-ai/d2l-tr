# İstatistik
:label:`sec_statistics`

Kuşkusuz, en iyi derin öğrenme uygulayıcılarından biri olmak için son teknoloji ürünü ve yüksek doğrulukta modelleri eğitme yeteneği çok önemlidir. Bununla birlikte, iyileştirmelerin ne zaman önemli olduğu veya yalnızca eğitim sürecindeki rastgele dalgalanmaların sonucu olduğu genellikle belirsizdir. Tahmini değerlerdeki belirsizliği tartışabilmek için biraz istatistik öğrenmemiz gerekir.

*İstatistiğin* en eski referansı, şifrelenmiş mesajları deşifre etmek için istatistiklerin ve sıklık analizinin nasıl kullanılacağına dair ayrıntılı bir açıklama veren $9.$ yüzyıldaki Arap bilim adamı Al-Kindi'ye kadar uzanabilir. 800 yıl sonra, modern istatistik, araştırmacıların demografik ve ekonomik veri toplama ve analizine odaklandığı 1700'lerde Almanya'da ortaya çıktı. Günümüzde istatistik, verilerin toplanması, işlenmesi, analizi, yorumlanması ve görselleştirilmesi ile ilgili bilim konusudur. Dahası, temel istatistik teorisi akademi, endüstri ve hükümet içindeki araştırmalarda yaygın olarak kullanılmaktadır.

Daha özel olarak, istatistik *tanımlayıcı istatistik* ve *istatistiksel çıkarım* diye bölünebilir. İlki, *örneklem* olarak adlandırılan gözlemlenen verilerden bir koleksiyonunun özniteliklerini özetlemeye ve göstermeye odaklanır. Örneklem bir *popülasyondan* alınmıştır, benzer bireyler, öğeler veya deneysel ilgi alanlarımıza ait olayların toplam kümesini belirtir. Tanımlayıcı istatistiğin aksine *istatistiksel çıkarım*, örneklem dağılımının popülasyon dağılımını bir dereceye kadar kopyalayabileceği varsayımlarına dayanarak, bir popülasyonun özelliklerini verilen *örneklemlerden* çıkarsar.

Merak edebilirsiniz: "Makine öğrenmesi ile istatistik arasındaki temel fark nedir?" Temel olarak, istatistik çıkarım sorununa odaklanır. Bu tür problemler, nedensel çıkarım gibi değişkenler arasındaki ilişkiyi modellemeyi ve A/B testi gibi model parametrelerinin istatistiksel olarak anlamlılığını test etmeyi içerir. Buna karşılık, makine öğrenmesi, her bir parametrenin işlevselliğini açıkça programlamadan ve anlamadan doğru tahminler yapmaya vurgu yapar.

Bu bölümde, üç tür istatistik çıkarım yöntemini tanıtacağız: Tahmin edicileri değerlendirme ve karşılaştırma, hipotez (denence) testleri yürütme ve güven aralıkları oluşturma. Bu yöntemler, belirli bir popülasyonun özelliklerini, yani gerçek $\theta$ parametresi gibi, anlamamıza yardımcı olabilir. Kısacası, belirli bir popülasyonun gerçek parametresinin, $\theta$, skaler bir değer olduğunu varsayıyoruz. $\theta$'nin bir vektör veya tensör olduğu durumu genişletmek basittir, bu nedenle tartışmamızda onu es geçiyoruz.

## Tahmincileri Değerlendirme ve Karşılaştırma

İstatistikte, bir *tahminci*, gerçek $\theta$ parametresini tahmin etmek için kullanılan belirli örneklemlerin bir fonksiyonudur. {$x_1, x_2, \ldots, x_n$} örneklerini gözlemledikten sonra $\theta$ tahmini için $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ yazacağız.

Tahmincilerin basit örneklerini daha önce şu bölümde görmüştük :numref:`sec_maximum_likelihood`. Bir Bernoulli rastgele değişkeninden birkaç örneğiniz varsa, rastgele değişkenin olma olasılığı için maksimum olabilirlik tahmini, gözlemlenenlerin sayısını sayarak ve toplam örnek sayısına bölerek elde edilebilir. Benzer şekilde, bir alıştırma sizden bir miktar örnek verilen bir Gauss'un ortalamasının maksimum olabilirlik tahmininin tüm örneklerin ortalama değeriyle verildiğini göstermenizi istiyor. Bu tahminciler neredeyse hiçbir zaman parametrenin gerçek değerini vermezler, ancak ideal olarak çok sayıda örnek için tahmin yakın olacaktır.

Örnek olarak, ortalama sıfır ve varyans bir olan bir Gauss rasgele değişkeninin gerçek yoğunluğunu, bu Gauss'tan bir dizi örnek ile aşağıda gösteriyoruz. Her noktanın $y$ koordinatı görünür ve orijinal yoğunluk ile olan ilişki daha net fark edilecek şekilde oluşturduk.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Örnek veri noktaları ve y koordinatı oluştur
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Gerçek yoğunluğu hesapla
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Sonuçları çiz
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Örnek veri noktaları ve y koordinatı oluşturun
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Gerçek yoğunluğu hesapla
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Sonuçları çiz
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Örnek veri noktaları ve y koordinatı oluşturun
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Gerçek yoğunluğu hesapla
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Sonuçları çiz
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

$\hat{\theta}_n $ parametresinin bir tahmincisini hesaplamanın birçok yolu olabilir. Bu bölümde, tahmincileri değerlendirmek ve karşılaştırmak için üç genel yöntem sunuyoruz: Ortalama hata karesi, standart sapma ve istatistiksel yanlılık.

### Ortalama Hata Karesi

Tahmin edicileri değerlendirmek için kullanılan en basit ölçüt, bir tahmincinin *ortalama hata karesi (MSE)* (veya $l_2$ kaybı) olarak tanımlanabilir.

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

Bu, gerçek değerden ortalama kare sapmayı ölçümlemizi sağlar. MSE her zaman negatif değildir. Eğer :numref:`sec_linear_regression` içinde okuduysanız, bunu en sık kullanılan bağlanım (regresyon) kaybı işlevi olarak tanıyacaksınız. Bir tahminciyi değerlendirmek için bir ölçü olarak, değeri sıfıra ne kadar yakınsa, tahminci gerçek $\theta$ parametresine o kadar yakın olur.

### İstatistiksel Yanlılık

MSE doğal bir ölçü sağlar, ancak onu büyük yapabilecek birden fazla farklı vakayı kolayca hayal edebiliriz. İki temel önemli olay veri kümesindeki rastgelelik nedeniyle tahmincideki dalgalanma ve tahmin prosedürüne bağlı olarak tahmincideki sistematik hatadır.

Öncelikle sistematik hatayı ölçelim. Bir $\hat{\theta}_n $ tahmincisi için *istatistiksel yanlılığın* matematiksel gösterimi şu şekilde tanımlanabilir:

$$\mathrm{yanlılık}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\mathrm{yanlılık}(\hat{\theta}_n) = 0$ olduğunda, $\hat{\theta}_n$ tahmin edicisinin beklentisinin parametrenin gerçek değerine eşit olduğuna dikkat edin. Bu durumda, $\hat{\theta}_n$'nin yansız bir tahminci olduğunu söylüyoruz. Genel olarak, yansız bir tahminci, yanlı bir tahminciden daha iyidir çünkü beklenen değeri gerçek parametre ile aynıdır.

Bununla birlikte, yanlı tahmin edicilerin pratikte sıklıkla kullanıldığının farkında olunması gerekir. Yansız tahmin edicilerin başka varsayımlar olmaksızın var olmadığı veya hesaplamanın zor olduğu durumlar vardır. Bu, bir tahmincide önemli bir kusur gibi görünebilir, ancak pratikte karşılaşılan tahmin edicilerin çoğu, mevcut örneklerin sayısı sonsuza giderken sapmanın sıfır olma eğiliminde olması açısından en azından asimptotik (kavuşma doğrusu) olarak tarafsızdır: $\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$.

### Varyans ve Standart Sapma

İkinci olarak tahmincideki rastgeleliği ölçelim. Eğer :numref:`sec_random_variables` bölümünü anımsarsak, *standart sapma* (veya *standart hata*), varyansın kare kökü olarak tanımlanır. Bir tahmincinin dalgalanma derecesini, o tahmincinin standart sapmasını veya varyansını ölçerek ölçebiliriz.

$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

Şunları karşılaştırmak önemlidir :eqref:`eq_var_est` ile :eqref:`eq_mse_est`. Bu denklemde gerçek popülasyon değeri $\theta$ ile değil, bunun yerine beklenen örneklem ortalaması $E(\hat{\theta}_n)$ ile karşılaştırıyoruz. Bu nedenle, tahmincinin gerçek değerden ne kadar uzakta olduğunu ölçmüyoruz, bunun yerine tahmincinin dalgalanmasını ölçüyoruz.

### Yanlılık-Varyans Ödünleşmesi

Bu iki ana bileşenin ortalama hata karesine (MSE) katkıda bulunduğu sezgisel olarak açıktır. Biraz şok edici olan şey, bunun aslında ortalama hata karesinin bu iki ve ek üçüncü bir parçaya *ayrıştırılması* olduğunu gösterebilmemizdir. Yani, ortalama hata karesini yanlılığın (ek girdi) karesinin, varyansın ve indirgenemeyen hatanın toplamı olarak yazabiliriz.

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \mathrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \mathrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (\mathrm{yanlılık} [\hat{\theta}_n])^2 + \mathrm{Var} (\hat{\theta}_n) + \mathrm{Var} [\theta].\\
\end{aligned}
$$

Yukarıdaki formülü *yanlılık-varyans ödünleşmesi* olarak adlandırıyoruz. Ortalama hata karesi kesin olarak üç hata kaynağına bölünebilir: Yüksek yanlılıktan, yüksek varyanstan ve indirgenemez hatadan kaynaklı hata. Yanlılık hatası genellikle basit bir modelde (doğrusal bağlanım modeli gibi) görülür, çünkü öznitelikler ve çıktılar arasındaki yüksek boyutsal ilişkileri çıkaramaz. Bir model yüksek yanlılık hatasından muzdaripse, (:numref:`sec_model_selection`) bölümünde açıklandığı gibi genellikle *eksik öğrenme* veya *esneklik* eksikliği olduğunu söylüyoruz. Yüksek varyans, genellikle eğitim verilerine öğrenen çok karmaşık bir modelden kaynaklanır. Sonuç olarak, *aşırı öğrenen* bir model, verilerdeki küçük dalgalanmalara duyarlıdır. Bir modelin varyansı yüksekse, genellikle (:numref:`sec_model_selection`) içinde tanıtıldığı gibi *aşırı öğrenme* ve *genelleme* yoksunluğu olduğunu söyleriz. İndirgenemez hata, $\theta$'nın kendisindeki gürültünün sonucudur.

### Kodda Tahmincileri Değerlendirme

Bir tahmincinin standart sapması, bir tensör `a` için basitçe `a.std()` çağırarak uygulandığından, onu atlayacağız ancak istatistiksel yanlılık ve ortalama hata karesini uygulayacağız.

```{.python .input}
# İstatistiksel yanlılık
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Ortalama kare hatası
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# İstatistiksel yanlılık
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Ortalama kare hatası
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# İstatistiksel yanlılık
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Ortalama kare hatası
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

Yanlılık-varyans ödünleşmesinin denklemini görsellemek için, $\mathcal{N}(\theta, \sigma^2)$ normal dağılımını $10.000$ örnekle canlandıralım. Burada $\theta = 1$ ve $\sigma = 4$ olarak kullanıyoruz. Tahminci verilen örneklerin bir fonksiyonu olduğu için, burada örneklerin ortalamasını bu normal dağılımdaki, $\mathcal{N}(\theta, \sigma^2)$, gerçek $\theta$ için bir tahminci olarak kullanıyoruz.

```{.python .input}
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

Tahmincimizin yanlılık karesi ve varyansının toplamını hesaplayarak ödünleşme denklemini doğrulayalım. İlk önce, tahmincimizin MSE'sini hesaplayın.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

Ardından, aşağıdaki gibi $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{yanlılık} (\hat{\theta}_n)]^2$'yi hesaplıyoruz. Gördüğünüz gibi, iki değer, sayısal kesinliğe uyuyor.

```{.python .input}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## Hipotez (Denence) Testleri Yürütme

İstatistiksel çıkarımda en sık karşılaşılan konu hipotez testidir. Hipotez testi 20. yüzyılın başlarında popüler hale gelirken, ilk kullanım 1700'lerde John Arbuthnot'a kadar takip edilebilir. John, Londra'da 80 yıllık doğum kayıtlarını takip etti ve her yıl kadından daha fazla erkeğin doğduğu sonucuna vardı. Bunu takiben, modern anlamlılık testi, $p$-değerini ve Pearson'ın ki-kare testini icat eden Karl Pearson, Student (öğrenci) t dağılımının babası William Gosset ve sıfır hipotezini ve anlamlılık testini başlatan Ronald Fisher'ın zeka mirasıdır.

*Hipotez testi*, bir popülasyon hakkındaki varsayılan ifadeye karşı bazı kanıtları değerlendirmenin bir yoludur. Varsayılan ifadeyi, gözlemlenen verileri kullanarak reddetmeye çalıştığımız, *sıfır hipotezi*, $H_0$, olarak adlandırıyoruz. Burada, istatistiksel anlamlılık testi için başlangıç ​​noktası olarak $H_0$'ı kullanıyoruz. *Alternatif hipotez* $H_A$ (veya $H_1$), sıfır hipotezine karşıt bir ifadedir. Bir sıfır hipotez, genellikle değişkenler arasında bir ilişki olduğunu varsayan açıklayıcı bir biçimde ifade edilir. İçeriğini olabildiğince açık bir şekilde yansıtmalı ve istatistik teorisi ile test edilebilir olmalıdır.

Kimyager olduğunuzu hayal edin. Laboratuvarda binlerce saat geçirdikten sonra, kişinin matematiği anlama yeteneğini önemli ölçüde arttırabilecek yeni bir ilaç geliştiriyorsunuz. Sihirli gücünü göstermek için onu test etmeniz gerekir. Doğal olarak, ilacı almak ve matematiği daha iyi öğrenmelerine yardımcı olup olmayacağını görmek için bazı gönüllülere ihtiyacınız olabilir. Nasıl başlayacaksınız?

İlk olarak, dikkatle rastgele seçilmiş iki grup gönüllüye ihtiyacınız olacak, böylece bazı ölçütlerle ölçülen matematik anlama yetenekleri arasında hiçbir fark olmayacak. Bu iki grup genellikle test grubu ve kontrol grubu olarak adlandırılır. *Test grubu* (veya *tedavi grubu*) ilacı deneyimleyecek bir grup kişidir, *kontrol grubu* ise bir kıyaslama olarak bir kenara bırakılan kullanıcı grubunu temsil eder, yani, ilaç almak dışında aynı ortam şartlarına sahipler. Bu şekilde, bağımsız değişkenin tedavideki etkisi dışında tüm değişkenlerin etkisi en aza indirilir.

İkincisi, ilacı bir süre aldıktan sonra, iki grubun matematik anlayışını, yeni bir matematik formülü öğrendikten sonra gönüllülerin aynı matematik testlerini yapmasına izin vermek gibi aynı ölçütlerle ölçmeniz gerekecektir. Ardından, performanslarını toplayabilir ve sonuçları karşılaştırabilirsiniz. Bu durumda, sıfır hipotezimiz, muhtemelen iki grup arasında hiçbir fark olmadığı ve alternatifimiz olduğu şeklinde olacaktır.

Bu hala tam olarak resmi (nizamlara uygun) değil. Dikkatlice düşünmeniz gereken birçok detay var. Örneğin, matematik anlama yeteneklerini test etmek için uygun ölçütler nelerdir? İlacınızın etkinliğini iddia edebileceğinizden emin olabilmeniz için testinizde kaç gönüllü var? Testi ne kadar süreyle koşturmalısınız? İki grup arasında bir fark olup olmadığına nasıl karar veriyorsunuz? Yalnızca ortalama performansla mı ilgileniyorsunuz, yoksa puanların değişim aralığını da mı önemsiyorsunuz? Ve bunun gibi.

Bu şekilde, hipotez testi, deneysel tasarım ve gözlemlenen sonuçlarda kesinlik hakkında akıl yürütme için bir çerçeve sağlar. Şimdi sıfır hipotezinin gerçek olma ihtimalinin çok düşük olduğunu gösterebilirsek, onu güvenle reddedebiliriz.

Hipotez testiyle nasıl çalışılacağına dair hikayeyi tamamlamak için, şimdi bazı ek terminolojiyle tanışmamız ve yukarıdaki bazı kavramlarımızı kurallara uygun halde işlememiz gerekiyor.

### İstatistiksel Anlamlılık

*İstatistiksel anlamlılık*, sıfır hipotezin, $H_0$, reddedilmemesi gerektiğinde yanlışlıkla reddedilme olasılığını ölçer, yani,

$$ \text{istatistiksel anlamlılık }= 1 - \alpha = 1 - P(H_0  \text{ reddet} \mid H_0 \text{ doğru} ).$$

Aynı zamanda *1. tür hata* veya *yanlış pozitif* olarak da anılır. $\alpha$, *anlamlılık düzeyi* olarak adlandırılır ve yaygın olarak kullanılan değeri $\% 5$, yani $1- \alpha = \% 95$ şeklindedir. Anlamlılık düzeyi, gerçek bir sıfır hipotezi reddettiğimizde almaya istekli olduğumuz risk seviyesi olarak açıklanabilir.

:numref:`fig_statistical_significance`, iki örneklemli bir hipotez testinde gözlemlerin değerlerini ve belirli bir normal dağılımın gelme olasılığını gösterir. Gözlem veri örneği $\% 95$ eşiğinin dışında yer alırsa, sıfır hipotez varsayımı altında çok olası olmayan bir gözlem olacaktır. Dolayısıyla, sıfır hipotezde yanlış bir şeyler olabilir ve onu reddedeceğiz.

![İstatistiksel anlamlılık.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`

### İstatistiksel Güç

*İstatistiksel Güç* (veya *duyarlılık*), reddedilmesi gerektiğinde sıfır hipotezin, $H_0$, reddedilme olasılığını ölçer, yani,

$$ \text{istatistiksel güç }= 1 - \beta = 1 - P(H_0 \text{ rededememe } \mid H_0 \text{ yanlış} ).$$

Bir *1. tür hata*nın, doğru olduğunda sıfır hipotezin reddedilmesinden kaynaklanan bir hata olduğunu hatırlayın, oysa *2. tür hata* yanlış olduğunda sıfır hipotezin reddedilmemesinden kaynaklanır. 2. tür hata genellikle $\beta$ olarak belirtilir ve bu nedenle ilgili istatistiksel güç $1-\beta$ olur.

Sezgisel olarak, istatistiksel güç, testimizin istenen bir istatistiksel anlamlılık düzeyindeyken minimum büyüklükte gerçek bir tutarsızlığı ne kadar olasılıkla tespit edeceği şeklinde yorumlanabilir. $\% 80$, yaygın olarak kullanılan bir istatistiksel güç eşiğidir. İstatistiksel güç ne kadar yüksekse, gerçek farklılıkları tespit etme olasılığımız o kadar yüksektir.

İstatistiksel gücün en yaygın kullanımlarından biri, ihtiyaç duyulan örnek sayısını belirlemektir. Sıfır hipotezini yanlış olduğunda reddetme olasılığınız, yanlış olma derecesine (*etki boyutu* olarak bilinir) ve sahip olduğunuz örneklerin sayısına bağlıdır. Tahmin edebileceğiniz gibi, küçük etki boyutları, yüksek olasılıkla tespit edilebilmesi için çok fazla sayıda örnek gerektirir. Ayrıntılı olarak türetmek için bu kısa ek bölümün kapsamı dışında, örnek olarak, örneğimizin sıfır ortalama bir varyanslı Gauss'tan geldiğine dair bir sıfır hipotezi reddedebilmek isterken, örneklemimizin ortalamasının aslında bire yakın olduğuna inanıyoruz, bunu yalnızca $8$'lik örneklem büyüklüğünde kabul edilebilir hata oranları ile yapabiliriz. Bununla birlikte, örnek popülasyonumuzun gerçek ortalamasının $0.01$'e yakın olduğunu düşünürsek, farkı tespit etmek için yaklaşık $80000$'lik bir örneklem büyüklüğüne ihtiyacımız olur.

Gücü bir su filtresi olarak hayal edebiliriz. Bu benzetmede, yüksek güçlü bir hipotez testi, sudaki zararlı maddeleri olabildiğince azaltacak yüksek kaliteli bir su filtreleme sistemi gibidir. Öte yandan, daha küçük bir tutarsızlık, bazı nispeten küçük maddelerin boşluklardan kolayca kaçabildiği düşük kaliteli bir su filtresine benzer. Benzer şekilde, istatistiksel güç yeterince yüksek güce sahip değilse, bu test daha küçük tutarsızları yakalayamayabilir.

### Test İstatistiği

Bir *test istatistiği* $T(x)$, örnek verilerin bazı özelliklerini özetleyen bir sayıdır. Böyle bir istatistiği tanımlamanın amacı, farklı dağılımları ayırt etmemize ve hipotez testimizi yürütmemize izin vermesidir. Kimyager örneğimize geri dönersek, bir popülasyonun diğerinden daha iyi performans gösterdiğini göstermek istiyorsak, ortalamayı test istatistiği olarak almak mantıklı olabilir. Farklı test istatistiği seçenekleri, büyük ölçüde farklı istatistiksel güce sahip istatistiksel testlere yol açabilir.

Genellikle, $T(X)$ (sıfır hipotezimiz altındaki test istatistiğinin dağılımı), en azından yaklaşık olarak, sıfır hipotezi kapsamında değerlendirildiğinde normal dağılım gibi genel bir olasılık dağılımını izleyecektir. Açıkça böyle bir dağılım elde edebilir ve daha sonra veri kümemizdeki test istatistiğimizi ölçebilirsek, istatistiğimiz beklediğimiz aralığın çok dışındaysa sıfır hipotezi güvenle reddedebiliriz. Bunu nicel hale getirmek bizi $p$-değerleri kavramına götürür.

### $p$-Değeri

$p$-değeri (veya *olasılık değeri*), sıfır hipotezinin *doğru* olduğu varsayılarak, $T(X)$'in en az gözlenen test istatistiği $T(x)$ kadar uç olma olasılığıdır, yani

$$ p\text{-değeri} = P_{H_0}(T(X) \geq T(x)).$$

$p$-değeri önceden tanımlanmış ve sabit bir istatistiksel anlamlılık düzeyi $\alpha$ değerinden küçükse veya ona eşitse, sıfır hipotezini reddedebiliriz. Aksi takdirde, sıfır hipotezi reddetmek için kanıtımız olmadığı sonucuna varacağız. Belirli bir popülasyon dağılımı için, *reddetme bölgesi*, istatistiksel anlamlılık düzeyi $\alpha$'dan daha küçük bir $p$ değerine sahip tüm noktaların içerildiği aralık olacaktır.

### Tek Taraflı Test ve İki Taraflı Test

Normalde iki tür anlamlılık testi vardır: Tek taraflı test ve iki taraflı test. *Tek taraflı test* (veya *tek kuyruklu test*), sıfır hipotez ve alternatif hipotezin yalnızca bir tarafta olduğunda geçerlidir. Örneğin, sıfır hipotez $\theta$ gerçek parametresinin $c$ değerinden küçük veya ona eşit olduğunu belirtebilir. Alternatif hipotez, $\theta$ nın $c$'den büyük olması olacaktır. Yani, reddetme bölgesi, örneklem dağılımının sadece bir tarafındadır. Tek taraflı testin aksine, *iki taraflı test* (veya *iki kuyruklu test*), reddetme bölgesi örneklem dağılımının her iki tarafında olduğunda uygulanabilir. Bu durumda bir örnek, $\theta$ gerçek parametresinin $c$ değerine eşit olduğunu belirten bir sıfır hipotez ifadesine sahip olma olabilir. Alternatif hipotez, $\theta$'nın $c$'ye eşit olmamasıdır.

### Hipotez Testinin Genel Adımları

Yukarıdaki kavramlara aşina olduktan sonra, hipotez testinin genel adımlarından geçelim.

1. Soruyu belirtin ve sıfır hipotezi, $H_0$, oluşturun.
2. İstatistiksel anlamlılık düzeyini $\alpha$'yı ve bir istatistiksel güç ($1-\beta$)'yı ayarlayın.
3. Deneyler yoluyla numuneler alın. İhtiyaç duyulan örnek sayısı istatistiksel güce ve beklenen etki büyüklüğüne bağlı olacaktır.
4. Test istatistiğini ve $p$-değerini hesaplayın.
5. $p$-değeri ve istatistiksel anlamlılık düzeyi $\alpha$ bağlı olarak sıfır hipotezi tutma veya reddetme kararını verin.

Bir hipotez testi yapmak için, bir sıfır hipotez ve almaya istekli olduğumuz bir risk seviyesi tanımlayarak başlıyoruz. Sonra, sıfır hipotezine karşı kanıt olarak test istatistiğinin aşırı bir değerini alarak numunenin (örneklemin) test istatistiğini hesaplıyoruz. Test istatistiği reddetme bölgesi dahilindeyse, alternatif lehine sıfır hipotezi reddedebiliriz.

Hipotez testi, klinik araştırmalar ve A/B testi gibi çeşitli senaryolarda uygulanabilir.

## Güven Aralıkları Oluşturma

Bir $\theta$ parametresinin değerini tahmin ederken, $\hat \theta$ gibi nokta tahmincileri, belirsizlik kavramı içermedikleri için sınırlı fayda sağlar. Daha ziyade, yüksek olasılıkla gerçek $\theta$ parametresini içeren bir aralık oluşturabilirsek çok daha iyi olurdu. Yüzyıl önce bu tür fikirlerle ilgileniyor olsaydınız, 1937'de güven aralığı kavramını tanıtan Jerzy Neyman'ın "Klasik Olasılık Teorisine Dayalı İstatistiksel Tahmin Teorisinin Ana Hatları (Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability)"'nı okumaktan heyecan duyardınız :cite:`Neyman.1937`.

Faydalı olması için, belirli bir kesinlik derecesi için bir güven aralığı mümkün olduğu kadar küçük olmalıdır. Nasıl türetileceğini görelim.

### Tanım

Matematiksel olarak, $\theta$ gerçek parametresi için *güven aralığı*, örnek verilerden şu şekilde hesaplanan bir $C_n$ aralığıdır.

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

Burada $\alpha \in (0, 1)$ ve $1 - \alpha$, aralığın *güven düzeyi* veya *kapsamı* olarak adlandırılır. Bu, yukarıda tartıştığımız anlam düzeyiyle aynı $\alpha$'dır.

Unutmayın :eqref:`eq_confidence`  $C_n$ değişkeni hakkındadır, sabit $\theta$ ile ilgili değildir. Bunu vurgulamak için, $P_{\theta} (\theta \in C_n)$ yerine $P_{\theta} (C_n \ni \theta)$ yazıyoruz.

### Yorumlama

$\% 95$ güven aralığını, gerçek parametrenin $\% 95$ olduğundan emin olabileceğiniz bir aralık olarak yorumlamak çok caziptir, ancak bu maalesef doğru değildir. Gerçek parametre sabittir ve rastgele olan aralıktır. Bu nedenle, bu yordamla çok sayıda güven aralığı oluşturduysanız, oluşturulan aralıkların $\% 95$'inin gerçek parametreyi içereceğini söylemek daha iyi bir yorum olacaktır.

Bu bilgiçlik gibi görünebilir, ancak sonuçların yorumlanmasında gerçek etkileri olabilir. Özellikle, çok nadiren yeterince yaptığımız sürece, *neredeyse kesin* gerçek değeri içermediğimiz aralıklar oluşturarak :eqref:`eq_confidence`i tatmin edebiliriz. Bu bölümü cazip ama yanlış üç ifade sunarak kapatıyoruz. Bu noktaların derinlemesine bir tartışması şu adreste bulunabilir :cite:`Morey.Hoekstra.Rouder.ea.2016`.

* **Yanılgı 1**. Dar güven aralıkları, parametreyi tam olarak tahmin edebileceğimiz anlamına gelir.
* **Yanılgı 2**. Güven aralığı içindeki değerlerin, aralığın dışındaki değerlere göre gerçek değer olma olasılığı daha yüksektir.
* **Yanılgı 3**. Gözlemlenen belirli bir $\% 95$ güven aralığının gerçek değeri içerme olasılığı $\% 95$'tir.

Güven aralıklarının narin nesneler olduğunu söylemek yeterli. Ancak yorumlamasını net tutarsanız, bunlar güçlü araçlar olabilir.

### Bir Gauss Örneği

En klasik örneği, bilinmeyen ortalama ve varyansa sahip bir Gauss dağılımının ortalaması için güven aralığını tartışalım. Gauss dağılımdan, $\mathcal{N}(\mu, \sigma^2)$'dan, $n$ örnek, $\{x_i\}_{i=1}^n$, topladığımızı varsayalım. Ortalama ve standart sapma için tahmincileri şu şekilde hesaplayabiliriz:

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

Şimdi rastgele değişkeni düşünürsek

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

$n-1$ *serbestlik derecesinde*, *Student-t (Öğrenci-t) dağılımı* adı verilen iyi bilinen bir dağılımı izleyen rastgele bir değişken elde ederiz.

Bu dağılım çok iyi incelenmiştir ve örneğin, $n \rightarrow \infty$ iken, yaklaşık olarak standart bir Gauss olduğu bilinir ve dolayısıyla bir tabloda Gauss b.y.f'nin değerlerine bakarak $T$ değerinin zamanın en az $\% 95$'inde  $[- 1.96, 1.96]$ aralığında olduğu sonucuna varabiliriz. $n$ değerinin sonlu değerleri için, aralığın biraz daha büyük olması gerekir, ancak bunlar iyi bilinmekte ve tablolarda önceden hesaplanmaktadır.

Böylece, büyük $ n $ için diyebiliriz ki,

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

Bunu her iki tarafı da $\hat\sigma_n /\sqrt{n}$ ile çarpıp ardından $\hat\mu_n$ ekleyerek yeniden düzenleriz,

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

Böylece, $\% 95$'lik güven aralığımızı bulduğumuzu biliyoruz:
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

Şunu söylemek güvenlidir: :eqref:`eq_gauss_confidence` istatistikte en çok kullanılan formüllerden biridir. İstatistik tartışmamızı uygulama ile kapatalım. Basit olması için, asimptotik (kavuşma doğrusal) rejimde olduğumuzu varsayıyoruz. Küçük $N$ değerleri, programlanarak veya bir $t$-tablosundan elde edilen `t_star`'ın doğru değerini içermelidir.

```{.python .input}
# Örnek sayısı
N = 1000

# Örnek veri kümesi
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Öğrenci t-dağılımının c.d.f.'sine bak
t_star = 1.96

# Aralık oluştur
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch, varsayılan olarak Bessel'in düzeltmesini kullanır; 
# bu, numpy'de varsayılan ddof=0 yerine ddof=1 kullanılması anlamına gelir. 
# ddof=0'ı taklit etmek için unbiased=False kullanabiliriz.

# Örnek sayısı
N = 1000

# Örnek veri kümesi
samples = torch.normal(0, 1, size=(N,))

# Öğrenci t-dağılımının c.d.f.'sine bak
t_star = 1.96

# Aralık oluştur
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Örnek sayısı
N = 1000

# Örnek veri kümesi
samples = tf.random.normal((N,), 0, 1)

# Öğrenci t-dağılımının c.d.f.'sine bak
t_star = 1.96

# Aralık oluştur
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## Özet

* İstatistik, çıkarım sorunlarına odaklanırken, derin öğrenme, açıkça programlamadan ve anlamadan doğru tahminler yapmaya vurgu yapar.
* Üç yaygın istatistik çıkarım yöntemi vardır: Tahmincileri değerlendirme ve karşılaştırma, hipotez testleri yürütme ve güven aralıkları oluşturma.
* En yaygın üç tahminci vardır: İstatistiksel yanlılık, standart sapma ve ortalama hata karesi.
* Bir güven aralığı, örneklerle oluşturabileceğimiz gerçek bir popülasyon parametresinin tahmini aralığıdır.
* Hipotez testi, bir popülasyonla ilgili varsayılan ifadeye karşı bazı kanıtları değerlendirmenin bir yoludur.

## Alıştırmalar

1. $X_1, X_2, \ldots, X_n \overset {\text{iid}}{\sim} \mathrm{Tekdüze} (0, \theta)$ olsun, burada "iid" *bağımsız ve aynı şekilde dağılmış* anlamına gelir. Aşağıdaki $\theta$ tahmincilerini düşünün:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * $\hat{\theta}$ için istatistiksel yanlılığı, standart sapmayı ve ortalama hata karesini bulunuz.
    * $\tilde{\theta}$ için istatistiksel yanlılığı, standart sapmayı ve ortalama hata karesini bulunuz.
    * Hangi tahminci daha iyi?
1. Girişteki kimyager örneğimiz için, iki taraflı bir hipotez testi yapmak için 5 adımı türetebilir misiniz? İstatistiksel anlamlılık düzeyini $\alpha = 0.05$ ve istatistiksel gücü $1 - \beta =0.8$ alınız.
1. $100$ tane bağımsız olarak oluşturulan veri kümesi için güven aralığı kodunu $N = 2$ ve $\alpha = 0.5$ ile çalıştırın ve ortaya çıkan aralıkları çizin (bu durumda `t_star = 1.0`). Gerçek ortalama olan $0$'ı içermekten çok uzak olan birkaç çok kısa aralık göreceksiniz. Bu, güven aralığının yorumlamasıyla çelişiyor mu? Yüksek hassasiyetli tahminleri belirtmek için kısa aralıklar kullanmakta kendinizi rahat hissediyor musunuz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1103)
:end_tab:
