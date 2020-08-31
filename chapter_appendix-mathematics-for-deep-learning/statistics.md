# İstatistik
:label:`sec_statistics`

Kuşkusuz, en iyi derin öğrenme uygulayıcılarından biri olmak için son teknoloji ürünü ve yüksek doğrulukta modelleri eğitme yeteneği çok önemlidir. Bununla birlikte, iyileştirmelerin ne zaman önemli olduğu veya yalnızca eğitim sürecindeki rastgele dalgalanmaların sonucu olduğu genellikle belirsizdir. Tahmini değerlerdeki belirsizliği tartışabilmek için biraz istatistik öğrenmemiz gerekir.

*İstatistiğin* en eski referansı, şifrelenmiş mesajları deşifre etmek için istatistiklerin ve sıklık analizinin nasıl kullanılacağına dair ayrıntılı bir açıklama veren $9.$-yüzyıldaki Arap bilim adamı Al-Kindi'ye kadar uzanabilir. 800 yıl sonra, modern istatistik, araştırmacıların demografik ve ekonomik veri toplama ve analizine odaklandığı 1700'lerde Almanya'da ortaya çıktı. Günümüzde istatistik, verilerin toplanması, işlenmesi, analizi, yorumlanması ve görselleştirilmesi ile ilgili bilim konusudur. Dahası, temel istatistik teorisi akademi, endüstri ve hükümet içindeki araştırmalarda yaygın olarak kullanılmaktadır.

Daha özel olarak, istatistik *tanımlayıcı istatistik* ve *istatistiksel çıkarım* diye bölünebilir. İlki, *örneklem* olarak adlandırılan gözlemlenen verilerden bir koleksiyonunun özelliklerini özetlemeye ve göstermeye odaklanır. Örneklem bir *popülasyondan* alınmıştır, benzer bireyler, öğeler veya deneysel ilgi alanlarımıza ait olayların toplam kümesini belirtir. Tanımlayıcı istatistiğin aksine *istatistiksel çıkarım*, örneklem dağılımının popülasyon dağılımını bir dereceye kadar kopyalayabileceği varsayımlarına dayanarak, bir popülasyonun özelliklerini verilen *örneklemlerden* çıkarsar.

Merak edebilirsiniz: "Makine öğrenmesi ile istatistik arasındaki temel fark nedir?" Temel olarak, istatistik çıkarım sorununa odaklanır. Bu tür problemler, nedensel çıkarım gibi değişkenler arasındaki ilişkiyi modellemeyi ve A/B testi gibi model parametrelerinin istatistiksel olarak anlamlılığını test etmeyi içerir. Buna karşılık, makine öğrenmesi, her bir parametrenin işlevselliğini açıkça programlamadan ve anlamadan doğru tahminler yapmaya vurgu yapar.

Bu bölümde, üç tür istatistik çıkarım yöntemini tanıtacağız: tahmin edicileri değerlendirme ve karşılaştırma, hipotez testleri yürütme ve güven aralıkları oluşturma. Bu yöntemler, belirli bir popülasyonun özelliklerini, yani gerçek $\theta$ parametresi gibi, anlamamıza yardımcı olabilir. Kısacası, belirli bir popülasyonun gerçek parametresinin, $\theta$, skaler bir değer olduğunu varsayıyoruz. $\theta$'nin bir vektör veya tensör olduğu durumu genişletmek basittir, bu nedenle tartışmamızda onu es geçiyoruz.

## Tahmincileri Değerlendirme ve Karşılaştırma

İstatistikte, bir *tahminci*, gerçek $\theta$ parametresini tahmin etmek için kullanılan belirli örneklemlerin bir fonksiyonudur. {$x_1, x_2, \ ldots, x_n$} örneklerini gözlemledikten sonra $\theta$ tahmini için $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ yazacağız .

Tahmincilerin basit örneklerini daha önce şu bölümde görmüştük :numref:`sec_maximum_likelihood`. Bir Bernoulli rastgele değişkeninden birkaç örneğiniz varsa, rastgele değişkenin olma olasılığı için maksimum olabilirlik tahmini, gözlemlenenlerin sayısını sayarak ve toplam örnek sayısına bölerek elde edilebilir. Benzer şekilde, bir alıştırma sizden bir miktar örnek verilen bir Gauss'un ortalamasının maksimum olabilirlik tahmininin tüm örneklerin ortalama değeriyle verildiğini göstermenizi istiyor. Bu tahminciler neredeyse hiçbir zaman parametrenin gerçek değerini vermezler, ancak ideal olarak çok sayıda örnek için tahmin yakın olacaktır.

Örnek olarak, ortalama sıfır ve varyans bir olan bir Gauss rasgele değişkeninin gerçek yoğunluğunu, bu Gauss'tan bir dizi örnek ile aşağıda gösteriyoruz. Her noktanın $y$ koordinatı görünür ve orijinal yoğunluk ile olan ilişki daha net fark edilecek şekilde oluşturduk.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[0:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
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

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[0:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

$\hat{\theta}_n $ parametresinin bir tahmincisini hesaplamanın birçok yolu olabilir. Bu bölümde, tahmincileri değerlendirmek ve karşılaştırmak için üç genel yöntem sunuyoruz: ortalama hata karesi, standart sapma ve istatistiksel yanlılık.

### Ortalama Hata Karesi

Tahmin edicileri değerlendirmek için kullanılan en basit ölçüt, bir tahmincinin *ortalama hata karesi (MSE)* (veya $l_2$ kaybı) olarak tanımlanabilir.

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

Bu, gerçek değerden ortalama kare sapmayı ölçümlemizi sağlar. MSE her zaman negatif değildir. Eğer :numref:`sec_linear_regression`'yi okuduysanız, bunu en sık kullanılan bağlanım (regresyon) kaybı işlevi olarak tanıyacaksınız. Bir tahminciyi değerlendirmek için bir ölçü olarak, değeri sıfıra ne kadar yakınsa, tahminci gerçek $\theta$ parametresine o kadar yakın olur.

### İstatistiksel Yanlılık

MSE doğal bir ölçü sağlar, ancak onu büyük yapabilecek birden fazla farklı vakayı kolayca hayal edebiliriz. İki temel önemli olay veri kümesindeki rastgelelik nedeniyle tahmincideki dalgalanma ve tahmin prosedürüne bağlı olarak tahmincideki sistematik hatadır.

Öncelikle sistematik hatayı ölçelim. Bir $\hat{\theta}_n $ tahmincisi için *istatistiksel yanlılığın* matematiksel gösterimi şu şekilde tanımlanabilir:

$$\mathrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\mathrm{bias}(\hat{\theta}_n) = 0$ olduğunda, $\hat{\theta}_n$ tahmin edicisinin beklentisinin parametrenin gerçek değerine eşit olduğuna dikkat edin. Bu durumda, $\hat{\theta}_n$'nin yansız bir tahminci olduğunu söylüyoruz. Genel olarak, yansız bir tahminci, yanlı bir tahminciden daha iyidir çünkü beklenen değeri gerçek parametre ile aynıdır.

Bununla birlikte, yanlı tahmin edicilerin pratikte sıklıkla kullanıldığının farkında olunması gerekir. Yansız tahmin edicilerin başka varsayımlar olmaksızın var olmadığı veya hesaplamanın zor olduğu durumlar vardır. Bu, bir tahmincide önemli bir kusur gibi görünebilir, ancak pratikte karşılaşılan tahmin edicilerin çoğu, mevcut örneklerin sayısı sonsuza giderken sapmanın sıfır olma eğiliminde olması açısından en azından asimptotik (kavuşma doğrusu) olarak tarafsızdır: $\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$.

### Varyans ve Standart Sapma

İkinci olarak tahmincideki rastgeleliği ölçelim. Eğer :numref:`sec_random_variables`yi anımsarsak, *standart sapma* (veya *standart hata*), varyansın kare kökü olarak tanımlanır. Bir tahmincinin dalgalanma derecesini, o tahmincinin standart sapmasını veya varyansını ölçerek ölçebiliriz.

$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

Şunları karşılaştırmak önemlidir :eqref:`eq_var_est` ile :eqref:`eq_mse_est`. Bu denklemde gerçek popülasyon değeri $\theta$ ile değil, bunun yerine beklenen örneklem ortalaması $E(\hat{\theta}_n)$ ile karşılaştırıyoruz. Bu nedenle, tahmincinin gerçek değerden ne kadar uzakta olduğunu ölçmüyoruz, bunun yerine tahmincinin dalgalanmasını ölçüyoruz.

### Yanlılık-Varyans Takası

Bu iki bileşenin ortalama hata karesine katkıda bulunduğu sezgisel olarak açıktır. Biraz şok edici olan şey, bunun aslında ortalama hata karesinin iki parçaya *ayrıştırılması* olduğunu gösterebilmemizdir. Yani, ortalama hata karesini varyansın ve yanlılığın karesi toplamı olarak yazabiliriz.

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - E(\hat{\theta}_n) + E(\hat{\theta}_n) - \theta)^2] \\
 &= E[(\hat{\theta}_n - E(\hat{\theta}_n))^2] + E[(E(\hat{\theta}_n) - \theta)^2] \\
 &= \mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2.\\
\end{aligned}
$$

Yukarıdaki formülü *yanlılık-varyans takası* olarak adlandırıyoruz. Ortalama hata karesi kesin olarak iki hata kaynağına bölünebilir: Yüksek yanlılıktan kaynaklanan hata ve yüksek varyans kaynaklı hata. Bir yandan, yanlılık hatası genellikle basit bir modelde (doğrusal bağlanım modeli gibi) görülür, çünkü özellikler ve çıktılar arasındaki yüksek boyutsal ilişkileri çıkaramaz. Bir model yüksek yanlılık hatasından muzdaripse, (:numref:`sec_model_selection`) bölümünde açıklandığı gibi genellikle *eksik öğrenme* veya *genelleme* eksikliği olduğunu söylüyoruz. Diğer taraftan, diğer hata kaynağı---yüksek varyans, genellikle eğitim verilerine öğrenen çok karmaşık bir modelden kaynaklanır. Sonuç olarak, *aşırı öğrenen* bir model, verilerdeki küçük dalgalanmalara duyarlıdır. Bir modelin varyansı yüksekse, genellikle (:numref:`sec_model_selection`)'de tanıtıldığı gibi *aşırı öğrenme* ve *esneklik* yoksunluğu olduğunu söyleriz.

### Kodda Tahmincileri Değerlendirme

Bir tahmincinin standart sapması, MXNet'te bir tensör `a` için basitçe `a.std()` çağırarak uygulandığından, onu atlayacağız ancak MXNet'te istatistiksel yanlılık ve ortalama hata karesini uygulayacağız.

```{.python .input}
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

Yanlılık-varyans takasının denklemini görsellemek için, $\mathcal{N}(\theta, \sigma^2)$ normal dağılımını $10.000$ örnekle canlandıralım. Burada bir $\theta = 1$ ve $\sigma = 4$ kullanıyoruz. Tahminci verilen örneklerin bir fonksiyonu olduğu için, burada örneklerin ortalamasını bu normal dağılımdaki, $\mathcal{N}(\theta, \sigma^2)$, gerçek $\theta$ için bir tahminci olarak kullanıyoruz.

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

Tahmincimizin yanlılık karesi ve varyansının toplamını hesaplayarak takas denklemini doğrulayalım. İlk önce, tahmincimizin MSE'sini hesaplayın.

```{.python .input}
mse(samples, theta_true)
```

```{.python .input}
#@tab pytorch
mse(samples, theta_true)
```

Ardından, aşağıdaki gibi $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$'yi hesaplıyoruz. Gördüğünüz gibi, iki değer sayısal kesinliğe kadar uyuyor.

```{.python .input}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

## Conducting Hypothesis Tests


The most commonly encountered topic in statistical inference is hypothesis testing. While hypothesis testing was popularized in the early 20th century, the first use can be traced back to John Arbuthnot in the 1700s. John tracked 80-year birth records in London and concluded that more men were born than women each year. Following that, the modern significance testing is the intelligence heritage by Karl Pearson who invented $p$-value and Pearson's chi-squared test), William Gosset who is the father of Student's t-distribution, and Ronald Fisher who initialed the null hypothesis and the significance test.

A *hypothesis test* is a way of evaluating some evidence against the default statement about a population. We refer the default statement as the *null hypothesis* $H_0$, which we try to reject using the observed data. Here, we use $H_0$ as a starting point for the statistical significance testing. The *alternative hypothesis* $H_A$ (or $H_1$) is a statement that is contrary to the null hypothesis. A null hypothesis is often stated in a declarative form which posits a relationship between variables. It should reflect the brief as explicit as possible, and be testable by statistics theory.

Imagine you are a chemist. After spending thousands of hours in the lab, you develop a new medicine which can dramatically improve one's ability to understand math. To show its magic power, you need to test it. Naturally, you may need some volunteers to take the medicine and see whether it can help them learn math better. How do you get started?

First, you will need carefully random selected two groups of volunteers, so that there is no difference between their math understanding ability measured by some metrics. The two groups are commonly referred to as the test group and the control group. The *test group* (or *treatment group*) is a group of individuals who will experience the medicine, while the *control group* represents the group of users who are set aside as a benchmark, i.e., identical environment setups except taking this medicine. In this way, the influence of all the variables are minimized, except the impact of the independent variable in the treatment.

Second, after a period of taking the medicine, you will need to measure the two groups' math understanding by the same metrics, such as letting the volunteers do the same tests after learning a new math formula. Then, you can collect their performance and compare the results.  In this case, our null hypothesis will be that there is no difference between the two groups, and our alternate will be that there is.

This is still not fully formal.  There are many details you have to think of carefully. For example, what is the suitable metrics to test their math understanding ability? How many volunteers for your test so you can be confident to claim the effectiveness of your medicine? How long should you run the test? How do you decide if there is a difference between the two groups?  Do you care about the average performance only, or do you also the range of variation of the scores. And so on.

In this way, hypothesis testing provides a framework for experimental design and reasoning about certainty in observed results.  If we can now show that the null hypothesis is very unlikely to be true, we may reject it with confidence.

To complete the story of how to work with hypothesis testing, we need to now introduce some additional terminology and make some of our concepts above formal.


### Statistical Significance

The *statistical significance* measures the probability of erroneously rejecting the null hypothesis, $H_0$, when it should not be rejected, i.e.,

$$ \text{statistical significance }= 1 - \alpha = P(\text{reject } H_0 \mid H_0 \text{ is true} ).$$

It is also referred to as the *type I error* or *false positive*. The $\alpha$, is called as the *significance level* and its commonly used value is $5\%$, i.e., $1-\alpha = 95\%$. The level of statistical significance level can be explained as the level of risk that we are willing to take, when we reject a true null hypothesis.

:numref:`fig_statistical_significance` shows the observations' values and probability of a given normal distribution in a two-sample hypothesis test. If the observation data point is located outsides the $95\%$ threshold, it will be a very unlikely observation under the null hypothesis assumption. Hence, there might be something wrong with the null hypothesis and we will reject it.

![Statistical significance.](../img/statistical_significance.svg)
:label:`fig_statistical_significance`


### Statistical Power

The *statistical power* (or *sensitivity*) measures the probability of reject the null hypothesis, $H_0$, when it should be rejected, i.e.,

$$ \text{statistical power }= P(\text{reject } H_0  \mid H_0 \text{ is false} ).$$

Recall that a *type I error* is error caused by rejecting the null hypothesis when it is true, whereas a *type II error* is resulted from failing to reject the null hypothesis when it is false. A type II error is usually denoted as $\beta$, and hence the corresponding statistical power is $1-\beta$.


Intuitively, statistical power can be interpreted as how likely our test will detect a real discrepancy of some minimum magnitude at a desired statistical significance level. $80\%$ is a commonly used statistical power threshold. The higher the statistical power, the more likely we are to detect true differences.

One of the most common uses of statistical power is in determining the number of samples needed.  The probability you reject the null hypothesis when it is false depends on the degree to which it is false (known as the *effect size*) and the number of samples you have.  As you might expect, small effect sizes will require a very large number of samples to be detectable with high probability.  While beyond the scope of this brief appendix to derive in detail, as an example, want to be able to reject a null hypothesis that our sample came from a mean zero variance one Gaussian, and we believe that our sample's mean is actually close to one, we can do so with acceptable error rates with a sample size of only $8$.  However, if we think our sample population true mean is close to $0.01$, then we'd need a sample size of nearly $80000$ to detect the difference.

We can imagine the power as a water filter. In this analogy, a high power hypothesis test is like a high quality water filtration system that will reduce harmful substances in the water as much as possible. On the other hand, a smaller discrepancy is like a low quality water filter, where some relative small substances may easily escape from the gaps. Similarly, if the statistical power is not of enough high power, then the test may not catch the smaller discrepancy.


### Test Statistic

A *test statistic* $T(x)$ is a scalar which summarizes some characteristic of the sample data.  The goal of defining such a statistic is that it should allow us to distinguish between different distributions and conduct our hypothesis test.  Thinking back to our chemist example, if we wish to show that one population performs better than the other, it could be reasonable to take the mean as the test statistic.  Different choices of test statistic can lead to statistical test with drastically different statistical power.

Often, $T(X)$ (the distribution of the test statistic under our null hypothesis) will follow, at least approximately, a common probability distribution such as a normal distribution when considered under the null hypothesis. If we can derive explicitly such a distribution, and then measure our test statistic on our dataset, we can safely reject the null hypothesis if our statistic is far outside the range that we would expect.  Making this quantitative leads us to the notion of $p$-values.


### $p$-value

The $p$-value (or the *probability value*) is the probability that $T(X)$ is at least as extreme as the observed test statistic $T(x)$ assuming that the null hypothesis is *true*, i.e.,

$$ p\text{-value} = P_{H_0}(T(X) \geq T(x)).$$

If the $p$-value is smaller than or equal to a predefined and fixed statistical significance level $\alpha$, we may reject the null hypothesis. Otherwise, we will conclude that we are lack of evidence to reject the null hypothesis. For a given population distribution, the *region of rejection* will be the interval contained of all the points which has a $p$-value smaller than the statistical significance level $\alpha$.


### One-side Test and Two-sided Test

Normally there are two kinds of significance test: the one-sided test and the two-sided test. The *one-sided test* (or *one-tailed test*) is applicable when the null hypothesis and the alternative hypothesis only have one direction. For example, the null hypothesis may state that the true parameter $\theta$ is less than or equal to a value $c$. The alternative hypothesis would be that $\theta$ is greater than $c$. That is, the region of rejection is on only one side of the sampling distribution.  Contrary to the one-sided test, the *two-sided test* (or *two-tailed test*) is applicable when the region of rejection is on both sides of the sampling distribution. An example in this case may have a null hypothesis state that the true parameter $\theta$ is equal to a value $c$. The alternative hypothesis would be that $\theta$ is not equal to $c$.


### General Steps of Hypothesis Testing

After getting familiar with the above concepts, let us go through the general steps of hypothesis testing.

1. State the question and establish a null hypotheses $H_0$.
2. Set the statistical significance level $\alpha$ and a statistical power ($1 - \beta$).
3. Obtain samples through experiments.  The number of samples needed will depend on the statistical power, and the expected effect size.
4. Calculate the test statistic and the $p$-value.
5. Make the decision to keep or reject the null hypothesis based on the $p$-value and the statistical significance level $\alpha$.

To conduct a hypothesis test, we start by defining a null hypothesis and a level of risk that we are willing to take. Then we calculate the test statistic of the sample, taking an extreme value of the test statistic as evidence against the null hypothesis. If the test statistic falls within the reject region, we may reject the null hypothesis in favor of the alternative.

Hypothesis testing is applicable in a variety of scenarios such as the clinical trails and A/B testing.


## Constructing Confidence Intervals


When estimating the value of a parameter $\theta$, point estimators like $\hat \theta$ are of limited utility since they contain no notion of uncertainty. Rather, it would be far better if we could produce an interval that would contain the true parameter $\theta$ with high probability.  If you were interested in such ideas a century ago, then you would have been excited to read "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" by Jerzy Neyman :cite:`Neyman.1937`, who first introduced the concept of confidence interval in 1937.

To be useful, a confidence interval should be as small as possible for a given degree of certainty. Let us see how to derive it.


### Definition

Mathematically, a *confidence interval* for the true parameter $\theta$ is an interval $C_n$ that computed from the sample data such that

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

Here $\alpha \in (0, 1)$, and $1 - \alpha$ is called the *confidence level* or *coverage* of the interval. This is the same $\alpha$ as the significance level as we discussed about above.

Note that :eqref:`eq_confidence` is about variable $C_n$, not about the fixed $\theta$. To emphasize this, we write $P_{\theta} (C_n \ni \theta)$ rather than $P_{\theta} (\theta \in C_n)$.

### Interpretation

It is very tempting to interpret a $95\%$ confidence interval as an interval where you can be $95\%$ sure the true parameter lies, however this is sadly not true.  The true parameter is fixed, and it is the interval that is random.  Thus a better interpretation would be to say that if you generated a large number of confidence intervals by this procedure, $95\%$ of the generated intervals would contain the true parameter.

This may seem pedantic, but it can have real implications for the interpretation of the results.  In particular, we may satisfy :eqref:`eq_confidence` by constructing intervals that we are *almost certain* do not contain the true value, as long as we only do so rarely enough.  We close this section by providing three tempting but false statements.  An in-depth discussion of these points can be found in :cite:`Morey.Hoekstra.Rouder.ea.2016`.

* **Fallacy 1**. Narrow confidence intervals mean we can estimate the parameter precisely.
* **Fallacy 2**. The values inside the confidence interval are more likely to be the true value than those outside the interval.
* **Fallacy 3**. The probability) that a particular observed $95\%$ confidence interval contains the true value is $95\%$.

Sufficed to say, confidence intervals are subtle objects.  However, if you keep the interpretation clear, they can be powerful tools.

### A Gaussian Example

Let us discuss the most classical example, the confidence interval for the mean of a Gaussian of unknown mean and variance.  Suppose we collect $n$ samples $\{x_i\}_{i=1}^n$ from our Gaussian $\mathcal{N}(\mu, \sigma^2)$.  We can compute estimators for the mean and standard deviation by taking

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

If we now consider the random variable

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

we obtain a random variable following a well-known distribution called the *Student's t-distribution on* $n-1$ *degrees of freedom*.

This distribution is very well studied, and it is known, for instance, that as $n\rightarrow \infty$, it is approximately a standard Gaussian, and thus by looking up values of the Gaussian c.d.f. in a table, we may conclude that the value of $T$ is in the interval $[-1.96, 1.96]$ at least $95\%$ of the time.  For finite values of $n$, the interval needs to be somewhat larger, but are well known and precomputed in tables.

Thus, we may conclude that for large $n$,

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

Rearranging this by multiplying both sides by $\hat\sigma_n/\sqrt{n}$ and then adding $\hat\mu_n$, we obtain

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

Thus we know that we have found our $95\%$ confidence interval:
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

It is safe to say that :eqref:`eq_gauss_confidence` is one of the most used formula in statistics.  Let us close our discussion of statistics by implementing it.  For simplicity, we assume we are in the asymptotic regime.  Small values of $N$ should include the correct value of `t_star` obtained either programmatically or from a $t$-table.

```{.python .input}
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

## Summary

* Statistics focuses on inference problems, whereas deep learning emphasizes on making accurate predictions without explicitly programming and understanding.
* There are three common statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.
* There are three most common estimators: statistical bias, standard deviation, and mean square error.
* A confidence interval is an estimated range of a true population parameter that we can construct by given the samples.
* Hypothesis testing is a way of evaluating some evidence against the default statement about a population.


## Exercises

1. Let $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$, where "iid" stands for *independent and identically distributed*. Consider the following estimators of $\theta$:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Find the statistical bias, standard deviation, and mean square error of $\hat{\theta}.$
    * Find the statistical bias, standard deviation, and mean square error of $\tilde{\theta}.$
    * Which estimator is better?
1. For our chemist example in introduction, can you derive the 5 steps to conduct a two-sided hypothesis testing? Given the statistical significance level $\alpha = 0.05$ and the statistical power $1 - \beta = 0.8$.
1. Run the confidence interval code with $N=2$ and $\alpha = 0.5$ for $100$ independently generated dataset, and plot the resulting intervals (in this case `t_star = 1.0`).  You will see several very short intervals which are very far from containing the true mean $0$.  Does this contradict the interpretation of the confidence interval?  Do you feel comfortable using short intervals to indicate high precision estimates?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/419)
:end_tab:
