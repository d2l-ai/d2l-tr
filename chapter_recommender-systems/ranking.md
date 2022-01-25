# Tavsiye Sistemleri için Kişiselleştirilmiş Sıralama

Eski bölümlerde, sadece açık geribildirim dikkate alındı ve modeller gözlemlenen derecelendirmeler üzerinde eğitildi ve test edildi. Bu tür yöntemlerin iki dezavantajları vardır: Birincisi, çoğu geri bildirim açık değildir, ancak gerçek dünya senaryolarında örtük değildir ve açık geri bildirimlerin toplanması daha pahalı olabilir. İkincisi, kullanıcıların çıkarları için öngörücü olabilecek gözlenmeyen kullanıcı öğe çiftleri tamamen göz ardı edilir ve bu yöntemler, derecelendirmelerin rastgele eksik olmadığı durumlarda, kullanıcıların tercihleri nedeniyle uygun değildir. Gözlemlenmeyen kullanıcı öğesi çiftleri, gerçek negatif geri bildirim (kullanıcılar öğelerle ilgilenmez) ve eksik değerlerin bir karışımıdır (kullanıcı gelecekte öğelerle etkileşime girebilir). Biz sadece matris çarpanlara ve AutoRec gözlenmeyen çiftleri görmezden geliyoruz. Açıkçası, bu modeller gözlenen ve gözlemlenmeyen çiftler arasında ayrım yapamaz ve genellikle kişiselleştirilmiş sıralama görevleri için uygun değildir. 

Bu amaçla, örtük geri bildirimlerden sıralamalı öneri listeleri oluşturmayı hedefleyen bir öneri modeli sınıfı popülerlik kazanmıştır. Genel olarak, kişiselleştirilmiş sıralama modelleri noktasal, çift veya listwise yaklaşımlarla optimize edilebilir. Sivri yaklaşımlar, bir seferde tek bir etkileşimi dikkate alır ve bireysel tercihleri tahmin etmek için bir sınıflandırıcı veya bir regresörü eğitir. Matris çarpanları ve AutoRec noktasal hedeflerle optimize edilmiştir. İkili yaklaşımlar her kullanıcı için bir çift öğe dikkate alır ve bu çift için en uygun siparişi yaklaşık olarak değerlendirmeyi amaçlar. Genellikle, çift yönlü yaklaşımlar sıralama görevi için daha uygundur, çünkü göreli düzeni tahmin etmek sıralamanın niteliğini anımsatır. Listwise yaklaşımları, örneğin Normalleştirilmiş İndirimli Toplu Kazanç ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)) gibi sıralama önlemlerini doğrudan optimize etmek gibi öğelerin tüm listesinin sıralamasını yaklaşık olarak değerlendirir. Bununla birlikte, listwise yaklaşımları noktasal veya çift yönlü yaklaşımlardan daha karmaşık ve bilgi işlem yoğundur. Bu bölümde iki çift hedef/kayıp, Bayesian Kişiselleştirilmiş Sıralama kaybı ve Menteşe kaybı ve bunların ilgili uygulamaları sunulacaktır. 

## Bayesian Kişiselleştirilmiş Sıralama Kaybı ve Uygulaması

Bayesian kişiselleştirilmiş sıralama (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009`, maksimum arka tahmin ediciden elde edilen çift kişiselleştirilmiş bir sıralama kaybıdır. Birçok mevcut öneri modelinde yaygın olarak kullanılmaktadır. BPR'nin eğitim verileri hem pozitif hem de negatif çiftlerden (eksik değerler) oluşur. Kullanıcının, gözlemlenmeyen diğer tüm öğeler üzerinde olumlu öğeyi tercih ettiği varsayılır. 

Resmi olarak, eğitim verileri, $u$ kullanıcısı $u$ öğesinin $j$ öğesi üzerinde $i$ öğesini tercih ettiğini temsil eden $(u, i, j)$ şeklinde tuples tarafından oluşturulmuştur. Posterior olasılığın en üst düzeye çıkarılmasını amaçlayan BPR'nin Bayesian formülasyonu aşağıda verilmiştir: 

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

$\Theta$ keyfi bir öneri modelinin parametrelerini temsil ettiği yerde, $>_u$, $u$ kullanıcısı için tüm öğelerin istenen kişiselleştirilmiş toplam sıralamasını temsil eder. Kişiselleştirilmiş sıralama görevi için genel optimizasyon kriterini elde etmek için maksimum arka tahmin ediciyi formüle edebiliriz. 

$$
\begin{aligned}
\text{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$

$D := \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ eğitim setidir, $I^+_u$ kullanıcının $u$ sevdiği öğeleri, $I$ tüm öğeleri belirten $I$ ve $I \backslash I^+_u$, kullanıcının sevdiği öğeler hariç diğer tüm öğeleri belirten $I \backslash I^+_u$ ile 73229363614 numaralı kullanıcının $i$ ve 7322936363614'e tahmini puanlarıdır. $\hat{y}_{ui}$ ve $\hat{y}_{uj}$, $u$ kullanıcısının $u$ öğesine tahmini puanlarıdır. 20, sırasıyla. Önceki $p(\Theta)$, sıfır ortalama ve varyans kovaryans matrisi $\Sigma_\Theta$ ile normal bir dağılımdır. Burada, $\Sigma_\Theta = \lambda_\Theta I$ izin veriyoruz. 

! [Illustration of Bayesian Personalized Ranking](../img/rec-ranking.svg) Temel sınıfı `mxnet.gluon.loss.Loss` uygulayacak ve Bayesian kişiselleştirilmiş sıralama kaybını oluşturmak için `forward` yöntemini geçersiz kılacağız. Kayıp sınıfını ve np modülünü içe aktararak başlıyoruz.

```{.python .input  n=5}
from mxnet import gluon, np, npx
npx.set_np()
```

BPR kaybının uygulanması aşağıdaki gibidir.

```{.python .input  n=2}
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Menteşe Kaybı ve Uygulanması

Sıralama için Menteşe kaybı, SVM'ler gibi sınıflandırıcılarda sıklıkla kullanılan gluon kütüphanesinde sağlanan [menteşe kaybı](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) farklı bir şekle sahiptir. Tavsiye sistemlerinde sıralamada kullanılan kayıp aşağıdaki formdadır. 

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

burada $m$ güvenlik marjı boyutudur. Negatif öğeleri pozitif maddelerden uzaklaştırmayı amaçlıyor. BPR'ye benzer şekilde, mutlak çıkışlar yerine pozitif ve negatif numuneler arasındaki mesafeyi optimize etmeyi amaçlar ve önerici sistemlere çok uygun hale getirir.

```{.python .input  n=3}
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

Bu iki kayıp tavsiye kişiselleştirilmiş sıralama için değiştirilebilir. 

## Özet

- Tavsiye sistemlerinde kişiselleştirilmiş sıralama görevi için üç tür sıralama kaybı vardır, yani noktalı, çift yönlü ve listwise yöntemleri.
- İki çift kaybeder, Bayesian kişiselleştirilmiş sıralama kaybı ve menteşe kaybı, birbirinin yerine kullanılabilir.

## Egzersizler

- BPR ve menteşe kaybı varyantı var mı?
- BPR veya menteşe kaybı kullanan herhangi bir öneri modeli bulabilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:
