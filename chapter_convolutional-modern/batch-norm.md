# Toplu Normalleştirme
:label:`sec_batch_norm`

Derin sinir ağlarını eğitmek zordur. Üstelik makul bir süre içinde yakınsamalarını sağlamak çetrefilli olabilir. Bu bölümde, derin ağların :cite:`Ioffe.Szegedy.2015` yakınsamasını sürekli olarak hızlandıran popüler ve etkili bir teknik olan *toplu normalleştirme*yi tanıtıyoruz. Daha sonra :numref:`sec_resnet` içinde kapsanan artık bloklarla birlikte toplu normalleştirme, uygulayıcıların 100'den fazla katmanlı ağları rutin olarak eğitmelerini mümkün kılmıştır.

## Derin Ağları Eğitme

Toplu normalleştirmeyi anlamak için, özellikle makine öğrenmesi modellerini ve sinir ağlarını eğitirken ortaya çıkan birkaç pratik zorluğu gözden geçirelim.

İlk olarak, veri ön işleme ile ilgili seçimler genellikle nihai sonuçlarda muazzam bir fark yaratmaktadır. Ev fiyatlarını tahmin etmek için MLP'i uygulamamızı hatırlayın (:numref:`sec_kaggle_house`). Gerçek verilerle çalışırken ilk adımımız, girdi özniteliklerimizin her birinin sıfır ortalaması ve bir varyansı olması için standartlaştırmaktı. Sezgisel olarak, bu standartleştirme eniyileyicilerimizle uyumlu olur, çünkü parametreleri benzer ölçekte bir *önsel dağılıma* yerleştirir.

İkincisi, tipik bir MLP veya CNN için, eğittiğimiz gibi, ara katmanlardaki değişkenler (örneğin, MLP'deki afin dönüşüm çıktıları), geniş çapta değişen büyüklüklere sahip değerler alabilir: Model parametrelerinde hem girdiden çıktıya kadar olan katmanlar boyunca, hem de aynı katmandaki birimler arasında ve zaman içinde yaptığımız güncellemeler nedeniyle. Toplu normalleştirmenin mucitleri, bu tür değişkenlerin dağılımındaki bu kaymanın ağın yakınsamasını engelleyebileceğini gayri resmi olarak önerdi. Sezgisel olarak, bir katmanın değişken değerleri başka bir katmandakilerin 100 katı kadarsa, bunun öğrenme oranlarında telafi edici ayarlamaları gerektirebileceğini varsayıyoruz.

Üçüncü olarak, daha derin ağlar karmaşıktır ve aşırı öğrenmeye yatkındır. Bu, düzenlileştirmenin daha kritik hale geldiği anlamına gelir.

Toplu normalleştirme bireysel katmanlara uygulanır (isteğe bağlı olarak, hepsi için) ve şu şekilde çalışır: Her eğitim yinelemesinde, ilk olarak, ortalamalarını çıkararak ve standart sapmaları ile bölerek (toplu normalleştirme), ki her ikisi de mevcut minigrup istatistiklerinden tahmin edilir, girdileri normalleştiririz. Daha sonra, bir ölçek katsayısı ve bir ölçek ofseti (sabit kaydırma değeri) uyguluyoruz. Tam olarak, *toplu normalleştirme* adı *toplu* istatistiklerine dayanan bu *normalleştirme*den kaynaklanmaktadır.

Boyutu 1 olan minigruplarla toplu normalleştirmeyi uygulamaya çalışırsak, hiçbir şey öğrenemeyeceğimizi unutmayın. Bunun nedeni, ortalamayı çıkardıktan sonra, her gizli birimin 0 değerini alacak olmasıdır! Tahmin edebileceğiniz gibi, ki bu nedenle toplu normalleştirmeye koca bir bölümü ayırıyoruz, yeterince büyük minigruplarla, yaklaşım etkili ve istikrarlı olduğunu kanıtlıyor. Buradan alınması gereken esas, toplu normalleştirmeyi uygularken, grup boyutunun seçiminin toplu normalleştirmesiz durumda olduğundan daha önemli olabileceğidir.

Biçimsel olarak, bir minigrup olan $\mathcal{B}$'ye dahil olan $\mathbf {x} \in \mathcal {B}$, toplu normalleştirmeye ($\mathrm{BN}$) girdi olursa, toplu normalleştirme $\mathbf{x}$ aşağıdaki ifadeye dönüşür:

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$ 
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm` denkleminde, $\hat{\boldsymbol{\mu}}_\mathcal{B}$ örneklem ortalaması ve $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ minigrup $\mathcal{B}$'nın örneklem standart sapmasıdır. Standartlaştırma uygulandıktan sonra, ortaya çıkan minigrup sıfır ortalama ve birim varyansa sahiptir. Birim varyans seçimi (diğer bazı sihirli sayılara karşı) keyfi bir seçim olduğundan, genel olarak eleman-yönlü  *ölçek parametresi* $\boldsymbol{\gamma}$'yı ve *kayma parametresi*  $\boldsymbol{\beta}$'yı dahil ederiz ve onlar $\mathbf{x}$ ile aynı şekle sahiptirler. $\boldsymbol{\gamma}$ ve $\boldsymbol{\beta}$'in diğer model parametreleriyle birlikte öğrenilmesi gereken parametreler olduğunu unutmayın.

Sonuç olarak, ara katmanlar için değişken büyüklükleri eğitim sırasında ıraksamaz, çünkü toplu normalleştirme bunları belirli bir ortalama ve boyuta ($\hat{\boldsymbol{\mu}}_\mathcal{B}$ ve ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ üzerinden) aktif olarak ortalar ve yeniden ölçeklendirir. Uygulayıcının sezgi veya bilgeliğinin bir parçası, toplu normalleştirmenin daha saldırgan öğrenme oranlarına izin vermesi gibi görünmesidir.

Resmi olarak, :eqref:`eq_batchnorm` denklemindeki $\hat{\boldsymbol{\mu}}_\mathcal{B}$ ve ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$'yi aşağıdaki gibi hesaplıyoruz:

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

Deneysel varyans tahmininin kaybolabileceği durumlarda bile sıfıra bölmeyi denemediğimizden emin olmak için varyans tahminine küçük bir sabit $\epsilon > 0$ eklediğimizi unutmayın. $\hat{\boldsymbol{\mu}}_\mathcal{B}$ ve ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ tahminleri, gürültülü ortalama ve varyans tahminleri kullanarak ölçekleme sorununa karşı koymaktadır. Bu gürültücülüğün bir sorun olması gerektiğini düşünebilirsiniz. Anlaşılacağı gibi, bu aslında faydalıdır.

Bunun derin öğrenmede yinelenen bir tema olduğu ortaya çıkıyor. Teorik olarak henüz iyi ortaya çıkarılamayan nedenlerden dolayı, eniyilemedeki çeşitli gürültü kaynakları genellikle daha hızlı eğitime ve daha az aşırı öğrenmeye neden olur: Bu varyasyon bir düzenlileştirme biçimi olarak kendini gösteriyor. Bazı ön araştırmalarda, :cite:`Teye.Azizpour.Smith.2018` ve :cite:`Luo.Wang.Shao.ea.2018`, toplu normalleşmenin özelliklerini sırasıyla Bayesian önselleri ve cezaları ile ilişkilendiriyor. Özellikle, bu durum $50 \sim 100$ aralığındaki orta boy minigrup boyutları için toplu normalleştirmenin neden en iyi şekilde çalıştığı bulmacasına biraz ışık tutuyor.

Eğitilmiş bir modeli sabitlerken, ortalama ve varyansı tahmin etmek için tüm veri kümesini kullanmayı tercih edeceğimizi düşünebilirsiniz. Eğitim tamamlandıktan sonra, dahil olduğu gruba bağlı olarak neden aynı imgenin farklı şekilde sınıflandırılmasını isteyelim? Eğitim sırasında, modelimizi her güncellediğimizde tüm veri örnekleri için ara değişkenler değiştiği için bu mutlak hesaplama uygulanabilir değildir. Bununla birlikte, model eğitildikten sonra, her katmanın değişkenlerinin ortalamalarını ve varyanslarını tüm veri kümesine dayalı olarak hesaplayabiliriz. Aslında bu, toplu normalleştirme kullanan modeller için standart bir uygulamadır ve böylece toplu normalleştirme katmanları *eğitim modu* (minigrup istatistiklerine göre normalleştirme) ve *tahmin modunda* (veri kümesi istatistiklerine göre normalleştirme) farklı şekilde çalışır.

Şimdi toplu normalleştirmenin pratikte nasıl çalıştığına bir göz atmaya hazırız.

## Toplu Normalleştirme Katmanları

Tam bağlı katmanlar ve evrişimli katmanlar için toplu normalleştirme uygulamaları biraz farklıdır. Her iki durumu aşağıda tartışıyoruz. Toplu normalleştirme ve diğer katmanlar arasındaki önemli bir farkın olduğunu hatırlayın; toplu normalleştirme bir seferinde bir minigrup üzerinde çalıştığı için, diğer katmanlarına ilerlerken daha önce yaptığımız gibi toplu iş boyutunu göz ardı edemeyiz.

### Tam Bağlı Katmanlar

Tam bağlı katmanlara toplu normalleştirme uygularken, orijinal makale, toplu normalleştirmeyi afin dönüşümden sonra ve doğrusal olmayan etkinleştirme işlevinden önce ekler (daha sonraki uygulamalar etkinleştirme işlevlerinden hemen sonra toplu normalleştirme ekleyebilir) :cite:`Ioffe.Szegedy.2015`. $\mathbf{x}$ ile tam bağlı katmana girdisi, $\mathbf{W}\mathbf{x} + \mathbf{b}$ afin dönüşümü (ağırlık parametresi $\mathbf{W}$ ve ek girdi parametresi $\mathbf{b}$ ile) ve $\phi$ ile etkinleştirme fonksiyonunu ifade ederse, tam bağlı bir katman çıktısının, $\mathbf{h}$, toplu normalleştirme etkinleştirilmiş hesaplanmasını aşağıdaki gibi ifade edebiliriz:

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Ortalama ve varyansın dönüşümün uygulandığı *aynı* minigrup üzerinde hesaplandığını hatırlayın.

### Evrişimli Katmanlar

Benzer şekilde, evrişimli katmanlarla, evrişimden sonra ve doğrusal olmayan etkinleştirme işlevinden önce toplu normalleştirme uygulayabiliriz. Evrişim birden fazla çıktı kanalı olduğunda, bu kanalların çıktılarının *her biri* için toplu normalleştirme gerçekleştirmemiz gerekir ve her kanalın kendi ölçek ve kayma parametreleri vardır, bunların her ikisi de skalerdir. Minigruplarımızın $m$ örnekleri içerdiğini ve her kanal için evrişim çıktısının $p$ yüksekliği ve $q$ genişliği olduğunu varsayalım. Evrişimli katmanlar için, çıktı kanalı başına $m \cdot p \cdot q$ eleman üzerinde toplu normalleştirmeyi aynı anda gerçekleştiriyoruz. Böylece, ortalama ve varyansı hesaplarken tüm mekansal konumlar üzerinde değerleri biraraya getiririz ve sonuç olarak her mekansal konumdaki değeri normalleştirmek için belirli bir kanal içinde aynı ortalama ve varyansı uygularız.

### Tahmin Sırasında Toplu Normalleştirme

Daha önce de belirttiğimiz gibi, toplu normalleştirme genellikle eğitim modunda ve tahmin modunda farklı davranır. İlk olarak, her birinin minigruplardan tahmin edilmesinden kaynaklanan örneklem ortalamasındaki ve örneklem varyansındaki gürültü, modeli eğittikten sonra artık arzu edilen birşey değildir. İkincisi, iş başına toplu normalleştirme istatistiklerini hesaplama lüksüne sahip olmayabiliriz. Örneğin, modelimizi bir seferde bir öngörü yapmak için uygulamamız gerekebilir.

Genellikle, eğitimden sonra, değişken istatistiklerin kararlı tahminlerini hesaplamak ve daha sonra bunları tahmin zamanında sabitlemek için veri kümesinin tamamını kullanırız. Sonuç olarak, toplu normalleştirme, eğitim sırasında ve test zamanında farklı davranır. Hattan düşürmenin de bu özelliği sergilediğini hatırlayın.

## (**Sıfırdan Uygulama**)

Aşağıda, tensörlerle sıfırdan bir toplu normalleştirme tabakası uyguluyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Mevcut modun eğitim modu mu yoksa tahmin modu mu olduğunu
    # belirlemek için `autograd`'ı kullan
    if not autograd.is_training():
        # Tahmin modu ise, hareketli ortalama ile elde edilen ortalama
        # ve varyansı doğrudan kullan
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Tam bağlı bir katman kullanırken, öznitelik 
            # boyutundaki ortalamayı ve varyansı hesapla
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # İki boyutlu bir evrişim katmanı kullanırken, kanal 
            # boyutundaki (axis=1) ortalamayı ve varyansı hesaplayın. 
            # Burada, yayın işleminin daha sonra gerçekleştirilebilmesi 
            # için `X`'in şeklini korumamız gerekiyor.
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # Eğitim modunda, standardizasyon için mevcut ortalama ve 
        # varyans kullanılır
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Hareketli ortalamayı kullanarak ortalamayı ve varyansı güncelle
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Ölçek ve kaydırma
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Mevcut modun eğitim modu mu yoksa tahmin modu mu olduğunu
    # belirlemek için `is_grad_enabled`'ı kullanın
    if not torch.is_grad_enabled():
        # Tahmin modu ise, hareketli ortalama ile elde edilen ortalama
        # ve varyansı doğrudan kullan
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Tam bağlı bir katman kullanırken, öznitelik 
            # boyutundaki ortalamayı ve varyansı hesapla
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # İki boyutlu bir evrişim katmanı kullanırken, kanal 
            # boyutundaki (axis=1) ortalamayı ve varyansı hesaplayın. 
            # Burada, yayın işleminin daha sonra gerçekleştirilebilmesi 
            # için `X`'in şeklini korumamız gerekiyor.
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # Eğitim modunda, standardizasyon için mevcut ortalama ve 
        # varyans kullanılır
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Hareketli ortalamayı kullanarak ortalamayı ve varyansı güncelle
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Ölçek ve kaydırma
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Hareketli varyansın karekökünün tersini eleman-yönlü hesaplayın
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Ölçek ve kaydırma
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

[**Artık uygun bir `BatchNorm` katmanı oluşturabiliriz.**] Katmanımız `gamma` ölçeği ve `beta` kayması için uygun parametreleri koruyacaktır, bunların her ikisi de eğitim sırasında güncellenecektir. Ayrıca, katmanımız modelin tahmini sırasında sonraki kullanım için ortalamaların ve varyansların hareketli ortalamalarını koruyacaktır.

Algoritmik ayrıntıları bir kenara bırakırsak, katmanın uygulanmasının altında yatan tasarım desenine dikkat edin. Tipik olarak, matematiği ayrı bir işlevde tanımlarız, varsayalım `batch_norm`. Daha sonra bu işlevselliği, verileri doğru cihazın bağlamına taşıma, gerekli değişkenleri tahsis etme ve ilkleme, hareketli ortalamaları takip etme (ortalama ve varyans için) vb. gibi çoğunlukla kayıt tutma konularını ele alan özel bir katmana kaynaştırıyoruz. Bu model, matematiğin basmakalıp koddan temiz bir şekilde ayrılmasını sağlar. Ayrıca, kolaylık sağlamak için burada girdi şeklini otomatik olarak çıkarma konusunda endişelenmediğimizi, bu nedenle özniteliklerin sayısını belirtmemiz gerektiğini unutmayın. Kaygılanmayın, derin öğrenme çerçevesindeki üst düzey toplu normalleştirme API'leri bunu bizim için halledecek; bunu daha sonra göreceğiz.

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: Tam bağlı bir katman için çıktıların sayısı veya 
    # evrişimli bir katman için çıktı kanallarının sayısı. 
    # `num_dims`: Tam bağlı katman için 2 ve evrişimli katman için 4
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Ölçek parametresi ve kaydırma parametresi (model parametreleri) 
        # sırasıyla 1 ve 0 olarak ilklenir
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # Model parametresi olmayan değişkenler 0 ve 1 olarak ilklenir.
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # Ana bellekte `X` yoksa, `moving_mean` ve `moving_var`'ı `X`in 
        # bulunduğu cihaza kopyalayın
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Güncellenen `moving_mean` ve `moving_var`'ı kaydedin
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: Tam bağlı bir katman için çıktıların sayısı veya 
    # evrişimli bir katman için çıktı kanallarının sayısı. 
    # `num_dims`: Tam bağlı katman için 2 ve evrişimli katman için 4
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # Ölçek parametresi ve kaydırma parametresi (model parametreleri) 
        # sırasıyla 1 ve 0 olarak ilklenir
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # Model parametresi olmayan değişkenler 0 ve 1 olarak ilklenir.
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # Ana bellekte `X` yoksa, `moving_mean` ve `moving_var`'ı `X`in 
        # bulunduğu cihaza kopyalayın
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Güncellenen `moving_mean` ve `moving_var`'ı kaydedin
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # Ölçek parametresi ve kaydırma parametresi (model parametreleri) 
        # sırasıyla 1 ve 0 olarak ilklenir
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # Model parametresi olmayan değişkenler 0 olarak ilklenir.
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

## [**LeNet'te Toplu Normalleştirme Uygulaması**]

Bağlamda `BatchNorm`'un nasıl uygulanacağını görmek için, aşağıda geleneksel bir LeNet modeline (:numref:`sec_lenet`) uyguluyoruz. Toplu normalleştirmenin, evrişimli katmanlardan veya tam bağlı katmanlardan sonra, ancak karşılık gelen etkinleştirme işlevlerinden önce uygulandığını hatırlayın.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Sahip olduğumuz CPU/GPU cihazlarını kullanabilmek için model oluşturma 
# veya derlemenin `strategy.scope()` içinde olması için bunun 
# `d2l.train_ch6`ya geçirilecek bir fonksiyon olması gerektiğini hatırlayın.
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

Daha önce olduğu gibi, [**ağımızı Fashion-MNIST veri kümesi üzerinde eğiteceğiz**]. Bu kod, LeNet'i ilk eğittiğimizdeki ile neredeyse aynıdır (:numref:`sec_lenet`). Temel fark, daha büyük öğrenme oranıdır.

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

[**`gamma` ölçek parametresine**] ve ilk toplu normalleştirme katmanından öğrenilen [**`beta` kayma parametresine**] bir göz atalım.

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

## [**Kısa Uygulama**]

Kendimizi tanımladığımız `BatchNorm` sınıfıyla karşılaştırıldığında, doğrudan derin öğrenme çerçevesinden üst seviye API'lerde tanımlanan `BatchNorm` sınıfını kullanabiliriz. Kod, yukarıdaki uygulamamız ile hemen hemen aynı görünüyor.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

Aşağıda, [**modelimizi eğitmek için aynı hiper parametreleri kullanıyoruz.**] Özel uygulamamız Python tarafından yorumlanılır iken, her zamanki gibi üst seviye API sürümünün kodu C++ veya CUDA için derlendiğinden, çok daha hızlı çalıştığını unutmayın.

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tartışma

Sezgisel olarak, toplu normalleştirmenin eniyileme alanını daha pürüzsüz hale getirdiği düşünülmektedir. Bununla birlikte, derin modelleri eğitirken gözlemlediğimiz olgular için kurgusal sezgiler ve gerçek açıklamalar arasında ayrım yaparken dikkatli olmalıyız. İlk etapta daha basit derin sinir ağlarının (MLP'ler ve geleneksel CNN'ler) neden genelleştirildiğini bile bilmediğimizi hatırlayın. Hattan düşürme ve ağırlık sönümü ile bile, görünmeyen verilere genelleme kabiliyetleri geleneksel öğrenme-kuramsal genelleme garantileri ile açıklanamayacak kadar esnek kalırlar.

Toplu normalleştirmeyi öneren orijinal çalışmada, yazarlar, güçlü ve kullanışlı bir araç tanıtmanın yanı sıra, neden çalıştığına dair bir açıklama sundular: *Dahili eşdeğişken kayması*nı (internal covariate shift) azaltmak. Muhtemelen *dahili eşdeğişken kayması* ile yazarlar, yukarıda ifade edilen sezgiye benzer bir şey ifade ettiler - değişken değerlerinin dağılımının eğitim boyunca değiştiği düşüncesi. Bununla birlikte, bu açıklamayla ilgili iki mesele vardı: i) Bu değişim, *eşdeğişken kayması*ndan çok farklıdır, bir yanlış isimlendirmeye yol açar. ii) Açıklama az belirtilmiş bir sezgi sunar ancak *neden tam olarak bu teknik çalışır* sorusunu titiz bir açıklama isteyen açık bir soru olarak bırakır. Bu kitap boyunca, uygulayıcıların derin sinir ağlarının gelişimine rehberlik etmek için kullandıkları sezgileri aktarmayı amaçlıyoruz. Bununla birlikte, bu rehber sezgileri yerleşik bilimsel gerçeklerden ayırmanın önemli olduğuna inanıyoruz. Sonunda, bu materyale hakim olduğunuzda ve kendi araştırma makalelerinizi yazmaya başladığınızda, teknik iddialar ve önseziler arasında konumunuzu bulmak için net olmak isteyeceksiniz.

Toplu normalleştirmenin başarısını takiben, başarısının *dahili eşdeğişken kayması* açısından açıklanması, teknik yazındaki tartışmalarda ve makine öğrenmesi araştırmasının nasıl sunulacağı konusunda daha geniş bir söylemde defalarca ortaya çıkmıştır. 2017 NeurIPS konferansında Zaman Testi Ödülü'nü kabul ederken verdiği unutulmaz bir konuşmada Ali Rahimi, modern derin öğrenme pratiğini simyaya benzeten bir argümanda odak noktası olarak *dahili eşdeğişken kayması*nı kullandı. Daha sonra, örnek makine öğrenmesindeki sıkıntılı eğilimleri özetleyen bir konum kağıdında (position paper) ayrıntılı olarak yeniden gözden geçirildi :cite:`Lipton.Steinhardt.2018`. Diğer yazarlar toplu normalleştirmenin başarısı için alternatif açıklamalar önerdiler, bazıları toplu normalleştirmenin başarısının bazı yönlerden orijinal makalede :cite:`Santurkar.Tsipras.Ilyas.ea.2018` iddia edilenlerin tersi olan davranışları sergilemesinden geldiğini iddia ettiler.

*Dahili eşdeğişken kayması*nın, teknik makine öğrenmesi yazınında benzer şekilde her yıl yapılan belirsiz iddiaların herhangi birinden daha fazla eleştiriye layık olmadığını not ediyoruz. Muhtemelen, bu tartışmaların odak noktası olarak tınlamasını (rezonans), hedef kitledeki geniş tanınabilirliğine borçludur. Toplu normalleştirme, neredeyse tüm konuşlandırılmış imge sınıflandırıcılarında uygulanan vazgeçilmez bir yöntem olduğunu kanıtlamıştır, tekniği tanıtan makaleye on binlerce atıf kazandırmıştır.

## Özet

* Model eğitimi sırasında, toplu normalleştirme, minigrubun ortalama ve standart sapmasını kullanarak sinir ağının ara çıktısını sürekli olarak ayarlar, böylece sinir ağı boyunca her katmandaki ara çıktının değerleri daha kararlı olur.
* Tam bağlı katmanlar ve evrişimli katmanlar için toplu normalleştirme yöntemleri biraz farklıdır.
* Bir hattan düşürme katmanı gibi, toplu normalleştirme katmanları da eğitim modunda ve tahmin modunda farklı hesaplama sonuçlarına sahiptir.
* Toplu normalleştirmenin, öncelikle düzenlileştirme olmak üzere birçok yararlı yan etkisi vardır. Öte yandan, orijinal motivasyondaki dahili eşdeğişken kaymasını azaltma geçerli bir açıklama gibi görünmüyor.

## Alıştırmalar

1. Toplu normalleştirmeden önce ek girdi parametresini tam bağlı katmandan veya evrişimli katmandan kaldırabilir miyiz? Neden?
1. Toplu normalleştirme olan ve olmayan LeNet için öğrenme oranlarını karşılaştırın.
    1. Eğitim ve test doğruluğundaki artışı çizin.
    1. Öğrenme oranını ne kadar büyük yapabilirsiniz?
1. Her katmanda toplu normalleştirmeye ihtiyacımız var mı? Deneyle gözlemleyebilir misiniz?
1. Toplu normalleştirme ile hattan düşürmeyi yer değiştirebilir misiniz? Davranış nasıl değişir?
1. `beta` ve `gamma` parametrelerini sabitleyin, sonuçları gözlemleyin ve analiz edin.
1. Toplu normalleştirmenin diğer uygulamalarını görmek amacıyla üst seviye API'lerden `BatchNorm` için çevrimiçi belgelerini gözden geçirin.
1. Araştırma fikirleri: Uygulayabileceğiniz diğer normalleştirme dönüşümleri düşünün? Olasılık integral dönüşümü uygulayabilir misiniz? Tam kerteli kovaryans tahminine ne dersiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/330)
:end_tab:
