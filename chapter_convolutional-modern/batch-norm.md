# Toplu Normalleştirme
:label:`sec_batch_norm`

Derin sinir ağlarını eğitmek zordur. Üstelik makul bir süre içinde yakınsamalarını sağlamak çetrefilli olabilir. Bu bölümde, derin ağların :cite:`Ioffe.Szegedy.2015` yakınsamasını sürekli olarak hızlandıran popüler ve etkili bir teknik olan *toplu normalleştirme*'yi tanıtıyoruz. Daha sonra :numref:`sec_resnet`'te kapsanan artık bloklarla birlikte toplu normalleştirme, uygulayıcıların 100'den fazla katmanlı ağları rutin olarak eğitmelerini mümkün kılmıştır.

## Derin Ağları Eğitme

Toplu normalleştirmeyi motive etmek için, özellikle makine öğrenimi modellerini ve sinir ağlarını eğitirken ortaya çıkan birkaç pratik zorluğu gözden geçirelim.

İlk olarak, veri önişleme ile ilgili seçimler genellikle nihai sonuçlarda muazzam bir fark yaratmaktadır. Ev fiyatlarını tahmin etmek için MLP'lerin uygulamamızı hatırlayın (:numref:`sec_kaggle_house`). Gerçek verilerle çalışırken ilk adımımız, giriş özelliklerimizin her birinin sıfır ortalaması ve bir varyansı olması için standartlaştırmaktı. Sezgisel olarak, bu standardizasyon optimizatörlerimizle iyi oynar, çünkü parametreleri benzer bir ölçekte bir öncelikli* koyar.

İkincisi, tipik bir MLP veya CNN için, antrenman yaptığımız gibi, ara katmanlardaki değişkenler (örneğin, MLP'deki afin dönüşüm çıkışları), geniş çapta değişen büyüklüklere sahip değerler alabilir: hem girişten çıkışa katmanlar boyunca, aynı katmandaki birimler arasında hem de modele yapılan güncellemelerimiz nedeniyle zamanla parametreleri. Toplu normalleştirmenin mucitleri, bu tür değişkenlerin dağılımında bu sürüklenmenin ağın yakınsamasını engelleyebileceğini gayri resmi olarak önerdi. Sezgisel olarak, bir katmanın başka bir katmanın 100 katı değişken değerleri varsa, bunun öğrenme oranlarında telafi edici ayarlamaları gerektirebileceğini varsayıyoruz.

Üçüncü olarak, daha derin ağlar karmaşık ve kolayca aşırı uydurma yeteneğine sahiptir. Bu, düzenliliğin daha kritik hale geldiği anlamına gelir.

Toplu normalleştirme bireysel katmanlara uygulanır (isteğe bağlı olarak, hepsi için) ve aşağıdaki gibi çalışır: Her eğitim yinelemesinde, ilk olarak, ortalamalarını çıkararak ve her ikisi de istatistiklerine göre tahmin edilmektedir standart sapma ile bölünerek (toplu normalleştirme) girişleri normalleştirmek Mevcut minibatch. Daha sonra, bir ölçek katsayısı ve bir ölçek ofseti uyguluyoruz. Tam olarak, *toplu normalleştirme* adını türeten *toplu işlem* istatistiklerine dayanan bu *normalizasyon* kaynaklanmaktadır.

Boyut 1 minibatch'lerle toplu normalleştirmeyi uygulamaya çalışırsak, hiçbir şey öğrenemeyeceğimizi unutmayın. Bunun nedeni, araçları çıkardıktan sonra, her gizli birim 0 değerini alacaktı! Tahmin edebileceğiniz gibi, toplu normalleştirmeye bütün bir bölümü ayırdığımızdan, yeterince büyük minibatch'lerle, yaklaşım etkili ve istikrarlı bir şekilde kanıtlıyor. Burada bir paket, toplu normalleştirmeyi uygularken, parti boyutunun seçiminin toplu normalleştirmeden daha önemli olabileceğidir.

Resmi olarak, $\mathbf{x} \in \mathcal{B}$ tarafından bir minibatch $\mathcal{B}$ olan toplu normalleştirme ($\mathrm{BN}$) için bir giriş gösteren toplu normalleştirme $\mathbf{x}$ aşağıdaki ifadeye göre dönüştürür:

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm`'te, $\hat{\boldsymbol{\mu}}_\mathcal{B}$ örnek ortalaması ve $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ minibatch $\mathcal{B}$'nın örnek standart sapması olur. Standardizasyon uygulandıktan sonra, ortaya çıkan minibatch sıfır ortalama ve birim varyansa sahiptir. Birim varyansı seçimi (diğer bazı sihirli sayılara karşı) keyfi bir seçim olduğundan, genel olarak eleman olarak dahil ederiz
*ölçek parametresi* $\boldsymbol{\gamma}$ ve*shift parametresi* $\boldsymbol{\beta}$
$\mathbf{x}$ ile aynı şekle sahip. $\boldsymbol{\gamma}$ ve $\boldsymbol{\beta}$'in diğer model parametreleriyle birlikte öğrenilmesi gereken parametreler olduğunu unutmayın.

Sonuç olarak, ara katmanlar için değişken büyüklükleri eğitim sırasında ayrılamaz, çünkü toplu normalleştirme bunları belirli bir ortalama ve boyuta ($\hat{\boldsymbol{\mu}}_\mathcal{B}$ ve ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ üzerinden) aktif olarak merkezler ve yeniden ölçeklendirir. Uygulayıcının sezgi veya bilgeliğinin bir parçası, toplu normalleştirmenin daha agresif öğrenme oranlarına izin vermesi gibi görünmesidir.

Resmi olarak, :eqref:`eq_batchnorm` içinde $\hat{\boldsymbol{\mu}}_\mathcal{B}$ ve ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ aşağıdaki gibi hesaplıyoruz:

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

Ampirik varyans tahmininin kaybolabileceği durumlarda bile sıfıra bölünmeyi denemediğimizden emin olmak için varyans tahminine küçük bir sabit $\epsilon > 0$ eklediğimizi unutmayın. $\hat{\boldsymbol{\mu}}_\mathcal{B}$ ve ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ tahminleri, gürültülü ortalama ve varyans tahminleri kullanarak ölçekleme sorununa karşı koymaktadır. Bu gürültüsünüzün bir sorun olması gerektiğini düşünebilirsiniz. Anlaşıldığı gibi, bu aslında faydalıdır.

Bu derin öğrenmede yinelenen bir tema olduğu ortaya çıkıyor. Teorik olarak henüz iyi karakterize edilmeyen nedenlerden dolayı, optimizasyondaki çeşitli gürültü kaynakları genellikle daha hızlı eğitime ve daha az aşırı takılmaya neden olur: bu varyasyon bir düzenlilik biçimi olarak hareket eder. Bazı ön araştırmalarda, :cite:`Teye.Azizpour.Smith.2018` ve :cite:`Luo.Wang.Shao.ea.2018`, toplu normalleşmenin özelliklerini sırasıyla Bayesian sabıkası ve cezaları ile ilişkilendirir. Özellikle, bu durum $50 \sim 100$ aralığındaki orta boy mini batches boyutları için toplu normalleştirmenin neden en iyi şekilde çalıştığını bulmacaya biraz ışık tutuyor.

Eğitimli bir modeli tamir ederken, ortalama ve varyansı tahmin etmek için tüm veri kümesini kullanmayı tercih edeceğimizi düşünebilirsiniz. Eğitim tamamlandıktan sonra, ikamet ettiği partiye bağlı olarak neden aynı görüntünün farklı şekilde sınıflandırılmasını isteriz? Eğitim sırasında, modelimizi her güncellediğimizde tüm veri örnekleri için ara değişkenler değiştiği için bu kesin hesaplama mümkün değildir. Bununla birlikte, model eğitildikten sonra, her katmanın değişkenlerinin araçlarını ve varyanslarını tüm veri kümesine dayalı olarak hesaplayabiliriz. Aslında bu, toplu normalleştirme kullanan modeller için standart bir uygulamadır ve böylece toplu normalleştirme katmanları *eğitim modu* (minibatch istatistiklerine göre normalleştirme) ve *tahmin modunda* (veri kümesi istatistiklerine göre normalleştirme) farklı şekilde çalışır.

Şimdi toplu normalleştirmenin pratikte nasıl çalıştığına bir göz atmaya hazırız.

## Toplu Normalleştirme Katmanları

Tam bağlı katmanlar ve kıvrımsal katmanlar için toplu normalleştirme uygulamaları biraz farklıdır. Her iki davada aşağıda tartışıyoruz. Toplu normalleştirme ve diğer katmanlar arasındaki önemli bir farkın, toplu normalleştirme aynı anda tam bir mini batch üzerinde çalıştığı için, diğer katmanları tanıtırken daha önce yaptığımız gibi toplu iş boyutunu göz ardı edemeyiz olduğunu hatırlayın.

### Tam Bağlı Katmanlar

Tam bağlı katmanlara toplu normalleştirme uygularken, orijinal kağıt, afin dönüşümden sonra ve doğrusal olmayan aktivasyon işlevinden önce toplu normalleştirmeyi ekler (daha sonraki uygulamalar etkinleştirme işlevlerinden hemen sonra toplu normalleştirme ekleyebilir) :cite:`Ioffe.Szegedy.2015`. $\mathbf{x}$ ile tam bağlı katmana girişi, $\mathbf{W}\mathbf{x} + \mathbf{b}$ afin dönüşümü (ağırlık parametresi $\mathbf{W}$ ve önyargı parametresi $\mathbf{b}$ ile) ve $\phi$ ile aktivasyon fonksiyonunu ifade ederek, toplu normalleştirme etkin, tam bağlı bir katman çıkışının hesaplanmasını ifade edebiliriz $\mathbf{h}$ aşağıdaki gibi:

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Dönüşümün uygulandığı *same* mini batch üzerinde ortalama ve varyans hesaplanır hatırlayın.

### Konvolüsyonel Katmanlar

Benzer şekilde, evrimsel katmanlarla, konvolüsyondan sonra ve doğrusal olmayan aktivasyon işlevinden önce toplu normalleştirme uygulayabiliriz. Evrişim birden fazla çıkış kanalı olduğunda, bu kanalların çıkışlarının*her bir* için toplu normalleştirme gerçekleştirmemiz gerekir ve her kanalın kendi ölçek ve kaydırma parametreleri vardır, bunların her ikisi de skaler. Minibatch'lerimizin $m$ örnekleri içerdiğini ve her kanal için evrimin çıkışının $p$ ve genişliği $q$ olduğunu varsayalım. Evrimsel katmanlar için, çıkış kanalı başına $m \cdot p \cdot q$ eleman üzerinde her parti normalleştirmesini aynı anda gerçekleştiriyoruz. Böylece, ortalama ve varyansı hesaplarken tüm mekansal konumlar üzerinde değerleri toplarız ve sonuç olarak her mekansal konumdaki değeri normalleştirmek için belirli bir kanal içinde aynı ortalama ve varyansı uygularız.

### Tahmin Sırasında Toplu Normalleştirme

Daha önce de belirttiğimiz gibi, toplu normalleştirme genellikle eğitim modunda ve tahmin modunda farklı davranır. İlk olarak, numune ortalamadaki gürültü ve minibatchlerde her birinin tahmin edilmesinden kaynaklanan örnek varyansı, modeli eğittikten sonra artık arzu edilmez. İkincisi, toplu iş başına normalleştirme istatistiklerini hesaplama lüksüne sahip olmayabiliriz. Örneğin, bir seferde bir öngörü yapmak için modelimizi uygulamamız gerekebilir.

Genellikle, eğitimden sonra, değişken istatistiklerin kararlı tahminlerini hesaplamak ve daha sonra bunları tahmin zamanında düzeltmek için veri kümesinin tamamını kullanırız. Sonuç olarak, toplu normalleştirme, eğitim sırasında ve test zamanında farklı davranır. Bırakmanın da bu özelliği sergilediğini hatırlayın.

## Sıfırdan Uygulama

Aşağıda, tensörlerle sıfırdan bir parti normalleştirme tabakası uyguluyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance element-wise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

Artık uygun bir `BatchNorm` katmanı oluşturabiliriz. Katmanımız `gamma` ölçeği ve `beta` kayması için uygun parametreleri koruyacaktır, bunların her ikisi de eğitim sırasında güncellenecektir. Ayrıca, katmanımız model tahmini sırasında sonraki kullanım için araçların ve varyansların hareketli ortalamalarını koruyacaktır.

Algoritmik ayrıntıları bir kenara bırakarak, katmanın uygulanmasının altında yatan tasarım desenine dikkat edin. Tipik olarak, matematiği ayrı bir işlevde tanımlarız, diyelim ki `batch_norm`. Daha sonra bu işlevselliği, verileri doğru cihaz bağlamına taşıma, gerekli değişkenleri tahsis etme ve başlatma, hareketli ortalamaları takip etme (ortalama ve varyans için burada), vb. gibi çoğunlukla defter tutma konularını ele alan özel bir katmana entegre ediyoruz. Bu model, matematiğin boilerplate kodundan temiz bir şekilde ayrılmasını sağlar. Ayrıca, kolaylık sağlamak için burada giriş şeklini otomatik olarak çıkarma konusunda endişelenmediğimizi, bu nedenle özelliklerin sayısını belirtmemiz gerektiğini unutmayın. Endişelenmeyin, derin öğrenme çerçevesindeki üst düzey toplu normalleştirme API'leri bunu bizim için halledecek ve bunu daha sonra göstereceğiz.

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.zeros(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
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
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.zeros,
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

## LeNet'te Toplu Normalleştirme Uygulaması

Bağlamda `BatchNorm`'in nasıl uygulanacağını görmek için, aşağıda geleneksel bir LeNet modeline (:numref:`sec_lenet`) uyguluyoruz. Toplu normalleştirmenin, konvolusyonel katmanlardan veya tam bağlı katmanlardan sonra ancak karşılık gelen etkinleştirme işlevlerinden önce uygulandığını hatırlayın.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
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
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
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

Daha önce olduğu gibi, ağımızı Moda-MNIST veri kümesi üzerinde eğiteceğiz. Bu kod, LeNet'i ilk eğittiğimizde neredeyse aynıdır (:numref:`sec_lenet`). Temel fark, önemli ölçüde daha büyük öğrenme oranıdır.

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

`gamma` ölçek parametresine ve ilk parti normalleştirme katmanından öğrenilen `beta` shift parametresine bir göz atalım.

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

## Özlü Uygulama

Kendimizi tanımladığımız `BatchNorm` sınıfıyla karşılaştırıldığında, doğrudan derin öğrenme çerçevesinden üst düzey API'lerde tanımlanan `BatchNorm` sınıfını kullanabiliriz. Kod, uygulamamız yukarıdaki uygulamamız ile hemen hemen aynı görünüyor.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
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
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
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
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
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

Aşağıda, modelimizi eğitmek için aynı hiperparametreleri kullanıyoruz. Özel uygulamamız Python tarafından yorumlanmalıdır iken kodu C++ veya CUDA için derlendiğinden, her zamanki gibi, üst düzey API varyantının çok daha hızlı çalıştığını unutmayın.

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## Tartışma

Sezgisel olarak, toplu normalleştirmenin optimizasyon manzarasını daha pürüzsüz hale getirdiği düşünülmektedir. Bununla birlikte, derin modelleri eğitirken gözlemlediğimiz fenomenler için spekülatif sezgiler ve gerçek açıklamalar arasında ayrım yapmak için dikkatli olmalıyız. Daha basit derin sinir ağlarının (MLP'ler ve konvansiyonel CNN'ler) neden ilk etapta genelleştirildiğini bile bilmediğimizi hatırlayın. Bırakma ve ağırlık bozulması ile bile, görünmeyen verilere genelleme kabiliyetleri geleneksel öğrenme-teorik genelleme garantileri ile açıklanamayacak kadar esnek kalırlar.

Toplu normalleştirmeyi öneren orijinal gazetede, yazarlar, güçlü ve kullanışlı bir araç tanıtmanın yanı sıra, neden çalıştığına dair bir açıklama sundular: *internal covariate shift* azaltarak. Muhtemelen *dahili eşdeğişken kayması* ile yazarlar, yukarıda ifade edilen sezgi gibi bir şey ifade ederdi- değişken değerlerin dağılımının eğitim boyunca değiştiği düşüncesi. Bununla birlikte, bu açıklamayla ilgili iki sorun vardı: i) Bu sürüklenme, *covariate shift*'den çok farklıdır, adı yanlış isimlendirir. ii) Açıklama az belirtilmiş bir sezgi sunar ancak *neden tam olarak bu tekniğin çalışır* titiz bir açıklama isteyen açık bir soru bırakır. Bu kitap boyunca, uygulayıcıların derin sinir ağlarının gelişimine rehberlik etmek için kullandıkları sezgileri aktarmayı amaçlıyoruz. Bununla birlikte, bu rehberlik sezgilerini yerleşik bilimsel gerçeklerden ayırmanın önemli olduğuna inanıyoruz. Sonunda, bu materyale hakim olduğunuzda ve kendi araştırma kağıtlarınızı yazmaya başladığınızda, teknik iddialar ve önseziler arasında yer almak için net olmak isteyeceksiniz.

Toplu normalleşmenin başarısını takiben, *internal covariate shift* açısından açıklaması, teknik literatürdeki tartışmalarda ve makine öğrenimi araştırmasının nasıl sunulacağı konusunda daha geniş bir söylemde defalarca ortaya çıkmıştır. 2017 NeurIPS konferansında Zaman Testi Ödülü'nü kabul ederken verilen unutulmaz bir konuşmada Ali Rahimi, modern derin öğrenme pratiğini simyaya benzeyen bir argümanda odak noktası olarak *internal covariate shift* kullandı. Daha sonra, örnek :cite:`Lipton.Steinhardt.2018` makine öğrenimindeki sıkıntılı eğilimleri özetleyen bir pozisyon kağıdında ayrıntılı olarak yeniden gözden geçirildi. Diğer yazarlar toplu normalleştirmenin başarısı için alternatif açıklamalar önerdiler, bazıları toplu normalleşmenin başarısının bazı yönlerden orijinal kağıt :cite:`Santurkar.Tsipras.Ilyas.ea.2018`'de iddia edilenlerin tersi olan sergileme davranışlarına rağmen geldiğini iddia ediyor.

*İç eşdeğişken kayması*, teknik makine öğrenimi literatüründe her yıl yapılan benzer şekilde belirsiz iddiaların herhangi birinden daha eleştirilere layık olmadığını not ediyoruz. Muhtemelen, bu tartışmaların odak noktası olarak rezonansı, hedef kitleye geniş tanınabilirliğine borçludur. Toplu normalleştirme, neredeyse tüm konuşlandırılmış görüntü sınıflandırıcılarında uygulanan vazgeçilmez bir yöntem kanıtlamıştır, tekniği on binlerce atıf getiren kağıt kazanç.

## Özet

* Model eğitimi sırasında, toplu normalleştirme, minibatch'ın ortalama ve standart sapmasını kullanarak sinir ağının ara çıkışını sürekli olarak ayarlar, böylece sinir ağı boyunca her katmandaki ara çıkışın değerleri daha kararlı olur.
* Tam bağlı katmanlar ve evrimsel katmanlar için toplu normalleştirme yöntemleri biraz farklıdır.
* Bir bırakma katmanı gibi, toplu normalleştirme katmanları da eğitim modunda ve tahmin modunda farklı hesaplama sonuçlarına sahiptir.
* Toplu normalleşmenin, öncelikle düzenliliğin birçok yararlı yan etkisi vardır. Öte yandan, iç kodeğişik kaymayı azaltmanın orijinal motivasyonu geçerli bir açıklama gibi görünmüyor.

## Egzersizler

1. Toplu normalleştirmeden önce önyargı parametresini tam bağlı katmandan veya konvolusyonel katmandan kaldırabilir miyiz? Neden?
1. Toplu normalleştirme ile ve olmadan LeNet için öğrenme oranlarını karşılaştırın.
    1. Eğitim ve test doğruluğundaki artışı çiz.
    1. Öğrenme oranını ne kadar büyük yapabilirsiniz?
1. Her katmanda toplu normalleştirmeye ihtiyacımız var mı? Deney mi?
1. Eğer toplu normalleştirme ile terk yerini alabilir? Davranış nasıl değişir?
1. `beta` ve `gamma` parametrelerini düzeltin ve sonuçları gözlemleyin ve analiz edin.
1. Toplu normalleştirme için diğer uygulamaları görmek için üst düzey API'lerden `BatchNorm` için çevrimiçi belgeleri gözden geçirin.
1. Araştırma fikirleri: uygulayabileceğiniz diğer normalleştirme dönüşümleri düşünün? Olasılık integral dönüşümü uygulayabilir misiniz? Tam rütbeli kovaryans tahminine ne dersin?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
