# Birden Fazla GPU Eğitimi
:label:`sec_multi_gpu`

Şimdiye kadar modellerin CPU'lar ve GPU'lar üzerinde nasıl verimli bir şekilde eğitileceğini tartıştık. Hatta derin öğrenme çerçevelerinin :numref:`sec_auto_para`'da hesaplama ve iletişimi otomatik olarak nasıl paralelleştirmesine izin verdiğini gösterdik. Ayrıca :numref:`sec_use_gpu`'te `nvidia-smi` komutunu kullanarak bir bilgisayardaki mevcut tüm GPU'ların nasıl listeleneceğini de gösterdik. Konuşmadığımız şey derin öğrenme eğitiminin nasıl paralelleştirileceğidir. Bunun yerine, bir şekilde verileri birden fazla cihaza böleceğini ve çalışmasını sağlayacağını ima ettik. Mevcut bölüm ayrıntıları doldurur ve sıfırdan başladığınızda bir ağın paralel olarak nasıl eğitileceğini gösterir. Üst düzey API'ler işlevselliğinden nasıl yararlanılacağı ile ilgili ayrıntılar :numref:`sec_multi_gpu_concise` için kümelenir. :numref:`sec_minibatch_sgd`'te açıklananlar gibi minibatch stokastik degrade iniş algoritmalarına aşina olduğunuzu varsayıyoruz. 

## Sorunu Bölmek

Basit bir bilgisayar görme sorunu ve biraz arkaik bir ağ ile başlayalım, örn. birden fazla katman kıvrım, havuzlama ve sonunda muhtemelen birkaç tamamen bağlı katmanla. Yani, LeNet :cite:`LeCun.Bottou.Bengio.ea.1998` veya AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`'e oldukça benzeyen bir ağ ile başlayalım. Birden fazla GPU (eğer bir masaüstü sunucusu ise 2, AWS g4dn.12xlarge örneğinde 4, p3.16xlarge üzerinde 8 veya p2.16xlarge üzerinde 16), aynı anda basit ve tekrarlanabilir tasarım seçimlerinden yararlanarak iyi bir hız elde edecek şekilde eğitimi bölümlemek istiyoruz. Sonuçta birden fazla GPU hem *bellek* hem de *hesaplama* yeteneğini artırır. Özetle, sınıflandırmak istediğimiz bir mini grup eğitim verisi göz önüne alındığında aşağıdaki seçeneklere sahibiz. 

İlk olarak, ağı birden fazla GPU arasında bölümleyebiliriz. Yani, her GPU belirli bir katmana akan verileri girdi olarak alır, verileri bir dizi sonraki katmanda işler ve daha sonra verileri bir sonraki GPU'ya gönderir. Bu, tek bir GPU'nun işleyebileceği şeylerle karşılaştırıldığında verileri daha büyük ağlarla işlememize olanak tanır. Ayrıca, GPU başına bellek ayak izi iyi kontrol edilebilir (toplam ağ ayak izinin bir kısmıdır). 

Ancak, katmanlar arasındaki arabirim (ve dolayısıyla GPU'lar) sıkı senkronizasyon gerektirir. Bu, özellikle hesaplamalı iş yükleri katmanlar arasında düzgün bir şekilde eşleştirilmemişse zor olabilir. Sorun çok sayıda GPU için daha da şiddetlenir. Katmanlar arasındaki arabirim, etkinleştirme ve degradeler gibi büyük miktarda veri aktarımı gerektirir. Bu, GPU veri yollarının bant genişliğini bunaltabilir. Ayrıca, bilgi işlem yoğun, ancak sıralı işlemler bölümleme için önemsiz değildir. Bu konuda en iyi çaba için örneğin :cite:`Mirhoseini.Pham.Le.ea.2017`'e bakın. Zor bir sorun olmaya devam ediyor ve önemsiz olmayan problemlerde iyi (doğrusal) ölçekleme elde etmenin mümkün olup olmadığı belirsizdir. Birden fazla GPU'ları birbirine zincirlemek için mükemmel çerçeve veya işletim sistemi desteği olmadığı sürece bunu önermiyoruz. 

İkincisi, işi katman olarak bölüşebiliriz. Örneğin, tek bir GPU'da 64 kanalı hesaplamak yerine, sorunu her biri 16 kanal için veri üreten 4 GPU'ya bölebiliriz. Aynı şekilde, tam bağlı bir katman için çıkış birimlerinin sayısını bölebiliriz. :numref:`fig_alexnet_original` (:cite:`Krizhevsky.Sutskever.Hinton.2012`'ten alınmıştır), bu stratejinin çok küçük bir bellek ayak izine sahip GPU'larla uğraşmak için kullanıldığı bu tasarımı göstermektedir (aynı anda 2 GB). Bu, kanalların (veya birim) sayısının çok küçük olmaması koşuluyla hesaplama açısından iyi ölçeklendirmeye izin verir. Ayrıca, kullanılabilir bellek doğrusal ölçeklendiğinden, birden fazla GPU giderek daha büyük ağları işleyebilir. 

![Model parallelism in the original AlexNet design due to limited GPU memory.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

Bununla birlikte, her katman diğer tüm katmanların sonuçlarına bağlı olduğundan, çok büyük* senkronizasyon veya bariyer işlemlerine ihtiyacımız var. Dahası, aktarılması gereken veri miktarı, katmanları GPU'lara dağıtırken potansiyel olarak daha büyüktür. Bu nedenle, bant genişliği maliyeti ve karmaşıklığı nedeniyle bu yaklaşımı önermiyoruz. 

Son olarak, verileri birden fazla GPU arasında bölümlendirebiliriz. Bu şekilde tüm GPU'lar farklı gözlemlerde de olsa aynı tür çalışmaları gerçekleştirir. Degradeler, eğitim verilerinin her mini toplu işleminden sonra GPU'lar arasında toplanır. Bu en basit yaklaşımdır ve her durumda uygulanabilir. Sadece her minibatch işleminden sonra senkronize etmemiz gerekiyor. Yani, diğerleri hala hesaplanırken degrade parametreleri alışverişinde başlamak son derece arzu edilir. Dahası, daha fazla sayıda GPU daha büyük mini batch boyutlarına yol açarak eğitim verimliliğini arttırır. Ancak, daha fazla GPU eklemek daha büyük modeller eğitmemize izin vermez. 

![Parallelization on multiple GPUs. From left to right: original problem, network partitioning, layerwise partitioning, data parallelism.](../img/splitting.svg)
:label:`fig_splitting`

Birden fazla GPU üzerinde farklı paralelleştirme yollarının karşılaştırılması :numref:`fig_splitting`'te tasvir edilmiştir. Büyük ve büyük, veri paralelliği, yeterince büyük belleğe sahip GPU'lara erişebilmemiz koşuluyla devam etmenin en uygun yoludur. Dağıtılmış eğitim için bölümleme ayrıntılı bir açıklaması için :cite:`Li.Andersen.Park.ea.2014` ayrıca bkz. GPU belleği derin öğrenmenin ilk günlerinde bir sorundu. Şimdiye kadar bu sorun en sıradışı durumlar dışında herkes için çözülmüştür. Biz aşağıdaki veri paralellik odaklanmak. 

## Veri Paralelliği

Bir makinede $k$ GPU olduğunu varsayalım. Eğitimli model göz önüne alındığında, her GPU, GPU'larda parametre değerleri aynı ve senkronize olmasına rağmen, bağımsız olarak tam bir model parametresi kümesini koruyacaktır. Örnek olarak, :numref:`fig_data_parallel` $k=2$ olduğunda veri paralelliği ile eğitim göstermektedir. 

![Calculation of minibatch stochastic gradient descent using data parallelism on two GPUs.](../img/data-parallel.svg)
:label:`fig_data_parallel`

Genel olarak, eğitim aşağıdaki gibi devam eder: 

* Eğitimin herhangi bir yinelemesinde, rastgele bir minibatch verildiğinde, partideki örnekleri $k$ bölüme ayırır ve GPU'lara eşit olarak dağıtırız.
* Her GPU, atandığı minibatch alt kümesine göre model parametrelerinin kaybını ve degradelerini hesaplar.
* $k$ GPU'larının her birinin yerel degradeleri, geçerli mini toplu iş stokastik degradeyi elde etmek için toplanır.
* Toplam degrade her GPU'ya yeniden dağıtılır.
* Her GPU, koruduğu model parametrelerinin tamamını güncelleştirmek için bu minibatch stokastik degradeyi kullanır.

Uygulamada $k$ GPU'larda antrenman yaparken minibatch boyutunu $k$ kat artırdığımızı* her GPU'nun sadece tek bir GPU üzerinde eğitim yapıyormuşuz gibi aynı miktarda iş yapmasını sağlar. 16 GPU sunucuda bu minibatch boyutunu önemli ölçüde artırabilir ve öğrenme oranını buna göre artırmamız gerekebilir. Ayrıca, :numref:`sec_batch_norm`'teki toplu normalleştirmenin, örneğin GPU başına ayrı bir toplu normalleştirme katsayısı tutarak ayarlanması gerektiğini unutmayın. Aşağıda, çoklu GPU eğitimini göstermek için bir oyuncak ağı kullanacağız.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**Bir Oyuncak Ağı**]

LeNet'i :numref:`sec_lenet`'te tanıtıldığı gibi kullanıyoruz (hafif modifikasyonlar ile). Parametre değişimini ve senkronizasyonu ayrıntılı olarak göstermek için sıfırdan tanımlıyoruz.

```{.python .input}
# Initialize model parameters
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## Veri Senkronizasyonu

Verimli çoklu GPU eğitimi için iki temel operasyona ihtiyacımız var. Öncelikle [**birden fazla cihaza parametre listesini dağıtabilir**] ve degradeleri (`get_params`) ekleme yeteneğine sahip olmamız gerekir. Parametreler olmadan ağı bir GPU üzerinde değerlendirmek imkansızdır. İkincisi, birden fazla cihazda parametreleri toplama yeteneğine ihtiyacımız var, yani bir `allreduce` işlevine ihtiyacımız var.

```{.python .input}
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

Model parametrelerini bir GPU'ya kopyalayarak deneyelim.

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

Henüz herhangi bir hesaplama yapmadığımız için, önyargı parametresi ile ilgili degrade hala sıfırdır. Şimdi birden fazla GPU arasında dağıtılmış bir vektör olduğunu varsayalım. Aşağıdaki [**`allreduce` işlevi tüm vektörleri ekler ve sonucu tüm GPU'lara geri gönderir**]. Bunun işe yaraması için verileri sonuçları toplayan cihaza kopyalamamız gerektiğini unutmayın.

```{.python .input}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

Farklı cihazlarda farklı değerlere sahip vektörler oluşturarak ve bunları toplayarak bunu test edelim.

```{.python .input}
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## Veri Dağıtımı

[**Bir minibatch birden çok GPU boyunca eşit olarak dağıtmak için**] basit bir yardımcı program işlevine ihtiyacımız var. Örneğin, iki GPU'da verilerin yarısının GPU'lardan birine kopyalanmasını istiyoruz. Daha kullanışlı ve daha özlü olduğu için, $4 \times 5$ matrisinde denemek için derin öğrenme çerçevesinden yerleşik işlevi kullanıyoruz.

```{.python .input}
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

Daha sonra yeniden kullanım için hem verileri hem de etiketleri bölen bir `split_batch` işlevi tanımlıyoruz.

```{.python .input}
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## Eğitim

Artık [**multi-GPU eğitimini tek bir minibatch'da**] uygulayabiliriz. Uygulaması öncelikle bu bölümde açıklanan veri paralellik yaklaşımına dayanmaktadır. Verileri birden fazla GPU arasında senkronize etmek için az önce tartıştığımız `allreduce` ve `split_and_load` yardımcı fonksiyonlarını kullanacağız. Paralellik elde etmek için herhangi bir özel kod yazmamıza gerek olmadığını unutmayın. Hesaplamalı grafiğin bir minibatch içindeki cihazlar arasında herhangi bir bağımlılığı olmadığından, paralel olarak *otomatik* yürütülür.

```{.python .input}
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

Şimdi [**eğitim fonksiyonu**] tanımlayabiliriz. Önceki bölümlerde kullanılanlardan biraz farklıdır: GPU'ları tahsis etmemiz ve tüm model parametrelerini tüm cihazlara kopyalamalıyız. Açıkçası her parti birden çok GPU ile başa çıkmak için `train_batch` işlevi kullanılarak işlenir. Kolaylık sağlamak (ve kodun özlü olması) için doğruluğu tek bir GPU üzerinde hesaplıyoruz, ancak diğer GPU'lar boşta olduğundan bu *verimsiz*.

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Bunun ne kadar iyi çalıştığını görelim [**tek bir GPU**]. İlk olarak 256 toplu boyutunu ve 0,2 öğrenme oranını kullanıyoruz.

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

Toplu iş boyutunu ve öğrenme oranını değişmeden tutarak ve [**GPU sayısını 2 olarak artırarak**], test doğruluğunun önceki deneye kıyasla kabaca aynı kaldığını görebiliriz. Optimizasyon algoritmaları açısından aynıdır. Ne yazık ki burada kazanılacak anlamlı bir hız yok: model sadece çok küçük; dahası, çoklu GPU eğitimini uygulamaya yönelik biraz sofistike yaklaşımımızın önemli Python yükünden muzdarip olduğu küçük bir veri kümesine sahibiz. Daha karmaşık modellerle ve daha gelişmiş paralelleşme yollarıyla karşılaşacağız. Bize Moda-MNIST için yine ne olacağını görelim.

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## Özet

* Derin ağ eğitimini birden fazla GPU üzerinden bölmenin birden çok yolu vardır. Katmanlar arasında, katmanlar arasında veya veriler arasında bölebiliriz. Eski ikisi sıkı bir şekilde koreografiye edilmiş veri aktarımlarına ihtiyaç duyar. Veri paralelliği en basit stratejidir.
* Veri paralel eğitimi basittir. Bununla birlikte, verimli olması için etkili mini batch boyutunu artırır.
* Veri paralelliğinde veriler birden çok GPU'ya bölünür; burada her GPU'nun kendi ileri ve geri çalışmasını yürütür ve daha sonra degradeler toplanır ve sonuçlar GPU'lara geri yayınlanır.
* Daha büyük minibüsler için biraz daha yüksek öğrenme oranları kullanabiliriz.

## Egzersizler

1. $k$ GPU'larda eğitim yaparken, minibatch boyutunu $b$'dan $k \cdot b$'e değiştirin, yani GPU sayısına göre ölçeklendirin.
1. Farklı öğrenme oranları için doğruluğu karşılaştırın. GPU sayısıyla nasıl ölçeklenir?
1. Farklı GPU'larda farklı parametreleri toplayan daha verimli bir `allreduce` işlevini uygulamak? Neden daha verimli?
1. Çoklu GPU test doğruluğu hesaplamasını gerçekleştirin.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1669)
:end_tab:
