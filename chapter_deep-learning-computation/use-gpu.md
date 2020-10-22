# GPU (Grafik İşleme Birimi)
:label:`sec_use_gpu`

Giriş bölümünde, son yirmi yılda hesaplamanın hızlı büyümesini tartıştık. Özetle, GPU performansı 2000 yılından bu yana her on yılda bir 1000 kat artmıştır. Bu büyük bir fırsat sunarken aynı zamanda bu tür bir performansın sağlanması için önemli bir gereksinim olduğunu da göstermektedir.

|Onyıl|Veri Kümesi|Bellek|Saniyedeki Kayan Virgüllü Sayı Hesaplaması|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (Boston'daki ev fiyatları)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optik karakter tanıma)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web sayfaları)|100 MB|1 GF (Intel Core)|
|2010|10 G (reklamlar)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social ağ)|100 GB|1 PF (NVIDIA DGX-2)|

Bu bölümde, araştırmanız için bu hesaplama performansından nasıl yararlanılacağını tartışmaya başlıyoruz. Öncelikle tek GPU'ları kullanarak ve daha sonra, birden çok GPU ve birden çok sunucuyu (birden çok GPU ile) nasıl kullanacağınızı tartışacağız.

Bu bölümde, hesaplamalar için tek bir NVIDIA GPU'nun nasıl kullanılacağını tartışacağız. Öncelikle, kurulu en az bir NVIDIA GPU'nuz olduğundan emin olun. Ardından, [CUDA'yı indirin](https://developer.nvidia.com/cuda-downloads) ve uygun yere kurmak için istemleri takip edin. Bu hazırlıklar tamamlandıktan sonra, `nvidia-smi`, komutu grafik kartı bilgilerini görüntülemek için kullanılabilir.

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
MXNet tensörünün NumPy'deki ile neredeyse aynı göründüğünü fark etmiş olabilirsiniz.

Ancak birkaç önemli farklılık var. MXNet'i NumPy'den ayıran temel özelliklerden biri, çeşitli donanım aygıtlarına desteğidir.

MXNet'te her dizilimin (array) bir bağlamı vardır. Şimdiye kadar, varsayılan olarak, tüm değişkenler ve ilişkili hesaplamalar CPU'ya atanmıştır. Tipik olarak, diğer bağlamlar çeşitli GPU'lar olabilir. İşleri birden çok sunucuya dağıttığımızda işler arap saçına dönebilir. Dizilimleri bağlamlara akıllıca atayarak, cihazlar arasında veri aktarımında harcanan zamanı en aza indirebiliriz. Örneğin, GPU'lu bir sunucuda sinir ağlarını eğitirken, genellikle modelin parametrelerinin GPU'da kalmasını tercih ederiz.

Ardından, MXNet'in GPU sürümünün kurulu olduğunu onaylamamız gerekiyor. MXNet'in bir CPU sürümü zaten kuruluysa, önce onu kaldırmamız gerekir. Örneğin, `pip uninstall mxnet` komutunu kullanın, ardından CUDA sürümünüze göre ilgili MXNet sürümünü kurun. CUDA 9.0'ın kurulu olduğunu varsayarsak, CUDA 9.0'ı destekleyen MXNet sürümünü `pip install mxnet-cu90` aracılığıyla kurabilirsiniz. Bu bölümdeki programları çalıştırmak için en az iki GPU'ya ihtiyacınız var.
:end_tab:

:begin_tab:`pytorch`
PyTorch'ta her dizilimin bir aygıtı vardır, biz onu genellikle bağlam olarak adlandırırız. Şimdiye kadar, varsayılan olarak, tüm değişkenler ve ilişkili hesaplama CPU'ya atanmıştır. Tipik olarak, diğer bağlamlar çeşitli GPU'lar olabilir. İşleri birden çok sunucuya dağıttığımızda işler arap saçına dönebilir. Dizileri bağlamlara akıllıca atayarak, cihazlar arasında veri aktarımında harcanan zamanı en aza indirebiliriz. Örneğin, GPU'lu bir sunucuda sinir ağlarını eğitirken, genellikle modelin parametrelerinin GPU'da kalmasını tercih ederiz.

Ardından, PyTorch'un GPU sürümünün kurulu olduğunu onaylamamız gerekiyor. PyTorch'un bir CPU sürümü zaten kuruluysa, önce onu kaldırmamız gerekir. Örneğin, `pip uninstall torch` komutunu kullanın, ardından CUDA sürümünüze göre ilgili PyTorch sürümünü kurun. CUDA 9.0'ın kurulu olduğunu varsayarsak, CUDA 9.0'ı destekleyen PyTorch sürümünü `pip install torch-cu90` aracılığıyla kurabilirsiniz. Bu bölümdeki programları çalıştırmak için en az iki GPU'ya ihtiyacınız var.
:end_tab:

Bunun çoğu masaüstü bilgisayar için abartılı olabileceğini, ancak bulutta (ör. AWS EC2 çoklu-GPU bulut sunucularını kullanarak) kolayca kullanılabileceğini unutmayın. Hemen hemen diğer bütün bölümler birden fazla GPU *gerektirmez*. Bunun burdaki kullanım nedeni, basitçe verilerin farklı cihazlar arasında nasıl aktığını göstermektir.

By default, tensors are created in the main memory and then uses the CPU to calculate it.

## Hesaplama Cihazları

Depolama ve hesaplama için CPU ve GPU gibi cihazları belirtebiliriz. Varsayılan olarak, ana bellekte tensörler oluşturulur ve ardından bunu hesaplamak için CPU'yu kullanır.

:begin_tab:`mxnet`
MXNet'te CPU ve GPU, `cpu()` ve `gpu()` ile gösterilebilir. `cpu()` (veya parantez içindeki herhangi bir tam sayı), tüm fiziksel CPU'lar ve bellek anlamına gelir. Bu, MXNet'in hesaplamalarının tüm CPU çekirdeklerini kullanmaya çalışacağı anlamına gelir. Ancak, `gpu()` yalnızca bir kartı ve ona denk gelen belleği temsil eder. Birden fazla GPU varsa, $i^\mathrm{th}$ GPU'yu ($i$ 0'dan başlar) temsil etmek için `gpu(i)`'yu kullanırız. Ayrıca, `gpu(0)` ve `gpu()` eşdeğerdir.
:end_tab:

:begin_tab:`pytorch`
PyTorch'ta CPU ve GPU, `torch.device('cpu')` ve `torch.cuda.device('cuda')` ile gösterilebilir. `cpu` aygıtının tüm fiziksel CPU'lar ve bellek anlamına geldiğine dikkat edilmelidir. Bu, PyTorch'un hesaplamalarının tüm CPU çekirdeklerini kullanmaya çalışacağı anlamına gelir. Bununla birlikte, bir `gpu` cihazı yalnızca bir kartı ve ona denk gelen belleği temsil eder. Birden çok GPU varsa, $i^\mathrm{th}$ GPU ($i$ 0'dan başlar) temsil etmek için `torch.cuda.device(f'cuda{i}')`yi kullanırız. Ayrıca `gpu:0` ve `gpu` eşdeğerdir.
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

Mevcut GPU adetini sorgulayabiliriz.

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

Şimdi, istenen GPU'lar var olmasa bile kodları çalıştırmamıza izin veren iki kullanışlı işlev tanımlıyoruz.

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [npx.gpu(i) for i in range(npx.num_gpus())]
    return ctxes if ctxes else [npx.cpu()]

try_gpu(), try_gpu(3), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return ctxes if ctxes else [torch.device('cpu')]

try_gpu(), try_gpu(3), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    ctxes = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return ctxes if ctxes else [tf.device('/CPU:0')]

try_gpu(), try_gpu(3), try_all_gpus()
```

## Tensörler and GPUlar

Varsayılan olarak, CPU'da tensörler oluşturulur. Tensörün bulunduğu cihazı sorgulayabiliriz.

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

Birden çok terimle çalışmak istediğimizde, aynı bağlamda olmaları gerektiğine dikkat etmemiz önemlidir. Örneğin, iki tensörü toplarsak, her iki argümanın da aynı cihazda olduğundan emin olmamız gerekir---aksi takdirde çerçeve, sonucu nerede saklayacağını ve hatta hesaplamayı nerede gerçekleştireceğine nasıl karar vereceğini bilemez.

### GPU'da Depolama

GPU'da bir tensör depolamanın birkaç yolu vardır. Örneğin, bir tensör oluştururken bir depolama cihazı belirleyebiliriz. Sonra, ilk `gpu`da tensör değişkeni `a`'yı oluşturuyoruz. `a`'yı yazdırırken aygıt bilgilerinin değiştiğine dikkat edin. Bir GPU'da oluşturulan tensör yalnızca o GPU'nun belleğini harcar. GPU bellek kullanımını görüntülemek için `nvidia-smi` komutunu kullanabiliriz. Genel olarak, GPU bellek sınırını aşan veriler oluşturmadığımızdan emin olmamız gerekir.

```{.python .input}
x = np.ones((2, 3), ctx=try_gpu())
x
```

```{.python .input}
#@tab pytorch
x = torch.ones(2, 3, device=try_gpu())
x
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    x = tf.ones((2, 3))
x
```

En az iki GPU'ya sahip olduğunuzu varsayarak, aşağıdaki kod ikinci GPU'da keyfi bir dizilim oluşturacaktır.

```{.python .input}
y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
y
```

```{.python .input}
#@tab pytorch
y = torch.randn(2, 3, device=try_gpu(1))
y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    y = tf.random.uniform((2, 3))
y
```

### Kopyalama

$\mathbf{x} + \mathbf{y}$'yı hesaplamak istiyorsak, bu işlemi nerede gerçekleştireceğimize karar vermemiz gerekir. Örneğin: numref:`fig_copyto`da gösterildiği gibi, $\mathbf{x}$'i ikinci GPU'ya aktarabilir ve işlemi orada gerçekleştirebiliriz. * Sadece `x + y` *toplamayın*, çünkü bu bir istisnayla sonuçlanacaktır. Koşma zamanı motoru ne yapacağını bilemez, veriyi aynı cihazda bulamaz ve başarısız olur.

![Copyto dizilimleri aynı cihaza kopyalar](../img/copyto.svg)
:label:`fig_copyto`

`copyto`, verileri toplayabileceğimiz şekilde başka bir cihaza kopyalar. $\mathbf{y}$ ikinci GPU'da olduğundan, ikisini toplayabilmemiz için önce $\mathbf{x}$'i oraya taşımamız gerekir.

```{.python .input}
z = x.copyto(try_gpu(1))
print(x)
print(z)
```

```{.python .input}
#@tab pytorch
z = x.cuda(1)
print(x)
print(z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    z = x
print(x)
print(z)
```

Artık veriler aynı GPU'da olduğuna göre (hem $\mathbf{z}$ hem de $\mathbf{y}$), onları toplayabiliriz.

```{.python .input}
#@tab all
y + z
```

:begin_tab:`mxnet`
`z` değişkeninizin halihazırda ikinci GPU'nuzda olduğunu hayal edin. Gene de `z.copyto(gpu(1))` çağırırsak ne olur? Değişken istenen cihazda zaten bulunsa bile, yeni bir kopya oluşturacak ve bellek tahsis edecektir! Kodumuzun çalıştığı ortama bağlı olarak, aynı cihazda iki değişkenin zaten var olduğu zamanlar vardır. Dolayısıyla, değişkenler şu anda farklı bağlamlarda yaşıyorsa yalnızca bir kopya yapmak isteriz. Bu durumlarda, `as_in_ctx()` çağırabiliriz. Değişken zaten belirtilen bağlamda yaşıyorsa, bu işlem-yok (no-op) demektir. Özellikle bir kopya yapmak istemediğiniz sürece, `as_in_ctx()` tercih edilen yöntemdir.
:end_tab:

:begin_tab:`pytorch`
`z` değişkeninizin halihazırda ikinci GPU'nuzda ver olduğunu hayal edin. Gene de `z.cuda(1)` diye çağırırsak ne olur? Kopyalamak ve yeni bellek ayırmak yerine `z`'yi döndürür.
:end_tab:

:begin_tab:`pytorch`
`z` değişkeninizin halihazırda ikinci GPU'nuzda ver olduğunu hayal edin. Aynı cihaz kapsamı altında gene de `z2 = z`'yi çağırırsak ne olur? Kopyalamak ve yeni bellek ayırmak yerine `z`'yi döndürür.
:end_tab:

```{.python .input}
z.as_in_ctx(try_gpu(1)) is z
```

```{.python .input}
#@tab pytorch
z.cuda(1) is z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    z2 = z
z2 is z
```

### Ek Notlar

İnsanlar hızlı olmalarını bekledikleri için makine öğrenmesi için GPU'ları kullanıyorlar. Ancak değişkenlerin bağlamlar arasında aktarılması yavaştır. Bu yüzden, yapmanıza izin vermeden önce yavaş bir şey yapmak istediğinizden %100 emin olmanızı istiyoruz. Çerçeve kopyayı çökmeden otomatik olarak yaptıysa, yavaş çalışan bir kod yazdığınızı fark etmeyebilirsiniz.

Ayrıca, cihazlar (CPU, GPU'lar, diğer makineler) arasında veri aktarımı, hesaplamadan *çok daha yavaş* bir şeydir. Ayrıca, daha fazla işleme ilerlemeden önce verilerin gönderilmesini (veya daha doğrusu alınmasını) beklememiz gerektiğinden bu paralelleştirmeyi çok daha zor hale getirir. Bu nedenle kopyalama işlemlerine büyük özen gösterilmelidir. Genel bir kural olarak, birçok küçük işlem, tek bir büyük işlemden çok daha kötüdür. Dahası, bir seferde birkaç işlem, koda serpiştirilmiş birçok tek işlemden çok daha iyidir (ne yaptığınızı biliyorsanız o ayrı). Bu durumda, bir aygıtın bir şey yapmadan önce bir diğerini beklemesi gerektiğinde bu tür işlemler onu engelleyebilir. Başka. Bu biraz, kahvenizi telefonla ön sipariş vermek ve siz istediğinizde hazır olduğunu öğrenmek yerine sırada bekleyerek sipariş etmek gibidir.

Son olarak, tensörleri yazdırdığımızda veya tensörleri NumPy formatına dönüştürdüğümüzde, veri ana bellekte değilse, çerçeve onu önce ana belleğe kopyalayacak ve bu da ek iletim yüküne neden olacaktır. Daha da kötüsü, şimdi Python'un her şeyi  tamamlanmasını beklemesine neden olan o korkunç Global Yorumlayıcı Kilidine tabidir.

## Sinir Ağları ve GPUlar

Benzer şekilde, bir sinir ağı modeli cihazları belirtebilir. Aşağıdaki kod, model parametrelerini GPU'ya yerleştirir (biraz daha yoğun işlem gerektireceklerinden, aşağıda modellerin GPU'larda nasıl çalıştırılacağına dair daha birçok örnek göreceğiz).

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

Girdi, GPU'da bir tensör olduğunda, Gluon sonucu aynı GPU'da hesaplayacaktır.

```{.python .input}
#@tab all
net(x)
```

Model parametrelerinin aynı GPU'da depolandığını doğrulayalım.

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

Kısacası tüm veriler ve parametreler aynı cihazda olduğu sürece modelleri verimli bir şekilde öğrenebiliriz. Sonraki kısımlarda bu tür birkaç örnek göreceğiz.

## Özet

* Depolama ve hesaplama için CPU veya GPU gibi cihazlar belirleyebiliriz. Varsayılan olarak, veriler ana bellekte oluşturulur ve ardından hesaplamalar için CPU kullanılır.
* Çerçeve, hesaplama için tüm girdi verilerinin *aynı cihazda* olmasını gerektirir, ister CPU ister aynı GPU olsun.
* Verileri dikkatsizce taşıyarak önemli bir performans kaybına uğrayabilirsiniz. Tipik bir hata şudur: GPU'daki her mini grup için kaybı hesaplamak ve bunu komut satırında kullanıcıya geri bildirmek (veya bir NumPy dizisinde kaydetmek), bu tüm GPU'ları durduran global yorumlayıcı kilidini tetikleyecektir. GPU içinde kayıt tutmak için bellek ayırmak ve yalnızca daha büyük kayıtları taşımak çok daha iyidir.

## Alıştırmalar

1. Büyük matrislerin çarpımı gibi daha büyük bir hesaplama görevi deneyiniz ve CPU ile GPU arasındaki hız farkını görünüz. Az miktarda hesaplama içeren bir göreve ne olur?
1. GPU'daki model parametrelerini nasıl okuyup yazmalıyız?
1. $100 \times 100$lük 1000 matris-matris çarpımını hesaplamak için gereken süreyi ölçünüz ve her seferde matris normu $\mathrm{tr} M M^\top$ sonucunu günlüğe kaydetme ile GPU'da günlük tutma ve yalnızca son sonucun aktarılmayı kıyaslayınız.
1. İki GPU'da iki matris-matris çarpımını aynı anda gerçekleştirme ile tek bir GPU'da sıralı gerçekleştirmenin ne kadar zaman aldığını ölçerek karşılaştırınız (İpucu: Neredeyse doğrusal bir ölçekleme görmelisiniz).

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/270)
:end_tab:
