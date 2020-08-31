# Naif (Saf) Bayes
:label:`sec_naive_bayes`

Önceki bölümler boyunca, olasılık teorisi ve rastgele değişkenler hakkında bilgi edindik. Bu teoriyi çalıştırmak için, *naif Bayes* sınıflandırıcısını tanıtalım. Bu, rakamların sınıflandırmasını yapmamıza izin vermek için olasılık temellerinden başka hiçbir şey kullanmaz.

Öğrenme tamamen varsayımlarda bulunmakla ilgilidir. Daha önce hiç görmediğimiz yeni bir veri noktasını sınıflandırmak istiyorsak, hangi veri noktalarının birbirine benzer olduğuna dair bazı varsayımlar yapmalıyız. Popüler ve oldukça net bir algoritma olan naif Bayes sınıflandırıcı, hesaplamayı basitleştirmek için tüm özelliklerin birbirinden bağımsız olduğunu varsayar. Bu bölümde, resimlerdeki karakterleri tanımak için bu modeli uygulayacağız.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

## Optik Karakter Tanıma

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`, yaygın olarak kullanılan veri kümelerinden biridir. Eğitim için 60.000 görüntü ve geçerleme için 10.000 görüntü içerir. Her görüntü, 0'dan 9'a kadar el yazısıyla yazılmış bir rakam içerir. Görev, her görüntüyü karşılık gelen rakama sınıflandırmaktır.

Gluon, veri kümesini İnternet'ten otomatik olarak almak için `data.vision` modülünde bir `MNIST` sınıfı sağlar.
Daha sonra, Gluon hali-hazırda indirilmiş yerel kopyayı kullanacaktır. `train` parametresinin değerini sırasıyla `True` veya `False` olarak ayarlayarak eğitim setini mi yoksa test setini mi talep ettiğimizi belirtiriz.
Her resim, hem genişliği hem de yüksekliği $28$ olan ve ($28$, $28$, $1$) şekilli gri tonlamalı bir resimdir. Son kanal boyutunu kaldırmak için özelleştirilmiş bir dönüşüm kullanıyoruz. Ek olarak, veri kümesi her pikseli işaretsiz $8$ bitlik bir tamsayı ile temsil eder. Problemi basitleştirmek için bunları ikili özellikler halinde nicelendiriyoruz.

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

Resmi ve ilgili etiketi içeren belirli bir örneğe erişebiliriz.

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

Burada `image` değişkeninde depolanan örneğimiz, yüksekliği ve genişliği $28$ piksel olan bir resme karşılık gelir.

```{.python .input}
image.shape, image.dtype
```

```{.python .input}
#@tab pytorch
image.shape, image.dtype
```

Kodumuz her görüntünün etiketini skaler olarak depolar. Türü $32$ bitlik bir tamsayıdır.

```{.python .input}
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

Aynı anda birden fazla örneğe de erişebiliriz.

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10,38)], 
                     dim=1).squeeze(0)
labels = torch.tensor([mnist_train[i][1] for i in range(10,38)])
images.shape, labels.shape
```

Bu örnekleri görselleştirelim.

```{.python .input}
d2l.show_images(images, 2, 9);
```

```{.python .input}
#@tab pytorch
d2l.show_images(images, 2, 9);
```

## Sınıflandırma için Olasılık Modeli

Bir sınıflandırma görevinde, bir örneği bir kategoriye eşleriz. Burada bir örnek gri tonlamalı $28\times 28$ resim ve kategori bir rakamdır. (Daha ayrıntılı bir açıklama için bakınız :numref:`sec_softmax`.)
Sınıflandırma görevini ifade etmenin doğal bir yolu, olasılık sorusudur: Özellikler (yani, görüntü pikselleri) verildiğinde en olası etiket nedir? $\mathbf x\in\mathbb R^d$ ile örneğin özelliklerini ve $y\in\mathbb R$ ile etiketini belirtiriz. Burada özellikler, $2$ boyutlu bir resmi $d = 28 ^ 2 = 784$ büyüklüğünde bir vektöre yeniden şekillendirebileceğimiz resim pikselleri ve etiketler rakamlardır.
Özellikleri verilen etiketin olasılığı $p(y  \mid  \mathbf{x})$ şeklindedir. Örneğimizde $y = 0, \ ldots, 9$ için $p(y  \mid  \mathbf{x})$ for $y=0, \ldots,9$ olan bu olasılıkları hesaplayabilirsek, sınıflandırıcı aşağıda verilen ifade ile tahminini, $\hat{y}$, yapacaktır:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Maalesef bu, her $\mathbf{x} = x_1, ..., x_d$ değeri için $p(y \mid \mathbf{x})$'yi tahmin etmemizi gerektirir. Her özelliğin $2$ değerden birini alabileceğini düşünün. Örneğin, $x_1 = 1$ özelliği, elma kelimesinin belirli bir belgede göründüğünü ve $x_1 = 0$ görünmediğini belirtebilir. Eğer $30$ tane bu tür ikili özelliklere sahip olsaydık, bu $\mathbf{x}$ girdi vektörünün $2^{30}$ (1 milyardan fazla!) olası değerlerinden herhangi birini sınıflandırmaya hazırlıklı olmamız gerektiği anlamına gelirdi.

Dahası, öğrenme nerede? İlgili etiketi tahmin etmek için her bir olası örneği görmemiz gerekiyorsa, o zaman gerçekten bir model öğrenmiyoruz, sadece veri setini ezberliyoruz.

## Naif Bayes Sınıflandırıcı

Neyse ki, koşullu bağımsızlık hakkında bazı varsayımlar yaparak, bazı tümevarımsal önyargılar sunabilir ve nispeten mütevazı bir eğitim örnekleri seçiminden genelleme yapabilen bir model oluşturabiliriz. Başlamak için, sınıflandırıcıyı şu şekilde ifade etmek için Bayes teoremini kullanalım:

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Paydanın normalleştirme teriminin $p(\ mathbf {x})$ olduğunu ve $y$ etiketinin değerine bağlı olmadığını unutmayın. Sonuç olarak, sadece payı farklı $y$ değerlerinde karşılaştırken endişelenmemiz gerekiyor. Paydanın hesaplanmasının zorlu olduğu ortaya çıksa bile, payı değerlendirebildiğimiz sürece onu görmezden gelerek kurtulabilirdik. Neyse ki, normalleştirme sabitini kurtarmak istesek, bunu da yapabilirdik. Normalleştirme terimini $\sum_y p(y \mid \mathbf{x}) = 1$ olduğundan her zaman kurtarabiliriz.

Şimdi $p( \mathbf{x}  \mid  y)$ üzerine odaklanalım. Zincir olasılık kuralını kullanarak $p( \mathbf{x}  \mid  y)$ terimini şu şekilde ifade edebiliriz:

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Tek başına bu ifade bizi daha ileriye götürmez. Yine de kabaca $2^ d$ tanee parametreyi tahmin etmeliyiz. Bununla birlikte, *etiketi verildiğinde özelliklerin koşullu olarak birbirinden bağımsız olduğunu varsayarsak*, aniden çok daha iyi durumda oluruz, çünkü bu terim $\prod_i p(x_i \mid y)$'a sadeleştirerek bize şu tahminciyi verir: 

$$ \hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Her $i$ ve $y$ için $\prod_i p(x_i = 1 \mid y)$'yi tahmin edebilir ve değerini $P_{xy} [i, y]$ olarak kaydedebiliriz, burada $P_{xy}$, $d \times n$ matristir; $n$ sınıf sayısı ve $y \in \{1, \ ldots, n\}$'dir. Ek olarak, her $y$ için $p(y)$ değerini tahmin ediyoruz ve $n$-uzunluk vektör olan $P_y$'ya, $P_y [y]$ olarak kaydediyoruz. Sonra herhangi bir yeni örnek için $\mathbf x$'yi hesaplayabiliriz:

$$ \hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d P_{xy}[x_i, y]P_y[y],$$
:eqlabel:`eq_naive_bayes_estimation`

üstelik herhangi bir $y$ için. Dolayısıyla, koşullu bağımsızlık varsayımımız, modelimizin karmaşıklığını özellik sayısına bağlı $\mathcal{O}(2^dn)$ üstel bir bağımlılıktan $\mathcal{O}(dn)$ olan doğrusal bir bağımlılığa almıştır.

## Eğitim

Şimdi sorun, $P_{xy}$ ve $P_y$'yi bilmiyor olmamızdır. Bu nedenle, önce bazı eğitim verileri verildiğinde bu değerleri tahmin etmemiz gerekiyor. Bu, modeli *eğitmektir*. $P_y$'yi tahmin etmek çok zor değil. Sadece $10$ sınıfla uğraştığımız için, her rakam için görülme sayısını, $n_y$, sayabilir ve bunu toplam veri miktarına $n$ bölebiliriz. Örneğin, 8 rakamı $n_8 = 5,800$ kez ortaya çıkarsa ve toplam $n = 60,000$ görüntümüz varsa, olasılık tahminimiz $ p(y=8) = 0,0967$ olur.

```{.python .input}
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], 
                dim=1).squeeze(0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

Şimdi biraz daha zor şeylere, $P_{xy}$'ye, geçelim. Siyah beyaz resimler seçtiğimiz için, $p(x_i \mid y)$, $i$ pikselinin $y$ sınıfı için açık olma olasılığını gösterir. Tıpkı daha önce olduğu gibi, bir olayın meydana geldiği $n_{iy}$ sayısını sayabilmemiz ve bunu $y$'nin toplam oluş sayısına bölebilmemiz gibi, yani $n_y$. Ancak biraz rahatsız edici bir şey var: Belirli pikseller asla siyah olmayabilir (örneğin, iyi kırpılmış görüntülerde köşe pikselleri her zaman beyaz olabilir). İstatistikçilerin bu sorunla baş etmeleri için uygun bir yol, tüm oluşumlara sözde sayımlar eklemektir. Bu nedenle, $n_{iy}$ yerine $n_{iy} + 1$ ve $n_y$ yerine $n_{y} + 1$ kullanıyoruz. Bu aynı zamanda *Laplace Düzleştirme (Smoothing)* olarak da adlandırılır. Geçici görünebilir, ancak Bayesci bir bakış açısından iyi motive edilmiş olabilir.

```{.python .input}
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

Bu $10$ \times 28 \times 28$ olasılıkları görselleştirerek (her sınıf için her piksel için) ortalama görünümlü rakamlar elde edebiliriz.

Şimdi yeni bir görüntüyü tahmin etmek için :eqref:`eq_naive_bayes_estimation`'yi kullanabiliriz. $\mathbf x$ verildiğinde, aşağıdaki işlevler her $y$ için $p(\mathbf x \mid y)p(y)$'yi hesaplar.

```{.python .input}
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

Bu korkunç bir şekilde yanlış gitti! Nedenini bulmak için piksel başına olasılıklara bakalım. Bunlar tipik $0,001$ ile $1$ arasındaki sayılardır. $784$ tanesini çarpıyoruz. Bu noktada, bu sayıları bir bilgisayarda sabit bir aralıkla hesapladığımızı belirtmekte fayda var, dolayısıyla bu kuvvet için de geçerli. Olan şu ki, *sayısal küçümenlik (underflow)* yaşıyoruz, yani tüm küçük sayıları çarpmak, sıfıra yuvarlanana kadar daha da küçük değerlere yol açar. Bunu teorik bir mesele olarak :numref:`sec_maximum_likelihood`da tartıştık, ancak burada pratikteki bir vaka olarak açıkça görüyoruz.

O bölümde tartışıldığı gibi, bunu $\log a b = \log a + \log b$ gerçeğini kullanarak, yani logaritma toplamaya geçerek düzeltiriz. Hem $a$ hem de $b$ küçük sayılar olsa bile, logaritma değerleri uygun bir aralıkta olacaktır.

```{.python .input}
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

Logaritma artan bir fonksiyon olduğundan, şu şekilde yeniden yazabiliriz :eqref:`eq_naive_bayes_estimation`:

$$ \hat{y} = \mathrm{argmax}_y \> \sum_{i=1}^d \log P_{xy}[x_i, y] + \log P_y[y].$$

Aşağıdaki kararlı sürümü uygulayabiliriz:

```{.python .input}
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Şimdi tahminin doğru olup olmadığını kontrol edebiliriz.

```{.python .input}
# Convert label which is a scalar tensor of int32 dtype
# to a Python scalar integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

Şimdi birkaç geçerleme örneği tahmin edersek, Bayes sınıflandırıcısının oldukça iyi çalıştığını görebiliriz.

```{.python .input}
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() 
            for x in X]

X = torch.stack([mnist_train[i][0] for i in range(10,38)], dim=1).squeeze(0)
y = torch.tensor([mnist_train[i][1] for i in range(10,38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Son olarak, sınıflandırıcının genel doğruluğunu hesaplayalım.

```{.python .input}
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_test))], 
                dim=1).squeeze(0)
y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

Modern derin ağlar $0,01$'den daha düşük hata oranlarına ulaşır. Nispeten düşük performans, modelimizde yaptığımız yanlış istatistiksel varsayımlardan kaynaklanmaktadır: Her pikselin yalnızca etikete bağlı olarak *bağımsızca* oluşturulduğunu varsaydık. İnsanların rakamları böyle yazmadığı açıktır ve bu yanlış varsayım, aşırı naif (Bayes) sınıflandırıcımızın çökmesine yol açtı.

## Özet
* Bayes kuralı kullanılarak, gözlenen tüm özelliklerin bağımsız olduğu varsayılarak bir sınıflandırıcı yapılabilir.
* Bu sınıflandırıcı, etiket ve piksel değerlerinin kombinasyonlarının olma sayısını sayarak bir veri kümesi üzerinde eğitilebilir.
* Bu sınıflandırıcı, istenmeyen elektronik posta (spam) tespiti gibi görevler için onlarca yıldır altın standarttı.

## Alıştırmalar
1. $[[0,0], [0,1], [1,0], [1,1]]$ veri kümesini iki öğenin XOR tarafından verilen etiketleri ile, $[0,1,1,0]$, düşünün. Bu veri kümesine dayanan bir naif Bayes sınıflandırıcısının olasılıkları nelerdir? Noktalarımızı başarıyla sınıflandırıyor mu? Değilse, hangi varsayımlar ihlal edilir?
1. Olasılıkları tahmin ederken Laplace düzleştirmeyi kullanmadığımızı ve eğitimde asla gözlenmeyen bir değer içeren bir veri noktasının test zamanında geldiğini varsayalım. Model ne çıkarır?
1. Naif Bayes sınıflandırıcısı, rastgele değişkenlerin bağımlılığının bir grafik yapısıyla kodlandığı belirli bir Bayes ağı örneğidir. Tam teorisi bu bölümün kapsamı dışında olsa da (tüm ayrıntılar için :cite:`Koller.Friedman.2009`a bakınız), XOR modelinde iki giriş değişkeni arasında açık bağımlılığa izin vermenin neden başarılı bir sınıflandırıcı oluşturmaya izin verdiğini açıklayıniz.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/418)
:end_tab:
