# Naif (Saf) Bayes
:label:`sec_naive_bayes`

Önceki bölümler boyunca, olasılık teorisi ve rastgele değişkenler hakkında bilgi edindik. Bu teoriyi uygulamaya koymak için, *naif Bayes* sınıflandırıcısını tanıtalım. Bu, rakamların sınıflandırmasını yapmamıza izin vermek için olasılık temellerinden başka hiçbir şey kullanmaz.

Öğrenme tamamen varsayımlarda bulunmakla ilgilidir. Daha önce hiç görmediğimiz yeni bir veri örneğini sınıflandırmak istiyorsak, hangi veri örneklerinin birbirine benzer olduğuna dair bazı varsayımlar yapmalıyız. Popüler ve oldukça net bir algoritma olan naif Bayes sınıflandırıcı, hesaplamayı basitleştirmek için tüm özniteliklerin birbirinden bağımsız olduğunu varsayar. Bu bölümde, imgelerdeki karakterleri tanımak için bu modeli uygulayacağız.

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

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Optik Karakter Tanıma

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`, yaygın olarak kullanılan veri kümelerinden biridir. Eğitim için 60.000 imge ve geçerleme için 10.000 imge içerir. Her imge, 0'dan 9'a kadar el yazısıyla yazılmış bir rakam içerir. Görev, her imgeyi karşılık gelen rakama sınıflandırmaktır.

Gluon, veri kümesini İnternet'ten otomatik olarak almak için `data.vision` modülünde bir `MNIST` sınıfı sağlar.
Daha sonra, Gluon hali-hazırda indirilmiş yerel kopyayı kullanacaktır. `train` parametresinin değerini sırasıyla `True` veya `False` olarak ayarlayarak eğitim kümesini mi yoksa test kümesini mi talep ettiğimizi belirtiriz.
Her resim, hem genişliği hem de yüksekliği $28$ olan ve ($28$, $28$, $1$) şekilli gri tonlamalı bir resimdir. Son kanal boyutunu kaldırmak için özelleştirilmiş bir dönüşüm kullanıyoruz. Ek olarak, veri kümesi her pikseli işaretsiz $8$ bitlik bir tamsayı ile temsil eder. Problemi basitleştirmek için bunları ikili öznitelikler halinde nicelendiriyoruz.

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# MNIST'in orijinal piksel değerleri 0-255 arasındadır (rakamlar uint8 olarak depolandığından). 
# Bu bölüm için (orijinal imgede) 128'den büyük piksel değerleri 1'e, 
# 128'den küçük değerler 0'a dönüştürülür. Nedeni için bölüm 18.9.2 ve 18.9.3'e bakın.
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

İmgeyi ve ilgili etiketi içeren belirli bir örneğe erişebiliriz.

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

Burada `image` değişkeninde saklanan örneğimiz, yüksekliği ve genişliği $28$ piksel olan bir imgeye karşılık gelir.

```{.python .input}
#@tab all
image.shape, image.dtype
```

Kodumuz her imgenin etiketini sayıl olarak depolar. Türü $32$ bitlik bir tamsayıdır.

```{.python .input}
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

Aynı anda birden fazla örneğe de erişebiliriz.

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

Bu örnekleri görselleştirelim.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## Sınıflandırma için Olasılık Modeli

Bir sınıflandırma görevinde, bir örneği bir kategoriye eşleriz. Burada bir örnek gri tonlamalı $28\times 28$ imge ve kategori bir rakamdır. (Daha ayrıntılı bir açıklama için bakınız :numref:`sec_softmax`.)
Sınıflandırma görevini ifade etmenin doğal bir yolu, olasılık sorusudur: Özellikler (yani, imge pikselleri) verildiğinde en olası etiket nedir? $\mathbf x\in\mathbb R^d$ ile örneğin özniteliklerini ve $y\in\mathbb R$ ile etiketini belirtiriz. Burada öznitelikler, $2$ boyutlu bir imgeyi $d = 28 ^ 2 = 784$ büyüklüğünde bir vektöre yeniden şekillendirebileceğimiz imge pikselleri ve etiketler rakamlardır.
Öznitelikleri verilen etiketin olasılığı $p(y  \mid  \mathbf{x})$ şeklindedir. Örneğimizde $y=0, \ldots, 9$ için $p(y  \mid  \mathbf{x})$ olan bu olasılıkları hesaplayabilirsek, sınıflandırıcı aşağıda verilen ifade ile tahminini, $\hat{y}$, yapacaktır:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

Maalesef bu, her $\mathbf{x} = x_1, ..., x_d$ değeri için $p(y \mid \mathbf{x})$'yi tahmin etmemizi gerektirir. Her özniteliğin $2$ değerden birini alabileceğini düşünün. Örneğin, $x_1 = 1$ özniteliği, elma kelimesinin belirli bir belgede göründüğünü ve $x_1 = 0$ görünmediğini belirtebilir. Eğer $30$ tane bu tür ikili özniteliğe sahip olsaydık, bu $\mathbf{x}$ girdi vektörünün $2^{30}$ (1 milyardan fazla!) olası değerlerinden herhangi birini sınıflandırmaya hazırlıklı olmamız gerektiği anlamına gelirdi.

Dahası, öğrenme nerede? İlgili etiketi tahmin etmek için her bir olası örneği görmemiz gerekiyorsa, o zaman gerçekten bir model öğrenmiyoruz, sadece veri kümesini ezberliyoruz.

## Naif Bayes Sınıflandırıcı

Neyse ki, koşullu bağımsızlık hakkında bazı varsayımlar yaparak, bazı tümevarımsal önyargılar sunabilir ve nispeten mütevazı bir eğitim örnekleri seçiminden genelleme yapabilen bir model oluşturabiliriz. Başlamak için, sınıflandırıcıyı şu şekilde ifade etmede Bayes teoremini kullanalım:

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Paydanın normalleştirme teriminin $p(\mathbf{x})$ olduğunu ve $y$ etiketinin değerine bağlı olmadığını unutmayın. Sonuç olarak, sadece payı farklı $y$ değerlerinde karşılaştırken endişelenmemiz gerekiyor. Paydanın hesaplanmasının zorlu olduğu ortaya çıksa bile, payı değerlendirebildiğimiz sürece onu görmezden gelerek kurtulabilirdik. Neyse ki, normalleştirme sabitini kurtarmak istesek, bunu da yapabilirdik. Normalleştirme terimini $\sum_y p(y \mid \mathbf{x}) = 1$ olduğundan her zaman kurtarabiliriz.

Şimdi $p( \mathbf{x}  \mid  y)$ üzerine odaklanalım. Zincir olasılık kuralını kullanarak $p( \mathbf{x}  \mid  y)$ terimini şu şekilde ifade edebiliriz:

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Tek başına bu ifade bizi daha ileriye götürmez. Yine de kabaca $2^d$ tane parametreyi tahmin etmeliyiz. Bununla birlikte, *etiketi verildiğinde özniteliklerin koşullu olarak birbirinden bağımsız olduğunu varsayarsak*, aniden çok daha iyi durumda oluruz, çünkü bu terim $\prod_i p(x_i \mid y)$'a sadeleşerek bize şu tahminciyi verir: 

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Her $i$ ve $y$ için $p(x_i=1 \mid y)$ tahmin edebilir ve değerini $P_{xy}[i, y]$ olarak kaydedebilirsek, burada $P_{xy}$ $n$ sınıf sayısı ve $y\in\{1, \ldots, n\}$ olan bir $d\times n$ matrisidir, o zaman bunu $p(x_i = 0 \mid y)$'i tahmin etmek için de kullanabiliriz, yani

$$ 
p(x_i = t_i \mid y) = 
\begin{cases}
    P_{xy}[i, y] & \text{öyle ki } t_i=1 ;\\
    1 - P_{xy}[i, y] & \text{öyle ki } t_i = 0 .
\end{cases}
$$

Ayrıca, her $y$ için $p(y)$ tahmininde bulunur ve $n$ uzunluğunda bir $P_y$ vektörü ile, $P_y[y]$'a kaydederiz. Ardından, herhangi bir yeni örnek $\mathbf t = (t_1, t_2, \ldots, t_d)$ için şunu hesaplayabiliriz;

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

üstelik herhangi bir $y$ için. Dolayısıyla, koşullu bağımsızlık varsayımımız, modelimizin karmaşıklığını öznitelik sayısına bağlı $\mathcal{O}(2^dn)$ üstel bir bağımlılıktan $\mathcal{O}(dn)$ olan doğrusal bir bağımlılığa almıştır.

## Eğitim

Şimdi sorun, $P_{xy}$ ve $P_y$'yi bilmiyor olmamızdır. Bu nedenle, önce bazı eğitim verileri verildiğinde bu değerleri tahmin etmemiz gerekiyor. Bu, modeli *eğitmektir*. $P_y$'yi tahmin etmek çok zor değil. Sadece $10$ sınıfla uğraştığımız için, her rakam için görülme sayısını, $n_y$, sayabilir ve bunu toplam veri miktarına $n$ bölebiliriz. Örneğin, 8 rakamı $n_8 = 5.800$ kez ortaya çıkarsa ve toplam $n = 60.000$ imgemiz varsa, olasılık tahminimiz $p(y=8) = 0.0967$ olur.

```{.python .input}
X, Y = mnist_train[:]  # Tüm eğitim örnekleri

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Şimdi biraz daha zor şeylere, $P_{xy}$'ye, geçelim. Siyah beyaz imgeler seçtiğimiz için, $p(x_i \mid y)$, $i$ pikselinin $y$ sınıfı için açık olma olasılığını gösterir. Tıpkı daha önce olduğu gibi, bir olayın meydana geldiği $n_{iy}$ sayısını sayabilmemiz ve bunu $y$'nin toplam oluş sayısına bölebilmemiz gibi, yani $n_y$. Ancak biraz rahatsız edici bir şey var: Belirli pikseller asla siyah olmayabilir (örneğin, iyi kırpılmış imgelerde köşe pikselleri her zaman beyaz olabilir). İstatistikçilerin bu sorunla baş etmeleri için uygun bir yol, tüm oluşumlara sözde sayımlar eklemektir. Bu nedenle, $n_{iy}$ yerine $n_{iy} + 1$ ve $n_y$ yerine $n_{y} + 1$ kullanıyoruz. Bu aynı zamanda *Laplace Düzleştirme (Smoothing)* olarak da adlandırılır. Geçici görünebilir, ancak Bayesci bir bakış açısından iyi motive edilmiş olabilir.

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

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 1), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

Bu $10 \times 28 \times 28$ olasılıkları görselleştirerek (her sınıf için her piksel için) ortalama görünümlü rakamlar elde edebiliriz.

Şimdi yeni bir imgeyi tahmin etmek için :eqref:`eq_naive_bayes_estimation` denklemini kullanabiliriz. $\mathbf x$ verildiğinde, aşağıdaki işlevler her $y$ için $p(\mathbf x \mid y)p(y)$'yi hesaplar.

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

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

Bu korkunç bir şekilde yanlış gitti! Nedenini bulmak için piksel başına olasılıklara bakalım. Bunlar tipik $0.001$ ile $1$ arasındaki sayılardır. $784$ tanesini çarpıyoruz. Bu noktada, bu sayıları bir bilgisayarda sabit bir aralıkla hesapladığımızı belirtmekte fayda var, dolayısıyla bu kuvvet için de geçerli. Olan şu ki, *sayısal küçümenlik (underflow)* yaşıyoruz, yani tüm küçük sayıları çarpmak, sıfıra yuvarlanana kadar daha da küçük değerlere yol açar. Bunu teorik bir mesele olarak :numref:`sec_maximum_likelihood içinde tartıştık, ancak burada pratikteki bir vaka olarak açıkça görüyoruz.

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

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Logaritma artan bir fonksiyon olduğundan, :eqref:`eq_naive_bayes_estimation` denklemini şu şekilde yeniden yazabiliriz:

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

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

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Şimdi tahminin doğru olup olmadığını kontrol edebiliriz.

```{.python .input}
# Karşılaştırma için int32 dtype skaler tensörü olan etiketi 
# Python skaler tamsayısına dönüştürün
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
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

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
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
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

Modern derin ağlar $0.01$'den daha düşük hata oranlarına ulaşır. Nispeten düşük performans, modelimizde yaptığımız yanlış istatistiksel varsayımlardan kaynaklanmaktadır: Her pikselin yalnızca etikete bağlı olarak *bağımsızca* oluşturulduğunu varsaydık. İnsanların rakamları böyle yazmadığı açıktır ve bu yanlış varsayım, aşırı naif (Bayes) sınıflandırıcımızın çökmesine yol açtı.

## Özet
* Bayes kuralı kullanılarak, gözlenen tüm özniteliklerin bağımsız olduğu varsayılarak bir sınıflandırıcı yapılabilir.
* Bu sınıflandırıcı, etiket ve piksel değerlerinin kombinasyonlarının olma sayısını sayarak bir veri kümesi üzerinde eğitilebilir.
* Bu sınıflandırıcı, istenmeyen elektronik posta (spam) tespiti gibi görevler için onlarca yıldır altın standarttı.

## Alıştırmalar
1. $[[0,0], [0,1], [1,0], [1,1]]$ veri kümesini iki öğenin XOR tarafından verilen etiketleri ile, $[0,1,1,0]$, düşünün. Bu veri kümesine dayanan bir naif Bayes sınıflandırıcısının olasılıkları nelerdir? Noktalarımızı başarıyla sınıflandırıyor mu? Değilse, hangi varsayımlar ihlal edilir?
1. Olasılıkları tahmin ederken Laplace düzleştirmeyi kullanmadığımızı ve eğitimde asla gözlenmeyen bir değer içeren bir veri örneğinin test zamanında geldiğini varsayalım. Model ne çıkarır?
1. Naif Bayes sınıflandırıcısı, rastgele değişkenlerin bağımlılığının bir grafik yapısıyla kodlandığı belirli bir Bayes ağı örneğidir. Tam teorisi bu bölümün kapsamı dışında olsa da (tüm ayrıntılar için :cite:`Koller.Friedman.2009` çalışmasına bakınız), XOR modelinde iki girdi değişkeni arasında açık bağımlılığa izin vermenin neden başarılı bir sınıflandırıcı oluşturmaya izin verdiğini açıklayıniz.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1101)
:end_tab:
