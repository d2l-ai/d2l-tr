# Sıfırdan Softmaks Regresyon Uygulaması Yaratma
:label:`sec_softmax_scratch`

(**Sıfırdan lineer regresyon uyguladığımız gibi,**) softmax regresyonunun da benzer şekilde temel olduğuna ve (**kanlı ayrıntılarını**) (~~softmax regresyon~~) ve nasıl kendinizin uygulanacağını bilmeniz gerektiğine inanıyoruz.
:numref:`sec_fashion_mnist` içinde yeni eklenen Fashion-MNIST veri kümesiyle çalışacağız, grup boyutu 256 olan bir veri yineleyicisi kuracağız.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Model Parametrelerini İlkletme

Doğrusal regresyon örneğimizde olduğu gibi, buradaki her örnek sabit uzunlukta bir vektörle temsil edilecektir. Ham veri kümesindeki her örnek $28 \times 28$'lik görseldir. Bu bölümde, [**her bir görseli 784 uzunluğundaki vektörler olarak ele alarak düzleştireceğiz**]. Gelecekte, görsellerdeki uzamsal yapıyı kullanmak için daha karmaşık stratejilerden bahsedeceğiz, ancak şimdilik her piksel konumunu yalnızca başka bir öznitelik olarak ele alıyoruz.

Softmaks regresyonunda, sınıflar kadar çıktıya sahip olduğumuzu hatırlayın. (**Veri kümemiz 10 sınıf içerdiğinden, ağımızın çıktı boyutu 10 olacaktır**). Sonuç olarak, ağırlıklarımız $784 \times 10$ matrisi ve ek girdiler $1 \times 10$ satır vektörü oluşturacaktır. Doğrusal regresyonda olduğu gibi, `W` ağırlıklarımızı Gauss gürültüsüyle ve ek girdilerimizi ilk değerini 0 alacak şekilde başlatacağız.

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Softmaks İşlemini Tanımlama

Softmaks regresyon modelini uygulamadan önce, toplam operatörünün bir tensörde belirli boyutlar boyunca nasıl çalıştığını :numref:`subseq_lin-alg-reduction` ve :numref:`subseq_lin-alg-non-reduction` içinde anlatıldığı gibi kısaca gözden geçirelim. [**Bir `X` matrisi verildiğinde, tüm öğeleri (varsayılan olarak) veya yalnızca aynı eksendeki öğeleri toplayabiliriz**], örneğin aynı sütun (eksen 0) veya aynı satır (eksen 1) üzerinden. `X` (2, 3) şeklinde bir tensör ise ve sütunları toplarsak, sonucun (3,) şeklinde bir vektör olacağını unutmayın. Toplam operatörünü çağırırken, üzerinde topladığımız boyutu daraltmak yerine esas tensördeki eksen sayısını korumayı belirtebiliriz. Bu, (1, 3) şeklinde iki boyutlu bir tensörle sonuçlanacaktır.

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

(**Artık softmaks işlemini uygulamaya**) hazırız. Softmaks'ın üç adımdan oluştuğunu hatırlayın: (i) Her terimin üssünü alıyoruz (`exp` kullanarak); (ii) her bir örnek için normalleştirme sabitini elde etmek için her satırı topluyoruz (toplu işte (batch) örnek başına bir satırımız vardır); (iii) her satırı normalleştirme sabitine bölerek sonucun toplamının 1 olmasını sağlıyoruz. Koda bakmadan önce, bunun bir denklem olarak nasıl ifade edildiğini hatırlayalım:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

Payda veya normalleştirme sabiti bazen *bölmeleme fonksiyonu* olarak da adlandırılır (ve logaritmasına log-bölmeleme fonksiyonu denir). Bu ismin kökenleri, ilgili bir denklemin bir parçacıklar topluluğu üzerindeki dağılımı modellediği [istatistiksel fizik](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))'tedir.

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # Burada yayin mekanizmasi uygulanir
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # Burada yayin mekanizmasi uygulanir
```

Gördüğünüz gibi, herhangi bir rastgele girdi için, [**her bir öğeyi negatif olmayan bir sayıya dönüştürüyoruz**]. Ayrıca, olasılık belirtmek için gerektiği gibi, her satırın toplamı 1'dir.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

Bu matematiksel olarak doğru görünse de, uygulamamızda biraz özensiz davrandık çünkü matrisin büyük veya çok küçük öğeleri nedeniyle sayısal taşma (overflow) veya küçümenliğe (underflow) karşı önlem alamadık.

## Modeli Tanımlama

Artık softmaks işlemini tanımladığımıza göre, [**softmaks regresyon modelini uygulayabiliriz**]. Aşağıdaki kod, girdinin ağ üzerinden çıktıya nasıl eşlendiğini tanımlar. Verileri modelimizden geçirmeden önce, `reshape` işlevini kullanarak toplu işteki (batch) her esas görseli bir vektör halinde düzleştirdiğimize dikkat edin.

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## Kayıp Fonksiyonunu Tanımlama

Daha sonra, :numref:`sec_softmax` içinde tanıtıldığı gibi, çapraz entropi kaybı işlevini uygulamamız gerekir. Bu, tüm derin öğrenmede en yaygın kayıp işlevi olabilir, çünkü şu anda sınıflandırma sorunları regresyon sorunlarından çok daha fazladır.

Çapraz entropinin, gerçek etikete atanan tahmin edilen olasılığın negatif log-olabilirliğini aldığını hatırlayın. Bir Python for-döngüsü (verimsiz olma eğilimindedir) ile tahminler üzerinde yinelemek yerine, tüm öğeleri tek bir operatörle seçebiliriz. 
Aşağıda, [**3 sınıf üzerinde tahmin edilen olasılıkların 2 örneğini ve bunlara karşılık gelen `y` etiketleriyle `y_hat` örnek verilerini oluşturuyoruz.**] `y` ile, ilk örnekte birinci sınıfın doğru tahmin olduğunu biliyoruz ve ikinci örnekte üçüncü sınıf temel referans değerdir. [**`y`'yi `y_hat` içindeki olasılıkların indeksleri olarak kullanarak,**] ilk örnekte birinci sınıfın olasılığını ve ikinci örnekte üçüncü sınıfın olasılığını seçiyoruz.

Aşağıda, 3 sınıf üzerinden tahmin edilen olasılıkların 2 örneğini içeren bir oyuncak verisi `y_hat`'i oluşturuyoruz. Ardından birinci örnekte birinci sınıfın olasılığını ve ikinci örnekte üçüncü sınıfın olasılığını seçiyoruz.

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Artık, (**çapraz entropi kaybı işlevini**) tek bir kod satırı ile verimli bir şekilde uygulayabiliriz.

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## Sınıflandırma Doğruluğu

Tahmin edilen olasılık dağılımı `y_hat` göz önüne alındığında, genellikle kesin bir tahmin vermemiz gerektiğinde tahmin edilen en yüksek olasılığa sahip sınıfı seçeriz. Aslında, birçok uygulama bir seçim yapmamızı gerektirir. Gmail, bir e-postayı "Birincil", "Sosyal", "Güncellemeler" veya "Forumlar" olarak sınıflandırmalıdır. Olasılıkları dahili olarak tahmin edebilir, ancak günün sonunda sınıflar arasından birini seçmesi gerekir.

Tahminler `y` etiket sınıfıyla tutarlı olduğunda doğrudur. Sınıflandırma doğruluğu, doğru olan tüm tahminlerin oranıdır. Doğruluğu doğrudan optimize etmek zor olabilse de (türevleri alınamaz), genellikle en çok önemsediğimiz performans ölçütüdür ve sınıflandırıcıları eğitirken neredeyse her zaman onu rapor edeceğiz.

Doğruluğu hesaplamak için aşağıdakileri yapıyoruz. İlk olarak, `y_hat` bir matris ise, ikinci boyutun her sınıf için tahmin puanlarını sakladığını varsayıyoruz. Her satırdaki en büyük girdi için dizine göre tahmin edilen sınıfı elde ederken `argmax` kullanırız. Ardından, [**tahmin edilen sınıfı gerçek referans değer `y` ile karşılaştırırız**]. Eşitlik operatörü `==` veri türlerine duyarlı olduğundan, `y_hat` veri türünü `y` ile eşleşecek şekilde dönüştürürüz. Sonuç, 0 (yanlış) ve 1 (doğru) girişlerini içeren bir tensördür. Toplamlarını almak doğru tahminlerin sayısını verir.

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Doğru tahminlerin sayısını hesaplayın."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

Önceden tanımlanan `y_hat` ve `y` değişkenlerini sırasıyla tahmin edilen olasılık dağılımları ve etiketler olarak kullanmaya devam edeceğiz. İlk örneğin tahmin sınıfının 2 olduğunu görebiliriz (satırın en büyük öğesi dizin 2 ile 0.6'dır), bu gerçek etiket, 0 ile tutarsızdır. İkinci örneğin tahmin sınıfı 2'dir (satırın en büyük öğesi 2 endeksi ile 0.5'tir) ve bu gerçek etiket 2 ile tutarlıdır. Bu nedenle, bu iki örnek için sınıflandırma doğruluk oranı 0.5'tir.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

Benzer şekilde, veri yineleyici `data_iter` aracılığıyla erişilen [**bir veri kümesindeki herhangi bir `net` modelinin doğruluğunu hesaplayabiliriz**].

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Bir veri kümesinde bir modelin doğruluğunu hesaplayın."""
    metric = Accumulator(2)  # Doğru tahmin sayısı, tahmin sayısı
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Bir veri kumesinde bir modelin doğruluğunu hesaplayın."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Modeli değerlendirme moduna kurun
    metric = Accumulator(2)  # Doğru tahmin sayısı, tahmin sayısı

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Burada, `Accumulator`, birden çok değişken üzerindeki toplamları biriktirmek için bir yardımcı sınıftır. Yukarıdaki `evaluate_accuracy` işlevinde, `Accumulator` örneğinde sırasıyla hem doğru tahminlerin sayısını hem de tahminlerin sayısını depolamak için 2 değişken oluştururuz. Veri kümesini yineledikçe her ikisi de zaman içinde birikecektir.

```{.python .input}
#@tab all
class Accumulator:  #@save
    """`n` değişken üzerinden toplamları biriktirmek için"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

[**`net` modelini rastgele ağırlıklarla başlattığımız için, bu modelin doğruluğu rastgele tahmin etmeye yakın olmalıdır**], yani 10 sınıf için 0.1 gibi.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## Eğitim

Softmaks regresyonu için [**eğitim döngüsü**], :numref:`sec_linear_scratch` içindeki doğrusal regresyon uygulamamızı okursanız, çarpıcı bir şekilde tanıdık gelebilir. Burada uygulamayı yeniden kullanılabilir hale getirmek için yeniden düzenliyoruz. İlk olarak, bir dönemi (epoch) eğitmek için bir işlev tanımlıyoruz. `updater`'in, grup boyutunu bağımsız değişken olarak kabul eden, model parametrelerini güncellemek için genel bir işlev olduğuna dikkat edin. `d2l.sgd` işlevinin bir sarmalayıcısı (wrapper) veya bir çerçevenin yerleşik optimizasyon işlevi olabilir.

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Bir modeli bir dönem içinde eğitme (Bölüm 3'te tanımlanmıştır)."""
    # Eğitim kaybı toplamı, eğitim doğruluğu toplamı, örnek sayısı
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Gradyanları hesaplayın ve parametreleri güncelleyin
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Eğitim kaybını ve doğruluğunu döndür
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Bölüm 3'te tanımlanan eğitim döngüsü."""
    # Modeli eğitim moduna kurun
    if isinstance(net, torch.nn.Module):
        net.train()
    # Eğitim kaybı toplamı, eğitim doğruluğu toplamı, örnek sayısı
    metric = Accumulator(3)
    for X, y in train_iter:
        # Gradyanları hesaplayın ve parametreleri güncelleyin
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # PyTorch yerleşik optimize edicisini ve kayıp kriterini kullanma
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # Özel olarak oluşturulmuş optimize edicisini ve kayıp ölçütünü kullanma
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Eğitim kaybını ve doğruluğunu döndür
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Bölüm 3'te tanımlanan eğitim döngüsü."""
     # Eğitim kaybı toplamı, eğitim doğruluğu toplamı, örnek sayısı
    metric = Accumulator(3)
    for X, y in train_iter:
        # Gradyanları hesaplayın ve parametreleri güncelleyin
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Kullanıcıların bu kitapta uygulayabilecekleri (tahminler, etiketler) 
            # yerine kayıp alımları (etiketler, tahminler) için Keras uygulamaları
            # ör. yukarıda uyguladığımız `cross_entropy`
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Varsayılan olarak Keras kaybı, bir toplu iş içindeki ortalama kaybı döndürür
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
     # Eğitim kaybını ve doğruluğunu döndür
    return metric[0] / metric[2], metric[1] / metric[2]
```

Eğitim işlevinin uygulamasını göstermeden önce, [**verileri animasyonda (canlandırma) çizen bir yardımcı program sınıfı tanımlıyoruz**]. Yine kitabın geri kalanında kodu basitleştirmeyi amaçlamaktadır.

```{.python .input}
#@tab all
class Animator:  #@save
    """Animasyonda veri çizdirme"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Çoklu çizgileri artarak çizdir
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Argumanları elde tutmak için bir lambda işlevi kullan
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Çoklu veri noktalarını şekile ekleyin
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

[~~Eğitim işlevi~~]
Aşağıdaki eğitim işlevi daha sonra, `num_epochs` ile belirtilen birden çok dönem için `train_iter` aracılığıyla erişilen bir eğitim veri kümesinde bir `net` modeli eğitir. Her dönemin sonunda model, `test_iter` aracılığıyla erişilen bir test veri kümesinde değerlendirilir. Eğitimin ilerlemesini görselleştirmek için `Animator` sınıfından yararlanacağız.

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Bir modeli eğitin (Bölüm 3'te tanımlanmıştır)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['egitim kaybi', 'egitim dogr', 'test dogr'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

Sıfırdan bir uygulama olarak, modelin kayıp fonksiyonunu 0.1 öğrenme oranıyla optimize ederek :numref:`sec_linear_scratch` içinde tanımlanan [**minigrup rasgele gradyan inişini kullanıyoruz**].

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """Minigrup rasgele gradyan inişini kullanarak parametreleri güncellemek için."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Şimdi [**modeli 10 dönem ile eğitiyoruz**]. Hem dönem sayısının (`num_epochs`) hem de öğrenme oranının (`lr`) ayarlanabilir hiper parametreler olduğuna dikkat edin. Değerlerini değiştirerek modelin sınıflandırma doğruluğunu artırabiliriz.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## Tahminleme

Artık eğitim tamamlandı, [**modelimiz bazı imgeleri sınıflandırmaya**] hazır. Bir dizi resim verildiğinde, bunların gerçek etiketlerini (metin çıktısının ilk satırı) ve modelden gelen tahminleri (metin çıktısının ikinci satırı) karşılaştıracağız.

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Etiketleri tahmin etme (Bölüm 3'te tanımlanmıştır)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Özet

* Softmaks regresyonu ile çok sınıflı sınıflandırma için modeller eğitebiliriz.
* Softmaks regresyonunun eğitim döngüsü doğrusal regresyondakine çok benzer: Verileri alın ve okuyun, modelleri ve kayıp fonksiyonlarını tanımlayın, ardından optimizasyon algoritmalarını kullanarak modelleri eğitin. Yakında öğreneceğiniz gibi, en yaygın derin öğrenme modellerinin benzer eğitim yordamları vardır.

## Alıştırmalar

1. Bu bölümde, softmaks işlemini matematiksel tanımına dayalı olarak doğrudan uyguladık. Bu hangi sorunlara neden olabilir? İpucu: $\exp(50)$'nin boyutunu hesaplamaya çalışın.
1. Bu bölümdeki `cross_entropy` işlevi, çapraz entropi kaybı işlevinin tanımına göre uygulandı. Bu uygulamadaki sorun ne olabilir? İpucu: Logaritmanın etki alanını düşünün.
1. Yukarıdaki iki sorunu çözmek için düşünebileceğiniz çözümler nelerdir?
1. En olası etiketi iade etmek her zaman iyi bir fikir midir? Örneğin, bunu tıbbi teşhis için yapar mıydınız?
1. Bazı özniteliklere dayanarak sonraki kelimeyi tahmin etmek için softmaks regresyonunu kullanmak istediğimizi varsayalım. Geniş bir kelime dağarcığı kullanımından ortaya çıkabilecek problemler nelerdir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/225)
:end_tab:
