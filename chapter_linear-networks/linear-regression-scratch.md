# Sıfırdan Doğrusal Regresyon Uygulaması Yaratma
:label:`sec_linear_scratch

Artık doğrusal regresyonun arkasındaki temel fikirleri anladığınıza göre, işe ellerimizi daldırarak kodda uygulamaya başlayabiliriz. Bu bölümde, veri komut işleme hattı (pipeline), model, kayıp fonksiyonu ve minigrup rasgele gradyan iniş optimize edici dahil olmak üzere tüm yöntemi sıfırdan uygulayacağız. Modern derin öğrenme çerçeveleri neredeyse tüm bu çalışmayı otomatikleştirebilirken, bir şeyleri sıfırdan uygulamak, ne yaptığınızı gerçekten bildiğinizden emin olmanın tek yoludur. Dahası, modelleri ihtiyacımıza göre özelleştirken, kendi katmanlarımızı veya kayıp işlevlerimizi tanımlama zamanı geldiğinde, kaputun altında işlerin nasıl ilerlediğini anlamak kullanışlı olacaktır. Bu bölümde, sadece tensörlere ve otomatik türev almaya güveneceğiz. Daha sonra, derin öğrenme çerçevelerinin çekici ek özelliklerden yararlanarak daha kısa bir uygulama sunacağız.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Veri Kümesini Oluşturma

İşleri basitleştirmek için, gürültülü doğrusal bir model için yapay bir veri kümesi oluşturacağız. Görevimiz, veri setimizde bulunan sonlu örnek kümesini kullanarak bu modelin parametrelerini elde etmek olacaktır. Verileri düşük boyutlu tutacağız, böylece kolayca görselleştirebiliriz. Aşağıdaki kod parçacığında, her biri standart bir normal dağılımdan örneklenmiş 2 öznitelikten oluşan 1000 örnekli bir veri kümesi oluşturuyoruz. Böylece, sentetik veri kümemiz matris $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$ olacaktır.

Veri kümemizi oluşturan gerçek parametreler $\mathbf{w} = [2, -3.4]^\top$ ve $b = 4.2$ olacaktır ve sentetik etiketlerimiz aşağıdaki $\epsilon$ gürültü terimli  doğrusal modele göre atanacaktır:

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

$\epsilon$'u öznitelikler ve etiketlerdeki olası ölçüm hatalarını yakalıyor diye düşünebilirsiniz. Standart varsayımların geçerli olduğunu ve böylece $\epsilon$ değerinin ortalaması 0 olan normal bir dağılıma uyduğunu varsayacağız. Problemimizi kolaylaştırmak için, standart sapmasını 0.01 olarak ayarlayacağız. Aşağıdaki kod, sentetik veri kümemizi üretir.

```{.python .input}
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = torch.zeros(size=(num_examples, len(w))).normal_()
    y = torch.matmul(X, w) + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = tf.zeros(shape=(num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, w) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, [num_examples])
    return X, y

true_w = tf.constant([2, -3.4], shape=(2, 1))
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

`Öznitelikler`deki (features değişkeni) her satırın 2 boyutlu bir veri örneğinden oluştuğuna ve `etiketlerdeki` (labels değişkeni) her satırın 1 boyutlu bir etiket değerinden (bir skaler) oluştuğuna dikkat edin.

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

İkinci özntelik `features[:, 1]` ve `labels` kullanılarak bir dağılım grafiği oluşturup ikisi arasındaki doğrusal korelasyonu net bir şekilde gözlemleyebiliriz.

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## Veri Kümesini Okuma

Model eğitimlerinin, veri kümesi üzerinde birden çok geçiş yapmaktan, her seferde bir minigrup örnek almaktan ve bunları modelimizi güncellemek için kullanmaktan oluştuğunu hatırlayın. Bu süreç, makine öğrenmesi algoritmalarını eğitmek için çok temel olduğundan, veri kümesini karıştırmak ve ona minigruplar halinde erişmek için bir yardımcı işlev tanımlamaya değer.

Aşağıdaki kodda, bu işlevselliğin olası bir uygulamasını göstermek için `data_iter` işlevini tanımlıyoruz. Fonksiyon, bir grup boyutunu, bir öznitelik matrisini ve bir etiket vektörünü alarak `batch_size` boyutundaki minigrupları verir. Her bir mini parti, bir dizi öznitelik ve etiketten oluşur.

```{.python .input}
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

Genel olarak, paralelleştirme işlemlerinde mükemmel olan GPU donanımından yararlanmak için makul boyutta minigruplar kullanmak istediğimizi unutmayın. Her örnek, modellerimiz üzerinden paralel olarak beslenebildiği ve her örnek için kayıp fonksiyonunun gradyanı da paralel olarak alınabildiğinden, GPU'lar, yüzlerce örneği yalnızca tek bir örneği işlemek için gerekebileceğinden çok daha az kısa sürede işlememize izin verir.

Biraz sezgi oluşturmak için, ilk küçük bir grup veri örneği okuyup yazdıralım. Her mini gruptaki özniteliklerin şekli bize hem minigrup boyutunu hem de girdi özniteliklerinin sayısını söyler. Aynı şekilde, minigrubumuzun etiketleri de `batch_size` ile verilen şekle sahip olacaktır.

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

Yinelemeyi çalıştırırken, tüm veri kümesi tükenene kadar art arda farklı minigruplar elde ederiz (bunu deneyin). Yukarıda uygulanan yineleme, eğitici amaçlar için iyi olsa da, gerçek problemlerde bizim başımızı belaya sokacak şekillerde verimsizdir. Örneğin, tüm verileri belleğe yüklememizi ve çok sayıda rastgele bellek erişimi gerçekleştirmemizi gerektirir. Derin öğrenme çerçevesinde uygulanan yerleşik yineleyiciler önemli ölçüde daha verimlidir ve hem dosyalarda depolanan verilerle hem de veri akışları aracılığıyla beslenen verilerle ilgilenebilirler.

## Model Parametrelerini İlkleme

Modelimizin parametrelerini minigrup rasgele gradyan inişiyle optimize etmeye başlamadan önce, ilk olarak bazı parametrelere ihtiyacımız var. Aşağıdaki kodda, ağırlıkları, ortalaması 0 ve standart sapması 0.01 olan normal bir dağılımdan rasgele sayılar örnekleyerek ve ek girdiyi 0 olarak ayarlayarak ilkliyoruz.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

Parametrelerimizi başlattıktan sonra, bir sonraki görevimiz, verilerimize yeterince iyi oturana kadar onları güncellemektir. Her güncelleme, parametrelere göre kayıp fonksiyonumuzun gradyanını almayı gerektirir. Bu gradyan göz önüne alındığında, her parametreyi kaybı azaltabilecek yönde güncelleyebiliriz.

Hiç kimse gradyanları açıkça hesaplamak istemediğinden (bu sıkıcı ve hataya açık), gradyanı hesaplamak için :numref:`sec_autograd`da tanıtıldığı gibi otomatik türev almayı kullanırız.

## Modeli Tanımlama

Daha sonra, modelimizi, onun girdileri ve parametreleri çıktıları ile ilişkilendirerek tanımlamalıyız. Doğrusal modelin çıktısını hesaplamak için, $\mathbf{X}$ girdi özniteliklerinin ve $\mathbf{w}$ model ağırlıklarının matris vektör nokta çarpımını alıp her bir örneğe $b$ ofsetini eklediğimizi hatırlayın. Aşağıda $\mathbf{Xw}$ bir vektör ve $b$ bir skalerdir. Yayma mekanizmasını şurada açıklandığı anımsayalım :numref:`subsec_broadcasting`. Bir vektör ve bir skaleri topladığımızda, skaler vektörün her bileşenine eklenir.

```{.python .input}
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return np.dot(X, w) + b
```

```{.python .input}
#@tab pytorch
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b
```

```{.python .input}
#@tab tensorflow
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return tf.matmul(X, w) + b
```

## Kayıp Fonksiyonunu Tanımlama

Modelimizi güncellemek, kayıp fonksiyonumuzun gradyanını almayı gerektirdiğinden, önce kayıp fonksiyonunu tanımlamalıyız. Burada kare kayıp fonksiyonunu şurada açıklandığı gibi kullanacağız :numref:`sec_linear_regression`. Uygulamada, `y` gerçek değerini tahmin edilen değer `y_hat` şekline dönüştürmemiz gerekir. Aşağıdaki işlev tarafından döndürülen sonuç da `y_hat` ile aynı şekle sahip olacaktır.

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## Optimizasyon Algoritmasını Tanımlama

:numref:`sec_linear_regression`da tartıştığımız gibi, doğrusal regresyon kapalı form bir çözüme sahiptir. Ancak, bu doğrusal regresyon hakkında bir kitap değil: Derin öğrenme hakkında bir kitap. Bu kitabın tanıttığı diğer modellerin hiçbiri analitik olarak çözülemediğinden, bu fırsatı minigrup rasgele gradyan inişinin ilk çalışan örneğini tanıtmak için kullanacağız.

Her adımda, veri setimizden rastgele alınan bir minigrup kullanarak, parametrelerimize göre kaybın gradyanını tahmin edeceğiz. Daha sonra kayıpları azaltabilecek yönde parametrelerimizi güncelleyeceğiz. Aşağıdaki kod, bir küme parametre, bir öğrenme hızı ve bir grup boyutu verildiğinde minigrup rasgele gradyan iniş güncellemesini uygular. Güncelleme adımının boyutu, öğrenme oranı `lr` tarafından belirlenir. Kaybımız, örneklerin minigrubu üzerinden bir toplam olarak hesaplandığından, adım boyutumuzu grup boyutuna (`batch_size`) göre normalleştiririz, böylece tipik bir adım boyutunun büyüklüğü, grup boyutu seçimimize büyük ölçüde bağlı olmaz.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

Finally, we will call the optimization algorithm `sgd` to update the model parameters.

## Eğitim

Artık tüm parçaları yerine koyduğumuza göre, ana eğitim döngüsünü uygulamaya hazırız. Bu kodu anlamanız çok önemlidir çünkü derin öğrenmede kariyeriniz boyunca neredeyse aynı eğitim döngülerini tekrar tekrar göreceksiniz.

Her yinelemede, bir minigrup eğitim örneği alacağız ve bir dizi tahmin elde etmek için bunları modelimizden geçireceğiz. Kaybı hesapladıktan sonra, her parametreye göre gradyanları depolayarak ağ üzerinden geriye doğru geçişi başlatırız. Son olarak, model parametrelerini güncellemek için optimizasyon algoritması `sgd`yi çağıracağız.

We elide these details for now but revise them later in :numref:`chap_optimization`.

Özetle, aşağıdaki döngüyü uygulayacağız:

* $(\mathbf {w}, b)$ parametrelerini ilkletin.
* Tamamlanana kadar tekrarlayın
     * Gradyanı hesaplayın: $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
     * Parametreleri güncelleyin: $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

Her bir *dönemde (epoch)*, eğitim veri kümesindeki her örnekten geçtikten sonra (örneklerin sayısının grup boyutuna bölünebildiği varsayılarak) tüm veri kümesini (`data_iter` işlevini kullanarak) yineleyeceğiz. Dönemlerin sayısı, `num_epochs`, ve öğrenme hızı, `lr`, burada sırasıyla 3 ve 0,03 olarak belirlediğimiz hiperparametrelerdir. Ne yazık ki, hiperparametrelerin belirlenmesi zordur ve deneme yanılma yoluyla bazı ayarlamalar gerektirir. Bu ayrıntıları şimdilik atlıyoruz, ancak daha sonra :numref:`chap_optimization` bölümünde tekrarlayacağız.

```{.python .input}
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

Bu durumda, veri kümemizi kendimiz sentezlediğimiz için, gerçek parametrelerin ne olduğunu tam olarak biliyoruz. Böylece eğitimdeki başarımızı, gerçek parametreleri eğitim döngümüz aracılığıyla öğrendiklerimizle karşılaştırarak değerlendirebiliriz. Gerçekten de birbirlerine çok yakın oldukları ortaya çıkıyor.

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

Parametreleri mükemmel bir şekilde elde ettiğimizi kabullenmememiz gerektiğini unutmayın. Bununla birlikte, makine öğrenmesinde, genellikle temeldeki gerçek parametreleri elde etmek ile daha az ilgileniriz ve yüksek derecede doğru tahminlere yol açan parametrelerle daha çok ilgileniriz. Neyse ki, zorlu optimizasyon problemlerinde bile, rasgele gradyan inişi, kısmen derin ağlar için, oldukça doğru tahmine götüren parametrelerin birçok konfigürasyonunun mevcut olmasından dolayı, genellikle dikkate değer ölçüde iyi çözümler bulabilir.

## Özet

* Derin bir ağın, katmanları veya süslü optimize edicileri tanımlamaya gerek kalmadan sadece tensörler ve otomatik türev alma kullanarak nasıl sıfırdan uygulanabileceğini ve optimize edilebileceğini gördük.
* Bu bölüm sadece mümkün olanın yüzeyini gösterir. İlerideki bölümlerde, yeni tanıttığımız kavramlara dayalı yeni modelleri açıklayacak ve bunları daha kısaca nasıl uygulayacağımızı öğreneceğiz.

## Alıştırmalar

1. Ağırlıkları sıfıra ilkleseydik ne olurdu? Algoritma yine de çalışır mıydı?
1. Voltaj ve akım arasında bir model bulmaya çalışan [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) olduğunuzu varsayın. Modelinizin parametrelerini öğrenmek için otomatik türev almayı kullanabilir misiniz?
1. Spektral enerji yoğunluğunu kullanarak bir nesnenin sıcaklığını belirlemek için [Planck Yasası](https://en.wikipedia.org/wiki/Planck%27s_law)'nı kullanabilir misiniz?
1. İkinci türevleri hesaplamak isterseniz karşılaşabileceğiniz sorunlar nelerdir? Onları nasıl düzeltirsiniz?
1. `squared_loss` işlevinde `reshape` işlevi neden gereklidir?
1. Kayıp işlevi değerinin ne kadar hızlı düştüğünü bulmak için farklı öğrenme oranları kullanarak deney yapın.
1. Örneklerin sayısı parti boyutuna bölünemezse, `data_iter` işlevinin davranışına ne olur?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/201)
:end_tab:
