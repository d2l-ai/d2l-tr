# Konvolüsyonel Sinir Ağları (LeNet)
:label:`sec_lenet`

Artık tam fonksiyonlu bir CNN'yi monte etmek için gerekli tüm malzemeye sahibiz. Görüntü verileriyle daha önceki karşılaşmamızda, Moda-MNIST veri setindeki giyim resimlerine bir softmax regresyon modeli (:numref:`sec_softmax_scratch`) ve bir MLP modeli (:numref:`sec_mlp_scratch`) uyguladık. Bu tür verileri softmax regresyon ve MLP'lere uygun hale getirmek için, önce $28\times28$ matrisinden her görüntüyü sabit uzunlukta $784$ boyutlu bir vektöre düzleştirdik ve daha sonra bunları tam bağlı katmanlarla işledik. Artık evrimsel katmanlar üzerinde bir tutamımız olduğuna göre, görüntülerimizdeki mekansal yapıyı koruyabiliriz. Tam bağlı katmanları evrimsel katmanlarla değiştirmenin ek bir avantajı olarak, çok daha az parametre gerektiren daha fazla parsimonious modellerin keyfini çıkaracağız.

Bu bölümde, ilk yayınlanan CNN'ler arasında bilgisayar görme görevlerinde performansından dolayı geniş bir dikkat çekmek için *LeNet* tanıtacağız. Model, :cite:`LeCun.Bottou.Bengio.ea.1998` görüntülerindeki el yazısı rakamları tanımak amacıyla AT&T Bell Labs'te araştırmacı olan Yann LeCun tarafından tanıtıldı (ve adlandırıldı). Bu çalışma, teknolojiyi geliştiren on yıllık bir araştırmanın doruk noktalarını temsil ediyordu. 1989'da LeCun, CNN'leri geri yayılım yoluyla başarılı bir şekilde eğitmek için ilk çalışmayı yayınladı.

LeNet, destek vektör makinelerinin performansıyla eşleşen olağanüstü sonuçlar elde etti, daha sonra denetimli öğrenmede baskın bir yaklaşım. LeNet sonunda ATM makinelerinde mevduat işlemek için rakamları tanımak için adapte edilmiştir. Bugüne kadar, bazı ATM'ler hala Yann ve meslektaşı Leon Bottou'nun 1990'larda yazdığı kodu çalıştırıyor!

## LeNet

Yüksek düzeyde, LeNet (LeNet-5) iki bölümden oluşur: (i) iki evrimsel katmandan oluşan bir evrimsel kodlayıcı; ve (ii) üç tam bağlı katmandan oluşan yoğun bir blok; Mimari :numref:`img_lenet`'te özetlenmiştir.

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

Her bir konvolüsyonel bloktaki temel birimler, bir konvolüsyonel tabaka, bir sigmoid aktivasyon fonksiyonu ve müteakip bir ortalama havuzlama işlemidir. ReLU'lar ve max-pooling daha iyi çalışırken, bu keşifler henüz 1990'larda yapılmamıştı. Her bir konvolüsyonel katman bir $5\times 5$ çekirdeği ve sigmoid aktivasyon işlevi kullanır. Bu katmanlar, mekansal olarak düzenlenmiş girdileri bir dizi iki boyutlu özellik eşlemelerine eşler ve genellikle kanal sayısını arttırır. İlk evrimsel tabaka 6 çıkış kanalına, ikincisi ise 16'ya sahiptir. Her $2\times2$ havuzlama işlemi (adım 2), uzamsal altörnekleme yoluyla boyutsallığı $4$ katına düşürür. Konvolusyonel blok tarafından verilen şekle sahip bir çıkış yayar (parti boyutu, kanal sayısı, yükseklik, genişlik).

Konvolusyonel bloktan yoğun bloğa çıktıyı geçirmek için, minibatchtaki her örneği düzleştirmeliyiz. Başka bir deyişle, bu dört boyutlu girdiyi alıp tam bağlı katmanlar tarafından beklenen iki boyutlu girdiye dönüştürüyoruz: bir hatırlatma olarak, arzu ettiğimiz iki boyutlu gösterim, minibatch örneklerini indekslemek için ilk boyutu kullanır ve ikincisini düz vektörü vermek için kullanır her örnek temsili. LeNet'in yoğun bloğu sırasıyla 120, 84 ve 10 çıkış ile üç tam bağlı katman içerir. Hala sınıflandırma gerçekleştirdiğimiz için, 10 boyutlu çıktı katmanı olası çıktı sınıflarının sayısına karşılık gelir.

LeNet'in içinde neler olup bittiğini gerçekten anladığınız noktaya gelirken, umarım aşağıdaki kod parçacığı sizi modern derin öğrenme çerçeveleri ile bu tür modellerin uygulanmasının son derece basit olduğuna ikna edecektir. Sadece `Sequential` bloğunu başlatmamız ve uygun katmanları birbirine bağlamamız gerekiyor.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # `Dense` will transform an input of the shape (batch size, number of
        # channels, height, width) into an input of the shape (batch size,
        # number of channels * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy, OneDeviceStrategy

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

Orijinal modelle küçük bir özgürlük aldık, son kattaki Gauss aktivasyonunu kaldırdık. Bunun dışında, bu ağ orijinal LeNet-5 mimarisiyle eşleşir.

Tek kanallı (siyah beyaz) $28 \times 28$ görüntüsünü ağ üzerinden geçirerek ve çıkış şeklini her katmanda yazdırarak, işlemlerinin :numref:`img_lenet_vert`'ten beklediğimiz şeyle hizaladığından emin olmak için modeli inceleyebiliriz.

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

Konvolusyonel blok boyunca her katmanda gösterim yüksekliğinin ve genişliğinin azaltıldığını (önceki katmanla karşılaştırıldığında) unutmayın. İlk kıvrımsal katman, $5 \times 5$ çekirdeğinin kullanılmasından kaynaklanan yükseklik ve genişlik azalmasını telafi etmek için 2 piksel dolgu kullanır. Buna karşılık, ikinci kıvrımlı tabaka dolgudan vazgeçer ve böylece yükseklik ve genişlik her ikisi de 4 piksel azaltılır. Katmanların yığınına çıktığımızda, kanalların sayısı, ilk konvolüsyonel tabakadan sonra girişteki 1'den 6'ya ve ikinci kıvrımsal tabakadan sonra 16'ya yükselir. Bununla birlikte, her bir havuzlama katmanı yüksekliği ve genişliği yarıya indirir. Son olarak, her tam bağlı katman boyutsallığı azaltır ve sonunda boyutu sınıf sayısıyla eşleşen bir çıktı yayar.

## Eğitim

Modeli uyguladığımıza göre, LeNet'in Moda-MNIST üzerinde nasıl olduğunu görmek için bir deney yapalım.

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

CNN'ler daha az parametre olsa da, her parametre daha fazla çarpmaya katılır çünkü benzer derin MLP'lerden daha hesaplanmaları daha pahalı olabilir. GPU'ya erişiminiz varsa, bu işlemi hızlandırmak için harekete geçirmek için iyi bir zaman olabilir.

:begin_tab:`mxnet, pytorch`
Değerlendirme için :numref:`sec_softmax_scratch`'te tarif ettiğimiz `evaluate_accuracy` işlevinde hafif bir değişiklik yapmamız gerekiyor. Tam veri kümesi ana bellekte olduğundan, model veri kümesiyle hesaplamak için GPU'yu kullanmadan önce GPU belleğine kopyalamamız gerekir.
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if not device:  # Query the first device where the first parameter is on
        device = list(net.collect_params().values())[0].list_ctx()[0]
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Ayrıca GPU'larla başa çıkmak için eğitim fonksiyonumuzu güncellememiz gerekiyor. :numref:`sec_softmax_scratch`'te tanımlanan `train_epoch_ch3`'in aksine, şimdi ileri ve geri yayılımı yapmadan önce her bir veri minibatchini belirlenen cihazımıza (umarım GPU) taşımamız gerekiyor.

Eğitim fonksiyonu `train_ch6`, :numref:`sec_softmax_scratch`'te tanımlanan `train_ch3`'ya da benzer. Birçok katman ileriye doğru ilerleyen ağları uygulayacağımızdan, öncelikle üst düzey API'lere güveneceğiz. Aşağıdaki eğitim işlevi, giriş olarak üst düzey API'lerden oluşturulan bir modeli varsayar ve buna göre optimize edilir. :numref:`subsec_xavier` yılında tanıtıldığı gibi Xavier başlatma kullanarak `device` argümanı ile belirtilen cihazdaki model parametrelerini başlatıyoruz. Tıpkı MLP'lerde olduğu gibi, kayıp fonksiyonumuz çapraz entropi ve minibatch stokastik gradyan iniş yoluyla en aza indiriyoruz. Her bir devirin çalışması on saniye sürdüğünden, eğitim kaybını daha sık görselleştiririz.

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the major difference from `d2l.train_epoch_ch3`
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

Şimdi LeNet-5 modelini eğitip değerlendirelim.

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## Özet

* CNN, evrimsel katmanları kullanan bir ağdır.
* Bir CNN'de, konvolüsyonları, doğrusal olmayanları ve (genellikle) havuzlama işlemlerini ara veriyoruz.
* Bir CNN'de, evrimsel katmanlar tipik olarak, gösterimlerin mekansal çözünürlüğünü yavaş yavaş azaltacak şekilde düzenlenir ve kanal sayısını arttırırlar.
* Geleneksel CNN'lerde, evrimsel bloklar tarafından kodlanan temsiller, çıktı yaymadan önce bir veya daha fazla tam bağlı katman tarafından işlenir.
* LeNet tartışmasız böyle bir ağın ilk başarılı dağıtımı oldu.

## Egzersizler

1. Ortalama havuzlama ile maksimum havuzlama değiştirin. Ne olur?
1. Doğruluğunu artırmak için LeNet'e dayalı daha karmaşık bir ağ oluşturmaya çalışın.
    1. Evrim penceresi boyutunu ayarlayın.
    1. Çıkış kanallarının sayısını ayarlayın.
    1. Etkinleştirme işlevini ayarlayın (örneğin, ReLU).
    1. Evrişim katmanlarının sayısını ayarlayın.
    1. Tam bağlı katmanların sayısını ayarlayın.
    1. Öğrenme oranlarını ve diğer eğitim ayrıntılarını ayarlayın (örneğin, başlatma ve epoch sayısı.)
1. Özgün MNIST veri kümesi üzerinde geliştirilmiş ağ deneyin.
1. Farklı girişler için LeNet'in birinci ve ikinci katmanının aktivasyonlarını gösterin (ör. kazak ve paltolar).

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:
