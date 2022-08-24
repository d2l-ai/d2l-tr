# Öğrenme Oranını Zamanlama
:label:`sec_scheduler`

Şimdiye kadar öncelikle ağırlık vektörlerinin güncellendikleri *oran* yerine ağırlık vektörlerinin nasıl güncelleneceğine dair optimizasyon *algoritmalarına* odaklandık. Bununla birlikte, öğrenme oranının ayarlanması genellikle gerçek algoritma kadar önemlidir. Göz önünde bulundurulması gereken bir dizi husus vardır: 

* En çok açıkçası öğrenme oranının *büyüklüğü* önemlidir. Çok büyükse, optimizasyon ıraksar, eğer çok küçükse, eğitilmesi çok uzun sürer veya yarı-optimal bir sonuç elde ederiz. Daha önce problemin sağlamlık sayısının önemli olduğunu gördük (ayrıntılar için bkz. :numref:`sec_momentum`). Sezgisel olarak, en hassas yöndeki değişim miktarının en az hassas olanına oranıdır.
* İkincisi, sönme oranı da aynı derecede önemlidir. Öğrenme oranı büyük kalırsa, en küçük seviyeyi atlayabilir ve böylece eniyiye ulaşamayabiliriz. :numref:`sec_minibatch_sgd` içinde bunu ayrıntılı olarak tartıştık ve :numref:`sec_sgd` içinde performans garantilerini analiz ettik. Kısacası, sönme oranını istiyoruz, ama muhtemelen dışbükey problemler için $\mathcal{O}(t^{-\frac{1}{2}})$ daha yavaş olanlar iyi bir seçim olurdu.
* Eşit derecede önemli olan bir diğer yön ise *ilklemedir*. Bu, hem parametrelerin başlangıçta nasıl ayarlandığı (ayrıntılar için bkz.:numref:`sec_numerical_stability`) hem de başlangıçta nasıl evrimleştikleri de ilgilidir. Bu, *ısınma* diye tabir edilir, yani başlangıçta çözüme doğru ne kadar hızlı ilerlemeye başladığımızla ilgilidir. Başlangıçta büyük adımlar yararlı olmayabilir, özellikle de ilk parametre kümesi rastgele olduğundan. İlk güncelleme yönergeleri de oldukça anlamsız olabilir.
* Son olarak, döngüsel öğrenme hızı ayarlaması gerçekleştiren bir dizi optimizasyon türü vardır. Bu, bu bölümün kapsamı dışındadır. Okuyucuya :cite:`Izmailov.Podoprikhin.Garipov.ea.2018` çalışmasındaki ayrıntıları gözden geçirmesini tavsiye ederiz, örneğin, parametrelerin izlediği *yolun* üzerinden ortalama alarak daha iyi çözümler elde etmek gibi.

Öğrenme oranlarını yönetmek için gereken çok fazla ayrıntı olduğu göz önüne alındığında, çoğu derin öğrenme çerçevesinin bununla otomatik olarak başa çıkabilmesi için araçlar vardır. Bu bölümde, farklı zamanlamaların doğruluk üzerindeki etkilerini gözden geçireceğiz ve ayrıca bunun bir *öğrenme oranı zamanlayıcısı* aracılığıyla nasıl verimli bir şekilde yönetilebileceğini göstereceğiz. 

## Basit Örnek Problem

Kolayca hesaplanabilecek kadar ucuz, ancak bazı önemli hususları göstermek için yeterince açık olmayan bir basit örnek problem ile başlıyoruz. Bunun için Fashion-MNIST'e uygulandığı gibi LeNet'in biraz modern bir sürümünü seçin (`relu` yerine `sigmoid` aktivasyon, MaxPooling (Maksimum Havuzlama) yerine AveragePooling (Ortalama Havuzlama)). Dahası, ağı performans için melezleştiriyoruz. Kodun çoğu standart olduğundan, daha fazla ayrıntılı tartışma yapmadan sadece temel bilgileri sunuyoruz. Gerektiğinde anımsamak için bkz. :numref:`chap_cnn`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Kod, evrişimli sinir ağları bölümünün lenet kısmında tanımlanan 
# `d2l.train_ch6` ile neredeyse aynıdır.
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Kod, evrişimli sinir ağları bölümünün lenet kısmında tanımlanan 
# `d2l.train_ch6` ile neredeyse aynıdır.
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# Kod, evrişimli sinir ağları bölümünün lenet kısmında tanımlanan 
# `d2l.train_ch6` ile neredeyse aynıdır.
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0, 
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Bu algoritmayı $0.3$ öğrenme oranı ve $30$ yinelemeyle eğitmek için varsayılan ayarlarla çalıştırırsak ne olacağına bir göz atalım. Test doğruluğu açısından ilerleme bir noktanın ötesinde dururken eğitim doğruluğunun nasıl artmaya devam ettiğine dikkat edin. Her iki eğri arasındaki boşluk aşırı öğrenmeyi işaret eder.

```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Zamanlayıcılar

Öğrenme oranını ayarlamanın bir yolu, her adımda açıkça ayarlamaktır. Bu, `set_learning_rate` yöntemi ile elverişli bir şekilde elde edilir. Her dönemden sonra (hatta her minigrup işleminden sonra), örneğin, optimizasyonun nasıl ilerlediğine yanıt olarak dinamik bir şekilde, aşağı doğru ayarlayabiliriz.

```{.python .input}
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

Daha genel olarak bir zamanlayıcı tanımlamak istiyoruz. Güncelleme sayısı ile çağrıldığında, öğrenme oranının uygun değerini döndürür. Öğrenme oranını $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$'ye ayarlayan basit bir tane tanımlayalım.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Davranışını bir dizi değer üzerinde çizelim.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Şimdi bunun Fashion-MNIST eğitimi için nasıl bittiğini görelim. Zamanlayıcıyı eğitim algoritmasına ek bir argüman olarak sağlıyoruz.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Bu eskisinden biraz daha iyi çalıştı. İki şey öne çıkıyor: Eğri daha öncekinden daha pürüzsüzdü. İkincisi, daha az aşırı öğrenme vardı. Ne yazık ki, belirli stratejilerin neden *teoride* daha az aşırı öğrenmeye yol açtığı tam olarak çözülmüş bir soru değil. Daha küçük bir adım boyutunun sıfıra yakın ve dolayısıyla daha basit olan parametrelere yol açacağına dair argümanlar vardır. Bununla birlikte, bu fenomeni tam olarak açıklamaz, çünkü gerçekten erken durmayız, sadece öğrenme oranını nazikçe azaltırız.

## Politikalar

Tüm öğrenme oranı zamanlayıcılarını kapsayamasak da, aşağıda popüler politikalara kısa bir genel bakış vermeye çalışıyoruz. Yaygın seçenekler polinom sönme ve parçalı sabit zamanlamalardır. Bunun ötesinde, kosinüs öğrenme oranı zamanlamalarının bazı problemler üzerinde deneysel olarak iyi çalıştığı tespit edilmiştir. Son olarak, bazı problemlerde, büyük öğrenme oranları kullanmadan önce optimize ediciyi ısıtmak faydalıdır. 

### Çarpan Zamanlayıcı

Bir polinom sönmesine bir alternatif, $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ için $\alpha \in (0, 1)$ olan çarpımsal bir çözüm olabilir. Öğrenme oranının makul bir alt sınırın ötesinde sönmesini önlemek için güncelleme denklemi genellikle $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$ olarak değiştirilir.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Bu, `lr_scheduler.FactorScheduler` nesnesi aracılığıyla MXNet'te yerleşik bir zamanlayıcı tarafından da gerçekleştirilebilir. Isınma süresi, ısınma modu (doğrusal veya sabit), istenen güncelleme sayısı, vb. gibi birkaç parametre daha alır; ileriye gidersek yerleşik zamanlayıcıları uygun olarak kullanacağız ve yalnızca işlevselliklerini burada açıklayacağız. Gösterildiği gibi, gerekirse kendi zamanlayıcınızı oluşturmak oldukça basittir. 

### Çok Çarpanlı Zamanlayıcı

Derin ağları eğitmek için yaygın bir strateji, öğrenme oranını parçalı sabit tutmak ve sık sık belirli bir miktarda azaltmaktır. Diğer bir deyişle, bir dizi zaman verildiğinde hızın ne zaman düşürüleceğidir, örneğin $s = \{5, 10, 20\}$ ve $t \in s$ için $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$'nın azalması. Değerlerin her adımda yarıya indirildiğini varsayarsak, bunu aşağıdaki gibi uygulayabiliriz.

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler) 
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
  
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Bu parçalı sabit öğrenme oranı zamanlamasının arkasındaki sezgi, ağırlık vektörlerinin dağılımı açısından sabit bir noktaya ulaşılana kadar optimizasyonun ilerlemesine izin vermesidir. O zaman (ve ancak o zaman) daha yüksek kaliteli bir vekil elde etmek için oranı iyi bir yerel minimuma düşürürüz. Aşağıdaki örnek, bunun nasıl biraz daha iyi çözümler üretebileceğini göstermektedir.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Kosinüs Zamanlayıcı

:cite:`Loshchilov.Hutter.2016` tarafından oldukça şaşırtıcı bir sezgisel yöntem önerildi. Başlangıçta öğrenme oranını çok büyük ölçüde düşürmek istemeyebileceğimiz ve dahası, çözümü sonunda çok küçük bir öğrenme oranı kullanarak “ince ayarlamak” isteyebileceğimiz gözlemine dayanıyor. Bu, $t \in [0, T]$ aralığındaki öğrenme oranları için aşağıdaki işlevsel formla kosinüs benzeri bir zamanlama ile sonuçlanır. 

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

Burada $\eta_0$ ilk öğrenme oranı, $\eta_T$ ise $T$ zamanındaki hedef orandır. Ayrıca, $t > T$ için değeri tekrar artırmadan $\eta_T$'ya sabitleriz. Aşağıdaki örnekte, maksimum güncelleme adımını $T = 20$'ye ayarladık.

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
  
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Bilgisayarla görme bağlamında bu zamanlama gelişmiş sonuçlara yol aça*bilir*. Yine de, bu tür iyileştirmelerin garanti edilmediğini unutmayın (aşağıda görülebileceği gibi).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Isınma

Bazı durumlarda parametrelerin ilklenmesi, iyi bir çözümü garanti etmek için yeterli değildir. Bu, özellikle kararsız optimizasyon sorunlarına yol açabilecek bazı gelişmiş ağ tasarımları için bir sorundur. Başlangıçta ıraksamayı önlemek için yeterince küçük bir öğrenme oranı seçerek bunu ele alabiliriz. Ne yazık ki bu ilerlemenin yavaş olduğu anlamına gelir. Tersine, büyük bir öğrenme oranı başlangıçta ıraksamaya yol açar. 

Bu ikilem için oldukça basit bir çözüm, öğrenme hızının başlangıçtaki maksimum değerine *arttığı* bir ısınma periyodu kullanmak ve hızı optimizasyon sürecinin sonuna kadar soğutmaktır. Basitlik için genellikle bu amaçla doğrusal bir artış kullanır. Bu, aşağıda belirtilen formda bir zamanlamaya yol açar.

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Ağın ilkin daha iyi yakınsadığına dikkat edin (özellikle ilk 5 dönemde performansı gözlemleyin).

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Isınma herhangi bir zamanlayıcıya uygulanabilir (sadece kosinüs değil). Öğrenme oranı programları ve daha birçok deney hakkında daha ayrıntılı bir tartışma için ayrıca bkz. :cite:`Gotmare.Keskar.Xiong.ea.2018`. Özellikle, ısınma fazının çok derin ağlardaki parametrelerin sapma miktarını sınırladığını buldular. Bu sezgisel olarak mantıklıdır, çünkü ağın başlangıçta ilerleme kaydetmesi en çok zaman alan bölümlerinde rastgele ilkleme nedeniyle önemli farklılıklar bekleyebiliriz. 

## Özet

* Eğitim sırasında öğrenme oranının azaltılması, doğruluğun iyileştirilmesine ve (en şaşırtıcı şekilde) modelde azaltılmış aşırı öğrenmeye neden olabilir.
* İlerlemenin düzleştiği her durumda öğrenme hızının parçalı azalması pratikte etkilidir. Esasen bu, uygun bir çözüme verimli bir şekilde yakınlaşmamızı ve ancak daha sonra öğrenme oranını azaltarak parametrelerin doğal varyansını azaltmamızı sağlar.
* Kosinüs zamanlayıcıları bazı bilgisayarla görme problemleri için popülerdir. Bu tür bir zamanlayıcı ile ilgili ayrıntılar için bkz. [GluonCV](http://gluon-cv.mxnet.io).
* Optimizasyondan önceki bir ısınma periyodu ıraksamayı önleyebilir.
* Optimizasyon, derin öğrenmede birden çok amaca hizmet eder. Eğitim hedefini en aza indirmenin yanı sıra, farklı optimizasyon algoritmaları ve öğrenme oranı zamanlama seçimleri, test kümesinde oldukça farklı miktarlarda genellemeye ve aşırı öğrenmeye (aynı miktarda eğitim hatası için) yol açabilir.

## Alıştırmalar

1. Belirli bir sabit öğrenme hızı için optimizasyon davranışıyla deneyler yapın. Bu şekilde elde edebileceğiniz en iyi model nedir?
1. Öğrenme oranındaki düşüşün üssünü değiştirirseniz yakınsama nasıl değişir? Deneylerde kolaylık sağlamak için `PolyScheduler`'i kullanın.
1. Kosinüs zamanlayıcısını büyük bilgisayarla görme problemlerine uygulayın, örn. ImageNet'i eğitin. Diğer zamanlayıcılara göre performansı nasıl etkiler?
1. Isınma ne kadar sürer?
1. Optimizasyon ve örneklemeyi ilişkilendirebilir misiniz? :cite:`Welling.Teh.2011`'ten Rasgele Gradyan Langevin Dinamik sonuçlarını kullanarak başlayın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1081)
:end_tab:
