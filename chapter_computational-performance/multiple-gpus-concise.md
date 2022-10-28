# Çoklu GPU için Özlü Uygulama
:label:`sec_multi_gpu_concise`

Her yeni model için sıfırdan paralellik uygulamak eğlenceli değildir. Ayrıca, yüksek performans için eşzamanlama araçlarının optimize edilmesinde önemli fayda vardır. Aşağıda, derin öğrenme çerçevelerinin üst düzey API'lerini kullanarak bunun nasıl yapılacağını göstereceğiz. Matematik ve algoritmalar :numref:`sec_multi_gpu`' içindekiler ile aynıdır. Şaşırtıcı olmayan bir şekilde, bu bölümün kodunu çalıştırmak için en az iki GPU'ya ihtiyacınız olacaktır.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Basit Örnek Bir Ağ**]

Hala yeterince kolay ve hızlı eğitilen :numref:`sec_multi_gpu` içindeki LeNet'ten biraz daha anlamlı bir ağ kullanalım. Bir ResNet-18 türevini :cite:`He.Zhang.Ren.ea.2016` seçiyoruz. Girdi imgeleri küçük olduğundan onu biraz değiştiriyoruz. Özellikle, :numref:`sec_resnet` içindekinden farkı, başlangıçta daha küçük bir evrişim çekirdeği, uzun adım ve dolgu kullanmamızdır. Ayrıca, maksimum ortaklama katmanını kaldırıyoruz.

```{.python .input}
#@save
def resnet18(num_classes):
    """Biraz değiştirilmiş ResNet-18 modeli."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # Bu model daha küçük bir evrişim çekirdeği, adım ve dolgu kullanır ve 
    # maksimum ortaklama katmanını kaldırır.
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """Biraz değiştirilmiş ResNet-18 modeli."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # Bu model daha küçük bir evrişim çekirdeği, adım ve dolgu kullanır ve 
    # maksimum ortaklama katmanını kaldırır.
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Ağ İlkleme

:begin_tab:`mxnet`
`initialize` işlevi, seçeceğimiz bir cihazda parametreleri ilklememizi sağlar. İlkleme yöntemleri üzerinde bir tazeleme için bkz. :numref:`sec_numerical_stability`. Özellikle kullanışlı olan şey, ağı aynı anda *birden çok* cihazda ilklememize de izin vermesidir. Bunun pratikte nasıl çalıştığını deneyelim.
:end_tab:

:begin_tab:`pytorch`
Eğitim döngüsünün içindeki ağı ilkleteceğiz. İlkleme yöntemleri üzerinde bir tazeleme için bkz. :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
net = resnet18(10)
# GPU'ların bir listesini alın
devices = d2l.try_all_gpus()
# Ağın tüm parametrelerini ilklet
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# GPU'ların bir listesini alın
devices = d2l.try_all_gpus()
# Ağı eğitim döngüsü içinde ilkleteceğiz
```

:begin_tab:`mxnet`
:numref:`sec_multi_gpu` içinde tanıtılan `split_and_load` işlevini kullanarak, bir minigrup veriyi bölebilir ve bölümleri `devices` değişkeni tarafından sağlanan cihazlar listesine kopyalayabiliriz. Ağ örneği *otomatik olarak*, ileri yayılmanın değerini hesaplamak için uygun GPU'yu kullanır. Burada 4 gözlem oluşturuyoruz ve bunları GPU'lara bölüyoruz.
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Veriler ağdan geçtiğinde, ilgili parametreler *verilerin geçtiği cihazda* ilkletilir. Bu, ilkleme işleminin cihaz başına temelinde gerçekleştiği anlamına gelir. İlkleme için GPU 0 ve GPU 1'i seçtiğimizden, ağ CPU'da değil, yalnızca orada ilkletilir. Aslında, parametreler CPU'da mevcut bile değildir. Parametreleri yazdırarak ve ortaya çıkabilecek hataları gözlemleyerek bunu doğrulayabiliriz.
:end_tab:

```{.python .input}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Ardından, [**doğruluğu değerlendirme**] kodunu (**birden çok cihazda paralel olarak**) çalışan bir kodla değiştirelim. Bu, :numref:`sec_lenet` içinden gelen `evaluate_accuracy_gpu` işlevinin yerini alır. Temel fark, ağı çağırmadan önce bir minigrubu bölmemizdir. Diğer her şey temelde aynıdır.
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Birden çok GPU kullanarak bir veri kümesindeki bir modelin doğruluğunu hesaplayın."""
    # Cihaz listesini sorgula
    devices = list(net.collect_params().values())[0].list_ctx()
    # Doğru tahmin sayısı, tahmin sayısını
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Paralel olarak çalıştır
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Eğitim**]

Daha önce olduğu gibi, eğitim kodunun verimli paralellik için birkaç temel işlevi yerine getirmesi gerekir: 

* Ağ parametrelerinin tüm cihazlarda ilklenmesi gerekir.
* Veri kümesi üzerinde yineleme yaparken minigruplar tüm cihazlara bölünmelidir.
* Kaybı ve gradyanı cihazlar arasında paralel olarak hesaplarız.
* Gradyanlar toplanır ve parametreler buna göre güncellenir.

Sonunda, ağın nihai performansını bildirmek için doğruluğu (yine paralel olarak) hesaplıyoruz. Eğitim rutini, verileri bölmemiz ve toplamamız gerekmesi dışında, önceki bölümlerdeki uygulamalara oldukça benzer.

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Bunun pratikte nasıl çalıştığını görelim. Isınma olarak [**ağı tek bir GPU'da eğitelim.**]

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Sonra [**eğitim için 2 GPU kullanıyoruz**]. :numref:`sec_multi_gpu` içinde değerlendirilen LeNet ile karşılaştırıldığında, ResNet-18 modeli oldukça daha karmaşıktır. Paralelleşmenin avantajını gösterdiği yer burasıdır. Hesaplama zamanı, parametreleri eşzamanlama zamanından anlamlı bir şekilde daha büyüktür. Paralelleştirme için ek yük daha az alakalı olduğundan, bu ölçeklenebilirliği artırır.

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Özet

:begin_tab:`mxnet`
* Gluon, bir bağlam listesi sağlayarak birden çok cihazda model ilkleme için en temel özellikleri sağlar.
:end_tab:

* Veriler, verilerin bulunabileceği cihazlarda otomatik olarak değerlendirilir.
* O cihazdaki parametrelere erişmeye çalışmadan önce her cihazdaki ağları ilklemeye özen gösterin. Aksi takdirde bir hatayla karşılaşırsınız.
* Optimizasyon algoritmaları otomatik olarak birden fazla GPU üzerinde toplanır.

## Alıştırmalar

:begin_tab:`mxnet`
1. Bu bölümde ResNet-18 kullanılıyor. Farklı dönemleri, toplu iş boyutlarını ve öğrenme oranlarını deneyin. Hesaplama için daha fazla GPU kullanın. Bunu 16 GPU ile (örn. bir AWS p2.16xlarge örneğinde) denerseniz ne olur?
1. Bazen, farklı cihazlar farklı bilgi işlem gücü sağlar. GPU'ları ve CPU'yu aynı anda kullanabiliriz. İşi nasıl bölmeliyiz? Çabaya değer mi? Neden? Neden olmasın?
1. `npx.waitall()`'ü atarsak ne olur? Paralellik için iki adıma kadar örtüşecek şekilde eğitimi nasıl değiştirirsiniz?
:end_tab:

:begin_tab:`pytorch`
1. Bu bölümde ResNet-18 kullanılıyor. Farklı dönemleri, toplu iş boyutlarını ve öğrenme oranlarını deneyin. Hesaplama için daha fazla GPU kullanın. Bunu 16 GPU ile (örn. bir AWS p2.16xlarge örneğinde) denerseniz ne olur?
1. Bazen, farklı cihazlar farklı bilgi işlem gücü sağlar. GPU'ları ve CPU'yu aynı anda kullanabiliriz. İşi nasıl bölmeliyiz? Çabaya değer mi? Neden? Neden olmasın?
:end_tab:

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1403)
:end_tab:
