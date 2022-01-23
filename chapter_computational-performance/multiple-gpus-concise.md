# Birden Çok GPU için Özlü Uygulama
:label:`sec_multi_gpu_concise`

Her yeni model için sıfırdan paralellik uygulanması eğlenceli değil. Ayrıca, yüksek performans için senkronizasyon araçlarının optimize edilmesinde önemli fayda vardır. Aşağıda, derin öğrenme çerçevelerinin üst düzey API'lerini kullanarak bunun nasıl yapılacağını göstereceğiz. Matematik ve algoritmalar :numref:`sec_multi_gpu`'teki ile aynıdır. Oldukça şaşırtıcı bir şekilde, bu bölümün kodunu çalıştırmak için en az iki GPU'ya ihtiyacınız olacaktır.

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

## [**Bir Oyuncak Ağı**]

Bize hala yeterince kolay ve hızlı eğitmek için :numref:`sec_multi_gpu` LeNet biraz daha anlamlı bir ağ kullanalım. Bir ResNet-18 varyantı :cite:`He.Zhang.Ren.ea.2016` seçiyoruz. Giriş görüntüleri küçük olduğundan biraz değiştiriyoruz. Özellikle, :numref:`sec_resnet` arasındaki fark, başlangıçta daha küçük bir evrişim çekirdeği, adım ve dolgu kullanmamızdır. Dahası, maksimum havuzlama katmanını kaldırıyoruz.

```{.python .input}
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
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
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
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
    """A slightly modified ResNet-18 model."""
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

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
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

## Ağ Başlatma

:begin_tab:`mxnet`
`initialize` işlevi, seçeceğimiz bir cihazda parametreleri başlatmamızı sağlar. Başlatma yöntemleri üzerinde bir yeniletici için bkz. :numref:`sec_numerical_stability`. Özellikle uygun olan şey, aynı zamanda şebekeyi aynı anda *çok* cihazlarda başlatmamıza izin vermesidir. Bunun pratikte nasıl çalıştığını deneyelim.
:end_tab:

:begin_tab:`pytorch`
Eğitim döngüsünün içindeki ağı başlatacağız. Başlatma yöntemleri üzerinde bir yeniletici için bkz. :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
:numref:`sec_multi_gpu`'te tanıtılan `split_and_load` işlevini kullanarak, bir minibatch veriyi bölebilir ve bölümleri `devices` değişkeni tarafından sağlanan cihazlar listesine kopyalayabiliriz. *otomatik olarak ağ örneği, ileri yayılma değerini hesaplamak için uygun GPU'yu kullanır. Burada 4 gözlem oluşturuyoruz ve bunları GPU'lar üzerinden bölüyoruz.
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Veri ağdan geçtikten sonra, ilgili parametreler başlatılır*cihazda* veri aktarılır*. Bu, başlatma işlemi cihaz başına temelinde gerçekleştiği anlamına gelir. Başlatma için GPU 0 ve GPU 1'i seçtiğimizden, ağ yalnızca CPU'da değil, orada başlatılır. Aslında, parametreler CPU'da bile mevcut değildir. Parametreleri yazdırarak ve ortaya çıkabilecek hataları gözlemleyerek bunu doğrulayabiliriz.
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
Ardından, kodu [**doğruluğu** değerlendirmek] için işe yarayan bir kodla değiştirelim (**birden fazla cihazda** paralel olarak). Bu `evaluate_accuracy_gpu` işlevinin :numref:`sec_lenet`'ten değiştirilmesi görevi görür. Temel fark, ağı çağırmadan önce bir mini batch paylaşmamızdır. Diğer her şey esasen aynıdır.
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Eğitim**]

Daha önce olduğu gibi, eğitim kodunun verimli paralellik için birkaç temel işlevi yerine getirmesi gerekir: 

* Ağ parametrelerinin tüm cihazlarda başlatılması gerekir.
* Veri kümesi üzerinde yineleme yaparken minibatches tüm cihazlara bölünmelidir.
* Kaybı ve degradeyi cihazlar arasında paralel olarak hesaplıyoruz.
* Degradeler toplanır ve parametreler buna göre güncellenir.

Sonunda, ağın nihai performansını bildirmek için doğruluğu (yine paralel olarak) hesaplıyoruz. Eğitim rutini, verileri bölmemiz ve toplamamız gerekmediği dışında, önceki bölümlerdeki uygulamalara oldukça benzer.

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

Bunun pratikte nasıl çalıştığını görelim. Isınma olarak [**ağı tek bir GPU'da eğitiyoruz**]

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Sonra [**eğitim için 2 GPU kullanıyoruz**]. :numref:`sec_multi_gpu`'te değerlendirilen LeNet ile karşılaştırıldığında, ResNet-18 modeli oldukça daha karmaşıktır. Paralelleşmenin avantajını gösterdiği yer burası. Hesaplama zamanı, parametreleri senkronize etme zamanından anlamlı bir şekilde daha büyüktür. Bu, paralelleştirme için ek yükü daha az alakalı olduğundan ölçeklenebilirliği artırır.

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Özet

:begin_tab:`mxnet`
* Gluon, bir bağlam listesi sağlayarak birden çok cihazda model başlatma için ilkel özellikleri sağlar.
:end_tab:

* Veriler, verilerin bulunabileceği cihazlarda otomatik olarak değerlendirilir.
* Bu cihazdaki parametrelere erişmeye çalışmadan önce her cihazdaki ağları başlatmaya özen gösterin. Aksi takdirde bir hatayla karşılaşırsınız.
* Optimizasyon algoritmaları otomatik olarak birden fazla GPU üzerinde toplanır.

## Egzersizler

:begin_tab:`mxnet`
1. Bu bölümde ResNet-18 kullanılır. Farklı çemleri, toplu iş boyutlarını ve öğrenme oranlarını deneyin. Hesaplama için daha fazla GPU kullanın. Bunu 16 GPU (örn. bir AWS p2.16xlarge örneğinde) denerseniz ne olur?
1. Bazen, farklı cihazlar farklı bilgi işlem gücü sağlar. GPU'ları ve CPU'yu aynı anda kullanabiliriz. İşi nasıl bölmeliyiz? Çabaya değer mi? Neden? Neden olmasın?
1. `npx.waitall()`'ü bırakırsak ne olur? Paralellik için iki adıma kadar çakışacak şekilde eğitimi nasıl değiştirirsiniz?
:end_tab:

:begin_tab:`pytorch`
1. Bu bölümde ResNet-18 kullanılır. Farklı çemleri, toplu iş boyutlarını ve öğrenme oranlarını deneyin. Hesaplama için daha fazla GPU kullanın. Bunu 16 GPU (örn. bir AWS p2.16xlarge örneğinde) denerseniz ne olur?
1. Bazen, farklı cihazlar farklı bilgi işlem gücü sağlar. GPU'ları ve CPU'yu aynı anda kullanabiliriz. İşi nasıl bölmeliyiz? Çabaya değer mi? Neden? Neden olmasın?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
