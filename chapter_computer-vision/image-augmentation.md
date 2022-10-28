# İmge Artırması
:label:`sec_image_augmentation`

:numref:`sec_alexnet` içinde, büyük veri kümelerinin, çeşitli uygulamalarda derin sinir ağlarının başarısı için bir ön koşul olduğunu belirttik.
*İmge artırma* 
eğitim imgelerinde bir dizi rastgele değişiklikten sonra benzer ama farklı eğitim örnekleri üretir ve böylece eğitim kümesinin boyutunu genişletir. Alternatif olarak, imge artırma, eğitim örneklerinin rastgele ayarlanması modellerin belirli niteliklere daha az güvenmesine ve böylece genelleme yeteneklerini geliştirmesine izin vermesi gerçeğiyle motive edilebilir. Örneğin, bir imgeyi, ilgi nesnesinin farklı pozisyonlarda görünmesini sağlamak için farklı şekillerde kırpabiliriz, böylece bir modelin nesnenin konumuna bağımlılığını azaltabiliriz. Ayrıca, modelin renge duyarlılığını azaltmak için parlaklık ve renk gibi faktörleri de ayarlayabiliriz. Muhtemelen zamanında imge artırmanın AlexNet'in başarısı için vazgeçilmez olduğu doğrudur. Bu bölümde bilgisayarla görmede yaygın olarak kullanılan bu tekniği tartışacağız.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## Yaygın İmge Artırma Yöntemleri

Yaygın imge artırma yöntemlerini incelememizde, aşağıdaki $400\times 500$ imgesini bir örnek olarak kullanacağız.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

Çoğu imge artırma yönteminin belirli bir rastgelelik derecesi vardır. İmge artırmanın etkisini gözlemlememizi kolaylaştırmak için, sonraki adımda `apply` yardımcı bir işlev tanımlıyoruz. Bu işlev `img` girdi imgesinde `aug` imge artırma yöntemini birden çok kez çalıştırır ve tüm sonuçları gösterir.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Döndürme ve Kırpma

:begin_tab:`mxnet`
[**Resmin sola ve sağa doğru döndürülmesi**] genellikle nesnenin kategorisini değiştirmez. Bu, en eski ve en yaygın kullanılan imge artırma yöntemlerinden biridir. Daha sonra, `transforms` modülünü `RandomFlipLeftRight` örneğini oluşturmak için kullanıyoruz, ki imgeyi sola ve sağa döndürebiliriz.
:end_tab:

:begin_tab:`pytorch`
[**Resmin sola ve sağa doğru döndürülmesi**] genellikle nesnenin kategorisini değiştirmez. Bu, en eski ve en yaygın kullanılan imge artırma yöntemlerinden biridir. Daha sonra, `transforms` modülünü `RandomHorizontalFlip` örneğini oluşturmak için kullanıyoruz, ki imgeyi sola ve sağa döndürebiliriz.
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**Yukarı ve aşağı döndürme**] sola ve sağa döndürmek kadar yaygın değildir. Ancak en azından bu örnek imge için, yukarı ve aşağı döndürmek tanımayı engellemez. Ardından, bir imgeyi %50 şansla yukarı ve aşağı döndürmek için `RandomFlipTopBottom` örneği oluşturuyoruz.
:end_tab:

:begin_tab:`pytorch`
[**Yukarı ve aşağı döndürme**] sola ve sağa döndürmek kadar yaygın değildir. Ancak en azından bu örnek imge için, yukarı ve aşağı döndürmek tanımayı engellemez. Ardından, bir imgeyi %50 şansla yukarı ve aşağı döndürmek için `RandomVerticalFlip` örneği oluşturuyoruz.
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

Kullandığımız örnek imgede, kedi imgenin ortasındadır, ancak bu genel olarak böyle olmayabilir. :numref:`sec_pooling` içinde, ortaklama katmanının evrişimli bir katmanın hedef konuma duyarlılığını azaltabileceğini açıkladık. Buna ek olarak, nesnelerin imgedeki farklı ölçeklerde farklı konumlarda görünmesini sağlamak için imgeyi rastgele kırpabiliriz, bu da bir modelin hedef konuma duyarlılığını da azaltabilir. 

Aşağıdaki kodda, her seferinde orijinal alanın $\%10 \sim \%100$ alanı olan bir alanı [**rastgele kırpıyoruz**] ve bu alanın genişliğinin yüksekliğine oranı $0.5 \sim 2$'den rastgele seçilir. Ardından, bölgenin genişliği ve yüksekliği 200 piksele ölçeklenir. Aksi belirtilmedikçe, bu bölümdeki $a$ ile $b$ arasındaki rastgele sayı, $[a, b]$ aralığından rastgele ve tek biçimli örneklemeyle elde edilen sürekli bir değeri ifade eder.

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Renkleri Değiştirme

Diğer bir artırma yöntemi ise renkleri değiştirmektir. İmge renginin dört cephesini değiştirebiliriz: Parlaklık, zıtlık, doygunluk ve renk tonu. Aşağıdaki örnekte, imgenin [**parlaklığını**] orijinal imgenin %50'si ($1-0.5$) ile %150'si ($1+0.5$) arasında bir değere değiştiriyoruz.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Benzer şekilde, imgenin [**tonunu rastgele**] değiştirebiliriz.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Ayrıca bir `RandomColorJitter` örneği oluşturabilir ve [**imgenin `brightness` (parlaklık), `contrast` (zıtlık), `saturation` (doygunluk) ve `hue`'ini (renk tonu) aynı anda nasıl rastgele değiştirebileceğimizi**] ayarlayabiliriz.

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Çoklu İmge Artırma Yöntemlerini Birleştirme

Pratikte, [**birden fazla imge artırma yöntemini**] birleştireceğiz. Örneğin, yukarıda tanımlanan farklı imge artırma yöntemlerini birleştirebilir ve bunları bir `Compose` (beste) örneği aracılığıyla her imgeye uygulayabiliriz.

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**İmge Artırması ile Eğitim**]

İmge artırma ile bir model eğitelim. Burada daha önce kullandığımız Fashion-MNIST veri kümesi yerine CIFAR-10 veri kümesini kullanıyoruz. Bunun nedeni, Fashion-MNIST veri kümesindeki nesnelerin konumu ve boyutu normalleştirilirken, CIFAR-10 veri kümesindeki nesnelerin rengi ve boyutunun daha önemli farklılıklara sahip olmasıdır. CIFAR-10 veri kümelerindeki ilk 32 eğitim imgesi aşağıda gösterilmiştir.

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Tahmin sırasında kesin sonuçlar elde etmek için genellikle sadece eğitim örneklerine imge artırma uygularız ve tahmin sırasında rastgele işlemlerle imge artırma kullanmayız. [**Burada sadece en basit rastgele sol-sağ döndürme yöntemini kullanıyoruz**]. Buna ek olarak, bir minigrup imgeyi derin öğrenme çerçevesinin gerektirdiği biçime dönüştürmek için `ToTensor` örneğini kullanıyoruz, yani 0 ile 1 arasında 32 bit kayan virgüllü sayıları (toplu iş boyutu, kanal sayısı, yükseklik, genişlik) şeklindedirler.

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
Ardından, imgeyi okumayı ve imge artırmasını uygulamayı kolaylaştırmak için yardımcı bir işlev tanımlıyoruz. Gluon'un veri kümeleri tarafından sağlanan `transform_first` işlevi, her eğitim örneğinin (imge ve etiket) ilk öğesine (resim ve etiket) imge artırma işlemini uygular. `DataLoader`'ya ayrıntılı bir giriş için lütfen :numref:`sec_fashion_mnist`'e bakın.
:end_tab:

:begin_tab:`pytorch`
Ardından, [**imgeyi okumayı ve imge artırmasını uygulamayı kolaylaştırmak için yardımcı bir işlev tanımlar**]. PyTorch'un veri kümesi tarafından sağlanan `transform` bağımsız değişkeni, imgeleri döndürmek için artırma uygular. `DataLoader`'a ayrıntılı bir giriş için lütfen :numref:`sec_fashion_mnist` kısmına bakın.
:end_tab:

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Çoklu-GPU Eğitimi

ResNet-18 modelini :numref:`sec_resnet` içindeki CIFAR-10 veri kümesi üzerinde eğitiyoruz. :numref:`sec_multi_gpu_concise` kısmından çoklu GPU eğitimine girişi hatırlayın. Aşağıda, [**birden çok GPU kullanarak modeli eğitmek ve değerlendirmek için bir işlev tanımlıyoruz**].

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Birden çok GPU'lu bir minigrup için eğitim (Bölüm 13'te tanımlanmıştır)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # `True` bayrağı, daha sonra yararlı olan eski gradyanlara sahip parametrelere izin verir (örneğin, BERT ince ayarında)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Birden çok GPU'lu bir minigrup için eğitim (Bölüm 13'te tanımlanmıştır)."""
    if isinstance(X, list):
        # BERT ince ayarı için gerekli (daha sonra ele alınacaktır)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Birden çok GPU'lu bir modeli eğitin (Bölüm 13'te tanımlanmıştır)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
    # Eğitim kaybının toplamı, eğitim doğruluğunun toplamı, örneklerin sayısı, tahminlerin sayısı
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Birden çok GPU'lu bir modeli eğitin (Bölüm 13'te tanımlanmıştır)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Eğitim kaybının toplamı, eğitim doğruluğunun toplamı, örneklerin sayısı, tahminlerin sayısı
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Şimdi modeli imge artırma ile eğitmek için [**`train_with_data_aug` işlevini tanımlayabiliriz**]. Bu işlev mevcut tüm GPU'ları alır, optimizasyon algoritması olarak Adam'ı kullanır, eğitim veri kümesine imge artırma uygular ve son olarak modeli eğitmek ve değerlendirmek için tanımlanmış `train_ch13` işlevini çağırır.

```{.python .input}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

Rastgele sol-sağ döndürmeye dayalı imge artırma kullanarak [**modeli eğitelim**].

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Özet

* İmge artırma, modellerin genelleme yeteneğini geliştirmek için mevcut eğitim verilerine dayanan rastgele imgeler üretir.
* Tahmin sırasında kesin sonuçlar elde etmek için genellikle sadece eğitim örneklerine imge artırma uygularız ve tahmin sırasında rastgele işlemlerle imge artırma kullanmayız.
* Derin öğrenme çerçeveleri, aynı anda uygulanabilen birçok farklı imge artırma yöntemi sağlar.

## Alıştırmalar

1. İmge artırma özelliğini, `train_with_data_aug(test_augs, test_augs)`, kullanmadan modeli eğitin. İmge artırmasını kullanırken ve kullanmazken eğitim ve test doğruluğunu karşılaştırın. Bu karşılaştırmalı deney, imge artırmanin aşırı uyumu azaltabileceği argümanını destekleyebilir mi? Neden?
1. CIFAR-10 veri kümesinde model eğitiminde birden çok farklı imge artırma yöntemini birleştirin. Test doğruluğunu arttırıyor mu? 
1. Derin öğrenme çerçevesinin çevrimiçi belgelerine bakın. Başka hangi imge artırma yöntemlerini de sağlar?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1404)
:end_tab:
