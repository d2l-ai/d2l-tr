# Kaggle'da İmge Sınıflandırması (CIFAR-10)
:label:`sec_kaggle_cifar10`

Şimdiye kadar, doğrudan tensör formatında imge veri kümelerini elde etmek için derin öğrenme çerçevelerinin üst düzey API'lerini kullanıyoruz. Ancak, özel imge veri kümeleri genellikle imge dosyaları halinde gelir. Bu bölümde, ham imge dosyalarından başlayacağız ve düzenleyeceğiz, okuyacağız, ardından bunları adım adım tensör formatına dönüştüreceğiz. 

Bilgisayarla görmede önemli bir veri kümesi olan :numref:`sec_image_augmentation` içinde CIFAR-10 veri kümesi ile deney yaptık. Bu bölümde, CIFAR-10 imge sınıflandırmasının Kaggle yarışmasını uygulamak için önceki bölümlerde öğrendiğimiz bilgileri uygulayacağız. (**Yarışmanın web adresi https://www.kaggle.com/c/cifar-10**) 

:numref:`fig_kaggle_cifar10` yarışmanın web sayfasındaki bilgileri gösterir. Sonuçları göndermek için bir Kaggle hesabına kayıt olmanız gerekir. 

![CIFAR-10 imge sınıflandırma yarışması web sayfası bilgileri. Yarışma veri kümesi "Data" ("Veri") sekmesine tıklanarak elde edilebilir.](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

## Veri Kümesini Elde Etme ve Düzenleme

Yarışma veri kümesi, sırasıyla 50000 ve 300000 imge içeren bir eğitim kümesi ve bir test kümesine ayrılmıştır. Test kümesinde, değerlendirme için 10000 imge kullanılacak, kalan 290000 imgeler değerlendirilmeyecek: Bunlar sadece test kümesinin *manuel* etiketli sonuçlarıyla hile yapmayı zorlaştırmak için dahil edilmiştir.
Bu veri kümesindeki imgeler, yüksekliği ve genişliği 32 piksel olan png renkli (RGB kanalları) imge dosyalarıdır. İmgeler, uçaklar, arabalar, kuşlar, kediler, geyik, köpekler, kurbağalar, atlar, tekneler ve kamyonlar olmak üzere toplam 10 kategoriyi kapsar. :numref:`fig_kaggle_cifar10` şeklinin sol üst köşesi veri kümesindeki uçakların, arabaların ve kuşların bazı imgelerini gösterir. 

### Veri Kümesini İndirme

Kaggle'a girdi yaptıktan sonra :numref:`fig_kaggle_cifar10` içinde gösterilen CIFAR-10 imge sınıflandırma yarışması web sayfasındaki “Veri” ("Data") sekmesine tıklayabilir ve “Tümünü İndir” ("Download All") butonuna tıklayarak veri kümesini indirebiliriz. İndirilen dosyayı `../data`'da açtıktan ve içinde `train.7z` ve `test.7z`'yı açtıktan sonra, tüm veri kümesini aşağıdaki yollarda bulacaksınız: 

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

`train` ve `test` dizinlerinin sırasıyla eğitim ve test imgelerini içerdiği, `trainLabels.csv` eğitim imgeleri için etiketler sağlar ve `sample_submission.csv` örnek bir gönderim dosyasıdır. 

Başlamayı kolaylaştırmak için [**ilk 1000 eğitim imgesi ve 5 rastgele test imgesi içeren veri kümesinin küçük ölçekli bir örneğini sağlıyoruz.**] Kaggle yarışmasının tam veri kümesini kullanmak için aşağıdaki `demo` değişkenini `False` olarak ayarlamanız gerekir.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# If you use the full dataset downloaded for the Kaggle competition, set
# `demo` to False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**Veri Kümesini Düzenleme**]

Model eğitimini ve testlerini kolaylaştırmak için veri kümeleri düzenlememiz gerekiyor. Önce csv dosyasındaki etiketleri okuyalım. Aşağıdaki işlev, dosya adının uzantısız kısmını etiketine eşleyen bir sözlük döndürür.

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# training examples:', len(labels))
print('# classes:', len(set(labels.values())))
```

Ardından, `reorg_train_valid` işlevini [**esas eğitim kümesinden geçerleme kümesini bölmek için tanımlıyoruz.**] Bu işlevdeki `valid_ratio` argümanı, geçerleme kümesindeki örneklerin sayısının orijinal eğitim kümesindeki örneklerin sayısına oranıdır. Daha somut olarak, sınıfın en az örnek içeren imge sayısı $n$ ve oranı da $r$ olsun. Geçerleme kümesi her sınıf için $\max(\lfloor nr\rfloor,1)$ imge ayırır. Örnek olarak `valid_ratio=0.1`'i kullanalım. Orijinal eğitim kümesi 50000 imgeye sahip olduğundan, `train_valid_test/train` yolunda eğitim için kullanılan 45000 imge olacak, diğer 5000 imge `train_valid_test/valid` yolunda geçerleme kümesi olarak bölünecek. Veri kümesini düzenledikten sonra, aynı sınıfın imgeleri aynı klasörün altına yerleştirilir.

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """Bir dosyayı hedef dizine kopyalayın."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Doğrulama kümesini orijinal eğitim kümesinden ayırın."""
    # Eğitim veri kümesinde en az örneğe sahip sınıfın örnek sayısı
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # Geçerleme kümesi için sınıf başına örnek sayısı
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

Aşağıdaki `reorg_test` işlevi [**tahmin sırasında veri yükleme için test kümesini düzenler.**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """Tahmin sırasında veri yüklemesi için test kümesini düzenleyin."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

Son olarak, `read_csv_labels`, `reorg_train_valid` ve `reorg_test` (**yukarıda tanımlanan işlevleri**) [**çağırmak**] için bir işlev kullanıyoruz.

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

Burada, veri kümesinin küçük ölçekli örneği için toplu iş boyutunu yalnızca 32 olarak ayarlıyoruz. Kaggle yarışmasının tüm veri kümesini eğitip test ederken, `batch_size` 128 gibi daha büyük bir tamsayıya ayarlanmalıdır. Eğitim örneklerinin %10'unu hiper parametrelerin ayarlanması için geçerleme kümesi olarak ayırdık.

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**İmge Artırma**]

Aşırı öğrenmeyi bertaraf etmek için imge artırımı kullanıyoruz. Örneğin, imgeler eğitim sırasında rastgele yatay olarak çevrilebilir. Renkli imgelerin üç RGB kanalı için standartlaştırma da gerçekleştirebiliriz. Aşağıda ayarlayabileceğiniz bu işlemlerin bazıları listelenmektedir.

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # İmgeyi hem yükseklik hem de genişlikte 40 piksellik bir kareye ölçeklendirin
    gluon.data.vision.transforms.Resize(40),
    # Orijinal imgenin alanının 0.64 ila 1 katı arasında küçük bir kare 
    # oluşturmak için hem yükseklik hem de genişlikte 40 piksellik bir kare 
    # imgeyi rastgele kırpın ve ardından hem yükseklik hem de genişlikte 
    # 32 piksellik bir kareye ölçeklendirin
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # İmgenin her kanalını standartlaştırın
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # İmgeyi hem yükseklik hem de genişlikte 40 piksellik bir kareye ölçeklendirin
    torchvision.transforms.Resize(40),
    # Orijinal imgenin alanının 0.64 ila 1 katı arasında küçük bir kare 
    # oluşturmak için hem yükseklik hem de genişlikte 40 piksellik bir kare 
    # imgeyi rastgele kırpın ve ardından hem yükseklik hem de genişlikte 
    # 32 piksellik bir kareye ölçeklendirin
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # İmgenin her kanalını standartlaştırın
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

Test sırasında, değerlendirme sonuçlarındaki rastgeleliği ortadan kaldırmak için yalnızca imgeler üzerinde standartlaştırma gerçekleştiriyoruz.

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## Veri Kümesini Okuma

Ardından, [**ham imge dosyalarından oluşan düzenlenmiş veri kümesini okuruz**]. Her örnek bir imge ve bir etiket içerir.

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

Eğitim sırasında [**yukarıda tanımlanan tüm imge artırım işlemlerini belirtmemiz**] gerekir. Geçerleme kümesi hiper parametre ayarlama sırasında model değerlendirmesi için kullanıldığında, imge artırımdan rastgelelik getirilmemelidir. Son tahminden önce, tüm etiketlenmiş verileri tam olarak kullanmak için modeli birleştirilmiş eğitim kümesi ve geçerleme kümesi üzerinde eğitiriz.

```{.python .input}
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**Modeli**] Tanımlama

:begin_tab:`mxnet`
Burada, :numref:`sec_resnet` içinde açıklanan uygulamadan biraz farklı olan `HybridBlock` sınıfına dayanan artık blokları inşa ediyoruz. Bu, hesaplama verimliliğini artırmak içindir.
:end_tab:

```{.python .input}
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
Ardından, ResNet-18 modelini tanımlıyoruz.
:end_tab:

```{.python .input}
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
Eğitim başlamadan önce :numref:`subsec_xavier` içinde açıklanan Xavier ilkletme işlemini kullanıyoruz.
:end_tab:

:begin_tab:`pytorch`
:numref:`sec_resnet`'te açıklanan ResNet-18 modelini tanımlıyoruz.
:end_tab:

```{.python .input}
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## [**Eğitim Fonksiyonunu**] Tanımlama

Modelleri seçeceğiz ve hiper parametreleri geçerleme kümesindeki modelin performansına göre ayarlayacağız. Aşağıda, model eğitim fonksiyonunu, `train`, tanımlıyoruz.

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Modeli Eğitme ve Geçerleme**]

Şimdi, modeli eğitebilir ve geçerleyebiliriz. Aşağıdaki tüm hiper parametreler ayarlanabilir. Örneğin, dönem sayısını artırabiliriz. `lr_period` ve `lr_decay` sırasıyla 4 ve 0.9 olarak ayarlandığında, optimizasyon algoritmasının öğrenme oranı her 4 dönem sonrasında 0.9 ile çarpılır. Sadece gösterim kolaylığı için, burada sadece 20 dönemlik eğitim yapıyoruz.

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**Test Kümesini Sınıflandırma**] ve Kaggle'da Sonuçları Teslim Etme

Hiper parametrelerle umut verici bir model elde ettikten sonra, modeli yeniden eğitmek ve test kümesini sınıflandırmak için tüm etiketli verileri (geçerleme kümesi dahil) kullanırız.

```{.python .input}
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

Yukarıdaki kod, biçimi Kaggle yarışmasının gereksinimini karşılayan bir `submission.csv` dosyası oluşturacaktır. Sonuçları Kaggle'a gönderme yöntemi, :numref:`sec_kaggle_house` içindeki yönteme benzerdir. 

## Özet

* Gerekli formata düzenledikten sonra ham imge dosyalarını içeren veri kümelerini okuyabiliriz.

:begin_tab:`mxnet`
* Bir imge sınıflandırma yarışmasında, evrişimli sinir ağlarını, imge artırmayı ve hibrit programlamayı kullanabiliriz.
:end_tab:

:begin_tab:`pytorch`
* Bir imge sınıflandırma yarışmasında evrişimli sinir ağları ve imge artırmayı kullanabiliriz.
:end_tab:

## Alıştırmalar

1. Bu Kaggle yarışması için CIFAR-10 veri kümesinin tamamını kullanın. Hiper parametreleri `batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50` ve `lr_decay = 0.1` olarak ayarlayın. Bu yarışmada hangi doğruluk ve sıralamayı elde edebileceğinizi görün. Onları daha da geliştirebilir misin?
1. İmge artırmayı kullanmadığınızda hangi doğruluğu elde edebilirsiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1479)
:end_tab:
