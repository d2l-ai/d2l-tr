# Kaggle'da Görüntü Sınıflandırması (CIFAR-10)
:label:`sec_kaggle_cifar10`

Şimdiye kadar, doğrudan tensör formatında görüntü veri kümelerini elde etmek için derin öğrenme çerçevelerinin üst düzey API'lerini kullanıyoruz. Ancak, özel görüntü veri kümeleri genellikle görüntü dosyaları şeklinde gelir. Bu bölümde, ham görüntü dosyalarından başlayacağız ve düzenleyeceğiz, okuyacağız, ardından bunları adım adım tensör formatına dönüştüreceğiz. 

Bilgisayar görüşünde önemli bir veri kümesi olan :numref:`sec_image_augmentation`'te CIFAR-10 veri kümesi ile deney yaptık. Bu bölümde, CIFAR-10 görüntü sınıflandırmasının Kaggle yarışmasını uygulamak için önceki bölümlerde öğrendiğimiz bilgileri uygulayacağız. (**Yarışmanın web adresi https://www.kaggle.com/c/cifar-10 **) 

:numref:`fig_kaggle_cifar10` yarışmanın web sayfasındaki bilgileri gösterir. Sonuçları göndermek için bir Kaggle hesabı kaydettirmeniz gerekir. 

![CIFAR-10 image classification competition webpage information. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-cifar10.png)
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

Yarışma veri seti, sırasıyla 50000 ve 300000 görüntü içeren bir eğitim seti ve bir test setine ayrılmıştır. Test setinde, değerlendirme için 10000 görüntü kullanılacak, kalan 290000 görüntüler değerlendirilmeyecek: sadece hile yapmayı zorlaştırmak için dahil edilirler
*manuel olarak test setinin sonuçları etiketli.
Bu veri kümeindeki görüntüler, yüksekliği ve genişliği 32 piksel olan png renkli (RGB kanalları) görüntü dosyalarıdır. Görüntüler, uçaklar, arabalar, kuşlar, kediler, geyik, köpekler, kurbağalar, atlar, tekneler ve kamyonlar olmak üzere toplam 10 kategoriyi kapsar. :numref:`fig_kaggle_cifar10`'ün sol üst köşesi veri kümesindeki uçakların, arabaların ve kuşların bazı görüntülerini gösterir. 

### Veri Kümesini İndirme

Kaggle'a giriş yaptıktan sonra :numref:`fig_kaggle_cifar10`'te gösterilen CIFAR-10 resim sınıflandırma yarışması web sayfasındaki “Veri” sekmesine tıklayabilir ve “Tümünü İndir” butonuna tıklayarak veri kümesini indirebiliriz. İndirilen dosyayı `../data`'te açtıktan ve içinde `train.7z` ve `test.7z`'yı açtıktan sonra, tüm veri kümesini aşağıdaki yollarda bulacaksınız: 

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

`train` ve `test` dizinlerinin sırasıyla eğitim ve test görüntülerini içerdiği, `trainLabels.csv` eğitim görüntüleri için etiketler sağlar ve `sample_submission.csv` örnek bir gönderim dosyasıdır. 

Başlamayı kolaylaştırmak için [**İlk 1000 eğitim görüntüsü ve 5 rastgele test görüntüsünü içeren veri kümesinin küçük ölçekli bir örneğini sağlıyoruz. **] Kaggle yarışmasının tam veri kümesini kullanmak için aşağıdaki `demo` değişkenini `False` olarak ayarlamanız gerekir.

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

### [**Veri Kümesini Düzenliyor**]

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

Ardından, `reorg_train_valid` işlevini [**orijinal eğitim setinden ayarlanan doğrulamayı bölmek için tanımlıyoruz. **] Bu işlevdeki `valid_ratio` argüman, doğrulamadaki örneklerin sayısının orijinal eğitim kümesindeki örneklerin sayısına oranıdır. Daha somut olarak, $n$'in en az örneklerle sınıfın görüntü sayısı olmasına izin verin ve $r$ oranı olsun. Doğrulama kümesi her sınıf için $\max(\lfloor nr\rfloor,1)$ görüntüyü ayırır. Örnek olarak `valid_ratio=0.1`'yı kullanalım. Orijinal eğitim seti sahip olduğundan 50000 resimler, olacak 45000 yolda eğitim için kullanılan resimler `train_valid_test/train`, diğer 5000 görüntüleri doğrulama yolunda ayarlanmış olarak ayrılacak iken `train_valid_test/valid`. Veri kümesini düzenledikten sonra, aynı sınıfın görüntüleri aynı klasörün altına yerleştirilir.

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
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

Aşağıdaki `reorg_test` işlevi [**tahmin sırasında veri yükleme için test setini düzenler.**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

Son olarak, `read_csv_labels`, `reorg_train_valid` ve `reorg_test` (** yukarıda tanımlanan işlevler**) [**invoke**] için bir işlev kullanıyoruz

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

Burada, veri kümesinin küçük ölçekli örneği için toplu iş boyutunu yalnızca 32 olarak ayarlıyoruz. Kaggle yarışmasının tüm veri kümesini eğitip test ederken, `batch_size` 128 gibi daha büyük bir tamsayıya ayarlanmalıdır. Eğitim örneklerinin%10'unu hiper parametrelerin ayarlanması için doğrulama kümesi olarak ayırdık.

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**Görüntü Artırma**]

Aşırı uyum sağlamak için görüntü büyütme kullanıyoruz. Örneğin, görüntüler eğitim sırasında rastgele yatay olarak çevrilebilir. Renkli görüntülerin üç RGB kanalı için standardizasyon da gerçekleştirebiliriz. Aşağıda ayarlayabileceğiniz bu işlemlerin bazıları listelenmektedir.

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    gluon.data.vision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    torchvision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

Test sırasında, değerlendirme sonuçlarındaki rastgeleliği ortadan kaldırmak için yalnızca görüntüler üzerinde standardizasyon gerçekleştiriyoruz.

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

Ardından, [**ham görüntü dosyalarından oluşan organize veri kümesini okudu**]. Her örnek bir görüntü ve bir etiket içerir.

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

During training,
we need to [**specify all the image augmentation operations defined above**].
When the validation set
is used for model evaluation during hyperparameter tuning,
no randomness from image augmentation should be introduced.
Before final prediction,
we train the model on the combined training set and validation set to make full use of all the labeled data.

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

## [**Model**] tanımlama

:begin_tab:`mxnet`
Burada, :numref:`sec_resnet`'te açıklanan uygulamadan biraz farklı olan `HybridBlock` sınıfına dayanan artık blokları inşa ediyoruz. Bu, hesaplama verimliliğini artırmak içindir.
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
Eğitim başlamadan önce :numref:`subsec_xavier`'te açıklanan Xavier başlatma işlemini kullanıyoruz.
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

## [**Eğitim Fonksiyonu**] tanımlama

Modelleri seçeceğiz ve hiper parametreleri doğrulama kümesindeki modelin performansına göre ayarlayacağız. Aşağıda, model eğitim fonksiyonunu tanımlıyoruz `train`.

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

## [**Modeli Eğitim ve Doğrulanma**]

Şimdi, modeli eğitebilir ve doğrulayabiliriz. Aşağıdaki tüm hiperparametreler ayarlanabilir. Örneğin, çak sayısını artırabiliriz. `lr_period` ve `lr_decay` sırasıyla 4 ve 0.9 olarak ayarlandığında, optimizasyon algoritmasının öğrenme hızı her 4 çağ sonrasında 0,9 ile çarpılır. Sadece gösteri kolaylığı için, burada sadece 20 çağından antrenman yapıyoruz.

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

## [**Test Setini Sınıflandırma**] ve Kaggle'da Sonuçları Gönderme

Hiperparametrelerle umut verici bir model elde ettikten sonra, modeli yeniden eğitmek ve test setini sınıflandırmak için tüm etiketli verileri (doğrulama seti dahil) kullanırız.

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

Yukarıdaki kod, biçimi Kaggle yarışmasının gereksinimini karşılayan bir `submission.csv` dosyası oluşturacaktır. Sonuçları Kaggle'a gönderme yöntemi, :numref:`sec_kaggle_house`'teki yönteme benzer. 

## Özet

* Gerekli formata düzenledikten sonra ham görüntü dosyalarını içeren veri kümelerini okuyabiliriz.

:begin_tab:`mxnet`
* Bir görüntü sınıflandırma yarışmasında, evrimsel sinir ağlarını, görüntü büyütme ve hibrit programlamayı kullanabiliriz.
:end_tab:

:begin_tab:`pytorch`
* Bir görüntü sınıflandırma yarışmasında evrimsel sinir ağları ve görüntü büyütme özelliğini kullanabiliriz.
:end_tab:

## Egzersizler

1. Bu Kaggle yarışması için CIFAR-10 veri setinin tamamını kullanın. Hiperparametreleri `batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50` ve `lr_decay = 0.1` olarak ayarlayın. Bu yarışmada hangi doğruluk ve sıralamayı elde edebileceğinizi görün. Onları daha da geliştirebilir misin?
1. Görüntü büyütmesini kullanmadığınızda hangi doğruluğu elde edebilirsiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1479)
:end_tab:
