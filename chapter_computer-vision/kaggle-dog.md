# Kaggle Üzerinde Köpek Cinsi Tanımlama (ImageNet Köpekler)

Bu bölümde, Kaggle'da köpek cinsi tanımlama problemini uygulayacağız. (**Bu yarışmanın web adresi: https://www.kaggle.com/c/dog-breed-identification**) 

Bu yarışmada 120 farklı köpek ırkı tanınacak. Aslında, bu yarışma için veri kümesi ImageNet veri kümesinin bir alt kümesidir. :numref:`sec_kaggle_cifar10` içindeki CIFAR-10 veri kümesindeki imgelerin aksine, ImageNet veri kümesindeki imgeler hem daha yüksek hem de daha geniş farklı boyutlardadır. :numref:`fig_kaggle_dog`, yarışmanın web sayfasındaki bilgileri gösterir. Sonuçlarınızı göndermek için bir Kaggle hesabına ihtiyacınız var. 

![Köpek cinsi tanımlama yarışması web sitesi. Yarışma veri kümesi "Data" ("Veri") sekmesine tıklanarak elde edilebilir.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## Veri Kümesini Elde Etme ve Düzenleme

Yarışma veri kümesi, sırasıyla üç RGB (renkli) kanalın 10222 ve 10357 JPEG, imgelerini içeren birer eğitim kümesine ve test kümesine ayrılmıştır. Eğitim veri kümesinde Labradors, Kaniş, Dachshunds, Samoyeds, Huskies, Chihuahuas ve Yorkshire Terriers gibi 120 köpek ırkı vardır. 

### Veri Kümesini İndirme

Kaggle'a giriş yaptıktan sonra, :numref:`fig_kaggle_dog` içinde gösterilen yarışma web sayfasındaki “Veri" ("Data") sekmesine tıklayabilir ve “Tümünü İndir" ("Download All") düğmesine tıklayarak veri kümesini indirebilirsiniz. İndirilen dosyayı `../data`'da açtıktan sonra, tüm veri kümesini aşağıdaki yollarda bulacaksınız: 

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

Yukarıdaki yapının `train/` ve `test/` klasörlerinin sırasıyla eğitim ve test köpek imgeleri içeren :numref:`sec_kaggle_cifar10` içindeki CIFAR-10 yarışmasına benzer olduğunu fark etmiş olabilirsiniz ve `labels.csv` eğitim imgeleri için etiketler içerir. Benzer şekilde, başlamayı kolaylaştırmak için, yukarıda belirtilen: `train_valid_test_tiny.zip` [**veri kümesinin küçük bir örneklemini sağlıyoruz**]. Kaggle yarışması için tam veri kümesini kullanacaksanız, aşağıdaki `demo` değişkenini `False` olarak değiştirmeniz gerekir.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# Kaggle yarışması için indirilen tam veri kümesini kullanıyorsanız, 
# aşağıdaki değişkeni `False` olarak değiştirin
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**Veri Kümesini Düzenleme**]

Veri kümesini :numref:`sec_kaggle_cifar10` içinde yaptığımız şeye benzer şekilde düzenleyebiliriz, yani esas eğitim kümesindeki bir geçerleme kümesini ayırabilir ve imgeleri etiketlere göre gruplandırılmış alt klasörlere taşıyabiliriz. 

Aşağıdaki `reorg_dog_data` işlevi eğitim veri etiketlerini okur, geçerleme kümesini böler ve eğitim kümesini düzenler.

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**İmge Artırma**]

Bu köpek cins veri kümesinin, imgeleri :numref:`sec_kaggle_cifar10` içindeki CIFAR-10 veri kümesinden daha büyük olan ImageNet veri kümesinin bir alt kümesi olduğunu hatırlayın. Aşağıda, nispeten daha büyük imgeler için yararlı olabilecek birkaç imge artırma işlemi listelenmektedir.

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Orijinal alanın 0.08 ila 1'i arasında bir alana ve 3/4 ile 4/3 arasında 
    # yükseklik-genişlik oranına sahip bir imge elde etmek için imgeyi 
    # rastgele kırpın. Ardından, yeni bir 224 x 224 imge oluşturmak için
    # imgeyi ölçeklendirin.
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Parlaklığı, zıtlığı ve doygunluğu rastgele değiştirin
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Rastgele gürültü ekle
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # İmgenin her kanalını standartlaştırın
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Orijinal alanın 0.08 ila 1'i arasında bir alana ve 3/4 ile 4/3 arasında 
    # yükseklik-genişlik oranına sahip bir imge elde etmek için imgeyi 
    # rastgele kırpın. Ardından, yeni bir 224 x 224 imge oluşturmak için
    # imgeyi ölçeklendirin.
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Parlaklığı, zıtlığı ve doygunluğu rastgele değiştirin
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Rastgele gürültü ekle
    torchvision.transforms.ToTensor(),
    # İmgenin her kanalını standartlaştırın
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

Tahmin sırasında yalnızca imge ön işleme işlemlerini rastgelelik olmadan kullanıyoruz.

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # İmgenin ortasından 224 x 224 karelik bir alanı kırpın
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # İmgenin ortasından 224 x 224 karelik bir alanı kırpın
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**Veri Kümesi Okuma**]

:numref:`sec_kaggle_cifar10` içinde olduğu gibi, ham imge dosyalarından oluşan düzenlenmiş veri kümesini okuyabiliriz.

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

Aşağıda, :numref:`sec_kaggle_cifar10` içinde olduğu gibi veri yineleyici örneklerini oluşturuyoruz.

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

## [**Önceden Eğitilmiş Modelleri İnce Ayarlama**]

Yine, bu yarışma için veri kümesi ImageNet veri kümesinin bir alt kümedir. Bu nedenle, tam ImageNet veri kümesinde önceden eğitilmiş bir model seçmek için :numref:`sec_fine_tuning` içinde tartışılan yaklaşımı kullanabilir ve bunu özel bir küçük ölçekli çıktı ağına beslenecek imge özniteliklerini ayıklamak için kullanabiliriz. Derin öğrenme çerçevelerinin üst seviye API'leri, ImageNet veri kümesi üzerinde önceden eğitilmiş geniş bir model yelpazesi sunar. Burada, bu modelin çıktı katmanının girdisini (yani ayıklanan öznitelikler) yeniden kullandığımız önceden eğitilmiş bir ResNet-34 modeli seçiyoruz. Daha sonra orijinal çıktı katmanını, iki tam bağlı katman yığını gibi eğitilebilecek küçük bir özel çıktı ağı ile değiştirebiliriz. :numref:`sec_fine_tuning` içindeki deneyden farklı olarak, aşağıdaki öznitelik ayıklamak için kullanılan önceden eğitilmiş modeli yeniden eğitmez. Bu, gradyanların depolanması için eğitim süresini ve hafızasını azaltır. 

Tüm ImageNet veri kümesi için üç RGB kanalının araçlarını ve standart sapmalarını kullanarak imgeleri standartlaştırdığımızı hatırlayın. Aslında, bu aynı zamanda ImageNet'te önceden eğitilmiş model tarafından uygulanan standartlaştırma işlemi ile de tutarlıdır.

```{.python .input}
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Yeni bir çıktı ağı tanımlayın
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 120 çıktı kategorisi var
    finetune_net.output_new.add(nn.Dense(120))
    # Çıktı ağını ilklet
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Model parametrelerini hesaplama için kullanılan CPU'lara veya 
    # GPU'lara dağıtın
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Yeni bir çıktı ağı tanımlayın (120 çıktı kategorisi var)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Modeli cihazlara taşı
    finetune_net = finetune_net.to(devices[0])
    # Öznitelik katmanlarının parametrelerini dondur
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

[**Kaybın hesaplanmasından**] önce, önce önceden eğitilmiş modelin çıktı katmanının girdisini, yani ayıklanan özniteliği, elde ederiz. Daha sonra bu özniteliği, kaybı hesaplamak için küçük özel çıktı ağımızın girdisi olarak kullanırız.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## [**Eğitim Fonksiyonunu**] Tanımlama

Modeli seçip, modelin geçerleme kümesindeki performansına göre hiper parametreleri ayarlayacağız. `train` model eğitim işlevi yalnızca küçük özel çıktı ağının parametrelerini yineler.

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Yalnızca küçük özel çıktı ağını eğitin
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Yalnızca küçük özel çıktı ağını eğitin
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Modeli Eğitme ve Geçerleme**]

Şimdi modeli eğitebilir ve geçerleyebiliriz. Aşağıdaki hiper parametrelerin tümü ayarlanabilir. Örneğin, dönem sayısı artırılabilir. `lr_period` ve `lr_decay` sırasıyla 2 ve 0.9 olarak ayarlandığından, eniyileme algoritmasının öğrenme oranı her 2 dönem sonrasında 0.9 ile çarpılır.

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**Test Kümesini Sınıflandırma**] ve Kaggle'da Sonuçları Teslim Etme

:numref:`sec_kaggle_cifar10` içindeki son adıma benzer şekilde, sonunda tüm etiketli veriler (geçerleme kümesi dahil) modeli eğitmek ve test kümesini sınıflandırmak için kullanılır. Sınıflandırma için eğitilmiş özel çıktı ağını kullanacağız.

```{.python .input}
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

Yukarıdaki kod, :numref:`sec_kaggle_house` içinde açıklanan şekilde Kaggle'a teslim edilecek bir `submission.csv` dosyası oluşturacaktır. 

## Özet

* ImageNet veri kümelerindeki imgeler CIFAR-10 imgelerinden (farklı boyutlarda) daha büyüktür. Farklı bir veri kümesindeki görevler için imge artırma işlemlerini değiştirebiliriz.
* ImageNet veri kümesinin bir alt kümesini sınıflandırmak için, öznitelikleri ayıklamak ve yalnızca özel bir küçük ölçekli çıktı ağı eğitebilmek için ImageNet veri kümesinin tüm ImageNet veri kümesinde önceden eğitilmiş modellerini kullanabiliriz. Bu, daha az hesaplama süresi ve bellek maliyetine yol açacaktır.

## Alıştırmalar

1. Tüm Kaggle yarışma veri kümesini kullanırken, diğer bazı hiper parametreleri `lr = 0.01`, `lr_period = 10` ve `lr_period = 10` ve `lr_decay = 0.1` olarak ayarlarken `batch_size` (toplu iş boyutu) ve `num_epochs` (dönem sayısı) artırdığınızda hangi sonuçları elde edebilirsiniz?
1. Bir önceden eğitilmiş daha derin model kullanırsanız daha iyi sonuçlar alır mısınız? Hiper parametreleri nasıl ayarlarsınız? Sonuçları daha da iyileştirebilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1481)
:end_tab:
