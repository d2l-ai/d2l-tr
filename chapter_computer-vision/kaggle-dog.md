# Kaggle üzerinde Köpek Irk Tanımlama (ImageNet Köpekler)

Bu bölümde, Kaggle'da köpek ırkı tanımlama problemini uygulayacağız. (**Bu yarışmanın web adresi https://www.kaggle.com/c/dog-breed-identification **) 

Bu yarışmada 120 farklı köpek ırkı tanınacak. Aslında, bu yarışma için veri kümesi ImageNet veri kümesinin bir alt kümesidir. :numref:`sec_kaggle_cifar10`'teki CIFAR-10 veri kümesindeki görüntülerin aksine, ImageNet veri kümesindeki görüntüler farklı boyutlarda hem daha yüksek hem de daha geniştir. :numref:`fig_kaggle_dog`, yarışmanın web sayfasındaki bilgileri gösterir. Sonuçlarınızı göndermek için bir Kaggle hesabına ihtiyacınız var. 

![The dog breed identification competition website. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-dog.jpg)
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

Yarışma veri kümesi, sırasıyla üç RGB (renkli) kanalın 10222 ve 10357 JPEG görüntüsünü içeren bir eğitim seti ve bir test setine ayrılmıştır. Eğitim veri kümesi arasında Labradors, Kaniş, Dachshunds, Samoyeds, Huskies, Chihuahuas ve Yorkshire Terriers gibi 120 köpek ırkı vardır. 

### Veri Kümesini İndirme

Kaggle'a giriş yaptıktan sonra, :numref:`fig_kaggle_dog`'te gösterilen yarışma web sayfasındaki “Veriler” sekmesine tıklayabilir ve “Tümünü İndir” düğmesine tıklayarak veri kümesini indirebilirsiniz. İndirilen dosyayı `../data`'te açtıktan sonra, tüm veri kümesini aşağıdaki yollarda bulacaksınız: 

* .. /data/dog-breed-identification/labels.csv
* .. /data/dog-breed-identification/sample_submission.csv
* .. /data/köpek ırkı tanımlama/tren
* .. /data/köpek ırkı tanımlama/test

Yukarıdaki yapının `train/` ve `test/` klasörleri sırasıyla eğitim ve test köpek görüntüleri içeren :numref:`sec_kaggle_cifar10` yılında CIFAR-10 yarışmasına benzer olduğunu fark etmiş olabilirsiniz ve `labels.csv` eğitim görüntüleri için etiketler içerir. Benzer şekilde, başlamayı kolaylaştırmak için, [**veri kümesinin küçük bir örneğini sağlıyoruz**] yukarıda belirtilen: `train_valid_test_tiny.zip`. Kaggle yarışması için tam veri kümesini kullanacaksanız, aşağıdaki `demo` değişkenini `False` olarak değiştirmeniz gerekir.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**Veri Kümesini Düzenliyor**]

Veri kümesini :numref:`sec_kaggle_cifar10`'te yaptığımız şeye benzer şekilde düzenleyebiliriz, yani orijinal eğitim kümesindeki bir doğrulama kümesini ayırabilir ve görüntüleri etiketlere göre gruplandırılmış alt klasörlere taşıyabiliriz. 

Aşağıdaki `reorg_dog_data` işlevi eğitim veri etiketlerini okur, doğrulama kümesini böler ve eğitim setini düzenler.

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

## [**Görüntü Artırma**]

Bu köpek cins veri kümesinin, görüntüleri :numref:`sec_kaggle_cifar10`'teki CIFAR-10 veri kümesinden daha büyük olan ImageNet veri kümesinin bir alt kümesi olduğunu hatırlayın. Aşağıda, nispeten daha büyük görüntüler için yararlı olabilecek birkaç görüntü büyütme işlemi listelenmektedir.

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

Tahmin sırasında yalnızca görüntü önişleme işlemlerini rastgelelik olmadan kullanıyoruz.

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**Veri Kümesi Okuma**]

:numref:`sec_kaggle_cifar10`'te olduğu gibi, ham görüntü dosyalarından oluşan organize veri kümesini okuyabiliriz.

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

Aşağıda, :numref:`sec_kaggle_cifar10`'te olduğu gibi veri yineleyici örneklerini oluşturuyoruz.

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

## [**Önceden Eğitimli Modelleri İnce Ayarlama**]

Yine, bu yarışma için veri kümesi ImageNet veri kümesinin bir alt kümedir. Bu nedenle, tam ImageNet veri kümesinde önceden eğitilmiş bir model seçmek için :numref:`sec_fine_tuning`'te tartışılan yaklaşımı kullanabilir ve bunu özel bir küçük ölçekli çıktı ağına beslenecek görüntü özelliklerini ayıklamak için kullanabiliriz. Derin öğrenme çerçevelerinin üst düzey API'leri, ImageNet veri kümesi üzerinde önceden eğitilmiş geniş bir model yelpazesi sunar. Burada, bu modelin çıktı katmanının girişini (yani çıkarılan özellikler) yeniden kullandığımız önceden eğitilmiş bir ResNet-34 modeli seçiyoruz. Daha sonra orijinal çıkış katmanını, iki tam bağlı katmanı istifleme gibi eğitilebilecek küçük bir özel çıkış ağı ile değiştirebiliriz. :numref:`sec_fine_tuning`'teki deneyden farklı olarak, aşağıdaki özellik çıkarımı için kullanılan önceden eğitilmiş modeli yeniden eğitmez. Bu, degradelerin depolanması için eğitim süresini ve hafızasını azaltır. 

Tam ImageNet veri kümesi için üç RGB kanalının araçlarını ve standart sapmalarını kullanarak görüntüleri standartlaştırdığımızı hatırlayın. Aslında, bu aynı zamanda ImageNet'te önceden eğitilmiş model tarafından standardizasyon işlemi ile de tutarlıdır.

```{.python .input}
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

[**Kayıp** hesaplanmadan önce] önce, önce önceden eğitilmiş modelin çıktı katmanının girişini elde ederiz, yani ayıklanan özellik. Daha sonra kaybı hesaplamak için küçük özel çıkış ağımızın giriş olarak bu özelliği kullanıyoruz.

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

## [**Eğitim Fonksiyonu**] Tanımı

Modelin seçip, modelin doğrulama kümesindeki performansına göre hiperparametreleri ayarlayacağız. `train` model eğitim işlevi yalnızca küçük özel çıkış ağının parametrelerini yineleyir.

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
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
    # Only train the small custom output network
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
            animator.add(epoch + 1, (None, valid_loss.detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**Modeli Eğitim ve Doğrulanma**]

Şimdi modeli eğitebilir ve doğrulayabiliriz. Aşağıdaki hiperparametrelerin tümü ayarlanabilir. Örneğin, çeyin sayısı artırılabilir. `lr_period` ve `lr_decay` sırasıyla 2 ve 0.9 olarak ayarlandığından, optimizasyon algoritmasının öğrenme hızı her 2 çağ sonrasında 0,9 ile çarpılır.

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

## [**Test Setini Sınıflandırma**] ve Kaggle'da Sonuçları Gönderme

:numref:`sec_kaggle_cifar10`'teki son adıma benzer şekilde, sonunda tüm etiketli veriler (doğrulama seti dahil) modeli eğitmek ve test setini sınıflandırmak için kullanılır. Sınıflandırma için eğitimli özel çıkış ağını kullanacağız.

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

Yukarıdaki kod, :numref:`sec_kaggle_house`'te açıklanan şekilde Kaggle'a gönderilecek bir `submission.csv` dosyası oluşturacaktır. 

## Özet

* ImageNet veri kümelerindeki görüntüler CIFAR-10 görüntülerinden daha büyüktür (farklı boyutlarda). Farklı bir veri kümesindeki görevler için görüntü büyütme işlemlerini değiştirebiliriz.
* ImageNet veri kümesinin bir alt kümesini sınıflandırmak için, özellikleri ayıklamak ve yalnızca özel bir küçük ölçekli çıktı ağı eğitebilmek için ImageNet veri kümesinin tam ImageNet veri kümelerinde önceden eğitilmiş modelleri kullanabiliriz. Bu, daha az hesaplama süresi ve bellek maliyetine yol açacaktır.

## Egzersizler

1. Tam Kaggle rekabet veri kümesini kullanırken, diğer bazı hiperparametreleri `lr = 0.01`, `lr_period = 10` ve `lr_period = 10` ve `lr_decay = 0.1` olarak ayarlarken `batch_size` (toplu iş boyutu) ve `num_epochs` (çağın sayısı) artırdığınızda hangi sonuçları elde edebilirsiniz?
1. Daha derin bir önceden eğitilmiş model kullanırsanız daha iyi sonuçlar alır mısınız? Hiperparametreleri nasıl ayarlarsınız? Sonuçları daha da geliştirebilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1481)
:end_tab:
