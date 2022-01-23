# Nesne Algılama Veri Kümesi
:label:`sec_object-detection-dataset`

Nesne algılama alanında MNIST ve Moda-MNIST gibi küçük veri kümesi yoktur. Nesne algılama modellerini hızlı bir şekilde göstermek için [**küçük bir veri kümesi topladık ve etiketledik]. İlk olarak, ofisimizden ücretsiz muzların fotoğraflarını çektik ve farklı rotasyon ve boyutlarda 1000 muz görüntüsü ürettik. Sonra her muz görüntüsünü arka plan görüntüsünde rastgele bir konuma yerleştirdik. Sonunda, resimlerdeki muzlar için sınırlayıcı kutuları etiketledik. 

## [**Veri Kümesi İndiriliyor**]

Tüm görüntü ve csv etiket dosyalarıyla birlikte muz algılama veri seti doğrudan internetten indirilebilir.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Veri Kümesini Okuma

Aşağıdaki `read_data_bananas` işlevinde [**muz algılama veri setini okuyun**] yapacağız. Veri kümesi, nesne sınıfı etiketleri için bir csv dosyası ve sol üst ve sağ alt köşelerde yer hakikat sınırlama kutusu koordinatları içerir.

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

Görüntüleri ve etiketleri okumak için `read_data_bananas` işlevini kullanarak, aşağıdaki `BananasDataset` sınıfı muz algılama veri kümesini yüklemek için [**özelleştirilmiş bir `Dataset` örneği** oluşturmamızı] sağlayacaktır.

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Son olarak, `load_data_bananas` işlevini [**eğitim ve test setleri için iki veri yineleme örneği döndürmek üzere tanımlıyoruz. **] Test veri kümesi için, rastgele sırayla okumaya gerek yoktur.

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

Bu minibatch içinde [**bir mini batch okuyalım ve hem resimlerin hem de etiketlerin şekillerini yazdıralım. Görüntü minibatch şekli, (parti boyutu, kanal sayısı, yükseklik, genişlik), tanıdık görünüyor: önceki görüntü sınıflandırma görevlerimizde olduğu gibi aynıdır. Etiket minibatch şekli (toplu boyutu, $m$, 5) olup, burada $m$, veri kümelerinde herhangi bir görüntünün sahip olduğu en büyük sınırlayıcı kutu sayısıdır. 

Minibatch'lerde hesaplama daha verimli olmasına rağmen, tüm görüntü örneklerinin birleştirme yoluyla bir mini batch oluşturmak için aynı sayıda sınırlayıcı kutuları içermesini gerektirir. Genel olarak, görüntüler farklı sayıda sınırlayıcı kutuya sahip olabilir; bu nedenle $m$'ten daha az sınırlayıcı kutuya sahip görüntüler, $m$'e ulaşılana kadar yasadışı sınırlayıcı kutularla doldurulacaktır. Daha sonra her sınırlayıcı kutunun etiketi 5 uzunluğunda bir dizi ile temsil edilir. Dizideki ilk öğe, sınırlayıcı kutudaki nesnenin sınıfıdır ve burada -1 dolgu için yasadışı bir sınırlama kutusunu gösterir. Dizinin kalan dört öğesi, sol üst köşenin ve sınırlayıcı kutunun sağ alt köşesinin ($x$, $y$) koordinat değerleridir (aralık 0 ile 1 arasındadır). Muz veri seti için, her görüntüde sadece bir sınırlayıcı kutu olduğundan $m=1$ sahibiz.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Gösteri**]

Etiketli zemin gerçeği sınırlayıcı kutularıyla on görüntü gösterelim. Muzun dönüşlerinin, boyutlarının ve konumlarının tüm bu görüntülerde değiştiğini görebiliyoruz. Tabii ki, bu sadece basit bir yapay veri kümesidir. Uygulamada, gerçek dünya veri kümeleri genellikle çok daha karmaşıktır.

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Özet

* Topladığımız muz algılama veri seti, nesne algılama modellerini göstermek için kullanılabilir.
* Nesne algılama için veri yükleme, görüntü sınıflandırmasına benzer. Bununla birlikte, nesne algılamasında etiketler ayrıca görüntü sınıflandırmasında eksik olan zemin gerçeği sınırlayıcı kutularla ilgili bilgileri de içerir.

## Egzersizler

1. Muz algılama veri kümelerinde yer hakikati sınırlayıcı kutularla diğer görüntüleri gösterin. Sınırlayıcı kutulara ve nesnelere göre nasıl farklılık gösterirler?
1. Nesne algılamaya rastgele kırpma gibi veri büyütme uygulamak istediğimizi söyleyin. Görüntü sınıflandırmasında bundan nasıl farklı olabilir? İpucu: Kırpılmış bir görüntü nesnenin yalnızca küçük bir bölümünü içeriyorsa ne olur?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
