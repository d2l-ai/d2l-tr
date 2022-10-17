# Nesne Algılama Veri Kümesi
:label:`sec_object-detection-dataset`

Nesne algılama alanında MNIST ve Fashion-MNIST gibi küçük bir veri kümeleri bulunmamaktadır. Nesne algılama modellerini hızlı bir şekilde göstermek için [**küçük bir veri kümesi topladık ve etiketledik**]. İlk olarak ofisimizdeki bedava muzların fotoğraflarını çektik ve farklı dönüşlerde ve boyutlarda 1000 muz imgesi oluşturduk. Sonra her bir muz imgesini bir arka plan görüntüsü üzerinde rastgele bir konuma yerleştirdik. Sonunda, resimlerdeki bu muzlar için kuşatan kutuları etiketledik. 

## [**Veri Kümesini İndirme**]

Tüm imgeler ve csv etiket dosyasıyla birlikte muz algılama veri kümesi doğrudan internetten indirilebilir.

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

Aşağıdaki `read_data_bananas` işlevinde [**muz algılama veri kümesini okuyacağız**]. Veri kümesi,  bir csv dosyasında nesne sınıfı etiketlerini ve gerçek referans değeri kuşatan kutunun sol üst ve sağ alt köşelerdeki koordinatları içerir.

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """Muz algılama veri kümesi imgelerini ve etiketlerini okuyun."""
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
        # Burada `target` (hedef), tüm imgelerin aynı muz sınıfına 
        # sahip olduğu (sınıf, sol üst x, sol üst y, sağ alt x, sağ alt y) 
        # içerir (indeks 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Muz algılama veri kümesi imgelerini ve etiketlerini okuyun."""
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
        # Burada `target` (hedef), tüm imgelerin aynı muz sınıfına 
        # sahip olduğu (sınıf, sol üst x, sol üst y, sağ alt x, sağ alt y) 
        # içerir (indeks 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

İmgeleri ve etiketleri okumak için `read_data_bananas` işlevini kullanarak, aşağıdaki `BananasDataset` sınıfı, muz algılama veri kümesini yüklemek için [**özelleştirilmiş bir `Dataset` örneği oluşturmamızı**] sağlayacaktır.

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """Muz algılama veri kümesini yüklemek için özelleştirilmiş bir veri kümesi."""
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
    """Muz algılama veri kümesini yüklemek için özelleştirilmiş bir veri kümesi."""    
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Son olarak, `load_data_bananas` fonksiyonunu [**hem eğitim hem de test kümeleri için iki veri yineleyici örneği döndürecek şekilde tanımlarız.**] Test veri kümesi için onu rastgele sırayla okumaya gerek yoktur.

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """Muz algılama veri kümesini yükleyin."""
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
    """Muz algılama veri kümesini yükleyin."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

[**Bir minigrup okuyalım ve bu minigruptaki hem imgelerin hem de etiketlerin şekillerini yazdıralım**]. İmge minigrubunun şekli (grup boyutu, kanal sayısı, yükseklik, genişlik) tanıdık görünüyor: Önceki imge sınıflandırma görevlerimizle aynı. Minigrup etiketinin şekli (parti boyutu, $m$, 5) şeklindedir; burada $m$, veri kümesinde herhangi bir imgenin sahip olabileceği mümkün olan en büyük kuşatan kutu sayısıdır.

Minigruplarda hesaplama daha verimli olmasına rağmen, tüm imge örneklerinin bitiştirme yoluyla bir minigrup oluşturması için aynı sayıda kuşatan kutu içermeleri gerektirir. Genel olarak, imgeler farklı sayıda kuşatan kutuya sahip olabilir; bu nedenle $m$'den daha az kuşatan kutuya sahip imgeler, $m$'e ulaşılana kadar geçersiz kuşatan kutularla doldurulacaktır. Daha sonra her kuşatan kutunun etiketi 5 uzunluğunda bir dizi ile temsil edilir. Dizideki ilk öğe, kuşatan kutudaki nesnenin sınıfıdır ve burada -1 dolgu için geçersiz bir kuşatan kutusunu gösterir. Dizinin kalan dört öğesi, kuşatan kutunun sol üst köşesinin ve sağ alt köşesinin ($x$, $y$) koordinat değerleridir (aralık 0 ile 1 arasındadır). Muz veri kümesi için, her imgede sadece bir kuşatan kutu olduğundan elimizde $m=1$ var.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Kanıtlama**]

Gerçek referans değeri etiketli kuşatan kutularıyla on imge gösterelim. Muzun dönüşlerinin, boyutlarının ve konumlarının tüm bu imgelerde değiştiğini görebiliyoruz. Tabii ki, bu sadece basit bir yapay veri kümesidir. Uygulamada, gerçek dünya veri kümeleri genellikle çok daha karmaşıktır.

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

* Topladığımız muz algılama veri kümesi, nesne algılama modellerini göstermek için kullanılabilir.
* Nesne algılama için veri yükleme, imge sınıflandırmasındakine benzer. Bununla birlikte, nesne algılamasında etiketler ayrıca imge sınıflandırmasında eksik olan gerçek referans değeri kuşatan kutularla ilgili bilgileri de içerir.

## Alıştırmalar

1. Muz algılama veri kümelerinde gerçek referans değeri kuşatan kutularla diğer imgeleri gösterin. Kuşatan kutulara ve nesnelere göre nasıl farklılık gösterirler?
1. Nesne algılamaya rastgele kırpma gibi veri artırımı uygulamak istediğimizi varsayalım. Bu imge sınıflandırmasındakinden nasıl farklı olabilir? İpucu: Kırpılmış bir imge nesnenin yalnızca küçük bir bölümünü içeriyorsa ne olur?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1608)
:end_tab:
