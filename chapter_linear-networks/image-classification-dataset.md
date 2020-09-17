# İmge Sınıflandırma Veri Kümesi
:label:`sec_fashion_mnist`

İmge sınıflandırması için yaygın olarak kullanılan veri kümelerinden biri MNIST veri kümesidir :cite:`LeCun.Bottou.Bengio.ea.1998`. Bir kıyaslama veri kümesi olarak iyi bir çalışma gerçekleştirmiş olsa da, günümüz standartlarına göre basit modeller bile %95'in üzerinde sınıflandırma doğruluğu elde ettiğinden daha güçlü modeller ile daha zayıf olanları ayırt etmek için uygun değildir. Bugün, MNIST bir kıyaslama ölçütü olmaktan çok makullük (sanity) kontrolü işlevi görüyor. Biraz daha ileriye gitmek için, önümüzdeki bölümlerdeki tartışmamızı niteliksel olarak benzer, ancak nispeten karmaşık olan 2017'de piyasaya sürülen Fashion-MNIST veri kümesine odaklayacağız :cite:`Xiao.Rasul.Vollgraf.2017`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## Veri Kümesini Okuma

Fashion-MNIST veri kümesini çerçevemizdeki yerleşik işlevler aracılığıyla indirebilir ve belleğe okuyabiliriz.

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST, her biri eğitim kümesinde 6000, test kümesinde ise 1000 ile temsil edilen 10 kategorideki görsellerden oluşmaktadır. Sonuç olarak eğitim kümesi ve test kümesi sırasıyla 60000 ve 10000 görüntü içermektedir.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

Her girdi imgesinin yüksekliği ve genişliği 28 pikseldir. Veri kümesinin, kanal sayısı 1 olan gri tonlamalı görüntülerden oluştuğuna dikkat edin. Kısaca, bu kitapta yüksekliği $h$ genişliği $w$ piksel olan herhangi bir görüntünün şekli $h \times w$ veya ($h$, $w$)'dir.

```{.python .input}
#@tab mxnet, pytorch
mnist_train[0][0].shape
```

```{.python .input}
#@tab tensorflow
mnist_train[0][0].shape
```

Fashion-MNIST'teki görüntüler şu kategorilerle ilişkilidir: tişört, pantolon, kazak, elbise, ceket, sandalet, gömlek, spor ayakkabı, çanta ve ayak bileği hizası bot. Aşağıdaki işlev, sayısal etiket indeksleri ve metindeki adları arasında dönüştürme yapar.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

Şimdi bu örnekleri görselleştirmek için bir işlev oluşturabiliriz.

```{.python .input}
#@tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Eğitim veri kümesindeki ilk birkaç örnek için görüntüler ve bunlara karşılık gelen etiketler (metin olarak) aşağıdadır.

```{.python .input}
X, y = mnist_train[:18]
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## Minigrup Okuma

Eğitim ve test kümelerinden okurken hayatımızı kolaylaştırmak için sıfırdan bir tane oluşturmak yerine yerleşik veri yineleyiciyi kullanıyoruz. Her yinelemede, bir yükleyicinin her seferinde grup (`batch_size`) boyutundaki bir veri mini grubunu okuduğunu hatırlayın. Ayrıca eğitim verisi yineleyicisi için örnekleri rastgele karıştırıyoruz.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data expect for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

Eğitim verilerini okurken geçen süreye bakalım.

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## Her Şeyi Bir Araya Getirme

Şimdi Fashion-MNIST veri kümesini alan ve okuyan `load_data_fashion_mnist` fonksiyonunu tanımlıyoruz. Hem eğitim kümesi hem de geçerleme kümesi için veri yineleyicileri döndürür. Ek olarak, imgeleri başka bir şekle yeniden boyutlandırmak için isteğe bağlı bir argüman kabul eder.

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

Aşağıda, `resize` bağımsız değişkenini belirterek `load_data_fashion_mnist` işlevinin görüntüyü yeniden boyutlandırma özelliğini test ediyoruz.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

Artık ilerleyen bölümlerde Fashion-MNIST veri kümesiyle çalışmaya hazırız.

## Özet

* Fashion-MNIST, 10 kategoriyi temsil eden resimlerden oluşan bir giyim sınıflandırma veri setidir. Bu veri kümesini, çeşitli sınıflandırma algoritmalarını değerlendirmek için sonraki bölümlerde kullanacağız.
* Yüksekliği $h$ genişliği $w$ piksel olan herhangi bir görüntünün şeklini $h \times w$ veya ($h$, $w$) olarak saklarız.
* Veri yineleyiciler, verimli performans için önemli bir bileşendir. Eğitim döngünüzü yavaşlatmaktan kaçınmak için yüksek performanslı hesaplamalardan yararlanan iyi uygulanmış veri yineleyicilerine güvenin.

## Alıştırmalar

1. `batch_size` değerini (örneğin 1'e) düşürmek okuma performansını etkiler mi?
1. Veri yineleyici performansı önemlidir. Mevcut uygulamanın yeterince hızlı olduğunu düşünüyor musunuz? İyileştirmek için çeşitli seçenekleri keşfediniz.
1. Çerçevenin çevrimiçi API belgelerine bakın. Başka hangi veri kümeleri mevcuttur?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/224)
:end_tab:
