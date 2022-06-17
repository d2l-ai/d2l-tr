# İmge Sınıflandırma Veri Kümesi
:label:`sec_fashion_mnist`

(~~MNIST veri kümesi, imge sınıflandırması için yaygın olarak kullanılan veri kümelerinden biridir, ancak bir kıyaslama veri kümesi olarak çok basittir. Benzer, ancak daha karmaşık Fashion-MNIST veri kümesini kullanacağız~~)

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

[**Fashion-MNIST veri kümesini çerçevemizdeki yerleşik işlevler aracılığıyla indirebilir ve belleğe okuyabiliriz**].

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# 'ToTensor', imge verilerini PIL türünden 32 bit kayan virgüllü sayı tensörlerine 
# dönüştürür. Tüm sayıları 255'e böler, böylece tüm piksel değerleri 0 ile 1 arasında olur.
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

Fashion-MNIST, her biri eğitim veri kümesinde 6000 görsel ve test veri kümesinde 1000 görsel ile temsil edilen 10 kategorideki görsellerden oluşur. Eğitim için değil, model performansını değerlendirmek için bir *test veri kümesi* (veya *test kümesi*) kullanılır.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

Her girdi imgesinin yüksekliği ve genişliği 28 pikseldir. Veri kümesinin, kanal sayısı 1 olan gri tonlamalı görsellerden oluştuğuna dikkat edin. Kısaca, bu kitapta yüksekliği $h$ genişliği $w$ piksel olan herhangi bir imgenin şekli $h \times w$ veya ($h$, $w$)'dir.

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

[~~Veri kümesini görselleştirmek için iki yardımcı işlev~~]

Fashion-MNIST'teki görseller şu kategorilerle ilişkilidir: Tişört, pantolon, kazak, elbise, ceket, sandalet, gömlek, spor ayakkabı, çanta ve ayak bileği hizası bot. Aşağıdaki işlev, sayısal etiket indeksleri ve metindeki adları arasında dönüştürme yapar.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Fashion-MNIST veri kümesi için metin etiketleri döndürün."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

Şimdi bu örnekleri görselleştirmek için bir işlev oluşturabiliriz.

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Görsellerin bir listesini çizin"""
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

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Görsellerin bir listesini çizin"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Eğitim veri kümesindeki ilk birkaç örnek için [**görseller ve bunlara karşılık gelen etiketler (metin olarak)**] aşağıdadır.

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
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

Eğitim ve test kümelerinden okurken hayatımızı kolaylaştırmak için sıfırdan bir tane oluşturmak yerine yerleşik veri yineleyiciyi kullanıyoruz. Her yinelemede, bir yineleyicinin [**her seferinde grup (`batch_size`) boyutundaki bir veri minigrubunu okuduğunu**] hatırlayın. Ayrıca eğitim verisi yineleyicisi için örnekleri rastgele karıştırıyoruz.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Windows dışında, verileri okumak için 4 işlem kullanın."""
    return 0 if sys.platform.startswith('win') else 4

# 'ToTensor', görsel verilerini uint8'den 32-bit kayan virgüllü sayıya dönüştürür. 
# Tüm sayıları 255'e böler, böylece tüm piksel değerleri 0 ile 1 arasında olur.
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Verileri okumak için 4 işlem kullanın."""
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

Şimdi [**Fashion-MNIST veri kümesini alan ve okuyan `load_data_fashion_mnist` fonksiyonunu**] tanımlıyoruz. Hem eğitim kümesi hem de geçerleme kümesi için veri yineleyicileri döndürür. Ek olarak, imgeleri başka bir şekle yeniden boyutlandırmak için isteğe bağlı bir argüman kabul eder.

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Fashion-MNIST veri kümesini indirin ve ardından belleğe yükleyin."""
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
    """Fashion-MNIST veri kümesini indirin ve ardından belleğe yükleyin."""
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
    """Fashion-MNIST veri kümesini indirin ve ardından belleğe yükleyin."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Tüm piksel değerleri 0 ile 1 arasında olacak şekilde tüm sayıları 255'e bölün, 
    # en sonunda bir grup boyutu ekleyin. Ayrıca etiketi int32'ye çevirin.
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

Aşağıda, `resize` bağımsız değişkenini belirterek `load_data_fashion_mnist` işlevinin görseli yeniden boyutlandırma özelliğini test ediyoruz.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

Artık ilerleyen bölümlerde Fashion-MNIST veri kümesiyle çalışmaya hazırız.

## Özet

* Fashion-MNIST, 10 kategoriyi temsil eden resimlerden oluşan bir giyim sınıflandırma veri kümesidir. Bu veri kümesini, çeşitli sınıflandırma algoritmalarını değerlendirmek için sonraki bölümlerde kullanacağız.
* Yüksekliği $h$ genişliği $w$ piksel olan herhangi bir imgenin şeklini $h \times w$ veya ($h$, $w$) olarak saklarız.
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
