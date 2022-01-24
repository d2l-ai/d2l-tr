# Tam Konvolsiyonel Ağlar
:label:`sec_fcn`

:numref:`sec_semantic_segmentation`'te tartışıldığı gibi, anlamsal segmentasyon görüntüleri piksel düzeyinde sınıflandırır. Tam bir evrimsel ağ (FCN), görüntü piksellerini :cite:`Long.Shelhamer.Darrell.2015` piksel sınıflarına dönüştürmek için bir evrimsel sinir ağı kullanır. Görüntü sınıflandırması veya nesne algılama için daha önce karşılaştığımız CNN'lerden farklı olarak, tam evrimsel bir ağ, ara özellik haritalarının yüksekliğini ve genişliğini girdi görüntüsüne geri dönüştürür: bu, :numref:`sec_transposed_conv`'da tanıtılan dönüştürülmüş evrimsel katman ile elde edilir. Sonuç olarak, sınıflandırma çıktısı ve girdi görüntüsü piksel düzeyinde bire bir yazışmaya sahiptir: herhangi bir çıkış pikselindeki kanal boyutu, girdi pikselinin sınıflandırma sonuçlarını aynı uzamsal konumda tutar.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
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
from torch.nn import functional as F
```

## Model

Burada tam evrimsel ağ modelinin temel tasarımını açıklıyoruz. :numref:`fig_fcn`'te gösterildiği gibi, bu model ilk olarak görüntü özelliklerini ayıklamak için bir CNN kullanır, daha sonra $1\times 1$ evrimsel katman aracılığıyla kanal sayısını sınıf sayısına dönüştürür ve son olarak özellik haritalarının yüksekliğini ve genişliğini tanıtılan dönüştürülmüş evrişim yoluyla giriş görüntüsüne dönüştürür içinde :numref:`sec_transposed_conv`. Sonuç olarak, model çıktısı girdi görüntüsüyle aynı yükseklik ve genişliğe sahiptir; burada çıktı kanalının giriş pikseli için öngörülen sınıfları aynı uzamsal konumda içerdiği yer alır. 

![Fully convolutional network.](../img/fcn.svg)
:label:`fig_fcn`

Aşağıda, [**görüntü özelliklerini ayıklamak için ImageNet veri kümesi üzerinde önceden eğitilmiş bir ResNet-18 modeli kullanıyor**] ve model örneğini `pretrained_net` olarak belirtiyoruz. Bu modelin son birkaç katmanı küresel ortalama havuzlama katmanı ve tam bağlı bir katman içerir: bunlar tamamen evrimsel ağda gerekli değildir.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

Ardından, [**tamamen evrimsel ağ örneği `net`**] oluşturuyoruz. Son genel ortalama havuzlama katmanı ve çıktıya en yakın tam bağlı katman dışında ResNet-18'deki tüm önceden eğitilmiş katmanları kopyalar.

```{.python .input}
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

Yüksekliği ve genişliği sırasıyla 320 ve 480 olan bir girdi göz önüne alındığında, `net`'ün ileri yayılımı giriş yüksekliğini ve genişliğini orijinalin 1/32'ine, yani 10 ve 15'e düşürür.

```{.python .input}
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

Ardından, [**çıkış kanallarının sayısını Pascal VOC2012 veri setinin sınıf sayısına (21) dönüştürmek için bir $1\times 1$ evrimsel katman kullanıyoruz.**] Son olarak, giriş görüntüsünün yüksekliğine ve genişliğine geri döndürmek için (**özellik haritalarının yüksekliğini ve genişliğini 32 kat artırma**) ihtiyacımız var. :numref:`sec_padding`'te bir kıvrımsal tabakanın çıkış şeklini nasıl hesaplayacağınızı hatırlayın. $(320-64+16\times2+32)/32=10$ ve $(480-64+16\times2+32)/32=15$'den bu yana, $(480-64+16\times2+32)/32=15$'den beri, $16$'a kadar çekirdeğin yüksekliğini ve genişliğini $64$'ya, dolgu $16$'a ayarlayarak $32$ adımıyla transpoze edilmiş bir kıvrımsal tabaka oluşturuyoruz. Genel olarak, $s$, $s$ dolgu $s/2$ ($s/2$ bir tamsayı varsayarak) ve çekirdeğin yüksekliği ve genişliği $2s$, dönüştürülmüş konvolüsyon $s$ kez giriş yüksekliğini ve genişliğini artıracağını görebilirsiniz.

```{.python .input}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**Transposed Konvolsyonel Katmanları Başlatılıyor**]

Dönüştürülmüş kıvrımlı katmanların özellik haritalarının yüksekliğini ve genişliğini artırabileceğini zaten biliyoruz. Görüntü işlemede, bir görüntüyü büyütmemiz gerekebilir, yani, *upsampling*.
*Bilineer enterpolasyon*
yaygın olarak kullanılan upsampling tekniklerinden biridir. Ayrıca, transpoze edilmiş kıvrımlı tabakaların başlatılması için de sıklıkla kullanılır. 

Bilineer enterpolasyonu açıklamak için, bir giriş görüntüsü göz önüne alındığında, yukarı örneklenen çıktı görüntüsünün her pikselini hesaplamak istediğimizi söyleyin. $(x, y)$ koordinatında çıktı görüntüsünün pikselini hesaplamak için, ilk harita $(x, y)$, giriş görüntüsünde $(x', y')$'yı koordine etmek için, örneğin giriş boyutunun çıkış boyutuna oranına göre. Eşlenen $x′$ and $y′$ gerçek sayılar olduğunu unutmayın. Ardından, giriş görüntüsünde $(x', y')$'yı koordine etmek için en yakın dört pikseli bulun. Son olarak, $(x, y)$ koordinatındaki çıktı görüntüsünün pikseli, giriş görüntüsündeki bu dört en yakın piksele ve $(x', y')$'dan göreli mesafelerine dayanarak hesaplanır.  

Bilineer enterpolasyonun örneklenmesi, aşağıdaki `bilinear_kernel` işlevi ile oluşturulmuş çekirdek ile transpolasyonel konvolüsyonel tabaka tarafından uygulanabilir. Alan sınırlamaları nedeniyle, algoritma tasarımı hakkında tartışılmadan sadece aşağıdaki `bilinear_kernel` işlevinin uygulanmasını sağlıyoruz.

```{.python .input}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

Let us [**experiment with upsampling of bilinear interpolation**] 
that is implemented by a transposed convolutional layer. 
We construct a transposed convolutional layer that 
doubles the height and weight,
and initialize its kernel with the `bilinear_kernel` function.

```{.python .input}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

`X` görüntüsünü okuyun ve yukarı örnekleme çıktısını `Y`'e atayın. Görüntüyü yazdırmak için kanal boyutunun konumunu ayarlamamız gerekiyor.

```{.python .input}
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

Gördüğümüz gibi, dönüştürülmüş kıvrımlı tabaka, görüntünün yüksekliğini ve genişliğini iki kat arttırır. Koordinatlardaki farklı ölçekler haricinde, ikili enterpolasyon ile büyütülmüş görüntü ve :numref:`sec_bbox`'te basılan orijinal görüntü aynı görünüyor.

```{.python .input}
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

[**Tamamen evrimsel bir ağda, ikili enterpolasyonun örneklenmesi ile dönüştürülmüş evrimsel tabakayı başlatırız. $1\times 1$ konvolsiyonel katman için Xavier başlatma kullanıyoruz.**]

```{.python .input}
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**Veri Kümesi Okuma**]

:numref:`sec_semantic_segmentation`'te tanıtıldığı gibi anlamsal segmentasyon veri kümesini okuduk. Rastgele kırpmanın çıktı görüntüsü şekli $320\times 480$ olarak belirtilir: hem yükseklik hem de genişlik $32$ ile bölünebilir.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**Eğitim**]

Şimdi inşa edilmiş tamamen evrimsel ağımızı eğitebiliriz. Buradaki kayıp fonksiyonu ve doğruluk hesaplaması, önceki bölümlerin görüntü sınıflandırılmasındakilerden farklı değildir. Her piksel için sınıfı tahmin etmek için dönüştürülmüş evrimsel katmanın çıkış kanalını kullandığımızdan, kanal boyutu kayıp hesaplamasında belirtilir. Buna ek olarak, doğruluk, tüm pikseller için tahmin edilen sınıfın doğruluğuna göre hesaplanır.

```{.python .input}
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**Prediction**]

Tahmin ederken, her kanaldaki giriş görüntüsünü standartlaştırmamız ve görüntüyü CNN'nin gerektirdiği dört boyutlu giriş formatına dönüştürmemiz gerekir.

```{.python .input}
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

Her pikselin [**tahmin edilen sınıfı**] görselleştirmek için, tahmin edilen sınıfı veri kümesindeki etiket rengine geri eşleriz.

```{.python .input}
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

Test veri kümelerindeki görüntüler boyut ve şekil bakımından farklılık gösterir. Model, bir giriş görüntüsünün yüksekliği veya genişliği 32 ile bölünmez olduğunda, 32 adımla transpoze edilmiş bir kıvrımlı katman kullandığından, dönüştürülmüş kıvrımlı katmanın çıkış yüksekliği veya genişliği giriş görüntüsünün şeklinden sapacaktır. Bu sorunu gidermek için, görüntüdeki 32 tamsayı katları olan yükseklik ve genişliğe sahip birden çok dikdörtgen alanı kırpabilir ve bu alanlardaki piksellerde ayrı ayrı ileriye yayılmasını gerçekleştirebiliriz. Bu dikdörtgen alanların birleşmesinin giriş görüntüsünü tamamen örtmesi gerektiğini unutmayın. Bir piksel birden fazla dikdörtgen alanla kaplandığında, aynı piksel için ayrı alanlardaki dönüştürülmüş evrişim çıktılarının ortalaması sınıfı tahmin etmek için softmax işlemine girilebilir. 

Basitlik açısından, sadece birkaç büyük test görüntüsü okuruz ve görüntünün sol üst köşesinden başlayarak tahmin için $320\times480$ alan kırpıyoruz. Bu test görüntüleri için kırpılmış alanlarını, tahmin sonuçlarını ve temel doğruluk satırını satır satır yazdırıyoruz.

```{.python .input}
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Özet

* Tam evrimsel ağ önce görüntü özelliklerini ayıklamak için bir CNN kullanır, daha sonra $1\times 1$ bir evrimsel katman aracılığıyla kanal sayısını sınıf sayısına dönüştürür ve son olarak özellik haritalarının yüksekliğini ve genişliğini dönüştürülmüş evrişim yoluyla girdi görüntüsüne dönüştürür.
* Tamamen evrimsel bir ağda, dönüştürülmüş evrimsel tabakayı başlatmak için ikili enterpolasyonun çoğaltılmasını kullanabiliriz.

## Egzersizler

1. Deneyde dönüştürülmüş evrimsel katman için Xavier başlatma kullanırsak, sonuç nasıl değişir?
1. Hiperparametreleri ayarlayarak modelin doğruluğunu daha da geliştirebilir misiniz?
1. Test görüntülerindeki tüm piksellerin sınıflarını tahmin edin.
1. Orijinal tamamen evrimsel ağ kağıdı, bazı ara CNN katmanlarının :cite:`Long.Shelhamer.Darrell.2015` çıkışlarını da kullanır. Bu fikri uygulamaya çalışın.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:
