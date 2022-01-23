# Nesne Algılama ve Sınırlama Kutuları
:label:`sec_bbox`

Önceki bölümlerde (örn. :numref:`sec_alexnet`—:numref:`sec_googlenet`), görüntü sınıflandırması için çeşitli modeller tanıttık. Görüntü sınıflandırma görevlerinde, resimde sadece bir* ana nesne olduğunu varsayıyoruz ve sadece kategorisini nasıl tanıyacağımıza odaklanıyoruz. Bununla birlikte, ilgi görüntüsünde sıklıkla *çok* nesne vardır. Sadece kategorilerini değil, aynı zamanda görüntüdeki belirli konumlarını da bilmek istiyoruz. Bilgisayar görüşünde, *nesne algılama* (veya *nesne tanıma) gibi görevlere atıfta bulunuyoruz. 

Nesne algılama birçok alanda yaygın olarak uygulanmıştır. Örneğin, çekilen video görüntülerindeki araçların, yayaların, yolların ve engellerin konumlarını tespit ederek seyahat rotalarını planlaması gerekir. Ayrıca, robotlar bu tekniği, bir ortamdaki gezinti boyunca ilgilenen nesneleri tespit etmek ve yerelleştirmek için kullanabilir. Ayrıca, güvenlik sistemlerinin davetsiz misafir veya bomba gibi anormal nesneleri algılaması gerekebilir. 

Sonraki birkaç bölümde, nesne algılama için çeşitli derin öğrenme yöntemlerini tanıtacağız. Nesnelerin*konumların* (veya *konumlar*) bir giriş ile başlayacağız.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Bu bölümde kullanılacak örnek görüntüyü yükleyeceğiz. Görüntünün sol tarafında bir köpek ve sağda bir kedi olduğunu görebiliriz. Bu görüntüdeki iki ana nesne bunlar.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Sınırlayıcı Kutular

Nesne algılamada, genellikle bir nesnenin uzamsal konumunu tanımlamak için bir *sınırlayıcı kutu* kullanırız. Sınırlayıcı kutu dikdörtgen olup, dikdörtgenin sol üst köşesinin $x$ ve $y$ koordinatları ve sağ alt köşenin koordinatları ile belirlenir. Yaygın olarak kullanılan bir diğer sınırlayıcı kutu gösterimi, sınırlayıcı kutu merkezinin $(x, y)$ eksen koordinatları ve kutunun genişliği ve yüksekliğidir. 

[**Burada** arasında dönüştürmek için işlevleri tanımlıyoruz**] bunlar (**iki gösterim**): `box_corner_to_center` iki köşeli temsilden merkez-genişlik yüksekliği sunumuna dönüştürür ve `box_center_to_corner` tersi. `boxes` giriş argümanı, $n$ sınırlayıcı kutuların sayısıdır burada iki boyutlu bir şekil tenörü ($n$, 4) olmalıdır.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Koordinat bilgilerine dayanarak [**köpeğin ve kedinin görüntüdeki sınırlayıcı kutularını tanımlayacağız. Görüntüdeki koordinatların kökeni görüntünün sol üst köşesidir ve sağ ve aşağı ise sırasıyla $x$ ve $y$ eksenlerinin pozitif yönleridir.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

İki sınırlayıcı kutu dönüştürme işlevinin doğruluğunu iki kez dönüştürerek doğrulayabiliriz.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Doğru olup olmadıklarını kontrol etmek için [**resimdeki sınırlayıcı kutuları çizim**] izin verin. Çizmeden önce, `bbox_to_rect` bir yardımcı işlevi tanımlayacağız. `matplotlib` paketinin sınırlayıcı kutu biçimindeki sınırlayıcı kutuyu temsil eder.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Görüntüye sınırlayıcı kutuları ekledikten sonra, iki nesnenin ana hatlarının temelde iki kutunun içinde olduğunu görebiliriz.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Özet

* Nesne algılaması sadece görüntüde ilgilenen tüm nesneleri değil, aynı zamanda konumlarını da tanır. Pozisyon genellikle dikdörtgen bir sınırlama kutusu ile temsil edilir.
* Yaygın olarak kullanılan iki sınırlayıcı kutu temsilleri arasında dönüştürebiliriz.

## Egzersizler

1. Başka bir görüntü bulun ve nesneyi içeren bir sınırlayıcı kutuyu etiketlemeyi deneyin. Etiketleme sınırlayıcı kutuları ve kategorileri karşılaştırın: hangi genellikle daha uzun sürer?
1. Neden giriş argümanı `boxes` arasında `box_corner_to_center` ve `box_center_to_corner` en içteki boyutu her zaman 4?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
