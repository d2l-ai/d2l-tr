# Nesne Algılama ve Kuşatan Kutular
:label:`sec_bbox`

Önceki bölümlerde (örn. :numref:`sec_alexnet`—-:numref:`sec_googlenet`), imge sınıflandırma için çeşitli modeller tanıttık. İmge sınıflandırma görevlerinde, resimde sadece *bir* ana nesne olduğunu varsayıyoruz ve sadece kategorisini nasıl tanıyacağımıza odaklanıyoruz. Bununla birlikte, ilgili imgede sıklıkla *çoklu* nesne vardır. Sadece kategorilerini değil, aynı zamanda imgedeki belirli konumlarını da bilmek istiyoruz. Bilgisayarla görmede, *nesne algılama* (veya *nesne tanıma*) gibi görevlere atıfta bulunuyoruz. 

Nesne algılama birçok alanda yaygın olarak uygulanmıştır. Örneğin, kendi kendine sürüş, çekilen video görüntülerinde araçların, yayaların, yolların ve engellerin konumlarını tespit ederek seyahat rotalarını planlamalıdır. Ayrıca, robotlar bu tekniği, bir ortamdaki gezinti boyunca ilgilenen nesneleri tespit etmek ve yerini belirlemek için kullanabilir. Ayrıca, güvenlik sistemlerinin davetsiz misafir veya bomba gibi sıradışı nesneleri algılaması gerekebilir. 

Sonraki birkaç bölümde, nesne algılama için çeşitli derin öğrenme yöntemlerini tanıtacağız. Nesnelerin *konumlarına* bir giriş ile başlayacağız.

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

Bu bölümde kullanılacak örnek imgeyi yükleyeceğiz. Görüntünün sol tarafında bir köpek ve sağda bir kedi olduğunu görebiliriz. Bu imgedeki iki ana nesne bunlardır.

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

## Kuşatan Kutular

Nesne algılamada, genellikle bir nesnenin uzamsal konumunu tanımlamak için bir *kuşatan kutu* kullanırız. Kuşatan kutu dikdörtgen olup, dikdörtgenin sol üst köşesinin $x$ ve $y$ koordinatları ve sağ alt köşenin koordinatları ile belirlenir. Yaygın olarak kullanılan bir diğer kuşatan kutu gösterimi, kuşatan kutu merkezinin $(x, y)$ eksen koordinatları ve kutunun genişliği ve yüksekliğidir. 

[**Burada**] (**iki temsil**) arasında dönüştürecek işlevleri tanımlıyoruz : 
`box_corner_to_center` iki köşeli temsilden merkez-genişlik yüksekliği temsiline dönüştürür ve `box_center_to_corner` tersini yapar. `boxes` girdi argümanı, iki boyutlu ($n$, 4) şekilli bir tensör olmalıdır, burada $n$ sınırlayıcı kutuların sayısıdır.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """(sol üst, sağ alt) konumundan (merkez, genişlik, yükseklik) konumuna dönüştür."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """(sol üst, sağ alt) konumundan (merkez, genişlik, yükseklik) konumuna dönüştür."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Koordinat bilgilerine göre [**resimdeki köpek ve kedinin kuşatan kutularını tanımlayacağız**]. İmgedeki koordinatların orijini imgenin sol üst köşesidir ve sağ ve aşağı sırasıyla $x$ ve $y$ eksenlerinin pozitif yönleridir.

```{.python .input}
#@tab all
# Burada `bbox` kuşatan kutunun kısaltmasıdır
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

İki kuşatan kutu dönüştürme işlevinin doğruluğunu iki kez dönüştürerek doğrulayabiliriz.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Doğru olup olmadıklarını kontrol etmek için [**resimdeki kuşatan kutuları çizelim**]. Çizmeden önce, `bbox_to_rect` yardımcı işlevini tanımlayacağız. `matplotlib` paketinin kuşatan kutu biçimindeki kuşatan kutusunu temsil eder.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Kuşatan kutuyu matplotlib biçimine dönüştürün."""
    # Kuşatan kutunun (sol üst x, sol üst y, sağ alt x, sağ alt y) biçimini matplotlib biçimine dönüştürün: ((sol üst x, sol üst y), genişlik, yükseklik)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

İmgeye kuşatan kutuları ekledikten sonra, iki nesnenin ana hatlarının temelde iki kutunun içinde olduğunu görebiliriz.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Özet

* Nesne algılama sadece imgede ilgili tüm nesneleri değil, aynı zamanda konumlarını da tanır. Pozisyon genellikle dikdörtgen bir kuşatan kutu ile temsil edilir.
* Yaygın olarak kullanılan iki kuşatan kutu temsili arasında dönüşüm yapabiliriz.

## Alıştırmalar

1. Başka bir imge bulun ve nesneyi içeren bir kuşatan kutuyu etiketlemeyi deneyin. Kuşatan kutuları ve kategorileri etiketlemeyi karşılaştırın: Hangisi genellikle daha uzun sürer?
1. Neden `box_center_to_corner` ve `box_corner_to_center` girdi bağımsız değişkeninin "kutuları" nın en içteki boyutu her zaman 4'tür?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1527)
:end_tab:
