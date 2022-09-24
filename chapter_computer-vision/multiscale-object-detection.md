# Çoklu Ölçekli Nesne Algılama
:label:`sec_multiscale-object-detection`

:numref:`sec_anchor` içinde, bir girdi imgesinin her pikselinde ortalanmış birden çok çapa kutusu oluşturduk. Esasen bu çapa kutuları imgenin farklı bölgelerinin örneklerini temsil eder. Ancak, *her* piksel için oluşturulmuşlarsa hesaplayamayacak kadar çok çapa kutusu elde edebiliriz. Bir $561 \times 728$'lik girdi imgesi düşünün. Merkez alarak her piksel için farklı şekillerde beş çapa kutusu oluşturulursa, iki milyondan fazla çapa kutusunun ($561  \times 728 \times 5$) imge üzerinde etiketlenmesi ve tahmin edilmesi gerekir.

## Çoklu Ölçekli Çapa Kutuları
:label:`subsec_multiscale-anchor-boxes`

Bir imgedeki çapa kutularını azaltmanın zor olmadığını fark edebilirsiniz. Örneğin, üzerlerinde ortalanmış çapa kutuları oluşturmak için girdi imgesindeki piksellerin küçük bir bölümünü tekdüze bir şekilde örnekleyebiliriz. Buna ek olarak, farklı ölçeklerde farklı boyutlarda farklı çapa kutuları üretebiliriz. Sezgisel olarak, daha küçük nesnelerin imgede daha büyük olanlardan daha fazla görünme olasılığı daha yüksektir. Örnek olarak, $1 \times 1$, $1 \times 2$ ve $2 \times 2$ nesneler $2 \times 2$'lik imgede sırasıyla 4, 2 ve 1 olası şekilde görünebilir. Bu nedenle, daha küçük nesneleri algılamak için daha küçük çapa kutuları kullanırken daha fazla bölgeyi örnekleyebiliriz, daha büyük nesneler için daha az bölgeyi örnekleyebiliriz. 

Birden çok ölçekte çapa kutuları nasıl oluşturulacağını göstermek için bir imgeyi okuyalım. Yüksekliği ve genişliği sırasıyla 561 ve 728 piksel olsun.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

:numref:`sec_conv_layer` içinde, evrişimli bir katmanın iki boyutlu bir dizi çıktısını bir öznitelik haritası olarak adlandırdığımızı hatırlayın. Öznitelik haritası şeklini tanımlayarak, herhangi bir imgedeki düzgün örneklenmiş çapa kutularının merkezlerini belirleyebiliriz. 

`display_anchors` işlevi aşağıda tanımlanmıştır. [**Çapa kutusu merkezi olarak her birim (piksel) ile öznitelik haritasında (`fmap`) çapa kutuları (`anchors`) oluştururuz.**] Çapa kutularındaki $(x, y)$ eksen koordinat değerleri (`anchors`), öznitelik haritasının genişlik ve yüksekliğine bölünmüş olduğundan (`fmap`), ki bu değerler 0 ve 1 arasındadır, öznitelik haritasındaki çapa kutularının göreli konumlarını belirtir. 

Çapa kutularının merkezleri (`anchors`) öznitelik haritasındaki tüm birimlere (`fmap`) yayıldığından, bu merkezler göreceli uzamsal konumları bakımından herhangi bir girdi imgesine tekdüze şekilde dağıtılmalıdır. Daha somut olarak, öznitelik haritasının sırasıyla `fmap_w` ve `fmap_h`, genişliği ve yüksekliği, göz önüne alındığında, aşağıdaki işlev, herhangi bir girdi imgesindeki `fmap_h` satırlarındaki ve `fmap_w` sütunlarındaki pikselleri *tekdüze* bir şekilde örnekleyecektir. Bu eşit örneklenen pikseller üzerinde ortalanan `s` ölçek çapa kutuları (`s` listenin uzunluğunun 1 olduğu varsayılarak) ve farklı en-boy oranları (`ratios`) oluşturulur.

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # İlk iki boyuttaki değerler çıktıyı etkilemez
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # İlk iki boyuttaki değerler çıktıyı etkilemez
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

İlk olarak, [**küçük nesnelerin algılanmasını düşünün**]. Görüntülendiğinde ayırt edilmeyi kolaylaştırmak için, buradaki farklı merkezlere sahip çapa kutuları çakışmaz: Çapa kutusu ölçeği 0.15 olarak ayarlanır ve öznitelik haritasının yüksekliği ve genişliği 4'e ayarlanır. Çapa kutularının merkezlerinin imge üzerindeki 4 satır ve  4 sütuna eşit olarak dağıtıldığını görebiliriz.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

[**Öznitelik haritasının yüksekliğini ve genişliğini yarıya indirmeye ve daha büyük nesneleri algılamak için daha büyük çapa kutuları kullanmaya**] geçiyoruz. Ölçek 0.4 olarak ayarlandığında, bazı çapa kutuları birbirleriyle çakışır.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Son olarak, [**öznitelik haritasının yüksekliğini ve genişliğini yarıya indirir ve çapa kutusu ölçeğini 0.8 seviyesine çıkarırız**]. Şimdi çapa kutusunun merkezi imgenin merkezidir.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Çoklu Ölçekli Algılama

Çoklu ölçekli çapa kutuları oluşturduğumuzdan, bunları farklı ölçeklerde çeşitli boyutlardaki nesneleri algılamak için kullanacağız. Aşağıda, :numref:`sec_ssd` içinde uygulayacağımız CNN tabanlı çoklu ölçekli nesne algılama yöntemini tanıtıyoruz. 

Bazı ölçekte, $h \times w$ şeklinde $c$ tane öznitelik haritası olduğunu varsayalım. :numref:`subsec_multiscale-anchor-boxes` içindeki yöntemi kullanarak, her kümenin aynı merkeze sahip $a$ çapa kutusuna sahip olduğu $hw$ çapa kutusu kümeleri oluştururuz. Örneğin, :numref:`subsec_multiscale-anchor-boxes` içindeki deneylerde ilk ölçekte, on (kanal sayısı) $4 \times 4$ öznitelik haritası verildiğinde, her kümenin aynı merkeze sahip 3 çapa kutusu içerdiği 16 adet çapa kutusu kümesi oluşturduk. Daha sonra, her çapa kutusu gerçek referans değerinin kuşatan kutusunun ofseti ve sınıfı ile etiketlenir. Geçerli ölçekte, nesne algılama modelinin, farklı kümelerin farklı merkezlere sahip olduğu girdi imgesindeki $hw$ çapa kutuları kümelerinin sınıflarını ve ofsetlerini tahmin etmesi gerekir. 

Buradaki $c$ öznitelik haritalarının girdi imgesine dayalı CNN ileri yayma tarafından elde edilen ara çıktılar olduğunu varsayalım. Her öznitelik haritasında $hw$ farklı uzamsal konum bulunduğundan, aynı uzamsal konumun $c$ birime sahip olduğu düşünülebilir. :numref:`sec_conv_layer` içindeki alıcı alan tanımına göre, öznitelik haritalarının aynı uzamsal konumundaki bu $c$ birimleri, girdi imgesinde aynı alıcı alana sahiptir: Aynı alıcı alanındaki girdi imgesi bilgisini temsil ederler. Bu nedenle, aynı uzamsal konumdaki öznitelik haritalarının $c$ birimi, bu uzamsal konum kullanılarak oluşturulan $a$ tane çapa kutusunun sınıflarına ve uzaklıklarına dönüştürebiliriz. Özünde, girdi imgesindeki alıcı alana yakın olan çapa kutularının sınıflarını ve ofsetlerini tahmin etmek için belirli bir alıcı alandaki girdi imgesinin bilgisini kullanırız. 

Farklı katmanlardaki öznitelik haritaları, girdi imgesinde farklı boyutlarda alıcı alanlara sahip olduğunda, farklı boyutlardaki nesneleri algılamak için kullanılabilirler. Örneğin, çıktı katmanına daha yakın olan öznitelik haritalarının birimlerinin daha geniş alıcı alanlara sahip olduğu bir sinir ağı tasarlayabiliriz, böylece girdi imgesinde daha büyük nesneleri algılayabilirler.

Özetle, çoklu ölçekli nesne algılama için derin sinir ağları aracılığıyla imgelerin katmansal temsillerini birden çok düzeyde kullanabiliriz. Bunun :numref:`sec_ssd` içinde somut bir örnekle nasıl çalıştığını göstereceğiz. 

## Özet

* Çoklu ölçekte, farklı boyutlardaki nesneleri algılamak için farklı boyutlarda çapa kutuları üretebiliriz.
* Öznitelik haritalarının şeklini tanımlayarak, herhangi bir imgedeki tekdüze örneklenmiş çapa kutularının merkezlerini belirleyebiliriz.
* Girdi imgesinde bir alıcı alana yakın olan çapa kutularının sınıflarını ve uzaklıklarını tahmin etmek için o belirli alıcı alandaki girdi imgesinin bilgisini kullanırız.
* Derin öğrenme sayesinde, çoklu ölçekli nesne algılama için imgelerin katmansal temsillerini birden çok düzeyde kullanabiliriz.

## Alıştırmalar

1. :numref:`sec_alexnet` içindeki tartışmalarımıza göre, derin sinir ağları imgeler için soyutlama düzeylerini artırarak hiyerarşik öznitelikleri öğrenir. Çoklu ölçekli nesne algılamada, farklı ölçeklerdeki öznitelik haritaları farklı soyutlama düzeylerine karşılık geliyor mu? Neden ya da neden değil?
1. :numref:`subsec_multiscale-anchor-boxes` içindeki deneylerde ilk ölçekte (`fmap_w=4, fmap_h=4`), çakışabilecek tekdüze dağıtılmış çapa kutuları oluşturun.
1. $1 \times c \times h \times w$ şeklinde bir öznitelik haritası değişkeni verilsin, burada $c$, $h$ ve $w$, öznitelik haritalarının sırasıyla kanal sayısı, yüksekliği ve genişliğidir. Bu değişkeni çapa kutularının sınıflarına ve ofsetlerine nasıl dönüştürebilirsiniz? Çıktının şekli nedir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1607)
:end_tab:
