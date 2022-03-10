# Çok Ölçekli Nesne Algılama
:label:`sec_multiscale-object-detection`

:numref:`sec_anchor`'te, bir girdi görüntüsünün her pikselinde ortalanmış birden çok çapa kutusu oluşturduk. Esasen bu çapa kutuları görüntünün farklı bölgelerinin örneklerini temsil eder. Bununla birlikte, *herşey* piksel için oluşturulmuşsa hesaplamak için çok fazla bağlantı kutusu ile sonuçlanabiliriz. Bir $561 \times 728$ giriş görüntüsü düşünün. Her piksel için merkezi olarak değişen şekillere sahip beş bağlantı kutusu oluşturulursa, görüntü üzerinde iki milyondan fazla bağlantı kutusu ($561 \times 728 \times 5$) etiketlenmeli ve tahmin edilmelidir. 

## Çok ölçekli çapa kutuları
:label:`subsec_multiscale-anchor-boxes`

Bir görüntüdeki bağlantı kutularını azaltmanın zor olmadığını fark edebilirsiniz. Örneğin, üzerinde merkezli bağlantı kutuları oluşturmak için giriş görüntüsünden piksellerin küçük bir bölümünü eşit bir şekilde örnekleyebiliriz. Buna ek olarak, farklı ölçeklerde farklı boyutlarda farklı çapa kutuları üretebiliriz. Sezgisel olarak, daha küçük nesnelerin görüntüde daha büyük olanlardan daha fazla görünme olasılığı daha yüksektir. Örnek olarak, $1 \times 1$, $1 \times 2$ ve $2 \times 2$ nesneler $2 \times 2$ görüntüsünde sırasıyla 4, 2 ve 1 olası yolla görünebilir. Bu nedenle, daha küçük nesneleri algılamak için daha küçük bağlantı kutuları kullanırken daha fazla bölgeyi örnekleyebiliriz, daha büyük nesneler için daha az bölgeyi örnekleyebiliriz. 

Birden çok ölçekte çapa kutuları nasıl oluşturulacağını göstermek için bir görüntüyü okuyalım. Yüksekliği ve genişliği sırasıyla 561 ve 728 pikseldir.

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

:numref:`sec_conv_layer`'te, evrimsel bir katmanın iki boyutlu bir dizi çıkışını bir özellik haritası olarak adlandırdığımızı hatırlayın. Özellik haritası şeklini tanımlayarak, herhangi bir görüntüdeki düzgün örneklenmiş çapa kutularının merkezlerini belirleyebiliriz. 

`display_anchors` işlevi aşağıda tanımlanmıştır. [**Çapa kutusu merkezi olarak her birim (piksel) ile özellik haritasında (`anchors`) bağlantı kutusu merkezi olarak bağlantı kutuları (`anchors`) oluştururuz.**] Çapa kutularındaki $(x, y)$ eksen koordinat değerleri (`anchors`) özellik eşleminin genişlik ve yüksekliğine bölünmüş olduğundan (`fmap`), bu değerler 0 arasındadır ve 1, hangi özellik haritasındaki bağlantı kutularının göreli konumlarını belirtir. 

Bağlantı kutularının merkezleri (`anchors`) özellik haritasındaki tüm birimlere (`fmap`) yayıldığından, bu merkezler göreceli uzamsal konumları bakımından herhangi bir girdi görüntüsüne eşit şekilde dağıtılmalıdır. Daha somut olarak, sırasıyla `fmap_w` ve `fmap_h` özellik haritasının genişliği ve yüksekliği göz önüne alındığında, aşağıdaki işlev, herhangi bir giriş görüntüsünde `fmap_h` satır ve `fmap_w` sütunlarında*homojen* örnek pikseller olacaktır. Bu eşit örneklenen pikseller üzerinde ortalanan `s` ölçek bağlantı kutuları (`s` listenin uzunluğunun 1 olduğu varsayılarak) ve farklı en boy oranları (`ratios`) oluşturulur.

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
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
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

İlk olarak, [**küçük nesnelerin algılanmasını düşünün**]. Görüntülendiğinde ayırt edilmeyi kolaylaştırmak için, burada farklı merkezlere sahip çapa kutuları çakışmaz: çapa kutusu ölçeği 0,15 olarak ayarlanır ve özellik eşleminin yüksekliği ve genişliği 4'e ayarlanır. Ankraj kutularının merkezlerinin 4 sıra ve görüntü üzerindeki 4 sütun eşit olarak dağıtıldığını görebiliriz.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

[**özellik eşlemesinin yüksekliğini ve genişliğini yarı yarıya indirir ve daha büyük nesneleri algılamak için daha büyük çapa kutuları kullanır**] seçeneğine geçiyoruz. Ölçek 0,4 olarak ayarlandığında, bazı bağlantı kutuları birbirleriyle çakışır.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Son olarak, [**özellik haritasının yüksekliğini ve genişliğini yarıya indirir ve çapa kutusu ölçeğini 0,8 seviyesine çıkarırız**]. Şimdi çapa kutusunun merkezi görüntünün merkezidir.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Çoklu Ölçü Algılama

Çok ölçekli çapa kutuları oluşturduğumuzdan, bunları farklı ölçeklerde çeşitli boyutlardaki nesneleri algılamak için kullanacağız. Aşağıda, :numref:`sec_ssd`'te uygulayacağımız CNN tabanlı çok ölçekli nesne algılama yöntemini tanıtıyoruz. 

Bazı ölçekte, $c$ şekil $h \times w$ özellik haritaları olduğunu söylüyorlar. :numref:`subsec_multiscale-anchor-boxes`'teki yöntemi kullanarak, $hw$ çapa kutuları setleri oluşturuyoruz, burada her setin aynı merkeze sahip $a$ çapa kutuları vardır. Örneğin, :numref:`subsec_multiscale-anchor-boxes`'teki deneylerde ilk ölçekte, on (kanal sayısı) $4 \times 4$ özellik haritaları verildiğinde, her setin aynı merkeze sahip 3 çapa kutusu içeren 16 adet çapa kutusu oluşturduk. Daha sonra, her çapa kutusu sınıfla etiketlenir ve zemin gerçeği sınırlayıcı kutulara göre ofset edilir. Geçerli ölçekte, nesne algılama modelinin, farklı kümelerin farklı merkezlere sahip olduğu giriş görüntüsündeki $hw$ bağlantı kutularının sınıflarını ve uzaklıklarını tahmin etmesi gerekir. 

Burada $c$ özellik eşlemeleri giriş görüntüsüne dayalı CNN ileri yayılımı tarafından elde edilen ara çıkışlar olduğunu varsayalım. Her özellik haritasında $hw$ farklı uzamsal pozisyon bulunduğundan, aynı uzamsal konum $c$ üniteye sahip olduğu düşünülebilir. :numref:`sec_conv_layer`'teki alıcı alanının tanımına göre, özellik haritalarının aynı uzamsal konumundaki bu $c$ birimleri girdi görüntüsünde aynı alıcı alana sahiptir: aynı alıcı alanındaki giriş görüntüsü bilgilerini temsil ederler. Bu nedenle, özellik haritalarının $c$ birimlerini aynı uzamsal konumdaki bu uzamsal konum kullanılarak oluşturulan $a$ çapa kutularının sınıflarına ve uzaklıklarına dönüştürebiliriz. Özünde, giriş görüntüsündeki alıcı alana yakın olan bağlantı kutularının sınıflarını ve uzaklıklarını tahmin etmek için belirli bir alıcı alandaki giriş görüntüsünün bilgilerini kullanırız. 

Farklı katmanlardaki özellik eşlemeleri, girdi görüntüsünde farklı boyutlarda alıcı alanlara sahip olduğunda, farklı boyutlardaki nesneleri algılamak için kullanılabilirler. Örneğin, çıktı katmanına daha yakın özellik eşlemeleri birimlerinin daha geniş alıcı alanlara sahip olduğu, böylece girdi görüntüsünden daha büyük nesneleri algılayabilecekleri bir sinir ağı tasarlayabiliriz. 

Özetle, çok ölçekli nesne algılaması için derin sinir ağları aracılığıyla görüntülerin katmanlı temsillerini birden çok düzeyde kullanabiliriz. Bunun :numref:`sec_ssd`'te somut bir örnekle nasıl çalıştığını göstereceğiz. 

## Özet

* Birden çok ölçekte, farklı boyutlarda nesneleri algılamak için farklı boyutlarda çapa kutuları üretebiliriz.
* Özellik haritalarının şeklini tanımlayarak, herhangi bir görüntüdeki düzgün örneklenmiş çapa kutularının merkezlerini belirleyebiliriz.
* Giriş görüntüsünde o alıcı alana yakın olan bağlantı kutularının sınıflarını ve uzaklıklarını tahmin etmek için belirli bir alıcı alandaki giriş görüntüsünün bilgilerini kullanırız.
* Derin öğrenme sayesinde, çok ölçekli nesne algılama için görüntülerin katmanlı temsillerini birden çok düzeyde kullanabiliriz.

## Egzersizler

1. :numref:`sec_alexnet`'teki tartışmalarımıza göre, derin sinir ağları görüntüler için soyutlama düzeylerini artırarak hiyerarşik özellikleri öğreniyor. Çok ölçekli nesne algılamada, farklı ölçeklerdeki özellik haritaları farklı soyutlama düzeylerine karşılık geliyor mu? Neden ya da neden olmasın?
1. :numref:`subsec_multiscale-anchor-boxes`'teki deneylerde ilk ölçekte (`fmap_w=4, fmap_h=4`), çakışabilecek düzgün dağıtılmış çapa kutuları oluşturun.
1. $1 \times c \times h \times w$ şeklindeki bir özellik eşleme değişkeni göz önüne alındığında; burada $c$, $h$ ve $w$ sırasıyla özellik eşlemelerinin kanal, yükseklik ve genişliklerinin sayısı yer alır. Bu değişkeni çapa kutularının sınıflarına ve uzaklıklarına nasıl dönüştürebilirsiniz? Çıkışın şekli nedir?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
