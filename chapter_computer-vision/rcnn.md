# Bölge tabanlı CNN'ler (R-CNN'ler)
:label:`sec_rcnn`

:numref:`sec_ssd`'te açıklanan tek atışta çoklu kutu algılamanın yanı sıra, bölge tabanlı CNN'ler veya CNN özniteliklerine (R-CNN) sahip bölgeler de nesne algılama :cite:`Girshick.Donahue.Darrell.ea.2014`'ya derin öğrenmeyi uygulamanın öncü yaklaşımları arasında yer almaktadır. Bu bölümde R-CNN ve onun iyileştirilmiş serilerini tanıtacağız: Hızlı R-CNN :cite:`Girshick.2015`, daha hızlı R-CNN :cite:`Ren.He.Girshick.ea.2015` ve maske R-CNN :cite:`He.Gkioxari.Dollar.ea.2017`. Sınırlı alan nedeniyle, sadece bu modellerin tasarımına odaklanacağız. 

## R-CNN'ler

*R-CNN* ilk olarak girdi imgesinden birçok (örneğin, 2000) *bölge önerisi* ayıklar (örneğin, çapa kutuları bölge önerileri olarak da kabul edilebilir), sınıflarını ve kuşatan kutuları (örneğin, ofsetler) etiketler :cite:`Girshick.Donahue.Darrell.ea.2014`. Sonra bir CNN, her bölge önerisi üzerinde ileri yaymayı gerçekleştirmek için öznitelikleri kullanır. Sonrasında, her bölge önerisinin öznitelikleri, bu bölge önerisinin sınıfını ve kuşatan kutusunu tahmin etmek için kullanılır. 

![R-CNN modeli.](../img/r-cnn.svg)
:label:`fig_r-cnn`

:numref:`fig_r-cnn` R-CNN modelini gösterir. Daha somut olarak, R-CNN aşağıdaki dört adımdan oluşur: 

1. :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013` girdi imgesinden birden fazla yüksek kaliteli bölge önerisi ayıklamak için *seçici arama* gerçekleştirin. Önerilen bu bölgeler genellikle farklı şekil ve boyutlarda çoklu ölçeklerde seçilir. Her bölge önerisi bir sınıf ve bir gerçek referans değeri kuşatan kutu ile etiketlenecektir.
1. Önceden eğitilmiş bir CNN seçin ve çıktı katmanından önce budayın. Her bölge önerisini ağın gerektirdiği girdi boyutuna yeniden boyutlandırın ve bölge önerisi için ayıklanan öznitelikleri ileri yayma yoluyla çıktılayın. 
1. Her bölge önerisinin çıkarılan özniteliklerini ve etiketlenmiş sınıfını örnek olarak ele alın. Her destek vektör makinesinin, örneğin belirli bir sınıfı içerip içermediğini ayrı ayrı belirlediği nesneleri sınıflandırmak için birden fazla destek vektör makinesini eğitin.
1. Örnek olarak her bölge önerisinin ayıklanan öznitelikleri ve etiketli kuşatan kutusunu alın. Gerçek referans değer kuşatan kutuyu tahmin etmek için doğrusal bağlanım modelini eğitin.

R-CNN modeli imge özniteliklerini etkili bir şekilde ayıklamak için önceden eğitilmiş CNN'ler kullansa da yavaştır. Tek bir girdi imgesinden binlerce bölge önerisini seçtiğimizi düşünün: Bu nesne algılamayı gerçekleştirmek için binlerce CNN ileri yaymasını gerektirir. Bu devasa bilgi işlem yükünden dolayı, R-CNN'lerin gerçek dünyadaki uygulamalarda yaygın olarak kullanılmasını mümkün değildir. 

## Hızlı R-CNN

Bir R-CNN'nin ana performans darboğazı, her bölge önerisi için hesaplamayı paylaşmadan bağımsız CNN ileri yaymasında yatmaktadır. Bu bölgelerde genellikle çakışmalar olduğundan, bağımsız öznitelik ayıklamaları çok fazla tekrarlanan hesaplamaya yol açar. *Hızlı R-CNN*'nin R-CNN'den sağladığı en önemli iyileştirmelerden biri, CNN ileri yaymasının yalnızca tüm imge üzerinde :cite:`Girshick.2015` gerçekleştirilmesidir.

![Hızlı R-CNN modeli.](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`

:numref:`fig_fast_r-cnn` hızlı R-CNN modelini açıklar. Başlıca hesaplamaları şöyledir: 

1. R-CNN ile karşılaştırıldığında, hızlı R-CNN'de, öznitelik ayıklama için CNN'nin girdisi, bireysel bölge önerilerinden ziyade tüm imgedir. Dahası, bu CNN eğitilebilir. Bir girdi imgesi göz önüne alındığında, CNN çıktısının şekli $1 \times c \times h_1  \times w_1$ olsun.
1. Seçici aramanın $n$ bölge önerileri oluşturduğunu varsayalım. Bu bölge önerileri (farklı şekillerde) CNN çıktısındaki ilgi alanlarını (farklı şekillerdeki) işaretler. Daha sonra kolayca birleştirilebilmesi için ilgili bu bölgelerin aynı şekle sahip öznitelikleri ayıklanır (yükseklik $h_2$ ve genişlik $w_2$ diye belirtilir). Bunu başarmak için hızlı R-CNN, *ilgili bölge alanı (RoI) ortaklama* katmanı sunar: CNN çıktısı ve bölge önerileri bu katmana girilir ve tüm bölge önerileri için daha da ayıklanmış $n \times c \times h_2 \times w_2$ şekilli bitiştirilmiş öznitelikleri ortaya çıkarır.
1. Tam bağlı bir katman kullanarak, bitiştirilmiş öznitelikleri $n \times d$'ün model tasarımına bağlı olduğu $n \times d$ şeklindeki bir çıktıya dönüştürün.
1. $n$ bölge önerilerinin her biri için sınıfı ve kuşatan kutuyu tahmin edin. Daha somut olarak, sınıf ve kuşatan kutu tahmininde, tam bağlı katman çıktısını $n \times q$ şeklinde ($q$ sınıf sayısıdır) ve $n \times 4$ şeklinde çıktıya dönüştürün. Sınıf tahmini softmaks bağlanım kullanır.

Hızlı R-CNN'de önerilen ilgi havuzlama katmanı bölgesi :numref:`sec_pooling`'te tanıtılan havuzlama katmanından farklıdır. Havuzlama katmanında, havuzlama penceresinin, dolgunun ve adımın boyutlarını belirterek çıkış şeklini dolaylı olarak kontrol ediyoruz. Buna karşılık, doğrudan ilgi havuzlama katmanı bölgesinde çıkış şeklini belirtebilirsiniz. 

Örneğin, her bölge için çıkış yüksekliğini ve genişliğini sırasıyla $h_2$ ve $w_2$ olarak belirtelim. $h \times w$ şeklindeki herhangi bir ilgi alanı penceresi için, bu pencere, her alt pencerenin şekli yaklaşık $(h/h_2) \times (w/w_2)$ olduğu $h_2 \times w_2$ bir alt pencere ızgarasına ayrılmıştır. Pratikte, herhangi bir alt pencerenin yüksekliği ve genişliği yuvarlanır ve en büyük eleman alt pencerenin çıkışı olarak kullanılacaktır. Bu nedenle, ilgi alanları farklı şekillere sahip olsa bile ilgi alanı havuzlama katmanı aynı şekle sahip özelliklerini ayıklayabilirsiniz. 

Açıklayıcı bir örnek olarak, :numref:`fig_roi`'te, $4 \times 4$ girişinde sol üst $3\times 3$ ilgi alanı seçilir. Bu ilgi alanı için $2\times 2$ çıktısı elde etmek için bir $2\times 2$ ilgi alanı havuzlama katmanı kullanıyoruz. Dört bölünmüş alt pencerenin her birinin 0, 1, 4 ve 5 (5 maksimumdur) öğelerini içerdiğini unutmayın; 2 ve 6 (6 maksimumdur); 8 ve 9 (maksimum 9); ve 10. 

![A $2\times 2$ region of interest pooling layer.](../img/roi.svg)
:label:`fig_roi`

Aşağıda ilgi havuzu katmanı bölgenin hesaplanmasını göstermektedir. CNN çıkarılan özelliklerin `X`'ün yüksekliği ve genişliğinin her ikisi de 4 olduğunu ve yalnızca tek bir kanal olduğunu varsayalım.

```{.python .input}
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

Giriş görüntüsünün yüksekliğinin ve genişliğinin hem 40 piksel olduğunu ve seçici aramanın bu görüntüde iki bölge önerisi oluşturduğunu varsayalım. Her bölge önerisi beş öğe olarak ifade edilir: nesne sınıfı, ardından sol üst ve sağ alt köşelerinin $(x, y)$-koordinatlarını takip eder.

```{.python .input}
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

`X`'in yüksekliği ve genişliği, giriş görüntüsünün yüksekliğinin ve genişliğinin $1/10$ olduğu için, iki bölge teklifinin koordinatları belirtilen `spatial_scale` bağımsız değişkenine göre 0.1 ile çarpılır. Daha sonra iki ilgi bölgesi `X` üzerinde sırasıyla `X[:, :, 0:3, 0:3]` ve `X[:, :, 1:4, 0:4]` olarak işaretlenmiştir. Son olarak $2\times 2$ ilgi havuzunun bulunduğu bölgede, her bir ilgi alanı, aynı şeklin $2\times 2$ özelliklerini daha fazla ayıklamak için bir alt pencere ızgarasına ayrılmıştır.

```{.python .input}
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

## Daha hızlı R-CNN

Nesne algılamada daha doğru olması için hızlı R-CNN modeli genellikle seçici aramada çok sayıda bölge teklifi üretmelidir. Doğruluk kaybı olmadan bölge tekliflerini azaltmak için*daha hızlı R-CNN* seçici aramayı *bölge önerisi ağı* :cite:`Ren.He.Girshick.ea.2015` ile değiştirmeyi önermektedir. 

![The faster R-CNN model.](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`

:numref:`fig_faster_r-cnn` daha hızlı R-CNN modelini gösterir. Hızlı R-CNN ile karşılaştırıldığında, daha hızlı R-CNN yalnızca bölge önerisi yöntemini seçici aramadan bölge önerisi ağına değiştirir. Modelin geri kalanı değişmeden kalır. Bölge önerisi ağı aşağıdaki adımlarda çalışır: 

1. CNN çıkışını $c$ kanal ile yeni bir çıkışa dönüştürmek için 1 dolgulu $3\times 3$ konvolüsyonel katman kullanın. Bu şekilde, CNN çıkarılan özellik haritalarının uzamsal boyutları boyunca her birim $c$ uzunluğunda yeni bir özellik vektörü alır.
1. Özellik eşlemelerinin her pikselinde ortalanan, farklı ölçeklerde ve en boy oranlarında birden çok bağlantı kutusu oluşturur ve bunları etiketleyin.
1. Her bir çapa kutusunun ortasındaki uzunluk-$c$ özellik vektörünü kullanarak, bu çapa kutusu için ikili sınıfı (arka plan veya nesneler) ve sınırlayıcı kutuyu tahmin edin.
1. Tahmin edilen sınıfları nesneler olan tahmin edilen sınırlayıcı kutuları düşünün. Maksimum olmayan bastırmayı kullanarak çakışan sonuçları kaldırın. Nesneler için kalan öngörülen sınırlama kutuları, ilgi alanı havuzu katmanı bölgesinin gerektirdiği bölge önerileridir.

Daha hızlı R-CNN modelinin bir parçası olarak, bölge önerisi ağının modelin geri kalanı ile birlikte eğitildiğini belirtmek gerekir. Başka bir deyişle, daha hızlı R-CNN'nin objektif işlevi yalnızca nesne algılamada sınıf ve sınırlayıcı kutu tahminini değil, aynı zamanda bölge önerisi ağındaki bağlantı kutularının ikili sınıfı ve sınırlayıcı kutu tahminini de içerir. Uçtan uca eğitim sonucunda bölge önerisi ağı, verilerden öğrenilen daha az sayıda bölge teklifiyle nesne algılamada doğru kalabilmek için yüksek kaliteli bölge tekliflerinin nasıl üretileceğini öğrenir. 

## Maske R-CNN

Eğitim veri kümesinde, nesnenin piksel düzeyindeki konumları görüntülerde de etiketlenmişse, *mask R-CNN*, nesne algılama :cite:`He.Gkioxari.Dollar.ea.2017`'ün doğruluğunu daha da geliştirmek için bu tür ayrıntılı etiketleri etkili bir şekilde kullanabilir. 

![The mask R-CNN model.](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

:numref:`fig_mask_r-cnn`'te gösterildiği gibi, R-CNN maskesi R-CNN daha hızlı R-CNN'e göre değiştirilir. Özellikle, maske R-CNN ilgi alanı havuzu katmanı ile değiştirir
*ilgi alanı (RoI) hizalama* katmanı. 
Bu ilgi alanı hizalama katmanı, özellik eşlemelerindeki uzamsal bilgileri korumak için ikili enterpolasyon kullanır; bu da piksel düzeyinde tahmin için daha uygundur. Bu katmanın çıktısı, ilgi çeken tüm bölgeler için aynı şekildeki özellik haritalarını içerir. Bunlar, yalnızca her bir ilgi alanı için sınıf ve sınırlayıcı kutuyu değil, aynı zamanda ek bir tamamen evrimsel ağ aracılığıyla nesnenin piksel düzeyinde konumunu da tahmin etmek için kullanılırlar. Bir görüntünün piksel düzeyinde semantiğini tahmin etmek için tamamen evrimsel bir ağ kullanma hakkında daha fazla ayrıntı, bu bölümün sonraki bölümlerinde sağlanacaktır. 

## Özet

* R-CNN, girdi görüntüsünden birçok bölge teklifini çıkarır, özelliklerini ayıklamak için her bölge önerisi üzerinde ileriye yayılmasını gerçekleştirmek için bir CNN kullanır, ardından bu özellikleri bu bölge teklifinin sınıfını ve sınırlayıcı kutuyu tahmin etmek için kullanır.
* R-CNN hızlı R-CNN önemli gelişmelerden biri CNN ileri yayılma sadece tüm görüntü üzerinde gerçekleştirilir olmasıdır. Aynı zamanda ilgi havuzu katmanı bölgesini tanıtır, böylece aynı şekle sahip özellikler daha farklı şekillere sahip ilgi alanları için çıkartılabilir.
* Daha hızlı R-CNN, hızlı R-CNN'de kullanılan seçici aramanın, ortak eğitimli bir bölge önerisi ağı ile değiştirir, böylece birincinin nesne algılamasında daha az sayıda bölge teklifi ile doğru kalabilmesi için.
* Daha hızlı R-CNN'e dayanan maske R-CNN, nesne algılamasının doğruluğunu daha da artırmak için piksel düzeyinde etiketlerden yararlanmak için ek olarak tamamen evrimsel bir ağ sunar.

## Egzersizler

1. Nesne algılamasını, sınırlayıcı kutuları ve sınıf olasılıklarını tahmin etme gibi tek bir regresyon sorunu olarak çerçeveleyebilir miyiz? YOLO model :cite:`Redmon.Divvala.Girshick.ea.2016` tasarımına başvurabilirsiniz.
1. Tek çekim çoklu kutu algılamasını bu bölümde tanıtılan yöntemlerle karşılaştırın. Onların büyük farklılıkları nelerdir? Şekil 2'den :cite:`Zhao.Zheng.Xu.ea.2019`'e başvurabilirsiniz.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/374)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1409)
:end_tab:
