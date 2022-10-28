# Dikkat İşaretleri
:label:`sec_attention-cues`

Bu kitaba gösterdiğiniz ilgi için teşekkür ederiz. Dikkat kıt bir kaynaktır: Şu anda bu kitabı okuyor ve gerisini görmezden geliyorsunuz. Böylece, paraya benzer şekilde, dikkatiniz bir fırsat maliyeti ile ödeniyor. Şu anda dikkat yatırımınızın değerli olmasını sağlamak için, güzel bir kitap üretmede dikkatimizi itinalı bir şekilde göstermeye son derece motive olduk. Dikkat, hayatın kemerindeki kilit taştır ve herhangi bir işin istisnacılığının anahtarını tutar. 

Ekonomi, kıt kaynakların tahsisini incelediğinden, insanların dikkatinin değiştirilebilecek sınırlı, değerli ve kıt bir hammadde olarak ele alındığı dikkat ekonomisi çağındayız. Yararlanmak için üzerinde çok sayıda iş modeli geliştirilmiştir. Müzik veya video akışı hizmetlerinde ya reklamlarına dikkat ediyoruz ya da bunları gizlemek için para ödüyoruz. Çevrimiçi oyunlar dünyasında büyüme için, ya yeni oyuncular çeken savaşlara katılmaya dikkat ediyoruz ya da anında güçlü olmak için para ödüyoruz. Hiçbir şey bedavaya gelmez. 

Sonuçta, çevremizdeki bilgiler kıt değilken dikkat kıttır. Görsel bir sahneyi incelerken, görsel sinirimiz saniyede $10^8$ bit ölçeğinde bilgi alır ve bu beynimizin tam olarak işleyebileceğini çok aşar. Neyse ki, atalarımız deneyimlerimizden (veri olarak da bilinir) öğrenmişlerdi ki, *tüm duyusal girdiler eşit yaratılmamıştır*. İnsanlık tarihi boyunca, ilgi duyulan bilginin yalnızca bir kısmına dikkati yöneltme yeteneği, beynimizin, avcıları, avları ve arkadaşları tespit etmek gibi, hayatta kalmak, büyümek ve sosyalleşmek için kaynakları daha akıllıca tahsis etmesini sağlamıştır. 

## Biyolojide Dikkat İşaretleri

Görsel dünyada dikkatimizin nasıl dağıtıldığını açıklamak için iki bileşenli bir çerçeve ortaya çıktı ve yaygın oldu. Bu fikir 1890'larda “Amerikan psikolojisinin babası” :cite:`James.2007` olarak kabul edilen William James'e dayanmaktadır. Bu çerçevede, konular seçici olarak dikkatin sahne ışığını hem *istemsiz işaret* hem de *istemli işaret* kullanarak yönlendirir. 

İstemsiz işaret, ortamdaki nesnelerin belirginliğine ve barizliğine dayanır. Önünüzde beş nesne olduğunu düşünün: :numref:`fig_eye-coffee` şeklinde gösterildiği bir gazete, bir araştırma makalesi, bir fincan kahve, bir defter ve bir kitap. Tüm kağıt ürünleri siyah beyaz basılıyken kahve fincanı kırmızıdır. Başka bir deyişle, bu kahve, bu görsel ortamda kendiliğinden belirgin ve bariz, otomatik ve istemsiz dikkat çekiyor. Böylece fovea'yı (görme keskinliğinin en yüksek olduğu makula merkezi) :numref:`fig_eye-coffee` şeklinde gösterildiği gibi kahvenin üzerine getirirsiniz. 

![Belirginliğe dayalı istemsiz işareti kullanarak (kırmızı fincan, kağıt olmayan), dikkat istemeden kahveye yönlendirilir.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

Kahve içtikten sonra kafeinlenmiş olursunuz ve kitap okumak istersiniz. Yani başınızı çevirin, gözlerinizi yeniden odaklayın ve :numref:`fig_eye-book` şeklinde tasvir edildiği gibi kitaba bakın. Kahve, belirginliğe göre seçme konusunda sizi önyargılı yapar, :numref:`fig_eye-coffee` şeklindeki durumdan farklı olarak, bu görev bağımlı durumda bilişsel ve istemli kontrol altında kitabı seçersiniz. Değişken seçim kriterlerine dayalı istemli işaret kullanarak, bu dikkat biçimi daha kasıtlıdır. Ayrıca deneğin gönüllü çabaları ile daha güçlüdür. 

![Göreve bağlı istemli işaret (kitap okumayı istemek) kullanılarak, dikkat istemli kontrol altındaki kitaba yönlendirilir.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`

## Sorgular, Anahtarlar ve Değerler

Dikkatli konuşlandırmayı açıklayan istemsiz ve istemli dikkat işaretlerinden esinlenerek, aşağıda bu iki dikkat işaretini birleştirerek dikkat mekanizmalarını tasarlamak için bir çerçeve anlatacağız. 

Başlangıç olarak, yalnızca istemsiz işaretlerin mevcut olduğu daha basit durumu göz önünde bulundurun. Duyusal girdiler üzerinden seçimi önyargılı hale getirmek için, basitçe parametreleştirilmiş tam bağlı bir katman veya hatta parametrelenmemiş maksimum veya ortalama ortaklama kullanabiliriz.

Bu nedenle, dikkat mekanizmalarını tam bağlı katmanlardan veya ortaklama katmanlarından ayıran şey, istemli işaretlerin dahil edilmesidir. Dikkat mekanizmaları bağlamında, istemli işaretlere *sorgular* olarak atıfta bulunuyoruz. Herhangi bir sorgu göz önüne alındığında, dikkat mekanizmaları duyusal girdiler üzerinde seçimi (örneğin, ara öznitelik temsilleri) *dikkat ortaklama* yoluyla yönlendirir. Bu duyusal girdilere dikkat mekanizmaları bağlamında *değerler* denir. Daha genel olarak, her değer bir *anahtar* ile eşleştirilir ve bu durum, bu duyusal girdinin istemsiz işareti olarak düşünülebilir. :numref:`fig_qkv` içinde gösterildiği gibi, verilen sorgu (istemli işaret), değerler (duyusal girdiler) üzerinde ek girdi seçimini yönlendiren anahtarlarla (istemsiz işaretler) etkileşime girebilmesi için dikkat ortaklamasını tasarlayabiliriz. 

![Dikkat mekanizmaları, sorguları (istemli işaretleri) ve anahtarları (istemsiz işaretleri) içeren dikkat ortaklama aracılığıyla değerler (duyusal girdiler) üzerinden seçimi önyargılı kılar.](../img/qkv.svg)
:label:`fig_qkv`

Dikkat mekanizmalarının tasarımı için birçok alternatif olduğunu unutmayın. Örneğin, :cite:`Mnih.Heess.Graves.ea.2014` pekiştirmeli öğrenme yöntemleri kullanılarak eğitilebilen türevlenemeyen bir dikkat modeli tasarlayabiliriz. :numref:`fig_qkv` şeklindeki çerçevenin hakimiyeti göz önüne alındığında, bu çerçevedeki modeller bu bölümdeki dikkatimizin merkezi olacak. 

## Dikkat Görselleştirme

Ortalama ortaklama ağırlıklarının tekdüze olduğu ağırlıklı bir girdi ortalaması olarak değerlendirilebilir. Pratikte dikkat ortaklama, verilen sorgu ile farklı anahtarlar arasında ağırlıkların hesaplandığı ağırlıklı ortalamayı kullanarak değerleri toplar.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

Dikkat ağırlıklarını görselleştirmek için `show_heatmaps` işlevini tanımlıyoruz. `matrices` girdisi (görüntüleme için satır sayısı, görüntüleme için sütun sayısı, sorgu sayısı, anahtar sayısı) biçimine sahiptir.

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Matrislerin ısı haritalarını gösterir"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

Gösterim için, dikkat ağırlığının yalnızca sorgu ve anahtar aynı olduğunda bir olduğu basit bir durum düşünün; aksi takdirde sıfırdır.

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

Sonraki bölümlerde, dikkat ağırlıklarını görselleştirmek için sıklıkla bu işlevi çağıracağız. 

## Özet

* İnsanın dikkati sınırlı, değerli ve kıt bir kaynaktır.
* Denekler hem istemsiz hem de istemli işaretleri kullanarak dikkati seçici olarak yönlendirir. Birincisi, belirginliğe dayanır ve ikincisi görev bağımlıdır.
* Dikkat mekanizmaları, istemli işaretlerin dahil edilmesi nedeniyle tam bağlı katmanlardan veya ortaklama katmanlarından farklıdır.
* Dikkat mekanizmaları, sorguları (istemli işaretler) ve anahtarları (istemsiz işaretler) içeren dikkat ortaklama aracılığıyla değerler (duyusal girdiler) üzerinden seçimi önyargılı kılar. Anahtarlar ve değerler eşleştirilir.
* Sorgular ve anahtarlar arasındaki dikkat ağırlıklarını görselleştirebiliriz.

## Alıştırmalar

1. Makine çevirisinde bir dizinin kodunu andıç andıç çözerken istemli işaret ne olabilir? İsteğe bağlı olmayan sinyaller ve duyusal girdiler nelerdir?
1. Rastgele bir $10 \times 10$'luk bir matris oluşturun ve her satırın geçerli bir olasılık dağılımı olduğundan emin olmak için softmaks işlemini kullanın. Çıktı dikkat ağırlıklarını görselleştirin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1710)
:end_tab:
