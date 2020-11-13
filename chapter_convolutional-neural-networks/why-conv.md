# Tam Bağlı Katmanlardan Konvolutions
:label:`sec_why-conv`

Bugüne kadar, şimdiye kadar tartıştığımız modeller, tablo verileriyle uğraşırken uygun seçenekler olmaya devam ediyor. Tablo olarak, verilerin, özelliklere karşılık gelen örneklere ve sütunlara karşılık gelen satırlardan oluştuğunu kastediyoruz. Tabla şeklinde verilerle, aradığımız kalıpların özellikler arasında etkileşimler içerebileceğini tahmin edebiliriz, ancak özelliklerin nasıl etkileşime girdiğine ilişkin herhangi bir yapı*öncelikli* varsayamayız.

Bazen, zanaatkar mimarilerin yapımına rehberlik etmek için gerçekten bilgi eksikliğimiz vardır. Bu gibi durumlarda, bir MLP yapabileceğimizin en iyisi olabilir. Bununla birlikte, yüksek boyutlu algısal veriler için, bu tür yapısız ağlar kullanışsız hale gelebilir.

Örneğin, kedileri köpeklerden ayırma örneğimize dönelim. Veri toplamada kapsamlı bir iş yaptığımızı, tek megapiksel fotoğrafların açıklamalı bir veri kümesini topladığımızı söyleyin. Bu, ağa her girişin bir milyon boyuta sahip olduğu anlamına gelir. Bin gizli boyutta agresif bir azalma bile $10^6 \times 10^3 = 10^9$ parametreleri ile karakterize tam bağlı bir katman gerektirir. Çok sayıda GPU'umuz, dağıtılmış optimizasyon için bir yeteneğimiz ve olağanüstü bir sabır olmadıkça, bu ağın parametrelerini öğrenmek mümkün olmayabilir.

Dikkatli bir okuyucu, bir megapiksel çözünürlüğün gerekli olmayabileceği temelinde bu argümana itiraz edebilir. Bununla birlikte, yüz bin pikselden paçayı kurtarabilsek de, 1000 büyüklüğündeki gizli katmanımız, görüntülerin iyi temsillerini öğrenmek için gereken gizli birimlerin sayısını büyük ölçüde hafife alıyor, bu nedenle pratik bir sistem hala milyarlarca parametre gerektirecektir. Dahası, bir sınıflandırıcıyı bu kadar çok parametre uyarak öğrenmek muazzam bir veri kümesini toplamayı gerektirebilir. Ve bugün hem insanlar hem de bilgisayarlar kedileri köpeklerden oldukça iyi ayırt edebiliyor, görünüşte bu sezgilerle çelişiyor. Bunun nedeni, görüntülerin insanlar ve makine öğrenimi modelleri tarafından istismar edilebilecek zengin bir yapı sergilemesidir. Konvolüsyonel sinir ağları (CNN), makine öğreniminin doğal görüntülerdeki bilinen yapıların bazılarını kullanmak için benimsediği yaratıcı bir yoldur.

## Değişmezlik

Görüntüdeki bir nesneyi algılamak istediğinizi düşünün. Nesneleri tanımak için kullandığımız yöntemin görüntüdeki nesnenin kesin konumu ile aşırı derecede ilgili olmaması mantıklı görünüyor. İdeal olarak, sistemimiz bu bilgiyi kullanmalıdır. Domuzlar genellikle uçmaz ve uçaklar genellikle yüzmez. Yine de, görüntünün en üstünde görünen bir domuzun olduğunu fark etmeliyiz. Burada “Waldo Nerede” çocuk oyunundan biraz ilham alabiliriz (:numref:`img_waldo`'te tasvir edilmiştir). Oyun faaliyetleri ile patlama kaotik sahneleri bir dizi oluşur. Waldo her birinde bir yerde ortaya çıkıyor, tipik olarak alışılmadık bir yerde gizleniyor. Okuyucunun amacı onu bulmak. Karakteristik kıyafetine rağmen, dikkat dağıtıcı çok sayıda nedeniyle şaşırtıcı derecede zor olabilir. Ancak, *Waldo'un neye benzediği*, Waldo'un nerede bulunduğuna bağlı değildir. Görüntüyü her yama için bir puan atayabilen bir Waldo dedektörü ile süpürebiliriz ve bu da yamanın Waldo içerme olasılığını gösterir. CNN'ler bu fikrini sistematize ederek, daha az parametre ile yararlı gösterimleri öğrenmek için kullanırlar.

![An image of the "Where's Waldo" game.](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`

Bilgisayar görüşü için uygun bir sinir ağı mimarisi tasarımımızı yönlendirmek için birkaç desiderata numaralandırarak bu sezgileri daha somut hale getirebiliriz:

1. En erken katmanlarda, ağımız görüntüde nerede göründüğüne bakılmaksızın aynı yamaya benzer şekilde yanıt vermelidir. Bu ilke*çeviri değişmezi* olarak adlandırılır.
1. Ağın en erken katmanları, uzak bölgelerdeki görüntünün içeriğine bakılmaksızın yerel bölgelere odaklanmalıdır. Bu, *yerlik* ilkesidir. Sonuç olarak, bu yerel temsiller tüm görüntü düzeyinde tahminler yapmak için toplanabilir.

Bunun nasıl matematiğe dönüştüğünü görelim.

## MLP'yi Kısıtlama

Başlangıç olarak, iki boyutlu görüntüler $\mathbf{X}$ ve $\mathbf{H}$ benzer şekilde matematikte matrisler olarak ve kodda iki boyutlu tensörler olarak temsil edilen $\mathbf{X}$ ve $\mathbf{H}$ aynı şekle sahip olduğu hemen gizli gösterimleri olarak bir MLP düşünebiliriz. Bırak içeri batsın. Şimdi sadece girdileri değil, aynı zamanda mekânsal yapıya sahip olarak gizli temsilleri de tasavvur ediyoruz.

$[\mathbf{X}]_{i, j}$ ve $[\mathbf{H}]_{i, j}$, sırasıyla giriş görüntüsünde ve gizli gösterimde pikseli ($i$, $j$) göstersin. Sonuç olarak, gizli birimlerin her birinin giriş piksellerinin her birinden giriş almasını sağlamak için ağırlık matrislerini kullanmaktan (daha önce MLP'lerde yaptığımız gibi) parametrelerimizi dördüncü dereceden ağırlık tensörleri $\mathsf{W}$ olarak temsil etmeye geçeceğiz. $\mathbf{U}$'nın önyargılar içerdiğini varsayalım, tam bağlı katmanı resmen olarak ifade edebiliriz.

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned},$$

$\mathsf{W}$'ten 73229363619'a kadar olan geçiş, her iki dördüncü mertebeden katsayılar arasında bire bir yazışma olduğundan, şimdilik tamamen kozmetik. Biz sadece $(k, l)$ aboneliklerini yeniden dizin $k = i+a$ ve $l = j+b$. Başka bir deyişle, $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$'i ayarladık. $a$ ve $b$ endeksleri, tüm görüntüyü kapsayan hem pozitif hem de negatif ofsetlerin üzerinden çalışır. $[\mathbf{H}]_{i, j}$ gizli temsil $[\mathbf{H}]_{i, j}$ herhangi bir konum ($i$, $j$) için, $x$, $(i, j)$ civarında ortalanmış ve $[\mathsf{V}]_{i, j, a, b}$ ağırlığında pikseller üzerinden toplanarak değerini hesaplarız.

### Çeviri Değişmezliği

Şimdi yukarıda kurulan ilk ilkeyi çağıralım: çeviri değişmezliği. Bu, $\mathbf{X}$ girişindeki bir kaymanın, gizli gösterimde $\mathbf{H}$'de bir kaymaya yol açması gerektiği anlamına gelir. Bu sadece $\mathsf{V}$ ve $\mathbf{U}$ aslında $(i, j)$'a bağımlı değilse mümkündür, yani $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ ve $\mathbf{U}$ bir sabit, diyelim ki $u$. Sonuç olarak, $\mathbf{H}$ tanımını basitleştirebiliriz:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

Bu bir konvolüsyon*! $[\mathbf{H}]_{i, j}$ değerini elde etmek için $(i, j)$'nın $[\mathbf{V}]_{a, b}$ katsayıları ile $(i+a, j+b)$ adresindeki pikselleri etkin bir şekilde ağırlaştırıyoruz. $[\mathbf{V}]_{a, b}$'ün artık görüntü içindeki konuma bağlı olmadığından $[\mathsf{V}]_{i, j, a, b}$'den çok daha az katsayıya ihtiyaç duyduğunu unutmayın. Önemli bir ilerleme kaydettik!

###  mahal

Şimdi ikinci prensibi çağıralım: yerellik. Yukarıda motive edildiği gibi, $[\mathbf{H}]_{i, j}$'te neler olup bittiğini değerlendirmek için ilgili bilgileri toplamak için $(i, j)$ konumundan çok uzaklara bakmak zorunda olmamamız gerektiğine inanıyoruz. Bu, bazı aralık dışında $|a|> \Delta$ veya $|b| > \Delta$, biz $[\mathbf{V}]_{a, b} = 0$ ayarlamanız gerektiği anlamına gelir. Eşdeğer olarak, $[\mathbf{H}]_{i, j}$ olarak yeniden yazabiliriz

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Özetle, :eqref:`eq_conv-layer`'ün bir *kıvrımsal katman* olduğuna dikkat edin.
*Konvolüsyonel sinir ağları* (CNN)
, evrimsel katmanlar içeren özel bir sinir ağları ailesidir. Derin öğrenme araştırma topluluğunda $\mathbf{V}$, bir *evrişim çekirdeği*, bir *filtre* veya genellikle öğrenilebilir parametreler olan katmanın *ağırlıkları* olarak adlandırılır. Yerel bölge küçük olduğunda, tam bağlı bir ağ ile karşılaştırıldığında fark dramatik olabilir. Daha önce, bir görüntü işleme ağındaki tek bir katmanı temsil etmek için milyarlarca parametre gerekebilirdik, şimdi genellikle girdilerin veya gizli temsillerin boyutsallığını değiştirmeden sadece birkaç yüze ihtiyacımız var. Parametrelerdeki bu ciddi azalma için ödenen fiyat, özelliklerimizin artık çeviri değişmez olması ve katmanımızın her gizli aktivasyonun değerini belirlerken yalnızca yerel bilgileri içerebilmesidir. Tüm öğrenme endüktif önyargı empoze bağlıdır. Bu önyargı gerçekle aynı fikirde olduğunda, görünmeyen verilere iyi genelleyen örnek verimli modeller elde ederiz. Ancak elbette, bu önyargılar gerçekle aynı fikirde değilse, örneğin, görüntülerin çeviri değişmez olmadığı ortaya çıktıysa, modellerimiz eğitim verilerimize uyması için bile mücadele edebilir.

## Konvolutions

Daha ileri gitmeden önce, yukarıdaki işlemin neden bir evrim olarak adlandırıldığını kısaca gözden geçirmeliyiz. Matematikte, iki fonksiyon arasındaki *kıvrım*, diyelim ki $f, g: \mathbb{R}^d \to \mathbb{R}$

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Yani, bir işlev “çevrilmiş” ve $\mathbf{x}$ ile kaydırıldığında $f$ ve $g$ arasındaki örtüşmeyi ölçüyoruz. Ayrık nesnelere sahip olduğumuzda, integral bir toplama dönüşür. Örneğin, $\mathbb{Z}$ üzerinde çalışan indeksi olan kare toplanabilir sonsuz boyutlu vektörler kümesindeki vektörler için aşağıdaki tanımı elde ederiz:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

İki boyutlu tensörler için, sırasıyla $f$ için $(a, b)$ ve $(i-a, j-b)$ için $(i-a, j-b)$ endeksleri ile karşılık gelen bir toplama sahibiz:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Bu, :eqref:`eq_conv-layer`'e benzer, büyük bir farkla. Bunun yerine $(i+a, j+b)$ kullanmak yerine farkı kullanıyoruz. Yine de, bu ayrımın çoğunlukla kozmetik olduğunu unutmayın çünkü :eqref:`eq_conv-layer` ve :eqref:`eq_2d-conv-discrete` arasındaki gösterimi her zaman eşleştirebiliriz. :eqref:`eq_conv-layer`'teki orijinal tanımımız, bir *çapraz korelasyonu daha doğru bir şekilde tanımlıyor. Aşağıdaki bölümde buna geri döneceğiz.

## “Waldo Nerede” Revisited

Waldo dedektörümüze dönersek, bunun neye benzediğini görelim. Konvolusyonel tabaka, belirli bir boyuttaki pencereleri seçer ve :numref:`fig_waldo_mask`'te gösterildiği gibi $\mathsf{V}$'e göre yoğunlukları ağırlaştırır. Bir model öğrenmeyi hedefleyebiliriz, böylece “waldoness” en yüksek nerede olursa olsun, gizli katman temsillerinde bir zirve bulmalıyız.

![Detect Waldo.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

### Kanallar
:label:`subsec_why-conv-channels`

Bu yaklaşımla ilgili tek bir sorun var. Şimdiye kadar, görüntülerin 3 kanaldan oluştuğunu görmezden geldik: kırmızı, yeşil ve mavi. Gerçekte, görüntüler iki boyutlu nesneler değil, yükseklik, genişlik ve kanal ile karakterize edilen üçüncü dereceden tensörlerdir, örn. şekil $1024 \times 1024 \times 3$ piksel. Bu eksenlerin ilk ikisi mekansal ilişkileri ilgilendirirken, üçüncüsü her piksel konumuna çok boyutlu bir temsil atama olarak kabul edilebilir. Biz böylece endeksi $\mathsf{X}$ olarak $[\mathsf{X}]_{i, j, k}$. Konvolüsyonel filtre buna göre adapte olmak zorundadır. $[\mathbf{V}]_{a,b}$ yerine artık $[\mathsf{V}]_{a,b,c}$ var.

Dahası, girdimiz üçüncü mertebeden bir tensörden oluştuğunda, gizli temsillerimizi üçüncü mertebeden tensörler $\mathsf{H}$ olarak benzer şekilde formüle etmek iyi bir fikir olduğu ortaya çıkıyor. Başka bir deyişle, her mekansal konuma karşılık gelen tek bir gizli gösterime sahip olmaktan ziyade, her mekansal konuma karşılık gelen gizli temsillerin tüm vektörünü istiyoruz. Gizli temsilleri, birbirinin üzerine yığılmış bir dizi iki boyutlu ızgarayı içeren olarak düşünebiliriz. Girdilerde olduğu gibi, bunlara bazen *kanallar* denir. Bunlara bazen *özellik haritaları* olarak da adlandırılırlar, çünkü her biri sonraki katmana uzamlaştırılmış bir öğrenilen özellik kümesi sağlar. Sezgisel olarak, girişlere daha yakın olan alt katmanlarda, bazı kanalların kenarları tanımak için uzmanlaşabileceğini, diğerlerinin dokuları tanıyabileceğini hayal edebilirsiniz.

Her iki girişte ($\mathsf{X}$) ve gizli temsillerde ($\mathsf{H}$) birden fazla kanalı desteklemek için $\mathsf{V}$:$[\mathsf{V}]_{a, b, c, d}$'ye dördüncü bir koordinat ekleyebiliriz. Sahip olduğumuz her şeyi bir araya getirmek:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

burada $d$ gizli temsillerde çıkış kanalları dizinleri $\mathsf{H}$. Sonraki konvolüsyonel tabaka, giriş olarak üçüncü mertebeden bir tensör olan $\mathsf{H}$ almaya devam edecektir. Daha genel olarak, :eqref:`eq_conv-layer-channels`, birden fazla kanal için bir evrimsel tabakanın tanımıdır; burada $\mathsf{V}$, katmanın bir çekirdeği veya filtresidir.

Halen ele almamız gereken birçok operasyon var. Örneğin, tüm gizli temsilleri tek bir çıktıda nasıl birleştireceğimizi bulmamız gerekiyor, örn. resimde herhangi bir yerde* var mı? Ayrıca işleri verimli bir şekilde nasıl hesaplayacağımıza, birden fazla katmanı nasıl birleştireceğimize, uygun etkinleştirme işlevlerine ve pratikte etkili ağlar oluşturmak için makul tasarım seçimlerinin nasıl yapılacağına karar vermeliyiz. Bölümün geri kalanında bu sorunlara dönüyoruz.

## Özet

* Görüntülerdeki çeviri değişmezliği, bir görüntünün tüm yamalarının aynı şekilde işleneceğini ima eder.
* Yerellik, karşılık gelen gizli temsilleri hesaplamak için yalnızca küçük bir piksel mahallesinin kullanılacağı anlamına gelir.
* Görüntü işlemede, kıvrımsal katmanlar genellikle tam bağlı katmanlardan çok daha az parametre gerektirir.
* CNNS, evrimsel katmanlar içeren özel bir sinir ağları ailesidir.
* Giriş ve çıkıştaki kanallar, modelimizin her mekansal konumda bir görüntünün birden çok yönünü yakalamalarına olanak tanır.

## Egzersizler

1. Evrişim çekirdeğinin boyutu $\Delta = 0$ olduğunu varsayalım. Bu durumda, evrişim çekirdeğinin her kanal kümesi için bağımsız olarak bir MLP uyguladığını gösterin.
1. Çeviri değişmezliği neden iyi bir fikir olmayabilir?
1. Bir görüntünün sınırındaki piksel konumlarına karşılık gelen gizli temsillerin nasıl tedavi edileceğine karar verirken hangi sorunlarla uğraşmalıyız?
1. Ses için benzer bir konvolüsyonel katmanı tanımlayın.
1. Kıvrımsal katmanların metin verileri için de geçerli olabileceğini düşünüyor musunuz? Neden ya da neden olmasın?
1. Bunu kanıtla $f * g = g * f$.

[Discussions](https://discuss.d2l.ai/t/64)
