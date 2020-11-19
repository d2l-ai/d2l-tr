# Tam Bağlı Katmanlardan Evrişimlere
:label:`sec_why-conv`

Bugüne kadar, buraya kadar tartıştığımız modeller, çizelge halindeki (tabular) veriyle uğraşırken uygun seçenekler olmaya devam ediyor. Çizelge hali ile, verinin, özniteliklere karşılık gelen örneklerden ve sütunlara karşılık gelen satırlardan oluştuğunu kastediyoruz. Çizelge veriyle, aradığımız desenlerin öznitelikler arasında etkileşimler içerebileceğini tahmin edebiliriz, ancak özniteliklerin nasıl etkileşime girdiğine ilişkin herhangi bir *önsel* yapıyı varsayamayız.

Bazen, zanaatkar (işini bilen) mimarilerin yapımına rehberlik etmek için gerçekten bilgi eksikliğimiz olur. Bu gibi durumlarda, bir MLP yapabileceğimizin en iyisi olabilir. Bununla birlikte, yüksek boyutlu algısal veriler için, bu tür yapı içermeyen ağlar kullanışsız hale gelebilir.

Mesela, kedileri köpeklerden ayırma örneğimize dönelim. Veri toplamada kapsamlı bir iş yaptığımızı, bir megapiksel fotoğraflardan açıklamalı bir veri kümesi topladık diyelim. Bu, ağa giren her girdinin bir milyon boyuta sahip olduğu anlamına gelir. Bin gizli boyutluk saldırgan bir azalma bile $10^6 \times 10^3 = 10^9$ parametre ile karakterize edilen tam bağlı bir katman gerektirir. Çok sayıda GPU'umuz, dağıtılmış eniyileme için bir marifetimiz ve olağanüstü bir sabrımız olmadıkça, bu ağın parametrelerini öğrenmek mümkün olmayabilir.

Dikkatli bir okuyucu, bu argümana bir megapiksel çözünürlüğün gerekli olmayabileceği temelinde itiraz edebilir. Bununla birlikte, yüz bin pikselden paçayı kurtarabilsek de, 1000 büyüklüğündeki gizli katmanımız, imgelerin iyi temsillerini öğrenmek için gereken gizli birimlerin sayısını büyük ölçüde hafife alıyor, bu nedenle pratik bir sistem hala milyarlarca parametre gerektirecektir. Dahası, bir sınıflandırıcıyı bu kadar çok parametre oturtarak öğrenmek muazzam bir veri kümesini toplamayı gerektirebilir. Ayrıca, bugün hem insanlar hem de bilgisayarlar kedileri köpeklerden oldukça iyi ayırt edebiliyor, bu da görünüşte bu sezgilerle çelişiyor. Bunun nedeni, imgelerin insanlar ve makine öğrenmesi modelleri tarafından istismar edilebilecek zengin bir yapı sergilemesidir. Evrişimli sinir ağları (CNN), makine öğrenmesinin doğal görüntülerdeki bilinen yapıların bazılarını kullanmak için benimsediği yaratıcı bir yoldur.

## Değişmezlik

İmgedeki bir nesneyi tespit etmek istediğinizi düşünün. Nesneleri tanımak için kullandığımız yöntemin görüntüdeki nesnenin kesin konumu ile aşırı derecede ilgili olmaması akla yatkın görünüyor. İdeal olarak, sistemimiz bu bilgiyi kullanmalıdır. Domuzlar genellikle uçmaz ve uçaklar genellikle yüzmez. Yine de, imgenin en üstünde görünen bir domuzu gene de fark etmeliyiz. Burada “Waldo Nerede” isimli çocuk oyunundan biraz ilham alabiliriz (:numref:`img_waldo`'te tasvir edilmiştir). Oyun kaotik sahneler ile dolu bir dizi faaliyetten oluşur. Waldo her birinde bir yerlerden ortaya çıkıyor, tipik olarakta alışılmadık bir yerde gizleniyor. Okuyucunun amacı onu bulmaktır. Karakteristik kıyafetine rağmen, dikkat dağıtıcı çok sayıda unsur nedeniyle şaşırtıcı derecede zor olabilir. Ancak, *Waldo'nun neye benzediği*, Waldo'nun nerede bulunduğuna bağlı değildir. İmgeyi her yama için bir puan atayabilen bir Waldo dedektörü ile tarayabiliriz ve bu da yamanın Waldo içerme olabilirliliğini gösterir. CNN'ler bu *konumsal değişmezlik* fikrini sistemleştirerek, daha az parametre ile yararlı gösterimleri öğrenmek için kullanırlar.

!["Waldo Nerede" oyundan bir resim.](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`

Bilgisayarla görmede uygun bir sinir ağı mimarisi tasarımımıza rehberlik etmesi için birkaç arzulanan şeyi numaralandırarak bu sezgileri daha somut hale getirebiliriz:

1. En erken katmanlarda, ağımız görüntüde nerede göründüğüne bakılmaksızın aynı yamaya benzer şekilde yanıt vermelidir. Bu ilke *çeviri değişmezliği* olarak adlandırılır.
1. Ağın en erken katmanları, uzak bölgelerdeki görüntünün içeriğine bakılmaksızın yerel bölgelere odaklanmalıdır. Bu, *yerellik* ilkesidir. Sonuç olarak, bu yerel temsiller tüm görüntü düzeyinde tahminler yapmak için toplanabilir.

Bunun matematik ile nasıl ifade edildiğini görelim.

## MLP'yi Kısıtlama

Başlangıç olarak, iki boyutlu $\mathbf{X}$ imgeler ile ve onların doğrudan saklı $\mathbf{H}$ gösterimlerinin matematikte benzer ebatta matrisler ve kodda iki boyutlu tensörler olarak temsil edildiği bir MLP düşünebiliriz. Bu aklımızda yeretsin. Şimdi sadece girdileri değil, aynı zamanda gizli temsilleri de konumsal yapıya sahip olarak tasavvur ediyoruz.

$[\mathbf{X}]_{i, j}$ ve $[\mathbf{H}]_{i, j}$'nin, sırasıyla girdi imgesinde ve gizli gösterimde ($i$, $j$) pikselini gösterdiğini varsayalım. Sonuç olarak, gizli birimlerin her birinin girdi piksellerinin her birinden girdi almasını sağlamak için ağırlık matrislerini kullanmaktan (daha önce MLP'lerde yaptığımız gibi) parametrelerimizi dördüncü dereceden ağırlık tensörleri $\mathsf{W}$ olarak temsil etmeye geçeceğiz. $\mathbf{U}$'nın ek girdiler içerdiğini varsayalım, tam bağlı katmanı biçimsel olarak şöyle ifade edebiliriz:

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned},$$

$\mathsf{W}$'ten $\mathsf{V}$'ye olan geçiş, her iki dördüncü mertebeden katsayılar arasında bire bir örtüşme olduğundan, şimdilik tamamen gösteriş için. Biz sadece $(k, l)$ altindisleri yeniden $k = i+a$ ve $l = j+b$ diye dizinliyoruz. Başka bir deyişle, $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$'i ayarladık. $a$ ve $b$ indisleri, tüm imgeyi kapsayan hem pozitif hem de negatif basit ek değerlerin üzerinden çalışır. $[\mathbf{H}]_{i, j}$ gizli gösterimindeki herhangi bir ($i$, $j$) konumunun değerini, $x$'deki, $(i, j)$ civarında ortalanmış ve $[\mathsf{V}]_{i, j, a, b}$ ile ağırlıklarıldırılmış pikseller üzerinden toplayarak hesaplarız.

### Çeviri Değişmezliği

Şimdi yukarıda bahsedilen ilk ilkeyi çağıralım: Çeviri değişmezliği. Bu, $\mathbf{X}$ girdisindeki bir kaymanın, gizli gösterimde $\mathbf{H}$'de bir kaymaya yol açması gerektiği anlamına gelir. Bu sadece $\mathsf{V}$ ve $\mathbf{U}$ aslında $(i, j)$'ye bağımlı değilse mümkündür, yani $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ ve $\mathbf{U}$ bir sabit, diyelim $u$ ise. Sonuç olarak, $\mathbf{H}$ tanımını basitleştirebiliriz:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

Bu bir *evrişim!* $[\mathbf{H}]_{i, j}$ değerini elde etmek için $(i, j)$'nın $[\mathbf{V}]_{a, b}$ katsayıları ile $(i+a, j+b)$ adresindeki pikselleri etkin bir şekilde ağırlıklandırıyoruz. $[\mathbf{V}]_{a, b}$'ün artık imge içindeki konuma bağlı olmadığından $[\mathsf{V}]_{i, j, a, b}$'den çok daha az katsayıya ihtiyaç duyduğunu unutmayın. Önemli bir ilerleme kaydettik!

###  Yerellik

Şimdi ikinci prensibi anımsıyalım: Yerellik. Yukarıda motive edildiği gibi, $[\mathbf{H}]_{i, j}$'te neler olup bittiğini değerlendirken ilgili bilgileri toplamak için $(i, j)$ konumundan çok uzaklara bakmak zorunda olmamamız gerektiğine inanıyoruz. Bu, bazı $|a|> \Delta$ veya $|b| > \Delta$ aralığı dışında, $[\mathbf{V}]_{a, b} = 0$ diye ayarlamanız gerektiği anlamına gelir. Eşdeğer olarak, $[\mathbf{H}]_{i, j}$ olarak yeniden yazabiliriz

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Özetle, :eqref:`eq_conv-layer` 'ün bir *evrişimli katman* olduğuna dikkat edin. *Evrişimli sinir ağları* (CNN), evrişimli katmanlar içeren özel bir sinir ağları ailesidir. Derin öğrenme araştırma topluluğunda $\mathbf{V}$, bir *evrişim çekirdeği*, bir *filtre* veya genellikle katmanın öğrenilebilir parametreler olan *ağırlıkları* olarak adlandırılır. Yerel bölge küçük olduğunda, tam bağlı bir ağ ile karşılaştırıldığında fark çarpıcı olabilir. Daha önce, bir imge işleme ağındaki tek bir katmanı temsil etmek için milyarlarca parametre gerekirken şimdi genellikle girdilerin veya gizli temsillerin boyutsallığını değiştirmeden sadece birkaç yüz taneye ihtiyacımız var. Parametrelerdeki bu ciddi azalma için ödenen bedel, özelliklerimizin artık çeviri değişmez olması ve katmanımızın her gizli etkinleştirmenin değerine karar verirken yalnızca yerel bilgileri içerebilmesidir. Tüm öğrenme tümevarımsal önyargının uygulmaya koymasına  bağlıdır. Bu önyargı gerçekle aynı yönde olduğunda, görünmeyen verilere iyi genelleyen örneklem-verimli modeller elde ederiz. Ancak elbette, bu önyargılar gerçekle aynı yönde değilse, örneğin, imgelerin çeviri değişmez olmadığı ortaya çıkarsa, modellerimiz eğitim verilerimize oturması için bile debelenmesi gerekir.

## Evrişimler

Daha ilerlemeden önce, yukarıdaki işlemin neden bir evrişim (birlikte evrimleşme) olarak adlandırıldığını kısaca gözden geçirmeliyiz. Matematikte, iki fonksiyon arasındaki *evrişim*, mesela $f, g: \mathbb{R}^d \to \mathbb{R}$ için, şöyle tanımlanır:

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Yani, bir işlev “çevrilmiş” ve $\mathbf{x}$ ile kaydırılmış olduğunda $f$ ve $g$ arasındaki örtüşmeyi ölçüyoruz. Ayrık nesnelere sahip olduğumuzda, integral bir toplama dönüşür. Örneğin, $\mathbb{Z}$ üzerinde çalışan indisi olan kare toplanabilir sonsuz boyutlu vektörler kümesinden vektörler için aşağıdaki tanımı elde ederiz:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

İki boyutlu tensörler için, sırasıyla $f$ için $(a, b)$ ve $g$ için $(i-a, j-b)$ indisleri ile karşılık gelen bir toplama işlemine sahibiz:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Bu, :eqref:`eq_conv-layer`'e benzer, ama büyük bir farkla. $(i+a, j+b)$ kullanmak yerine farkı kullanıyoruz. Yine de, bu ayrımın çoğunlukla gösterişsel olduğunu unutmayın, çünkü :eqref:`eq_conv-layer` ve :eqref:`eq_2d-conv-discrete` arasındaki gösterimi her zaman eşleştirebiliriz. :eqref:`eq_conv-layer`'teki orijinal tanımımız, daha doğru bir şekilde *çapraz korelasyonu* tanımlıyor. Aşağıdaki bölümde buna geri döneceğiz.

## “Waldo Nerede”ye Tekrar Bakış

Waldo dedektörümüze dönersek, bunun neye benzediğini görelim. Evrişimli tabaka, belirli bir boyuttaki pencereleri seçer ve :numref:`fig_waldo_mask`'te gösterildiği gibi $\mathsf{V}$'e göre yoğunlukları ağırlıklandırır. Bir model öğrenmeyi hedefleyebiliriz, böylece “Waldoluk” en yüksek nerede olursa olsun, gizli katman temsillerinde bir yüksek değer bulmalıyız.

![Waldo'yu bul.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

### Kanallar
:label:`subsec_why-conv-channels`

Bu yaklaşımla ilgili tek bir sorun var. Şimdiye kadar, imgelerin 3 kanaldan oluştuğunu görmezden geldik: Kırmızı, yeşil ve mavi. Gerçekte, imgeler iki boyutlu nesneler değil, yükseklik, genişlik ve kanal ile karakterize edilen üçüncü dereceden tensörlerdir, örn.  $1024 \times 1024 \times 3$ şekilli pikseller. Bu eksenlerin ilk ikisi konumsal ilişkileri ilgilendirirken, üçüncüsü her piksel konumuna çok boyutlu bir temsil atama olarak kabul edilebilir. Böylece, $\mathsf{X}$'i $[\mathsf{X}]_{i, j, k}$ olarak indisleriz. Evrişimli filtre buna göre uyarlanmak zorundadır. $[\mathbf{V}]_{a,b}$ yerine artık $[\mathsf{V}]_{a,b,c}$ var.

Dahası, girdimiz üçüncü mertebeden bir tensörden oluştuğu gibi, gizli temsillerimizi de benzer şekilde üçüncü mertebeden tensörler $\mathsf{H}$ olarak formüle etmenin iyi bir fikir olduğu ortaya çıkıyor. Başka bir deyişle, her uzaysal konuma karşılık gelen tek bir gizli gösterime sahip olmaktan ziyade, her uzaysal konuma karşılık gelen gizli temsillerin tüm vektörünü istiyoruz. Gizli temsilleri, birbirinin üzerine yığılmış bir dizi iki boyutlu ızgaralar topluluğu olarak düşünebiliriz. Girdilerde olduğu gibi, bunlara bazen *kanallar* denir. Bunlara bazen *öznitelik eşlemeleri* de denir, çünkü her biri sonraki katmana uzamlaştırılmış bir öğrenilmiş öznitelikler kümesi sağlar. Sezgisel olarak, girdilere daha yakın olan alt katmanlarda, bazı kanalların kenarları tanımak için uzmanlaşabileceğini, diğerlerinin de dokuları tanıyabileceğini tasavvur edebilirsiniz.

Hem girdide ($\mathsf{X}$) hem de gizli temsillerde ($\mathsf{H}$) birden fazla kanalı desteklemek için $\mathsf{V}$:$[\mathsf{V}]_{a, b, c, d}$'ye dördüncü bir koordinat ekleyebiliriz. Sahip olduğumuz her şeyi bir araya getirirsek:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

burada $d$, $\mathsf{H}$ gizli temsillerinde çıktı kanalları dizinler. Sonraki evrişimli tabaka, girdi olarak üçüncü mertebeden bir tensör olan $\mathsf{H}$'i almaya devam edecektir. Daha genel olarak, :eqref:`eq_conv-layer-channels`, birden fazla kanal için bir evrişimli tabakanın tanımıdır; burada $\mathsf{V}$, katmanın bir çekirdeği veya filtresidir.

Halen ele almamız gereken birçok işlem var. Örneğin, tüm gizli temsilleri tek bir çıktıda nasıl birleştireceğimizi bulmamız gerekiyor, örn. resimde *herhangi bir yerde* Waldo var mı? Ayrıca işleri verimli bir şekilde nasıl hesaplayacağımıza, birden fazla katmanı nasıl birleştireceğimize, uygun etkinleştirme işlevlerine ve pratikte etkili ağlar oluşturmak için makul tasarım seçimlerinin nasıl yapılacağına karar vermeliyiz. Bölümün geri kalanında bu sorunlara eğiliyoruz.

## Özet

* İmgelerdeki çeviri değişmezliği, bir imgenin tüm yamalarının aynı şekilde işleneceğini ima eder.
* Yerellik, karşılık gelen gizli temsilleri hesaplamak için yalnızca küçük bir piksel komşuluğunun kullanılacağı anlamına gelir.
* İmge işlemede, evrişimli katmanlar genellikle tam bağlı katmanlardan çok daha az parametre gerektirir.
* CNN'ler, evrişimli katmanlar içeren özel bir sinir ağları ailesidir.
* Girdi ve çıktıdaki kanallar, modelimizin bir görüntünün her uzaysal konumda birden çok yönünü yakalamasına olanak tanır.

## Alıştırmalar

1. Evrişim çekirdeğinin boyutu $\Delta = 0$ olduğunu varsayalım. Bu durumda, evrişim çekirdeğinin her kanal kümesi için bağımsız olarak bir MLP uyguladığını gösterin.
1. Çeviri değişmezliği neden iyi bir fikir olmayabilir?
1. Bir imgenin sınırındaki piksel konumlarına karşılık gelen gizli temsillerine nasıl muamele edileceğine karar verirken hangi sorunlarla baş etmeliyiz?
1. Ses için benzer bir evrişimli katman tanımlayın.
1. Evrişimli katmanların metin verileri için de geçerli olabileceğini düşünüyor musunuz? Neden ya da neden olmasın?
1. Bunu kanıtlayın: $f * g = g * f$.

[Tartışmalar](https://discuss.d2l.ai/t/64)
