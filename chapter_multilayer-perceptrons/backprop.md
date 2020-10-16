# İleri Yayma, Geriye Yayma ve Hesaplamalı Çizge
:label:`sec_backprop`

Şimdiye kadar modellerimizi minigrup rasgele gradyan inişi ile eğittik. Bununla birlikte, algoritmayı uyguladığımızda, yalnızca model aracılığıyla *ileri yayma* ile ilgili hesaplamalar hakkında endişelendik. Gradyanları hesaplama zamanı geldiğinde, çerçeve tarafından sağlanan geri yayma fonksiyonunu çalıştırdık.

Gradyanların otomatik olarak hesaplanması, derin öğrenme algoritmalarının uygulanmasını büyük ölçüde basitleştirir. Otomatik türev almadan önce, karmaşık modellerde yapılan küçük değişiklikler bile karmaşık türevlerin elle yeniden hesaplanmasını gerektiriyordu. Şaşırtıcı bir şekilde, akademik makaleler güncelleme kurallarını türetmek için çok sayıda sayfa ayırmak zorunda kalırdı. İlginç kısımlara odaklanabilmemiz için otomotik türeve güvenmeye devam etmemiz gerekse de, sığ bir derin öğrenme anlayışının ötesine geçmek istiyorsanız, bu gradyanların kaputun altında nasıl hesaplandığını *bilmelisiniz*.

Bu bölümde, geriye doğru yaymanın (daha yaygın olarak *geri yayma* veya *geriyay* olarak adlandırılır) ayrıntılarına derinlemesine dalacağız. Hem teknikler hem de uygulamaları hakkında bazı bilgiler vermek için birtakım temel matematik ve hesaplama çizgelerine güveniyoruz. Başlangıç ​​olarak, açıklamamızı ağırlık sönümü ($\ell_2$ düzenlileştirme) olan üç katmanlı (biri gizli) çok katmanlı algılayıcıya odaklıyoruz.


## İleri Yayma

İleri yayma, giriş katmanından çıktı katmanına sırayla sinir ağı için ara değişkenlerin (çıktılar dahil) hesaplanması ve depolanması anlamına gelir. Artık tek bir gizli katmana sahip derin bir ağın mekaniği üzerinde adım adım çalışacağız. Bu sıkıcı görünebilir ama funk virtüözü James Brown'un ebedi sözleriyle, "patron olmanın bedelini ödemelisiniz".

Kolaylık olması açısından, girdi örneğinin $\mathbf{x}\in \mathbb{R}^d$ olduğunu ve gizli katmanımızın bir ek girdi terimi içermediğini varsayalım. İşte ara değişkenimiz:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ gizli katmanın ağırlık parametresidir. $\mathbf{z} \in \mathbb{R}^h$ ara değişkenini $\phi$ etkinleştirme fonksiyonu üzerinden çalıştırdıktan sonra, $h$ uzunluğundaki gizli etkinleştirme vektörümüzü elde ederiz,

$$\mathbf{h}= \phi (\mathbf{z}).$$

$\mathbf{h}$ gizli değişkeni de bir ara değişkendir. Çıktı katmanının parametrelerinin yalnızca $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ ağırlığına sahip olduğunu varsayarsak, $q$ uzunluğunda vektörel bir çıktı katmanı değişkeni elde edebiliriz :

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Kayıp fonksiyonunun $l$ ve örnek etiketin $y$ olduğunu varsayarsak, tek bir veri örneği için kayıp terimini hesaplayabiliriz,

$$L = l(\mathbf{o}, y).$$

$\ell_2$ düzenlileştirmenin tanımına göre, $\lambda$ hiper parametresi verildiğinde, düzenlileştirme terimi:

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$

burada matrisin Frobenius normu, matris bir vektöre düzleştirildikten sonra uygulanan $L_2$ normudur. Son olarak, modelin belirli bir veri örneğine göre düzenlileştirilmiş kaybı şudur:

$$J = L + s.$$

Aşağıdaki tartışmada $J$'ye *amaç fonksiyonu* olarak atıfta bulunacağız.


## İleri Yaymanın Hesaplamalı Çizgesi

Hesaplamalı çizgeleri çizmek, hesaplamadaki operatörlerin ve değişkenlerin bağımlılıklarını görselleştirmemize yardımcı olur. :numref:`fig_forward`, yukarıda açıklanan basit ağ ile ilişkili çizgeyi içerir. Sol alt köşe girdiyi, sağ üst köşesi çıktıyı belirtir. Okların yönünün (veri akışını gösteren) esasen sağa ve yukarıya doğru olduğuna dikkat edin.

![Hesaplamalı Çizgesi](../img/forward.svg)
:label:`fig_forward`


## Geri Yayma

Geri yayma, sinir ağı parametrelerinin gradyanını hesaplama yöntemini ifade eder. Kısacası yöntem, analizden *zincir kuralına* göre ağı çıktıdan girdi katmanına ters sırada dolaşır. Algoritma, gradyanı bazı parametrelere göre hesaplarken gerekli olan tüm ara değişkenleri (kısmi türevler) depolar. $\mathsf{Y}=f(\mathsf{X})$ ve $\mathsf{Z}=g(\mathsf{Y}) = g \circ f(\mathsf{X})$, fonksiyonlarımız olduğunu $\mathsf{X}, \mathsf{Y}, \mathsf {Z}$ girdi ve çıktılarının rastgele şekilli tensörler olduğunu varsayalım. Zincir kuralını kullanarak, $\mathsf{Z}$'nin $\mathsf{X}$'e göre türevini hesaplayabiliriz. 

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Burada, aktarma ve girdi konumlarını değiştirme gibi gerekli işlemler gerçekleştirildikten sonra argümanlarını çarpmak için $\text{prod}$ operatörünü kullanıyoruz. Vektörler için bu basittir: Bu basitçe matris-matris çarpımıdır. Daha yüksek boyutlu tensörler için uygun muadili kullanırız. $\text{prod}$ operatörü tüm gösterim ek yükünü gizler.

Bir gizli katmana sahip basit ağın parametreleri $\mathbf{W}^{(1)}$ ve $\mathbf{W}^{(2)}$'dir. Geri yaymanın amacı, $\partial J/\partial \mathbf{W}^{(1)}$ ve $\partial J/\partial \mathbf{W}^{(2)}$ gradyanlarını hesaplamaktır. Bunu başarmak için, zincir kuralını uygularız ve sırayla her bir ara değişken ve parametrenin gradyanını hesaplarız. Hesaplama çizgesinin sonucuyla başlamamız ve parametrelere doğru yolumuza devam etmemiz gerektiğinden, hesaplamaların sırası ileri yaymada gerçekleştirilenlere göre tersine çevrilir. İlk adım $J=L+s$ amaç fonksiyonunun gradyanlarını $L$ kayıp terimi ve $s$ düzenlileştirme terimine göre hesaplamaktır.

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

Ardından, zincir kuralı ile $\mathbf{o}$ çıktı katmanının değişkenine göre amaç fonksiyonunun gradyanını hesaplıyoruz.

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Ardından, her iki parametreye göre düzenlileştirme teriminin gradyanlarını hesaplıyoruz.

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Şimdi, çıktı katmanına en yakın model parametrelerinin gradyanını, $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, hesaplayabiliriz. Zincir kuralını kullanalım:

$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)
= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.
$$

$\mathbf{W}^{(1)}$'e göre gradyanı elde etmek için, çıktı katmanı boyunca gizli katmana geri yaymaya devam etmemiz gerekir. Gizli katmanın $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ çıktılarına göre gradyan şu şekilde verilir:

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

$\phi$ etkinleştirme fonksiyonu eleman yönlü uygulandığından, $\mathbf{z}$ ara değişkeninin $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ gradyanını hesaplamak şunu gerektirir: $\odot$ ile gösterdiğimiz eleman yönlü çarpma operatörünü kullanın.

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Son olarak, girdi katmanına en yakın model parametrelerinin, $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$, gradyanını elde edebiliriz. Zincir kuralına göre hesaplarsak,

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## Model Eğitimi

Ağları eğitirken, ileri ve geri yayma birbirine bağlıdır. Özellikle, ileri yayma için, hesaplama çizgesini bağımlılıklar yönünde gezeriz ve yoldaki tüm değişkenleri hesaplarız. Bunlar daha sonra grafikteki hesaplama sırasının tersine çevrildiği geri yayma için kullanılır. Sonuçlardan biri, geri yayma tamamlanana kadar ara değerleri korumamız gerektiğidir. Bu aynı zamanda, geri yaymanın basit tahminden önemli ölçüde daha fazla bellek gerektirmesinin nedenlerinden biridir. Tensörleri gradyan olarak hesaplıyoruz ve zincir kuralını çağırmak için tüm ara değişkenleri korumamız gerekiyor. Diğer bir neden ise, tipik olarak birden fazla değişken içeren minigruplarla eğitim yapmamızdır, bu nedenle daha fazla ara etkinleştirmenin depolanması gerekir.

## Özet

* İleri yayma, ara değişkenleri sırayla hesaplar ve sinir ağı tarafından tanımlanan hesaplama çizgesi içinde depolar. Girdiden çıktı katmanına doğru ilerler.
* Geri yayma, sinir ağı içindeki ara değişkenlerin ve parametrelerin gradyanlarını ters sırayla sırayla hesaplar ve saklar.
* Derin öğrenme modellerini eğitirken, ileri yayma ve geri yayma birbirine bağlıdır.
* Eğitim, önemli ölçüde daha fazla bellek ve depolama alanı gerektirir.

## Alıştırma

1. $\mathbf{x}$'in bazı sayıl (skaler) fonksiyonu $f$'in girdilerinin $n \times m$ matrisler olduğunu varsayın. $\mathbf{x}$'e göre $f$'in gradyanının boyutsallığı nedir?
1. Bu bölümde açıklanan modelin gizli katmanına bir ek girdi ekleyiniz.
     * İlgili hesaplama çizgesini çizin.
     * İleri ve geri yayma denklemlerini türetiniz.
1. Mevcut bölümde açıklanan modeldeki eğitim ve çıkarım için bellek ayak izini hesaplayın.
1. *İkinci* türevleri hesaplamak istediğinizi varsayın. Hesaplama çizgesine ne olur? Hesaplamanın ne kadar sürmesini bekliyorsunuz?
1. Hesaplama çizgenizin GPU'nuz için çok büyük olduğunu varsayın.
     * Birden fazla GPU'ya bölebilir misiniz?
     * Daha küçük bir minigrup üzerinde eğitime göre avantajları ve dezavantajları nelerdir?

[Tartışmalar](https://discuss.d2l.ai/t/102)
