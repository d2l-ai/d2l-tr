# İleri Yayma, Geri Yayma ve Hesaplamalı Çizge
:label:`sec_backprop`

Şimdiye kadar modellerimizi minigrup rasgele gradyan inişi ile eğittik. Bununla birlikte, algoritmayı uyguladığımızda, yalnızca model aracılığıyla *ileri yayma* ile ilgili hesaplamalar hakkında endişelendik. Gradyanları hesaplama zamanı geldiğinde, derin öğrenme çerçevesi tarafından sağlanan geri yayma fonksiyonunu çalıştırdık.

Gradyanların otomatik olarak hesaplanması (otomatik türev alma), derin öğrenme algoritmalarının uygulanmasını büyük ölçüde basitleştirir. Otomatik türev almadan önce, karmaşık modellerde yapılan küçük değişiklikler bile karmaşık türevlerin elle yeniden hesaplanmasını gerektiriyordu. Şaşırtıcı bir şekilde, akademik makaleler güncelleme kurallarını türetmek için çok sayıda sayfa ayırmak zorunda kalırdı. İlginç kısımlara odaklanabilmemiz için otomatik türeve güvenmeye devam etmemiz gerekse de, sığ bir derin öğrenme anlayışının ötesine geçmek istiyorsanız, bu gradyanların kaputun altında nasıl hesaplandığını bilmelisiniz.

Bu bölümde, *geriye doğru yayma*nın (daha yaygın olarak *geri yayma* olarak adlandırılır) ayrıntılarına derinlemesine dalacağız. Hem teknikler hem de uygulamaları hakkında bazı bilgiler vermek için birtakım temel matematik ve hesaplama çizgelerine güveniyoruz. Başlangıç olarak, açıklamamızı ağırlık sönümlü ($L_2$ düzenlileştirme), bir gizli katmanlı MLP'ye odaklıyoruz.


## İleri Yayma

*İleri yayma* (veya *ileri iletme*), girdi katmanından çıktı katmanına sırayla bir sinir ağı için ara değişkenlerin (çıktılar dahil) hesaplanması ve depolanması anlamına gelir. Artık tek bir gizli katmana sahip bir sinir ağı mekaniği üzerinde adım adım çalışacağız. Bu sıkıcı görünebilir ama funk virtüözü James Brown'un ebedi sözleriyle, "patron olmanın bedelini ödemelisiniz".

Kolaylık olması açısından, girdi örneğinin $\mathbf{x}\in \mathbb{R}^d$ olduğunu ve gizli katmanımızın bir ek girdi terimi içermediğini varsayalım. İşte ara değişkenimiz:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ gizli katmanın ağırlık parametresidir. $\mathbf{z} \in \mathbb{R}^h$ ara değişkenini $\phi$ etkinleştirme fonksiyonu üzerinden çalıştırdıktan sonra, $h$ uzunluğundaki gizli etkinleştirme vektörümüzü elde ederiz,

$$\mathbf{h}= \phi (\mathbf{z}).$$

$\mathbf{h}$ gizli değişkeni de bir ara değişkendir. Çıktı katmanının parametrelerinin yalnızca $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ ağırlığına sahip olduğunu varsayarsak, $q$ uzunluğunda vektörel bir çıktı katmanı değişkeni elde edebiliriz:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Kayıp fonksiyonunun $l$ ve örnek etiketin $y$ olduğunu varsayarsak, tek bir veri örneği için kayıp terimini hesaplayabiliriz,

$$L = l(\mathbf{o}, y).$$

$L_2$ düzenlileştirmenin tanımına göre, $\lambda$ hiper parametresi verildiğinde, düzenlileştirme terimi şöyledir:

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

burada matrisin Frobenius normu, matris bir vektöre düzleştirildikten sonra uygulanan $L_2$ normudur. Son olarak, modelin belirli bir veri örneğine göre düzenlileştirilmiş kaybı şudur:

$$J = L + s.$$

Aşağıdaki tartışmada $J$'ye *amaç fonksiyonu* olarak atıfta bulunacağız.


## İleri Yaymanın Hesaplamalı Çizgesi

**Hesaplamalı çizgeleri** çizmek, hesaplamadaki operatörlerin ve değişkenlerin bağımlılıklarını görselleştirmemize yardımcı olur. :numref:`fig_forward`, yukarıda açıklanan basit ağ ile ilişkili çizgeyi içerir, öyleki kareler değişkenleri ve daireler işlemleri temsil eder. Sol alt köşe girdiyi, sağ üst köşesi çıktıyı belirtir. Okların yönlerinin (veri akışını gösteren) esasen sağa ve yukarıya doğru olduğuna dikkat edin.

![İleri yaymanın hesaplamalı çizgesi](../img/forward.svg)
:label:`fig_forward`


## Geri Yayma

*Geri yayma*, sinir ağı parametrelerinin gradyanını hesaplama yöntemini ifade eder. Kısacası yöntem, analizden *zincir kuralına* göre ağı çıktıdan girdi katmanına ters sırada dolaşır. Algoritma, gradyanı bazı parametrelere göre hesaplarken gerekli olan tüm ara değişkenleri (kısmi türevler) depolar. $\mathsf{Y}=f(\mathsf{X})$ ve $\mathsf{Z}=g(\mathsf{Y})$, fonksiyonlarımız olduğunu $\mathsf{X}, \mathsf{Y}, \mathsf {Z}$ girdi ve çıktılarının rastgele şekilli tensörler olduğunu varsayalım. Zincir kuralını kullanarak, $\mathsf{Z}$'nin $\mathsf{X}$'e göre türevini hesaplayabiliriz. 

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Burada, aktarma ve girdi konumlarını değiştirme gibi gerekli işlemler gerçekleştirildikten sonra argümanlarını çarpmak için $\text{prod}$ operatörünü kullanıyoruz. Vektörler için bu basittir: Bu basitçe matris-matris çarpımıdır. Daha yüksek boyutlu tensörler için uygun muadili kullanırız. $\text{prod}$ operatörü tüm gösterim ek yükünü gizler.

Hesaplamalı çizgesi :numref:`fig_forward` içinde gösterilen bir gizli katmana sahip basit ağın parametrelerinin $\mathbf{W}^{(1)}$ ve $\mathbf{W}^{(2)}$ olduğunu hatırlayalım. Geri yaymanın amacı, $\partial J/\partial \mathbf{W}^{(1)}$ ve $\partial J/\partial \mathbf{W}^{(2)}$ gradyanlarını hesaplamaktır. Bunu başarmak için, zincir kuralını uygularız ve sırayla her bir ara değişken ve parametrenin gradyanını hesaplarız. Hesaplamalı çizgenin sonucuyla başlamamız ve parametrelere doğru yolumuza devam etmemiz gerektiğinden, hesaplamaların sırası ileri yaymada gerçekleştirilenlere göre tersine çevrilir. İlk adım $J=L+s$ amaç fonksiyonunun gradyanlarını $L$ kayıp terimi ve $s$ düzenlileştirme terimine göre hesaplamaktır.

$$\frac{\partial J}{\partial L} = 1 \; \text{ve} \; \frac{\partial J}{\partial s} = 1.$$

Ardından, zincir kuralı ile $\mathbf{o}$ çıktı katmanının değişkenine göre amaç fonksiyonunun gradyanını hesaplıyoruz:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Ardından, her iki parametreye göre düzenlileştirme teriminin gradyanlarını hesaplıyoruz:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Şimdi, çıktı katmanına en yakın model parametrelerinin gradyanını, $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, hesaplayabiliriz. Zincir kuralını kullanalım:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

$\mathbf{W}^{(1)}$'e göre gradyanı elde etmek için, çıktı katmanı boyunca gizli katmana geri yaymaya devam etmemiz gerekir. Gizli katmanın $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ çıktılarına göre gradyan şu şekilde verilir:

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

$\phi$ etkinleştirme fonksiyonu eleman yönlü uygulandığından, $\mathbf{z}$ ara değişkeninin $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ gradyanını hesaplamak  $\odot$ ile gösterdiğimiz eleman yönlü çarpma operatörünü kullanmayı gerektirir:

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

## Sinir Ağları Eğitimi

Sinir ağlarını eğitirken, ileri ve geri yayma birbirine bağlıdır. Özellikle, ileri yayma için, hesaplamalı çizgeyi bağımlılıklar yönünde gezeriz ve yoldaki tüm değişkenleri hesaplarız. Bunlar daha sonra grafikteki hesaplama sırasının tersine çevrildiği geri yayma için kullanılır. 

Kafanızda canlandırmak için yukarıda belirtilen basit ağı örnek olarak alın.
Bir yandan, ileri yayma sırasında :eqref:`eq_forward-s` düzenlileştirme teriminin hesaplanması $\mathbf{W}^{(1)}$ ve $\mathbf{W}^{(2)}$ model parametrelerinin mevcut değerlerine bağlıdır. En son yinelemede geri yaymaya göre optimizasyon algoritması tarafından verilirler. Öte yandan, geri yayma sırasında :eqref:`eq_backprop-J-h` parametresi için gradyan hesaplaması, ileri yayma tarafından verilen gizli $\mathbf{h}$ değişkeninin mevcut değerine bağlıdır.


Bu nedenle, sinir ağlarını eğitirken, model parametreleri ilklendikten sonra, ileri yaymayı geri yayma ile değiştiririz, model parametrelerini geri yayma tarafından verilen gradyanları kullanarak güncelleriz. Geri yaymanın, tekrarlanan hesaplamaları önlemek için ileriye yaymadan depolanan ara değerleri yeniden kullandığını unutmayın.
Sonuçlardan biri, geri yayma tamamlanana kadar ara değerleri korumamız gerektiğidir. Bu aynı zamanda, eğitimin basit tahminden önemli ölçüde daha fazla bellek gerektirmesinin nedenlerinden biridir. Ayrıca, bu tür ara değerlerin boyutu, ağ katmanlarının sayısı ve iş boyutu ile kabaca orantılıdır. Bu nedenle, daha büyük toplu iş boyutlarını kullanarak daha derin ağlar eğitmek, kolaylıkla *yetersiz bellek* hatalarına yol açar.

## Özet

* İleri yayma, ara değişkenleri sırayla hesaplar ve sinir ağı tarafından tanımlanan hesaplamalı çizge içinde depolar. Girdiden çıktı katmanına doğru ilerler.
* Geri yayma, sinir ağı içindeki ara değişkenlerin ve parametrelerin gradyanlarını ters sırayla sırayla hesaplar ve saklar.
* Derin öğrenme modellerini eğitirken, ileri yayma ve geri yayma birbirine bağlıdır.
* Eğitim, tahminlemeden önemli ölçüde daha fazla bellek gerektirir.

## Alıştırma

1. $\mathbf{X}$'in bazı sayıl (skaler) fonksiyonu $f$'nin girdilerinin $n \times m$ matrisler olduğunu varsayın. $\mathbf{X}$'e göre $f$'nin gradyanının boyutsallığı nedir?
1. Bu bölümde açıklanan modelin gizli katmanına bir ek girdi ekleyiniz (düzenlileştirme teriminde ek girdiyi katmanız gerekmez).
     1. İlgili hesaplamalı çizgeyi çizin.
     1. İleri ve geri yayma denklemlerini türetiniz.
1. Bu bölümde açıklanan modeldeki eğitim ve tahmin için bellek ayak izini hesaplayın.
1. İkinci türevleri hesaplamak istediğinizi varsayın. Hesaplamalı çizgeye ne olur? Hesaplamanın ne kadar sürmesini bekliyorsunuz?
1. Hesaplamalı çizgenin GPU'nuz için çok büyük olduğunu varsayın.
     1. Birden fazla GPU'ya bölebilir misiniz?
     1. Daha küçük bir minigrup üzerinde eğitime göre avantajları ve dezavantajları nelerdir?

[Tartışmalar](https://discuss.d2l.ai/t/102)
