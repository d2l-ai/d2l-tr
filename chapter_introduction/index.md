# Giriş
:label:`chap_introduction`


Yakın zamana kadar, günlük etkileşimde bulunduğumuz hemen hemen her bilgisayar programı basit prensiplerle yazılım geliştiricileri tarafından kodlandı.
Bir e-ticaret platformunu yönetmek için bir uygulama yazmak istediğimizi varsayalım. Soruna kafa yorarak birkaç saat boyunca bir beyaz tahta etrafında toplandıktan sonra, muhtemelen kabaca böyle bir şeye benzeyebilecek bir çalışma çözümü buluruz:
(i) Kullanıcılar bir web tarayıcısında veya mobil uygulamada çalışan bir arabirim aracılığıyla uygulama ile etkileşimde bulunurlar,
(ii) uygulamamız, her kullanıcının durumunu takip etmek ve geçmiş işlemlerin kayıtlarını tutmak için ticari düzeyde bir veritabanı motoruyla etkileşime girer ve (iii) uygulamamızın merkezinde, uygulamamızın *iş mantığı* (*beyinleri* diyebilirsiniz) yöntemsel olarak ayrıntılı bir şekilde programımızın akla gelebilecek her durumda gerçekleştirmesi gereken eylemi açıklar.


Uygulamamızın *beyinlerini* oluşturmak için karşılaşacağımızı tahmin ettiğimiz her köşe vakasına adım atmamız ve uygun kuralları belirlediğimiz gerekir.
Bir müşteri alışveriş sepetine bir öğe eklemek için her tıkladığında, alışveriş sepeti veritabanı tablosuna bu kullanıcının kimliğini istenen ürünün kimliği ile ilişkilendirerek bir girdi ekleriz.
Sadece birkaç geliştirici ilk seferinde tamamen doğru hale getirebilirken (karışıklıkları çözmek için bazı test çalışmaları gerekebilir), çoğunlukla, *önceden gerçek bir müşteri bile görmeden* böyle bir programı basit prensiplerle yazabilir ve güvenle başlatabiliriz.
Genellikle de yeni durumlarda, işlevsel ürün ve sistemleri yönlendiren otomatik sistemleri basit prensiplerden tasarlama yeteneğimiz, dikkate değer bir bilişsel başarıdır.
Ayrıca zamanının $\%100$'ünde çalışan çözümler tasarlayabildiğinizde, *makine öğrenmesi kullanmamalısınız*.


Neyse ki, giderek artan makine öğrenmesi (MÖ) bilim insanları topluluğu için, otomatikleştirmek istediğimiz birçok görev insan yaratıcılığına bu kadar kolay boyun eğmiyor.
Beyaz tahta etrafında bildiğiniz en akıllı zihinlerle toplandığınızı hayal edin, ancak bu sefer aşağıdaki sorunlardan birini ele alıyorsunuz:

* Coğrafi bilgi, uydu görüntüleri ve yakın bir zaman penceresindeki geçmiş hava koşullarını göz önüne alındığında yarının hava durumunu tahmin eden bir program yazma.
* Serbest biçimli metinle ifade edilen bir soruyu alan ve onu doğru cevaplayan bir program yazma.
* Verilen bir imgedeki, her birinin etrafında çerçeve çizerek, içerdiği tüm insanları tanımlayabilen bir program yazma.
* Kullanıcılara keyif alabilecekleri ancak sıradan gezinmeleri esnasında karşılaşma olasılıkları yüksek olmayan ürünler sunan bir program yazma.


Bu vakaların her birinde, seçkin programcılar bile çözümleri sıfırdan kodlayamazlar.
Bunun nedenleri değişebilir. Bazen aradığımız program zaman içinde değişen bir kalıp takip eder ve programlarımızın adapte olması gerekir.
Diğer durumlarda, ilişki (pikseller ve soyut kategoriler arasında) çok karmaşık olabilir ve bilinçli anlayışımızın ötesinde binlerce veya milyonlarca hesaplama gerektirebilir.
(ki gözlerimiz görevi zahmetsizce yönetse bile). MÖ *deneyimlerden öğrenebilen* güçlü tekniklerin incelenmesidir.
Bir MÖ algoritması, tipik olarak gözlemsel veriler veya bir çevre ile etkileşimler şeklinde, daha fazla deneyim biriktirdikçe performansı artar.
Bunu, ne kadar deneyim kazanırsa kazansın, geliştiricilerin kendileri *öğrenip* yazılımın güncellenme zamanının geldiğine karar verene kadar, aynı iş mantığına göre çalışan gerekirci (deterministik) e-ticaret platformumuzla karşılaştırın.
Bu kitapta size makine öğrenmesinin temellerini öğreteceğiz ve özellikle de bilgisayarlı görme, doğal dil işleme, sağlık ve genomik gibi farklı alanlarda yenilikleri yönlendiren güçlü bir teknikler kümesine, yani derin öğrenmeye odaklanacağız.


## Motive Edici Bir Örnek

Yazmaya başlamadan önce, biz bu kitabın yazarları, işgücünün çoğu gibi, kafeinli olmak zorundaydık.
Arabaya bindik ve araba kullanmaya başladık.
Alex, bir iPhone kullanıp telefonun ses tanıma sistemini uyandırarak "Hey Siri" diye seslendi.
Sonra Mu "Blue Bottle kafesine yol tarifi" komutunu verdi.
Telefon komutun uyarlamasını (transkripsiyonunu) hızlı bir şekilde gösterdi.
Ayrıca yol tarifini istediğimizi fark etti ve talebimizi yerine getirmek için Maps (Haritalar) uygulamasını başlattı.
Bir kez başlatıldığında, Haritalar uygulaması bir dizi rota belirledi.
Her rotanın yanında, telefon tahmini bir yol süresi gösterdi.
Biz bu hikayeyi pedagojik (eğitbilimsel) rahatlık için üretirken, sadece birkaç saniye içinde, bir akıllı telefondaki günlük etkileşimlerimizin birkaç makine öğrenmesi modeliyle işbirligi yaptığını gösteriyoruz.


"Alexa", "Tamam, Google" veya "Siri" gibi bir *uyandırma kelimesine* yanıt vermek için bir program yazdığınızı düşünün.
Bir odada kendiniz bir bilgisayar ve kod düzenleyicisinden başka bir şey olmadan kodlamayı deneyin :numref:`fig_wake_word`.
Böyle bir programı basit ilkelerden (prensiplerden) nasıl yazarsınız?
Bir düşünün ... problem zor.
Mikrofon her saniye yaklaşık 44.000 örnek toplayacaktır.
Her örnek, ses dalgasının genliğinin bir ölçümüdür.
Hangi kural güvenilir bir şekilde, ses parçasının uyandırma sözcüğünü içerip içermediğine bağlı olarak bir ham ses parçasından emin ``{evet, hayır}`` tahminlerine eşleme yapabilir?
Sıkıştıysanız endişelenmeyin.
Böyle bir programı nasıl sıfırdan yazacağımızı bilmiyoruz.
Bu yüzden MÖ kullanıyoruz.


![Bir uyandırma kelimesi tanıma. ](../img/wake-word.svg)
:label:`fig_wake_word`

Olayın özünü şöyle açıklayabiliriz.
Çoğu zaman, bir bilgisayara girdilerden çıktılara nasıl eşleştirebileceğini açıklayamayı bilmediğimiz de bile, kendimiz yine de bu bilişsel başarıyı gerçekleştirebiliyoruz.
Diğer bir deyişle, "Alexa" kelimesini tanımak için *bir bilgisayarı nasıl programlayacağınızı* bilmeseniz bile siz *kendiniz* "Alexa" kelimesini tanıyabilirsiniz.
Bu yetenekle donanmış bizler ses örnekleri içeren büyük bir *veri kümesi* toplayabilir ve uyandırma kelimesini *içerenleri* ve *içermeyenleri* etiketleyebiliriz.
MÖ yaklaşımında, uyandırma kelimelerini tanımak için *açıktan* bir sistem tasarlamaya çalışmayız.
Bunun yerine, davranışı bir miktar *parametre* ile belirlenen esnek bir program tanımlarız.
Ardından, veri kümesini, ilgili görevdeki performans ölçüsüne göre, programımızın performansını artıran en iyi parametre kümesini belirlemek için kullanırız.

Parametreleri, çevirerek programın davranışını değiştirebileceğimiz düğmeler olarak düşünebilirsiniz.
Parametreleri sabitlendiğinde, programa *model* diyoruz.
Sadece parametreleri manipüle ederek üretebileceğimiz tüm farklı programlara (girdi-çıktı eşlemeleri) *model ailesi* denir.
Ve parametreleri seçmek için veri kümemizi kullanan *başkalaşım (meta) programına* *öğrenme algoritması* denir.

Devam etmeden ve öğrenme algoritmasını kullanmadan önce, sorunu kesin olarak tanımlamalı, girdi ve çıktıların kesin doğasını tespit etmeli ve uygun bir model ailesi seçmeliyiz.
Bu durumda, modelimiz *girdi* olarak bir ses parçasını alır ve *çıktı* olarak ``{evet, hayır}`` arasında bir seçim oluşturur.
Her şey plana göre giderse, modelin parçanın uyandırma kelimesini içerip içermediğine dair tahminleri genellikle doğru olacaktır.


Doğru model ailesini seçersek, o zaman model her "Alexa" kelimesini her duyduğunda ``evet``i seçecek düğmelerin bir ayarı olmalıdır.
Uyandırma kelimesinin kesin seçimi keyfi olduğundan, muhtemelen yeterince zengin bir model ailesine ihtiyacımız olacak, öyle ki düğmelerin başka bir ayarı ile, sadece "Kayısı" kelimesini duyduktan sonra da ``evet`` seçilebilsin.
Aynı model ailesinin *"Alexa"yı tanıma* ve *"Kayısı"yı tanıma* için uygun olması beklenir, çünkü sezgisel olarak benzer görevler gibi görünüyorlar.
Bununla birlikte, temel olarak farklı girdiler veya çıktılarla uğraşmak istiyorsak, resimlerden altyazılara veya İngilizce cümlelerden Çince cümlelere eşlemek istiyorsak mesela, tamamen farklı bir model ailesine ihtiyacımız olabilir.

Tahmin edebileceğiniz gibi, tüm düğmeleri rastgele bir şekilde ayarlarsak, modelimizin "Alexa", "Kayısı" veya başka bir İngilizce kelimeyi tanıması muhtemel değildir.
Derin öğrenmede, *öğrenme*, modelimizi istenen davranışa zorlayan düğmelerin doğru ayarını keşfettiğimiz süreçtir.

Gösterildiği gibi :numref:`fig_ml_loop`, eğitim süreci genellikle şöyle görünür:

1. Yararlı bir şey yapamayan rastgele başlatılan bir model ile başlayın.
1. Etiketli verilerinizin bir kısmını alın (örneğin, ses parçaları ve onlara karşılık gelen ``{evet, hayır}`` etiketleri).
1. Modelin bu örneklere göre daha az hata yapması için düğmeleri değiştirin.
1. Model harika olana kadar tekrarlayın.


[Tipik bir eğitim süreci.](../img/ml-loop.svg)
:label:`fig_ml_loop`

Özetlemek gerekirse, bir uyandırma kelimesi tanıyıcısını kodlamak yerine, büyük bir etiketli veri kümesi *sunarsak* uyandırma sözcüklerini tanımayı *öğrenebilen* bir program kodlarız.
Bu eylemi bir programın davranışını ona bir veri kümesi sunup *veri ile programlayarak* belirleme gibi düşünebilirsiniz.
Makine öğrenme sistemimize, aşağıdaki resimler gibi, birçok kedi ve köpek örneği sağlayarak bir kedi dedektörü "programlayabiliriz":


|kedi|kedi|köpek|köpek|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![cat3](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|


Bu şekilde dedektör, sonunda, bir kedi ise çok büyük bir pozitif sayı, bir köpekse çok büyük bir negatif sayı ve emin değilse sıfıra daha yakın bir şey yaymayı öğrenir ve bu,  MÖ'nin neler yapabileceğinin ancak yüzeyini kazır.

Derin öğrenme, makine öğrenmesi problemlerini çözmek için mevcut birçok popüler yöntemden sadece biridir.
Şimdiye kadar, derin öğrenme değil, yalnızca geniş kapsamlı makine öğrenmesi hakkında konuştuk. Derin öğrenmenin neden önemli olduğunu görmek amacıyla, birkaç önemli noktayı vurgulamak için bir anlığına durmalıyız.

Birincisi, şu ana kadar tartıştığımız problemler --- ham ses sinyalinden, görüntülerin ham piksel değerlerinden öğrenmek veya keyfi uzunluktaki cümleleri yabancı dillerdeki muadilleri arasında eşlemek --- derin öğrenmenin üstün olduğu ve geleneksel MÖ metotlarının sendelediği problemlerdir.
Derin modeller, birçok hesaplama *katmanını* öğrenmeleri anlamında *derindir*.
Bu çok katmanlı (veya hiyerarşik) modellerin, düşük seviyeli algısal verileri önceki araçların yapamayacağı bir şekilde ele alabildiği ortaya çıkıyor.
Eski günlerde, MÖ'yi bu sorunlara uygulamanın en önemli kısmı, verileri *sığ* modellere uygun bir biçime dönüştürmek için elle (manuel olarak) tasarlanmış yolları bulmaktan oluşuyordu.
Derin öğrenmenin önemli bir avantajı, sadece geleneksel öğrenme üretim hatlarının sonundaki *sığ* modellerin değil, aynı zamanda öznitelik mühendisliğinin emek yoğun sürecinin de yerini almasıdır.
İkincisi,  derin öğrenme, *alana özgü önişlemlemenin* çoğunu eleyerek, daha önce bilgisayarlı görme, konuşma tanıma, doğal dil işleme, tıbbi bilişim ve diğer uygulama alanlarını ayıran sınırların çoğunu ortadan kaldırıp, çeşitli sorunlarla mücadelede ortak kullanılabilecek bir küme araç sunar.

## Temel Bileşenler: Veri, Modeller ve Algoritmalar

*Uyandırma kelimesi* örneğimizde, ses parçaları ve ikili etiketlerden oluşan bir veri kümesi tanımladık ve parçalardan sınıflandırmalara bir eşlemeyi yaklaşık olarak nasıl eğitebileceğimize dair çok ciddi olmayan bir izlenim verdik.
Bu tarz problem, etiketlerinin bilindiği örneklerden oluşan bir veri kümesinin verildiği ve bilinen *girdiler* in belirli bir bilinmeyen *etiket* ini öngörmeye çalıştığımız, *gözetimli öğrenme* olarak adlandırılır ve bu birçok *çeşit* makine öğrenme problemlerinden sadece bir tanesidir.
Bir sonraki bölümde, farklı MÖ sorunlarına derinlemesine bakacağız.
İlk olarak, ne tür bir MÖ problemi olursa olsun, bizi takip edecek bazı temel bileşenlere daha fazla ışık tutmak istiyoruz:

1. Öğrenebileceğimiz *veriler*.
2. Verilerin nasıl dönüştürüleceğine dair bir *model*.
3. Modelimizin *kötülüğünü* ölçen bir *yitim* işlevi.
4. Kaybı en aza indirmede modelin parametrelerini ayarlamak için bir *algoritma*.


### Veri

Veri bilimini veri olmadan yapamayacağınızı söylemeye gerek yok.
Tam olarak veriyi neyin oluşturduğunu düşünerek yüzlerce sayfayı doldurabiliriz, ancak şimdilik pratik tarafta hata yapacağız ve bizi ilgilendiren temel özelliklere odaklanacağız.
Genellikle *örnekler* (*veri noktaları*, *örneklemler* veya *misaller* olarak da adlandırılır) derlemesiyle ilgileniriz.
Verilerle yararlı bir şekilde çalışmak için, genellikle uygun bir sayısal temsil (gösterim) bulmamız gerekir.
Her *örnek* tipik olarak *öznitelikler* adı verilen sayısal özelliklerden oluşur.
Yukarıdaki gözetimli öğrenme problemlerinde özel bir özellik *hedef* tahmini olarak adlandırılır (bazen *etiket* veya *bağımlı değişken* olarak da adlandırılır).
Modelin tahminlerini yapması gereken verilmiş özellikler daha sonra *öznitelikler* (veya bazen *girdiler*, *öndeğişkenler* veya *bağımsız değişkenler*) olarak adlandırılabilir.

Eğer görüntü verileriyle çalışıyorsak, her bir fotoğraf, her bir pikselin parlaklığına karşılık gelen sıralı bir sayısal değerler listesi ile temsil edilen bir *örnek* oluşturabilir.
$200\times200$ bir renkli fotoğraf, her bir uzamsal konum için kırmızı, yeşil ve mavi kanalların parlaklığına karşılık gelen $200\times200\times3=120000$ sayısal değerden oluşur.
Daha geleneksel bir görevde, yaş, yaşamsal belirtiler, teşhisler vb. gibi standart bir dizi özellik göz önüne alındığında, bir hastanın hayatta kalıp kalmayacağını tahmin etmeye çalışabiliriz.

Her örnek aynı sayıda sayısal değerle karakterize edildiğinde, verilerin *sabit uzunluklu* vektörlerden oluştuğunu söylüyoruz ve vektörlerin (sabit) uzunluğunu
verilerin *boyutluluğu* olarak tanımlıyoruz.
Tahmin edebileceğiniz gibi, sabit uzunluk uygun bir özellik olabilir.
Mikroskopi görüntülerinde kanseri tanımak için bir model eğitmek istersek, sabit uzunluktaki girdiler endişelenecek şeylerin sayısının bir tane azaldığı anlamına gelir.

Ancak, tüm veriler kolayca sabit uzunluklu vektörler olarak gösterilemez.
Mikroskop görüntülerinin standart ekipmanlardan gelmesini beklesek de, internetten toplanan görüntülerin aynı çözünürlük veya şekil ile ortaya çıkmasını bekleyemeyiz.
Görüntüler için, hepsini standart bir boyuta kırpmayı düşünebiliriz, ancak bu strateji bizi bir yere kadar götürür.
Kırpılan bölümlerde bilgi kaybetme riskiyle karşı karşıyayız.
Ayrıca, metin verileri sabit uzunluklu gösterimlere daha inatçı bir şekilde direnir.
Amazon, IMDB veya TripAdvisor gibi e-ticaret sitelerine bırakılan müşteri yorumlarını düşünün.
Bazıları kısadır: "berbat!". Diğerleri sayfalara yayılır.
Geleneksel yöntemlere göre derin öğrenmenin en büyük avantajlarından biri, modern modellerin *değişen uzunluktaki* verileri işleyebileceği göreceli yetenektir.

Genel olarak, ne kadar fazla veriye sahip olursak işimiz o kadar kolay olur.
Daha fazla veriye sahip olduğumuzda, daha güçlü modeller eğitebilir ve önceden tasarlanmış varsayımlara daha az bel bağlayabiliriz.
(Nispeten) küçükten büyük verilere rejim (düzen) değişikliği, modern derin öğrenmenin başarısına önemli bir katkıda bulunmaktadır.
İşin özünden bahsedersek, derin öğrenmedeki en heyecan verici modellerin çoğu büyük veri kümeleri olmadan çalışmaz.
Bazıları düşük veri düzeninde çalışır, ancak geleneksel yaklaşımlardan daha iyi değildir.

Son olarak, çok fazla veriye sahip olmak ve onu akıllıca işlemek yeterli değildir.
*Doğru* verilere ihtiyacımız vardır. Veriler hatalarla doluysa veya seçilen özellikler hedefteki ilgili miktarı öngörmüyorsa, öğrenme başarısız olacaktır.
Durum şu klişe ile iyi betimlenebilir: *çöp içeri, çöp dışarı*.
Ayrıca, kötü tahmin performansı tek olası sonuç değildir.
Tahminli polislik, özgeçmiş taraması ve borç verme için kullanılan risk modelleri gibi makine öğrenmesinin hassas uygulamalarında, özellikle çöp verilerinin sonuçlarına karşı dikkatli olmalıyız.
Yaygın bir hata modu, bazı insan gruplarının eğitim verilerinde temsil edilmediği veri kümelerinde gerçekleşir.
Gerçek hayatta, daha önce hiç siyah ten görmemiş bir cilt kanseri tanıma sistemi uyguladığınızı düşünün.
Başarısızlık ayrıca veriler sadece bazı grupları az temsil etmediğinde değil, aynı zamanda toplumsal önyargıları yansıttığı zaman da meydana gelebilir.
Örneğin, özgeçmişleri taramak için kullanılacak bir öngörü modeli eğitmek için geçmiş işe alım kararları kullanılıyorsa, makine öğrenme modelleri yanlışlıkla tarihi adaletsizlikleri yakalayıp onları otomatikleştirebilir.
Tüm bunların, veri bilimcisi aktif olarak komplo kurmadan ve hatta o farkında olmadan gerçekleşebileceğini unutmayın.


### Modeller


Çoğu makine öğrenmesi, verileri bir anlamda *dönüştürmeyi* içerir.
Fotoğrafları yiyen ve *güleryüzlülük* tahmin eden bir sistem kurmak isteyebiliriz.
Alternatif olarak, bir dizi sensör okuması almak ve okumaların *normal* ve *anormal* değerlerini tahmin etmek isteyebiliriz.
*Model* ile, bir tipteki verileri alana ve muhtemelen farklı tipte tahminler veren hesaplama makinelerini belirtiyoruz.
Özellikle verilerden tahmin yapabilecek istatistiksel modellerle ilgileniyoruz.
Basit modeller, uygun şekilde basit problemleri mükemmel bir şekilde çözebilirken, bu kitapta odaklandığımız problemler klasik yöntemlerin sınırlarını aşmaktadır.
Derin öğrenme, klasik yaklaşımlardan, esas olarak odaklandığı güçlü modeller kümesi ile ayrılır.
Bu modeller, yukarıdan aşağıya zincirlenmiş verilerin art arda dönüşümlerinden oluşur, bu nedenle adları *derin öğrenme*.
Derin sinir ağlarını tartışırken, bazı geleneksel yöntemleri de tartışacağız.

### Amaç işlevleri

Daha önce, makine öğrenmesini "deneyimden öğrenme" olarak tanıttık.
Burada *öğrenme* ile zamanla bazı görevlerde *iyileştirme* yi kastediyoruz.
Peki kim neyin bir iyileştirme oluşturduğunu söyleyecek?
Modelimizi güncellemeyi önerebileceğimizi düşünebilirsiniz ve bazı insanlar önerilen güncellemenin bir iyileştirme mi yoksa bir düşüş mü oluşturduğuna katılmayabilir.

Resmi bir matematiksel öğrenme makinesi sistemi geliştirmek için modellerimizin ne kadar iyi (ya da kötü) olduğuna dair kurallı ölçümlere ihtiyacımız var.
Makine öğrenmesi ve daha genel olarak optimizasyonda (eniyilemede), bunları amaç işlevleri olarak adlandırıyoruz.
Yaygın kanı olarak, genellikle objektif fonksiyonları tanımlarız, böylece *daha alt* *daha iyi* olur.
Bu sadece bir yaygın kanı. Daha yüksekken daha iyi olan herhangi bir $f$ işlevini alabilir ve $f'$ işlevini, niteliksel olarak özdeş $f' = -f$ şekilde ayarlayarak daha düşükken daha iyi yeni bir işleve dönüştürebilirsiniz.
Düşük daha iyi olduğu için, bu işlevlere bazen *yitim işlevleri* veya *maliyet işlevleri* denir.

Sayısal değerleri tahmin etmeye çalışırken, en yaygın amaç fonksiyonu hata karesi $(y-\hat{y})^2$'dır.
Sınıflandırma için en yaygın amaç fonksiyonu, hata oranını, yani tahminlerimizin gerçeğe değere uymadığı örneklerin oranını, en aza indirmektir.
Bazı hedeflerin (hata karesi gibi) optimize edilmesi kolaydır.
Diğerlerinin (hata oranı gibi) türevlerinin alınamaması veya diğer başka zorluklar nedeniyle doğrudan optimize edilmesi zordur.
Bu durumlarda, *vekil amaç* optimize etmek yaygındır.

Tipik olarak, yitim fonksiyonu modelin parametrelerine göre tanımlanır ve veri kümesine bağlıdır.
Modelimizin parametrelerinin en iyi değerleri, eğitim için toplanan *örneklerden* oluşan bir *eğitim kümesinde* meydana gelen kaybı en aza indirerek öğrenilir.
Bununla birlikte, eğitim verilerinde iyi performans gösterilmesi, (görülmeyen) test verileri üzerinde iyi performans göstereceğimizi garanti etmez.
Bu nedenle, genellikle mevcut verileri iki parçaya ayırmak isteyeceğiz: Eğitim verileri (model parametrelerini bulmak için) ve test verileri (değerlendirme için tutulan), ayrıca aşağıdaki iki sonucu rapor edeceğiz:

* ** Eğitim Hatası: **
Modelin eğitildiği verilerdeki hatadır.
Bunu, bir öğrencinin gerçek bir sınava hazırlamak için girdiği uygulama sınavlarındaki puanları gibi düşünebilirsiniz.
Sonuçlar cesaret verici olsa bile, bu final sınavında başarıyı garanti etmez.
* ** Test Hatası: **
Bu, görünmeyen bir test kümesinde oluşan hatadır.
Bu, eğitim hatasından önemli ölçüde sapabilir.
Bir model eğitim verileri üzerinde iyi performans gösterdiğinde, ancak bunu görünmeyen verilere genelleştiremediğinde, buna *aşırı öğrenme* diyoruz.
Gerçek yaşamda, bu, uygulama sınavlarında başarılı olunmasına rağmen gerçek sınavda çakmak gibidir.

### Optimizasyon (Eniyileme) algoritmaları

Bir kez veri kaynağı ve gösterim, bir model ve iyi tanımlanmış bir amaç fonksiyona sahip olduktan sonra, yitim fonksiyonunu en aza indirmek için mümkün olan en iyi parametreleri arayabilen bir algoritmaya ihtiyacımız var.
Sinir ağları için en popüler optimizasyon algoritmaları, gradyan (eğim) alçaltma olarak adlandırılan bir yaklaşımı izler. Kısacası, her adımda, her bir parametre için, bu parametreyi sadece küçük bir miktar bozarsanız eğitim kümesi kaybının nasıl hareket edeceğini (değişeceğini) kontrol ederler.
Daha sonra parametreyi kaybı azaltan yönde güncellerler.

## Makine Öğrenmesi Çeşitleri

Aşağıdaki bölümlerde, birkaç *çeşit* makine öğrenmesi problemini daha ayrıntılı olarak tartışacağız.
*Hedeflerin* bir listesiyle, yani makine öğrenmesinin yapmasını istediğimiz şeylerin bir listesiyle başlıyoruz.
Hedeflerin, veri türleri, modeller, eğitim teknikleri vb. dahil olmak üzere, *nasıl*  başarılabileceğine dair bir dizi teknik ile tamamlandığını unutmayın.
Aşağıdaki liste, okuyucuyu motive etmek ve kitap boyunca daha fazla sorun hakkında konuştuğumuzda bize ortak bir dil sağlamak için MÖ'nün uğraşabileceği sorunların sadece bir örneğidir.

### Gözetimli öğrenme

Gözetimli öğrenme *girdiler* verildiğinde *hedefleri* tahmin etme görevini ele alır.
Sık sık *etiket* adını verdiğimiz hedefler genellikle *y* ile gösterilir.
*Öznitellikler* veya eş değişkenler olarak da adlandırılan girdi verilerini genellikle $\mathbf{x}$ olarak belirtiriz.
Her (girdi, hedef) çiftine *örnek* veya *misal* denir.
Bazen, bağlam açık olduğunda, bir girdi topluluğuna atıfta bulunmak için örnekler terimini kullanabiliriz,
karşılık gelen hedefler bilinmese bile.
Belirli bir örneği, mesela $i$, bir altindis ile gösteririz, örneğin ($\mathbf{x}_i,y_i$).
Veri kümesi, $n$ taneli örnekli, $\{\mathbf{x}_i, y_i\}_{i=1}^n$, bir topluluktur.
Hedefimiz, $\mathbf{x}_i$ girdisini $f_{\theta}(\mathbf{x}_i)$ tahminiyle eşleyen bir $f_\theta$ modeli üretmektir.

Bu açıklamayı somut bir örneğe oturtalım; sağlık hizmetlerinde çalışıyorsak, bir hastanın kalp krizi geçirip geçirmeyeceğini tahmin etmek isteyebiliriz.
Bu gözlem, *kalp krizi* veya *kalp krizi yok*, $y$ etiketimiz olacaktır.
$\mathbf{x}$ girdi verileri, kalp atış hızı, diyastolik ve sistolik kan basıncı gibi hayati belirtiler olabilir.

Burada gözetim devreye girer, çünkü $\theta$ parametrelerini seçmek için, biz (gözetimciler) modele *etiketli örnekleri* ($\mathbf{x}_i,y_i$) içeren bir veri kümesi sağlıyoruz, ki bu kümedeki her örnek, $\mathbf{x}_i$, doğru etiketle eşleştirilmiştir.

Olasılıksal terimlerle, tipik olarak koşullu olasılığı, $P(y|x)$, tahmin etmekle ilgileniyoruz.
Makine öğrenmesi içindeki birçok paradigmadan sadece biri olsa da, gözetimli öğrenme, makine öğrenmesinin endüstrideki başarılı uygulamalarının çoğunu oluşturur.
Kısmen, bunun nedeni, birçok önemli görevin, belirli bir mevcut veri kümesi göz önüne alındığında bilinmeyen bir şeyin olasılığını tahmin etmek gibi net bir şekilde tanımlanabilmesidir; örneğin:

* CT görüntüsü verildiğinde kansere karşı kanser olmama tahmini.
* İngilizce bir cümle verildiğinde Fransızca doğru çevirisinin tahmini.
* Bir  hisse senedinin gelecek aydaki fiyatının bu ayın finansal raporlama verilerine dayalı tahmini.

Basit bir tanımla bile "girdilerden hedefleri tahmin et" diye tanımladığımız gözetimli öğrenme çok çeşitli şekillerde olabilir, (diğer hususların yanı sıra) girdilerin ve çıktıların türüne, boyutuna ve sayısına bağlı olarak çok sayıda modelleme kararı gerektirebilir.
Örneğin, dizileri işlemek için (metin dizeleri veya zaman serisi verileri gibi) farklı ve sabit uzunluklu vektör temsillerini işlemek için farklı modeller kullanırız.
Bu kitabın ilk 9 bölümünde bu sorunların birçoğunu derinlemesine ziyaret edeceğiz.

Gayri resmi olarak, öğrenme süreci şöyle görünür:
Ortak değişkenlerin bilindiği büyük bir örnek koleksiyonu alın ve her biri için doğru değer etiketlerini alarak rastgele bir altküme seçin.
Bazen bu etiketler zaten toplanmış verilerde olabilir (örn. Bir hasta sonraki yıl içinde öldü mü?) ve diğer zamanlarda verileri etiketlemek için insanların yorumlamalarını kullanmamız gerekebilir (örn. Görüntülere kategori atama).

Bu girdiler ve karşılık gelen etiketler birlikte eğitim kümesini oluştururlar.
Eğitim veri kümesini gözetimli bir öğrenme algoritmasına besleriz; veri kümesini girdi olarak alır ve başka bir işlev, *öğrenilen model*, verir.
Son olarak, çıktılarını karşılık gelen etiketin tahminleri olarak kullanarak önceden görülmemiş girdileri öğrenilen modele besleyebiliriz.
Tüm süreç şöyle çizilebilir :numref:`fig_supervised_learning`.

![Gözetimli öğrenme.](../img/supervised-learning.svg)
:label:`fig_supervised_learning`

#### Bağlanım

Belki de kafanıza sokmak için en basit gözetimli öğrenme görevi *bağlanım*'dır.
Örneğin, ev satışları veritabanından toplanan bir veri kümesini düşünün.
Her sıranın farklı bir eve karşılık geldiği bir tablo oluşturabiliriz ve her sütun, bir evin alanı, yatak odası sayısı, banyo sayısı ve şehir merkezine (yürüyüş) dakika sayısı  gibi ilgili bazı özelliklere karşılık gelir.
Bu veri kümesinde, her *örnek* belirli bir ev olacaktır ve karşılık gelen *öznitelik vektörü* tabloda bir satır olacaktır.

If you live in New York or San Francisco,
and you are not the CEO of Amazon, Google, Microsoft, or Facebook,
the (sq. footage, no. of bedrooms, no. of bathrooms, walking distance)
feature vector for your home might look something like: $[100, 0, .5, 60]$.
However, if you live in Pittsburgh, it might look more like $[3000, 4, 3, 10]$.
Feature vectors like this are essential
for most classic machine learning algorithms.
We will continue to denote the feature vector corresponding
to any example $i$ as $\mathbf{x}_i$ and we can compactly refer
to the full table containing all of the feature vectors as $X$.

What makes a problem a *regression* is actually the outputs.
Say that you are in the market for a new home.
You might want to estimate the fair market value of a house,
given some features like these.
The target value, the price of sale, is a *real number*.
If you remember the formal definition of the reals
you might be scratching your head now.
Homes probably never sell for fractions of a cent,
let alone prices expressed as irrational numbers.
In cases like this, when the target is actually discrete,
but where the rounding takes place on a sufficiently fine scale,
we will abuse language just a bit and continue to describe
our outputs and targets as real-valued numbers.


We denote any individual target $y_i$
(corresponding to example $\mathbf{x}_i$)
and the set of all targets $\mathbf{y}$
(corresponding to all examples $X$).
When our targets take on arbitrary values in some range,
we call this a regression problem.
Our goal is to produce a model whose predictions
closely approximate the actual target values.
We denote the predicted target for any instance $\hat{y}_i$.
Do not worry if the notation is bogging you down.
We will unpack it more thoroughly in the subsequent chapters.


Lots of practical problems are well-described regression problems.
Predicting the rating that a user will assign to a movie
can be thought of as a regression problem
and if you designed a great algorithm to accomplish this feat in 2009,
you might have won the [1-million-dollar Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize).
Predicting the length of stay for patients in the hospital
is also a regression problem.
A good rule of thumb is that any *How much?* or *How many?* problem
should suggest regression.

* "How many hours will this surgery take?": *regression*
* "How many dogs are in this photo?": *regression*.

However, if you can easily pose your problem as "Is this a _ ?",
then it is likely, classification, a different kind
of supervised problem that we will cover next.
Even if you have never worked with machine learning before,
you have probably worked through a regression problem informally.
Imagine, for example, that you had your drains repaired
and that your contractor spent $x_1=3$ hours
removing gunk from your sewage pipes.
Then she sent you a bill of $y_1 = \$350$.
Now imagine that your friend hired the same contractor for $x_2 = 2$ hours
and that she received a bill of $y_2 = \$250$.
If someone then asked you how much to expect
on their upcoming gunk-removal invoice
you might make some reasonable assumptions,
such as more hours worked costs more dollars.
You might also assume that there is some base charge
and that the contractor then charges per hour.
If these assumptions held true, then given these two data points,
you could already identify the contractor's pricing structure:
\$100 per hour plus \$50 to show up at your house.
If you followed that much then you already understand
the high-level idea behind linear regression
(and you just implicitly designed a linear model with a bias term).

In this case, we could produce the parameters
that exactly matched the contractor's prices.
Sometimes that is not possible, e.g., if some of
the variance owes to some factors besides your two features.
In these cases, we will try to learn models
that minimize the distance between our predictions and the observed values.
In most of our chapters, we will focus on one of two very common losses,
the L1 loss
where

$$l(y, y') = \sum_i |y_i-y_i'|$$

and the least mean squares loss, or
L2 loss
where

$$l(y, y') = \sum_i (y_i - y_i')^2.$$

As we will see later, the $L_2$ loss corresponds to the assumption
that our data was corrupted by Gaussian noise,
whereas the $L_1$ loss corresponds to an assumption
of noise from a Laplace distribution.

#### Classification

While regression models are great for addressing *how many?* questions,
lots of problems do not bend comfortably to this template.
For example, a bank wants to add check scanning to its mobile app.
This would involve the customer snapping a photo of a check
with their smart phone's camera
and the machine learning model would need to be able
to automatically understand text seen in the image.
It would also need to understand hand-written text to be even more robust.
This kind of system is referred to as optical character recognition (OCR),
and the kind of problem it addresses is called *classification*.
It is treated with a different set of algorithms
than those used for regression (although many techniques will carry over).

In classification, we want our model to look at a feature vector,
e.g., the pixel values in an image,
and then predict which category (formally called *classes*),
among some (discrete) set of options, an example belongs.
For hand-written digits, we might have 10 classes,
corresponding to the digits 0 through 9.
The simplest form of classification is when there are only two classes,
a problem which we call binary classification.
For example, our dataset $X$ could consist of images of animals
and our *labels* $Y$ might be the classes $\mathrm{\{cat, dog\}}$.
While in regression, we sought a *regressor* to output a real value $\hat{y}$,
in classification, we seek a *classifier*, whose output $\hat{y}$ is the predicted class assignment.

For reasons that we will get into as the book gets more technical,
it can be hard to optimize a model that can only output
a hard categorical assignment, e.g., either *cat* or *dog*.
In these cases, it is usually much easier to instead express
our model in the language of probabilities.
Given an example $x$, our model assigns a probability $\hat{y}_k$
to each label $k$. Because these are probabilities,
they need to be positive numbers and add up to $1$
and thus we only need $K-1$ numbers
to assign probabilities of $K$ categories.
This is easy to see for binary classification.
If there is a $0.6$ ($60\%$) probability that an unfair coin comes up heads,
then there is a $0.4$ ($40\%$) probability that it comes up tails.
Returning to our animal classification example,
a classifier might see an image and output the probability
that the image is a cat $P(y=\text{cat} \mid x) = 0.9$.
We can interpret this number by saying that the classifier
is $90\%$ sure that the image depicts a cat.
The magnitude of the probability for the predicted class
conveys one notion of uncertainty.
It is not the only notion of uncertainty
and we will discuss others in more advanced chapters.

When we have more than two possible classes,
we call the problem *multiclass classification*.
Common examples include hand-written character recognition
`[0, 1, 2, 3 ... 9, a, b, c, ...]`.
While we attacked regression problems by trying
to minimize the L1 or L2 loss functions,
the common loss function for classification problems is called cross-entropy.

Note that the most likely class is not necessarily
the one that you are going to use for your decision.
Assume that you find this beautiful mushroom in your backyard
as shown in :numref:`fig_death_cap`.

![Death cap---do not eat!](../img/death_cap.jpg)
:width:`200px`
:label:`fig_death_cap`

Now, assume that you built a classifier and trained it
to predict if a mushroom is poisonous based on a photograph.
Say our poison-detection classifier outputs
$P(y=\mathrm{death cap}|\mathrm{image}) = 0.2$.
In other words, the classifier is $80\%$ sure
that our mushroom *is not* a death cap.
Still, you'd have to be a fool to eat it.
That is because the certain benefit of a delicious dinner
is not worth a $20\%$ risk of dying from it.
In other words, the effect of the *uncertain risk*
outweighs the benefit by far. We can look at this more formally.
Basically, we need to compute the expected risk that we incur,
i.e., we need to multiply the probability of the outcome
with the benefit (or harm) associated with it:

$$L(\mathrm{action}| x) = E_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)].$$

Hence, the loss $L$ incurred by eating the mushroom
is $L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty$,
whereas the cost of discarding it is
$L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$.

Our caution was justified: as any mycologist would tell us,
the above mushroom actually *is* a death cap.
Classification can get much more complicated than just
binary, multiclass, or even multi-label classification.
For instance, there are some variants of classification
for addressing hierarchies.
Hierarchies assume that there exist some relationships among the many classes.
So not all errors are equal---if we must err, we would prefer
to misclassify to a related class rather than to a distant class.
Usually, this is referred to as *hierarchical classification*.
One early example is due to [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus), who organized the animals in a hierarchy.

In the case of animal classification,
it might not be so bad to mistake a poodle for a schnauzer,
but our model would pay a huge penalty
if it confused a poodle for a dinosaur.
Which hierarchy is relevant might depend
on how you plan to use the model.
For example, rattle snakes and garter snakes
might be close on the phylogenetic tree,
but mistaking a rattler for a garter could be deadly.

#### Tagging

Some classification problems do not fit neatly
into the binary or multiclass classification setups.
For example, we could train a normal binary classifier
to distinguish cats from dogs.
Given the current state of computer vision,
we can do this easily, with off-the-shelf tools.
Nonetheless, no matter how accurate our model gets,
we might find ourselves in trouble when the classifier
encounters an image of the Town Musicians of Bremen.

![A cat, a rooster, a dog and a donkey](../img/stackedanimals.jpg)
:width:`300px`


As you can see, there is a cat in the picture,
and a rooster, a dog, a donkey, and a bird,
with some trees in the background.
Depending on what we want to do with our model
ultimately, treating this as a binary classification problem
might not make a lot of sense.
Instead, we might want to give the model the option of
saying the image depicts a cat *and* a dog *and* a donkey
*and* a rooster *and* a bird.

The problem of learning to predict classes that are
*not mutually exclusive* is called multi-label classification.
Auto-tagging problems are typically best described
as multi-label classification problems.
Think of the tags people might apply to posts on a tech blog,
e.g., "machine learning", "technology", "gadgets",
"programming languages", "linux", "cloud computing", "AWS".
A typical article might have 5-10 tags applied
because these concepts are correlated.
Posts about "cloud computing" are likely to mention "AWS"
and posts about "machine learning" could also deal
with "programming languages".

We also have to deal with this kind of problem when dealing
with the biomedical literature, where correctly tagging articles is important
because it allows researchers to do exhaustive reviews of the literature.
At the National Library of Medicine, a number of professional annotators
go over each article that gets indexed in PubMed
to associate it with the relevant terms from MeSH,
a collection of roughly 28k tags.
This is a time-consuming process and the
annotators typically have a one year lag between archiving and tagging.
Machine learning can be used here to provide provisional tags
until each article can have a proper manual review.
Indeed, for several years, the BioASQ organization
has [hosted a competition](http://bioasq.org/) to do precisely this.


#### Search and ranking

Sometimes we do not just want to assign each example to a bucket
or to a real value. In the field of information retrieval,
we want to impose a ranking on a set of items.
Take web search for example, the goal is less to determine whether
a particular page is relevant for a query, but rather,
which one of the plethora of search results is *most relevant*
for a particular user.
We really care about the ordering of the relevant search results
and our learning algorithm needs to produce ordered subsets
of elements from a larger set.
In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference
between returning ``A B C D E`` and ``C A B E D``.
Even if the result set is the same,
the ordering within the set matters.

One possible solution to this problem is to first assign
to every element in the set a corresponding relevance score
and then to retrieve the top-rated elements.
[PageRank](https://en.wikipedia.org/wiki/PageRank),
the original secret sauce behind the Google search engine
was an early example of such a scoring system but it was
peculiar in that it did not depend on the actual query.
Here they relied on a simple relevance filter
to identify the set of relevant items
and then on PageRank to order those results
that contained the query term.
Nowadays, search engines use machine learning and behavioral models
to obtain query-dependent relevance scores.
There are entire academic conferences devoted to this subject.



#### Recommender systems
:label:`subsec_recommender_systems`

Recommender systems are another problem setting
that is related to search and ranking.
The problems are similar insofar as the goal
is to display a set of relevant items to the user.
The main difference is the emphasis on *personalization*
to specific users in the context of recommender systems.
For instance, for movie recommendations,
the results page for a SciFi fan and the results page
for a connoisseur of Peter Sellers comedies might differ significantly.
Similar problems pop up in other recommendation settings,
e.g., for retail products, music, or news recommendation.

In some cases, customers provide explicit feedback communicating
how much they liked a particular product
(e.g., the product ratings and reviews on Amazon, IMDB, GoodReads, etc.).
In some other cases, they provide implicit feedback,
e.g., by skipping titles on a playlist,
which might indicate dissatisfaction but might just indicate
that the song was inappropriate in context.
In the simplest formulations, these systems are trained
to estimate some score $y_{ij}$, such as an estimated rating
or the probability of purchase, given a user $u_i$ and product $p_j$.

Given such a model, then for any given user,
we could retrieve the set of objects with the largest scores $y_{ij}$,
which could then be recommended to the customer.
Production systems are considerably more advanced and take
detailed user activity and item characteristics into account
when computing such scores. :numref:`fig_deeplearning_amazon` is an example
of deep learning books recommended by Amazon based on personalization algorithms tuned to capture the author's preferences.

![Deep learning books recommended by Amazon.](../img/deeplearning_amazon.png)
:label:`fig_deeplearning_amazon`

Despite their tremendous economic value, recommendation systems
naively built on top of predictive models
suffer some serious conceptual flaws.
To start, we only observe *censored feedback*.
Users preferentially rate movies that they feel strongly about:
you might notice that items receive many 5 and 1 star ratings
but that there are conspicuously few 3-star ratings.
Moreover, current purchase habits are often a result
of the recommendation algorithm currently in place,
but learning algorithms do not always take this detail into account.
Thus it is possible for feedback loops to form
where a recommender system preferentially pushes an item
that is then taken to be better (due to greater purchases)
and in turn is recommended even more frequently.
Many of these problems about how to deal with censoring,
incentives, and feedback loops, are important open research questions.

#### Sequence Learning

So far, we have looked at problems where we have
some fixed number of inputs and produce a fixed number of outputs.
Before we considered predicting home prices from a fixed set of features: square footage, number of bedrooms,
number of bathrooms, walking time to downtown.
We also discussed mapping from an image (of fixed dimension)
to the predicted probabilities that it belongs to each
of a fixed number of classes, or taking a user ID and a product ID,
and predicting a star rating. In these cases,
once we feed our fixed-length input
into the model to generate an output,
the model immediately forgets what it just saw.

This might be fine if our inputs truly all have the same dimensions
and if successive inputs truly have nothing to do with each other.
But how would we deal with video snippets?
In this case, each snippet might consist of a different number of frames.
And our guess of what is going on in each frame might be much stronger
if we take into account the previous or succeeding frames.
Same goes for language. One popular deep learning problem
is machine translation: the task of ingesting sentences
in some source language and predicting their translation in another language.

These problems also occur in medicine.
We might want a model to monitor patients in the intensive care unit
and to fire off alerts if their risk of death
in the next 24 hours exceeds some threshold.
We definitely would not want this model to throw away
everything it knows about the patient history each hour
and just make its predictions based on the most recent measurements.

These problems are among the most exciting applications of machine learning
and they are instances of *sequence learning*.
They require a model to either ingest sequences of inputs
or to emit sequences of outputs (or both!).
These latter problems are sometimes referred to as ``seq2seq`` problems.  Language translation is a ``seq2seq`` problem.
Transcribing text from the spoken speech is also a ``seq2seq`` problem.
While it is impossible to consider all types of sequence transformations,
a number of special cases are worth mentioning:

**Tagging and Parsing**. This involves annotating a text sequence with attributes.
In other words, the number of inputs and outputs is essentially the same.
For instance, we might want to know where the verbs and subjects are.
Alternatively, we might want to know which words are the named entities.
In general, the goal is to decompose and annotate text based on structural
and grammatical assumptions to get some annotation.
This sounds more complex than it actually is.
Below is a very simple example of annotating a sentence
with tags indicating which words refer to named entities.

```text
Tom has dinner in Washington with Sally.
Ent  -    -    -     Ent      -    Ent
```


**Automatic Speech Recognition**. With speech recognition, the input sequence $x$
is an audio recording of a speaker (shown in :numref:`fig_speech`), and the output $y$
is the textual transcript of what the speaker said.
The challenge is that there are many more audio frames
(sound is typically sampled at 8kHz or 16kHz)
than text, i.e., there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word.
These are ``seq2seq`` problems where the output is much shorter than the input.

![`-D-e-e-p- L-ea-r-ni-ng-`](../img/speech.png)
:width:`700px`
:label:`fig_speech`

**Text to Speech**. Text-to-Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text
and the output $y$ is an audio file.
In this case, the output is *much longer* than the input.
While it is easy for *humans* to recognize a bad audio file,
this is not quite so trivial for computers.

**Machine Translation**. Unlike the case of speech recognition, where corresponding
inputs and outputs occur in the same order (after alignment),
in machine translation, order inversion can be vital.
In other words, while we are still converting one sequence into another,
neither the number of inputs and outputs nor the order
of corresponding data points are assumed to be the same.
Consider the following illustrative example
of the peculiar tendency of Germans
to place the verbs at the end of sentences.

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Did you already check out this excellent tutorial?
Wrong alignment:  Did you yourself already this excellent tutorial looked-at?
```


Many related problems pop up in other learning tasks.
For instance, determining the order in which a user
reads a Webpage is a two-dimensional layout analysis problem.
Dialogue problems exhibit all kinds of additional complications,
where determining what to say next requires taking into account
real-world knowledge and the prior state of the conversation
across long temporal distances. This is an active area of research.


### Unsupervised learning

All the examples so far were related to *Supervised Learning*,
i.e., situations where we feed the model a giant dataset
containing both the features and corresponding target values.
You could think of the supervised learner as having
an extremely specialized job and an extremely anal boss.
The boss stands over your shoulder and tells you exactly what to do
in every situation until you learn to map from situations to actions.
Working for such a boss sounds pretty lame.
On the other hand, it is easy to please this boss.
You just recognize the pattern as quickly as possible
and imitate their actions.

In a completely opposite way, it could be frustrating
to work for a boss who has no idea what they want you to do.
However, if you plan to be a data scientist, you'd better get used to it.
The boss might just hand you a giant dump of data and tell you to *do some data science with it!* This sounds vague because it is.
We call this class of problems *unsupervised learning*,
and the type and number of questions we could ask
is limited only by our creativity.
We will address a number of unsupervised learning techniques
in later chapters. To whet your appetite for now,
we describe a few of the questions you might ask:

* Can we find a small number of prototypes
that accurately summarize the data?
Given a set of photos, can we group them into landscape photos,
pictures of dogs, babies, cats, mountain peaks, etc.?
Likewise, given a collection of users' browsing activity,
can we group them into users with similar behavior?
This problem is typically known as *clustering*.
* Can we find a small number of parameters
that accurately capture the relevant properties of the data?
The trajectories of a ball are quite well described
by velocity, diameter, and mass of the ball.
Tailors have developed a small number of parameters
that describe human body shape fairly accurately
for the purpose of fitting clothes.
These problems are referred to as *subspace estimation* problems.
If the dependence is linear, it is called *principal component analysis*.
* Is there a representation of (arbitrarily structured) objects
in Euclidean space (i.e., the space of vectors in $\mathbb{R}^n$)
such that symbolic properties can be well matched?
This is called *representation learning* and it is used
to describe entities and their relations,
such as Rome $-$ Italy $+$ France $=$ Paris.
* Is there a description of the root causes
of much of the data that we observe?
For instance, if we have demographic data
about house prices, pollution, crime, location,
education, salaries, etc., can we discover
how they are related simply based on empirical data?
The fields concerned with *causality* and
*probabilistic graphical models* address this problem.
* Another important and exciting recent development in unsupervised learning
is the advent of *generative adversarial networks* (GANs).
These give us a procedural way to synthesize data,
even complicated structured data like images and audio.
The underlying statistical mechanisms are tests
to check whether real and fake data are the same.
We will devote a few notebooks to them.


### Interacting with an Environment

So far, we have not discussed where data actually comes from,
or what actually *happens* when a machine learning model generates an output.
That is because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data upfront,
then set our pattern recognition machines in motion
without ever interacting with the environment again.
Because all of the learning takes place
after the algorithm is disconnected from the environment,
this is sometimes called *offline learning*.
For supervised learning, the process looks like :numref:`fig_data_collection`.

![Collect data for supervised learning from an environment.](../img/data-collection.svg)
:label:`fig_data_collection`

This simplicity of offline learning has its charms.
The upside is we can worry about pattern recognition
in isolation, without any distraction from these other problems.
But the downside is that the problem formulation is quite limiting.
If you are more ambitious, or if you grew up reading Asimov's Robot Series,
then you might imagine artificially intelligent bots capable
not only of making predictions, but of taking actions in the world.
We want to think about intelligent *agents*, not just predictive *models*.
That means we need to think about choosing *actions*,
not just making *predictions*. Moreover, unlike predictions,
actions actually impact the environment.
If we want to train an intelligent agent,
we must account for the way its actions might
impact the future observations of the agent.


Considering the interaction with an environment
opens a whole set of new modeling questions.
Does the environment:

* Remember what we did previously?
* Want to help us, e.g., a user reading text into a speech recognizer?
* Want to beat us, i.e., an adversarial setting like spam filtering (against spammers) or playing a game (vs an opponent)?
* Not care (as in many cases)?
* Have shifting dynamics (does future data always resemble the past or do the patterns change over time, either naturally or in response to our automated tools)?

This last question raises the problem of *distribution shift*,
(when training and test data are different).
It is a problem that most of us have experienced
when taking exams written by a lecturer,
while the homeworks were composed by her TAs.
We will briefly describe reinforcement learning and adversarial learning,
two settings that explicitly consider interaction with an environment.


### Reinforcement learning

If you are interested in using machine learning
to develop an agent that interacts with an environment
and takes actions, then you are probably going to wind up
focusing on *reinforcement learning* (RL).
This might include applications to robotics,
to dialogue systems, and even to developing AI for video games.
*Deep reinforcement learning* (DRL), which applies
deep neural networks to RL problems, has surged in popularity.
The breakthrough [deep Q-network that beat humans at Atari games using only the visual input](https://www.wired.com/2015/02/google-ai-plays-atari-like-pros/),
and the [AlphaGo program that dethroned the world champion at the board game Go](https://www.wired.com/2017/05/googles-alphago-trounces-humans-also-gives-boost/) are two prominent examples.

Reinforcement learning gives a very general statement of a problem,
in which an agent interacts with an environment over a series of *timesteps*.
At each timestep $t$, the agent receives some observation $o_t$
from the environment and must choose an action $a_t$
that is subsequently transmitted back to the environment
via some mechanism (sometimes called an actuator).
Finally, the agent receives a reward $r_t$ from the environment.
The agent then receives a subsequent observation,
and chooses a subsequent action, and so on.
The behavior of an RL agent is governed by a *policy*.
In short, a *policy* is just a function that maps
from observations (of the environment) to actions.
The goal of reinforcement learning is to produce a good policy.

![The interaction between reinforcement learning and an environment.](../img/rl-environment.svg)

It is hard to overstate the generality of the RL framework.
For example, we can cast any supervised learning problem as an RL problem.
Say we had a classification problem.
We could create an RL agent with one *action* corresponding to each class.
We could then create an environment which gave a reward
that was exactly equal to the loss function
from the original supervised problem.

That being said, RL can also address many problems
that supervised learning cannot.
For example, in supervised learning we always expect
that the training input comes associated with the correct label.
But in RL, we do not assume that for each observation,
the environment tells us the optimal action.
In general, we just get some reward.
Moreover, the environment may not even tell us which actions led to the reward.

Consider for example the game of chess.
The only real reward signal comes at the end of the game
when we either win, which we might assign a reward of 1,
or when we lose, which we could assign a reward of -1.
So reinforcement learners must deal with the *credit assignment problem*:
determining which actions to credit or blame for an outcome.
The same goes for an employee who gets a promotion on October 11.
That promotion likely reflects a large number
of well-chosen actions over the previous year.
Getting more promotions in the future requires figuring out
what actions along the way led to the promotion.

Reinforcement learners may also have to deal
with the problem of partial observability.
That is, the current observation might not
tell you everything about your current state.
Say a cleaning robot found itself trapped
in one of many identical closets in a house.
Inferring the precise location (and thus state) of the robot
might require considering its previous observations before entering the closet.

Finally, at any given point, reinforcement learners
might know of one good policy,
but there might be many other better policies
that the agent has never tried.
The reinforcement learner must constantly choose
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies,
potentially giving up some short-run reward in exchange for knowledge.


#### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover, not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of
*special cases* of reinforcement learning problems.

When the environment is fully observed,
we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions
with initially unknown rewards, this problem
is the classic *multi-armed bandit problem*.



## Roots

Although many deep learning methods are recent inventions,
humans have held the desire to analyze data
and to predict future outcomes for centuries.
In fact, much of natural science has its roots in this.
For instance, the Bernoulli distribution is named after
[Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli), and the Gaussian distribution was discovered
by [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss).
He invented, for instance, the least mean squares algorithm,
which is still used today for countless problems
from insurance calculations to medical diagnostics.
These tools gave rise to an experimental approach
in the natural sciences---for instance, Ohm's law
relating current and voltage in a resistor
is perfectly described by a linear model.

Even in the middle ages, mathematicians had a keen intuition of estimates.
For instance, the geometry book of [Jacob Köbel (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) illustrates
averaging the length of 16 adult men's feet to obtain the average foot length.

![Estimating the length of a foot](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`

:numref:`fig_koebel` illustrates how this estimator works.
The 16 adult men were asked to line up in a row, when leaving church.
Their aggregate length was then divided by 16
to obtain an estimate for what now amounts to 1 foot.
This "algorithm" was later improved to deal with misshapen feet---the
2 men with the shortest and longest feet respectively were sent away,
averaging only over the remainder.
This is one of the earliest examples of the trimmed mean estimate.

Statistics really took off with the collection and availability of data.
One of its titans, [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), contributed significantly to its theory
and also its applications in genetics.
Many of his algorithms (such as Linear Discriminant Analysis)
and formula (such as the Fisher Information Matrix)
are still in frequent use today (even the Iris dataset
that he released in 1936 is still used sometimes
to illustrate machine learning algorithms).
Fisher was also a proponent of eugenics,
which should remind us that the morally dubious use of data science
has as long and enduring a history as its productive use
in industry and the natural sciences.

A second influence for machine learning came from Information Theory
[(Claude Shannon, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the Theory of computation via [Alan Turing (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing).
Turing posed the question "can machines think?”
in his famous paper [Computing machinery and intelligence](https://en.wikipedia.org/wiki/Computing_Machinery_and_Intelligence) (Mind, October 1950).
In what he described as the Turing test, a machine
can be considered intelligent if it is difficult
for a human evaluator to distinguish between the replies
from a machine and a human based on textual interactions.

Another influence can be found in neuroscience and psychology.
After all, humans clearly exhibit intelligent behavior.
It is thus only reasonable to ask whether one could explain
and possibly reverse engineer this capacity.
One of the oldest algorithms inspired in this fashion
was formulated by [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).
In his groundbreaking book The Organization of Behavior :cite:`Hebb.Hebb.1949`,
he posited that neurons learn by positive reinforcement.
This became known as the Hebbian learning rule.
It is the prototype of Rosenblatt's perceptron learning algorithm
and it laid the foundations of many stochastic gradient descent algorithms
that underpin deep learning today: reinforce desirable behavior
and diminish undesirable behavior to obtain good settings
of the parameters in a neural network.

Biological inspiration is what gave *neural networks* their name.
For over a century (dating back to the models of Alexander Bain, 1873
and James Sherrington, 1890), researchers have tried to assemble
computational circuits that resemble networks of interacting neurons.
Over time, the interpretation of biology has become less literal
but the name stuck. At its heart, lie a few key principles
that can be found in most networks today:

* The alternation of linear and nonlinear processing units, often referred to as *layers*.
* The use of the chain rule (also known as *backpropagation*) for adjusting parameters in the entire network at once.

After initial rapid progress, research in neural networks
languished from around 1995 until 2005.
This was due to a number of reasons.
Training a network is computationally very expensive.
While RAM was plentiful at the end of the past century,
computational power was scarce.
Second, datasets were relatively small.
In fact, Fisher's Iris dataset from 1932
was a popular tool for testing the efficacy of algorithms.
MNIST with its 60,000 handwritten digits was considered huge.

Given the scarcity of data and computation,
strong statistical tools such as Kernel Methods,
Decision Trees and Graphical Models proved empirically superior.
Unlike neural networks, they did not require weeks to train
and provided predictable results with strong theoretical guarantees.

## The Road to Deep Learning

Much of this changed with the ready availability of large amounts of data,
due to the World Wide Web, the advent of companies serving
hundreds of millions of users online, a dissemination of cheap,
high-quality sensors, cheap data storage (Kryder's law),
and cheap computation (Moore's law), in particular in the form of GPUs, originally engineered for computer gaming.
Suddenly algorithms and models that seemed computationally infeasible
became relevant (and vice versa).
This is best illustrated in :numref:`tab_intro_decade`.

:Dataset vs. computer memory and computational power

|Decade|Dataset|Memory|Floating Point Calculations per Second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (Nvidia C2050)|
|2020|1 T (social network)|100 GB|1 PF (Nvidia DGX-2)|
:label:`tab_intro_decade`

It is evident that RAM has not kept pace with the growth in data.
At the same time, the increase in computational power
has outpaced that of the data available.
This means that statistical models needed to become more memory efficient
(this is typically achieved by adding nonlinearities)
while simultaneously being able to spend more time
on optimizing these parameters, due to an increased compute budget.
Consequently, the sweet spot in machine learning and statistics
moved from (generalized) linear models and kernel methods to deep networks.
This is also one of the reasons why many of the mainstays
of deep learning, such as multilayer perceptrons
:cite:`McCulloch.Pitts.1943`, convolutional neural networks
:cite:`LeCun.Bottou.Bengio.ea.1998`, Long Short-Term Memory
:cite:`Hochreiter.Schmidhuber.1997`,
and Q-Learning :cite:`Watkins.Dayan.1992`,
were essentially "rediscovered" in the past decade,
after laying comparatively dormant for considerable time.

The recent progress in statistical models, applications, and algorithms,
has sometimes been likened to the Cambrian Explosion:
a moment of rapid progress in the evolution of species.
Indeed, the state of the art is not just a mere consequence
of available resources, applied to decades old algorithms.
Note that the list below barely scratches the surface
of the ideas that have helped researchers achieve tremendous progress
over the past decade.

* Novel methods for capacity control, such as Dropout
  :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`
  have helped to mitigate the danger of overfitting.
  This was achieved by applying noise injection :cite:`Bishop.1995`
  throughout the network, replacing weights by random variables
  for training purposes.
* Attention mechanisms solved a second problem
  that had plagued statistics for over a century:
  how to increase the memory and complexity of a system without
  increasing the number of learnable parameters.
  :cite:`Bahdanau.Cho.Bengio.2014` found an elegant solution
  by using what can only be viewed as a learnable pointer structure.
  Rather than having to remember an entire sentence, e.g.,
  for machine translation in a fixed-dimensional representation,
  all that needed to be stored was a pointer to the intermediate state
  of the translation process. This allowed for significantly
  increased accuracy for long sentences, since the model
  no longer needed to remember the entire sentence before
  commencing the generation of a new sentence.
* Multi-stage designs, e.g., via the Memory Networks (MemNets)
  :cite:`Sukhbaatar.Weston.Fergus.ea.2015` and the Neural Programmer-Interpreter :cite:`Reed.De-Freitas.2015`
  allowed statistical modelers to describe iterative approaches to reasoning. These tools allow for an internal state of the deep network
  to be modified repeatedly, thus carrying out subsequent steps
  in a chain of reasoning, similar to how a processor
  can modify memory for a computation.
* Another key development was the invention of GANs
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
  Traditionally, statistical methods for density estimation
  and generative models focused on finding proper probability distributions
  and (often approximate) algorithms for sampling from them.
  As a result, these algorithms were largely limited by the lack of
  flexibility inherent in the statistical models.
  The crucial innovation in GANs was to replace the sampler
  by an arbitrary algorithm with differentiable parameters.
  These are then adjusted in such a way that the discriminator
  (effectively a two-sample test) cannot distinguish fake from real data.
  Through the ability to use arbitrary algorithms to generate data,
  it opened up density estimation to a wide variety of techniques.
  Examples of galloping Zebras :cite:`Zhu.Park.Isola.ea.2017`
  and of fake celebrity faces :cite:`Karras.Aila.Laine.ea.2017`
  are both testimony to this progress.
  Even amateur doodlers can produce
  photorealistic images based on just sketches that describe
  how the layout of a scene looks like :cite:`Park.Liu.Wang.ea.2019`.
* In many cases, a single GPU is insufficient to process
  the large amounts of data available for training.
  Over the past decade the ability to build parallel
  distributed training algorithms has improved significantly.
  One of the key challenges in designing scalable algorithms
  is that the workhorse of deep learning optimization,
  stochastic gradient descent, relies on relatively
  small minibatches of data to be processed.
  At the same time, small batches limit the efficiency of GPUs.
  Hence, training on 1024 GPUs with a minibatch size of,
  say 32 images per batch amounts to an aggregate minibatch
  of 32k images. Recent work, first by Li :cite:`Li.2017`,
  and subsequently by :cite:`You.Gitman.Ginsburg.2017`
  and :cite:`Jia.Song.He.ea.2018` pushed the size up to 64k observations,
  reducing training time for ResNet50 on ImageNet to less than 7 minutes.
  For comparison---initially training times were measured in the order of days.
* The ability to parallelize computation has also contributed quite crucially
  to progress in reinforcement learning, at least whenever simulation is an
  option. This has led to significant progress in computers achieving
  superhuman performance in Go, Atari games, Starcraft, and in physics
  simulations (e.g., using MuJoCo). See e.g.,
  :cite:`Silver.Huang.Maddison.ea.2016` for a description
  of how to achieve this in AlphaGo. In a nutshell,
  reinforcement learning works best if plenty of (state, action, reward) triples are available, i.e., whenever it is possible to try out lots of things to learn how they relate to each
  other. Simulation provides such an avenue.
* Deep Learning frameworks have played a crucial role
  in disseminating ideas. The first generation of frameworks
  allowing for easy modeling encompassed
  [Caffe](https://github.com/BVLC/caffe),
  [Torch](https://github.com/torch), and
  [Theano](https://github.com/Theano/Theano).
  Many seminal papers were written using these tools.
  By now, they have been superseded by
  [TensorFlow](https://github.com/tensorflow/tensorflow),
  often used via its high level API [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), and [Apache MxNet](https://github.com/apache/incubator-mxnet). The third generation of tools, namely imperative tools for deep learning,
  was arguably spearheaded by [Chainer](https://github.com/chainer/chainer),
  which used a syntax similar to Python NumPy to describe models.
  This idea was adopted by both [PyTorch](https://github.com/pytorch/pytorch),
  the [Gluon API](https://github.com/apache/incubator-mxnet) of MXNet, and [Jax](https://github.com/google/jax).
  It is the latter group that this course uses to teach deep learning.

The division of labor between systems researchers building better tools
and statistical modelers building better networks
has greatly simplified things. For instance,
training a linear logistic regression model
used to be a nontrivial homework problem,
worthy to give to new machine learning
PhD students at Carnegie Mellon University in 2014.
By now, this task can be accomplished with less than 10 lines of code,
putting it firmly into the grasp of programmers.

## Success Stories

Artificial Intelligence has a long history of delivering results
that would be difficult to accomplish otherwise.
For instance, mail is sorted using optical character recognition.
These systems have been deployed since the 90s
(this is, after all, the source of the famous MNIST and USPS sets of handwritten digits).
The same applies to reading checks for bank deposits and scoring
creditworthiness of applicants.
Financial transactions are checked for fraud automatically.
This forms the backbone of many e-commerce payment systems,
such as PayPal, Stripe, AliPay, WeChat, Apple, Visa, MasterCard.
Computer programs for chess have been competitive for decades.
Machine learning feeds search, recommendation, personalization
and ranking on the Internet. In other words, artificial intelligence
and machine learning are pervasive, albeit often hidden from sight.

It is only recently that AI has been in the limelight, mostly due to
solutions to problems that were considered intractable previously.

* Intelligent assistants, such as Apple's Siri, Amazon's Alexa, or Google's
  assistant are able to answer spoken questions with a reasonable degree of
  accuracy. This includes menial tasks such as turning on light switches (a boon to the disabled) up to making barber's appointments and offering phone support dialog. This is likely the most noticeable sign that AI is affecting our lives.
* A key ingredient in digital assistants is the ability to recognize speech
  accurately. Gradually the accuracy of such systems has increased to the point
  where they reach human parity :cite:`Xiong.Wu.Alleva.ea.2018` for certain
  applications.
* Object recognition likewise has come a long way. Estimating the object in a
  picture was a fairly challenging task in 2010. On the ImageNet benchmark
  :cite:`Lin.Lv.Zhu.ea.2010` achieved a top-5 error rate of 28%. By 2017,
  :cite:`Hu.Shen.Sun.2018` reduced this error rate to 2.25%. Similarly, stunning
  results have been achieved for identifying birds, or diagnosing skin cancer.
* Games used to be a bastion of human intelligence.
  Starting from TDGammon [23], a program for playing Backgammon
  using temporal difference (TD) reinforcement learning,
  algorithmic and computational progress has led to algorithms
  for a wide range of applications. Unlike Backgammon,
  chess has a much more complex state space and set of actions.
  DeepBlue beat Garry Kasparov, Campbell et al.
  :cite:`Campbell.Hoane-Jr.Hsu.2002`, using massive parallelism,
  special purpose hardware and efficient search through the game tree.
  Go is more difficult still, due to its huge state space.
  AlphaGo reached human parity in 2015, :cite:`Silver.Huang.Maddison.ea.2016` using Deep Learning combined with Monte Carlo tree sampling.
  The challenge in Poker was that the state space is
  large and it is not fully observed (we do not know the opponents'
  cards). Libratus exceeded human performance in Poker using efficiently
  structured strategies :cite:`Brown.Sandholm.2017`.
  This illustrates the impressive progress in games
  and the fact that advanced algorithms played a crucial part in them.
* Another indication of progress in AI is the advent of self-driving cars
  and trucks. While full autonomy is not quite within reach yet,
  excellent progress has been made in this direction,
  with companies such as Tesla, NVIDIA,
  and Waymo shipping products that enable at least partial autonomy.
  What makes full autonomy so challenging is that proper driving
  requires the ability to perceive, to reason and to incorporate rules
  into a system. At present, deep learning is used primarily
  in the computer vision aspect of these problems.
  The rest is heavily tuned by engineers.

Again, the above list barely scratches the surface of where machine learning has impacted practical applications. For instance, robotics, logistics, computational biology, particle physics, and astronomy owe some of their most impressive recent advances at least in parts to machine learning. ML is thus becoming a ubiquitous tool for engineers and scientists.

Frequently, the question of the AI apocalypse, or the AI singularity
has been raised in non-technical articles on AI.
The fear is that somehow machine learning systems
will become sentient and decide independently from their programmers
(and masters) about things that directly affect the livelihood of humans.
To some extent, AI already affects the livelihood of humans
in an immediate way---creditworthiness is assessed automatically,
autopilots mostly navigate vehicles, decisions about
whether to grant bail use statistical data as input.
More frivolously, we can ask Alexa to switch on the coffee machine.

Fortunately, we are far from a sentient AI system
that is ready to manipulate its human creators (or burn their coffee).
First, AI systems are engineered, trained and deployed in a specific,
goal-oriented manner. While their behavior might give the illusion
of general intelligence, it is a combination of rules, heuristics
and statistical models that underlie the design.
Second, at present tools for *artificial general intelligence*
simply do not exist that are able to improve themselves,
reason about themselves, and that are able to modify,
extend and improve their own architecture
while trying to solve general tasks.

A much more pressing concern is how AI is being used in our daily lives.
It is likely that many menial tasks fulfilled by truck drivers
and shop assistants can and will be automated.
Farm robots will likely reduce the cost for organic farming
but they will also automate harvesting operations.
This phase of the industrial revolution
may have profound consequences on large swaths of society
(truck drivers and shop assistants are some
of the most common jobs in many states).
Furthermore, statistical models, when applied without care
can lead to racial, gender or age bias and raise
reasonable concerns about procedural fairness
if automated to drive consequential decisions.
It is important to ensure that these algorithms are used with care.
With what we know today, this strikes us a much more pressing concern
than the potential of malevolent superintelligence to destroy humanity.

## Summary

* Machine learning studies how computer systems can leverage *experience* (often data) to improve performance at specific tasks. It combines ideas from statistics, data mining, artificial intelligence, and optimization. Often, it is used as a means of implementing artificially-intelligent solutions.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. This is often accomplished by a progression of learned transformations.
* Much of the recent progress in deep learning has been triggered by an abundance of data arising from cheap sensors and Internet-scale applications, and by significant progress in computation, mostly through GPUs.
* Whole system optimization is a key component in obtaining good performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.

## Exercises

1. Which parts of code that you are currently writing could be "learned", i.e., improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
1. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using deep learning.
1. Viewing the development of artificial intelligence as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal (what is the fundamental difference)?
1. Where else can you apply the end-to-end training approach? Physics? Engineering? Econometrics?

[Discussions](https://discuss.d2l.ai/t/22)
