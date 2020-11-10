# Giriş
:label:`chap_introduction`


Yakın zamana kadar, günlük etkileşimde bulunduğumuz hemen hemen her bilgisayar programı basit prensiplerle yazılım geliştiricileri tarafından kodlandı.
Bir e-ticaret uygulaması yazmak istediğimizi varsayalım. Soruna bir beyaz tahta üzerinde birkaç saat kafa yorduktan sonra muhtemelen aşağıdaki gibi bir çözüme ulaşırız:
(i) Kullanıcılar bir web tarayıcısında veya mobil uygulamada çalışan bir arabirim aracılığıyla uygulama ile etkileşimde bulunurlar,
(ii) Uygulamamız, her kullanıcının durumunu takip etmek ve geçmiş işlemlerin kayıtlarını tutmak için ticari düzeyde bir veritabanı motoruyla etkileşime girer ve (iii) Uygulamamızın merkezindeki iş mantığı(uygulamanın beyni), farklı senaryolarda uygulamanın nasıl davranacağını belirler.


Uygulamamızın *beynini* oluşturmak için karşılaşacağımızı tahmin ettiğimiz her senaryoyu değerlendirerek uygun kuralları belirlememiz gerekir.
Bir müşteri alışveriş sepetine bir ürün eklemek için her tıkladığında, alışveriş sepeti veritabanı tablosuna bu müşterinin(kullanıcının) kimliğini istenen ürünün kimliği ile ilişkilendirerek bir kayıt ekleriz. Böyle bir programı basit prensiplerle yazabilir ve güvenle başlatabiliriz(çok az sayıda geliştirici ilk seferde tamamen doğru çalışan bir uygulama yazabilir, çoğunlukla hataları tespit etmek ve çözmek için de çalışmak gerekir). 
Genellikle yeni durumlarda, işlevsel ürün ve sistemleri yöneten/yönlendiren uygulamaları tasarlama yeteneğimiz, dikkate değer bir bilişsel başarıdır.
Ayrıca $\%100$ oranında işe yarayan çözümler tasarlayabildiğinizde, *makine öğrenmesi kullanmamalısınız*.

Giderek artan makine öğrenmesi(MÖ) uzmanı için ne mutlu ki, otomatikleştirmek istediğimiz birçok görev insan yaratıcılığına bu kadar kolay boyun eğmiyor.
Beyaz tahta etrafında bildiğiniz en akıllı zihinlerle toplandığınızı hayal edin, ancak bu sefer aşağıdaki sorunlardan birini ele alıyorsunuz:

* Coğrafi bilgi, uydu görüntüleri ve yakın bir zaman aralığındaki geçmiş hava koşulları göz önüne alındığında yarının hava durumunu tahmin eden bir program yazma.
* Serbest biçimli metinle ifade edilen bir soruyu alan ve onu doğru cevaplayan bir program yazma.
* Verilen bir fotoğrafın içerdiği tüm insanları her birinin etrafına çerçeve çizerek tanımlayabilen bir program yazma.
* Kullanıcılara internette gezinirken karşılaşma olasılıkları yüksek olmayan ancak keyif alabilecekleri ürünler sunan bir program yazma.


Bu durumların her birinde, seçkin programcılar bile çözümleri sıfırdan kodlayamazlar.
Bunun farklı nedenleri olabilir. Bazen aradığımız program zaman içinde değişen bir kalıp takip eder ve programlarımızın adapte olması gerekir.
Diğer durumlarda, ilişki (pikseller ve soyut kategoriler arasında) çok karmaşık olabilir ve bilinçli anlayışımızın ötesinde binlerce veya milyonlarca hesaplama gerekebilir (ki gözlerimiz bu görevi halihâzırda zahmetsizce yönetse bile). MÖ *deneyimlerden öğrenebilen* güçlü tekniklerin incelenmesidir.
Bir MÖ algoritmasının performansı, tipik gözlemsel veri veya bir çevre ile etkileşim şeklinde daha fazla deneyim biriktirdikçe artar.
Bunu, ne kadar deneyim kazanırsa kazansın, aynı iş mantığına göre çalışmaya devam eden(
geliştiricilerin kendileri *öğrenip* yazılımın güncellenme zamanının geldiğine karar verene kadar) deterministik (gerekirci) e-ticaret platformumuzla karşılaştırın.
Bu kitapta size makine öğrenmesinin temellerini öğreteceğiz ve özellikle de bilgisayarlı görme, doğal dil işleme, sağlık ve genomik gibi farklı alanlarda yenilikleri yönlendiren güçlü bir teknik altyapıya, yani derin öğrenmeye odaklanacağız.


## Motive Edici Bir Örnek

Bu kitabı yazmaya başlayabilmek için, birçok çalışan gibi, bol miktarda kahve tüketmemiz gerekiyordu. Arabaya bindik ve sürmeye başladık. Alex "Hey Siri" diye seslenerek iPhone'unun sesli asistan sistemini uyandırdı ve "Blue Bottle kafesine yol tarifi" komutunu verdi. Telefon komutun metnini (transkripsiyonunu) hızlı bir şekilde gösterdi. Ayrıca yol tarifini istediğimizi fark etti ve talebimizi yerine getirmek için Maps uygulamasını başlattı.
Maps uygulaması bir dizi rota belirledi, her rotanın yanında tahmini bir varış süresi de gösterdi. Bu hikaye, bir akıllı telefondaki günlük etkileşimlerimizin saniyeler içinde birkaç makine öğrenmesi modeliyle işbirligi yaptığını gösteriyor.


"Alexa", "OK, Google" veya "Hey Siri" gibi bir *uyandırma kelimesine* yanıt vermek için bir program yazdığınızı düşünün.
Bir odada kendiniz bir bilgisayar ve kod editöründen başka bir şey olmadan kodlamayı deneyin :numref:`fig_wake_word`.
Böyle bir programı basit ilkelerle (prensiplerle) nasıl yazarsınız?
Bir düşünün ... problem zor.
Mikrofon her saniye yaklaşık 44.000 örnek toplayacaktır.
Her örnek, ses dalgasının genliğinin bir ölçümüdür.
Hangi kural güvenilir bir şekilde, ses parçasının uyandırma sözcüğünü içerip içermediğine bağlı olarak bir ham ses parçasından emin ``{evet, hayır}`` tahminlerine eşleme yapabilir?
Cevabı bulmakta zorlanıyorsanız endişelenmeyin.
Böyle bir programı nasıl sıfırdan yazacağımızı bilmiyoruz.
Bu yüzden MÖ kullanıyoruz.


![Bir uyandırma kelimesi tanıma. ](../img/wake-word.svg)
:label:`fig_wake_word`

Olayın özünü şöyle açıklayabiliriz.
Çoğu zaman, bir bilgisayara girdilerle çıktıları nasıl eşleştirebileceğini açıklayamayı bilmediğimizde bile, kendimiz bu bilişsel başarıyı gerçekleştirebiliyoruz.
Diğer bir deyişle, "Alexa" kelimesini tanımak için *bir bilgisayarı nasıl programlayacağınızı* bilmeseniz bile siz *kendiniz* "Alexa" kelimesini tanıyabilirsiniz.
Bu yetenekle donanmış bizler ses örnekleri içeren büyük bir *veri kümesi* toplayabilir ve uyandırma kelimesi *içerenleri* ve *içermeyenleri* etiketleyebiliriz.
MÖ yaklaşımında, uyandırma kelimelerini tanımak için *açıktan* bir sistem tasarlamaya çalışmayız.
Bunun yerine, davranışı bir miktar *parametre* ile belirlenen esnek bir program tanımlarız.
Ardından, veri kümesini, ilgili görevdeki performans ölçüsüne göre, programımızın performansını artıran en iyi parametre kümesini belirlemek için kullanırız.

Parametreleri, çevirerek programın davranışını değiştirebileceğimiz düğmeler olarak düşünebilirsiniz.
Parametreleri sabitlendiğinde, programa *model* diyoruz.
Sadece parametreleri manipüle ederek üretebileceğimiz tüm farklı programlara (girdi-çıktı eşlemeleri) *model ailesi* denir.
Ve parametreleri seçmek için veri kümemizi kullanan * meta(başkalaşım) programa* *öğrenme algoritması* denir.

Devam etmeden ve öğrenme algoritmasını kullanmadan önce, sorunu kesin olarak tanımlamalı, girdi ve çıktıların kesin doğasını tespit etmeli ve uygun bir model ailesi seçmeliyiz.
Bu durumda, modelimiz *girdi* olarak bir ses parçasını alır ve *çıktı* olarak ``{evet, hayır}`` arasında bir seçim oluşturur.
Her şey plana göre giderse, modelin parçanın uyandırma kelimesini içerip içermediğine dair tahminleri genellikle doğru olacaktır.


Doğru model ailesini seçersek, o zaman model "Alexa" kelimesini her duyduğunda ``evet``i seçecek düğmelerin bir ayarı olmalıdır.
Uyandırma kelimesinin kesin seçimi keyfi olduğundan, muhtemelen yeterince zengin bir model ailesine ihtiyacımız olacak, öyle ki düğmelerin başka bir ayarı ile, sadece "Kayısı" kelimesini duyduktan sonra da ``evet`` seçilebilsin.
Aynı model ailesinin *"Alexa"yı tanıma* ve *"Kayısı"yı tanıma* için uygun olması beklenir, çünkü sezgisel olarak benzer görevler gibi görünüyorlar.
Bununla birlikte, temel olarak farklı girdiler veya çıktılarla uğraşmak istiyorsak, resimlerden altyazılara veya İngilizce cümlelerden Çince cümlelere eşlemek istiyorsak mesela, tamamen farklı bir model ailesine ihtiyacımız olabilir.

Tahmin edebileceğiniz gibi, tüm düğmeleri rastgele bir şekilde ayarlarsak, modelimizin "Alexa", "Kayısı" veya başka bir kelimeyi tanıması muhtemel değildir.
Derin öğrenmede, *öğrenme*, modelimizi istenen davranışa zorlayan düğmelerin doğru ayarını keşfettiğimiz süreçtir.

Gösterildiği gibi :numref:`fig_ml_loop`, eğitim süreci genellikle şöyle görünür:

1. Yararlı bir şey yapamayan rastgele başlatılan bir model ile başlayın.
1. Etiketli verilerinizin bir kısmını alın (örneğin, ses parçaları ve onlara karşılık gelen ``{evet, hayır}`` etiketleri).
1. Modelin bu örneklere göre daha az hata yapması için düğmelerin ayarlarını değiştirin.
1. Model harika olana kadar tekrarlayın.


[Tipik bir eğitim süreci.](../img/ml-loop.svg)
:label:`fig_ml_loop`

Özetlemek gerekirse, bir uyandırma kelimesi tanıyıcısını kodlamak yerine, büyük bir etiketli veri kümesi *sunarsak* uyandırma sözcüklerini tanımayı *öğrenebilen* bir program kodlarız.
Bu eylemi bir programın davranışını ona bir veri kümesi sunup *veri ile programlayarak* belirleme gibi düşünebilirsiniz.
MÖ sistemimize, aşağıdaki resimler gibi, birçok kedi ve köpek örneği sağlayarak bir kedi dedektörü "programlayabiliriz":


|kedi|kedi|köpek|köpek|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![cat3](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|


Bu şekilde dedektör, sonunda, bir kedi ise çok büyük bir pozitif sayı, bir köpekse çok büyük bir negatif sayı ve emin değilse sıfıra daha yakın bir şey yaymayı öğrenir ve bu,  MÖ'nin neler yapabileceğinin ancak yüzeyini kazır.

Derin öğrenme(DÖ), makine öğrenmesi problemlerini çözmek için mevcut birçok popüler yöntemden sadece biridir.
Şimdiye kadar, derin öğrenme hakkında değil, yalnızca geniş kapsamlı makine öğrenmesi hakkında konuştuk. Derin öğrenmenin neden önemli olduğunu görmek amacıyla, birkaç önemli noktayı vurgulamak için bir anlığına durmalıyız.

Birincisi, şu ana kadar tartıştığımız problemler --- ham ses sinyalinden, görüntülerin ham piksel değerlerinden öğrenmek veya keyfi uzunluktaki cümleleri yabancı dillerdeki muadilleri ile eşlemek --- derin öğrenmenin üstün olduğu ve geleneksel MÖ metotlarının sendelediği problemlerdir.
Derin modeller, birçok hesaplama *katmanını* öğrenmeleri anlamında *derindir*.
Bu çok katmanlı (veya hiyerarşik) modellerin, düşük seviyeli algısal verileri önceki araçların yapamayacağı bir şekilde ele alabildiği ortaya çıkıyor.
Eski günlerde, MÖ'yi bu sorunlara uygulamanın en önemli kısmı, veriyi *sığ* modellere uygun bir biçime dönüştürmek için elle (manuel olarak) tasarlanmış yolları bulmaktan oluşuyordu.
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

New York veya San Francisco'da yaşıyorsanız ve Amazon, Google, Microsoft veya Facebook'un CEO'su değilseniz, (metrekare bilgileri, yatak odası sayısı, banyo sayısı, yürüme mesafesi) eviniz için vektör özelliği şuna benzeyebilir: $[100, 0, .5, 60]$.
Ancak, Pittsburgh'da yaşıyorsanız, daha çok $[3000, 4, 3, 10]$ gibi görünebilir.
Bunun gibi öznitelik vektörleri, çoğu klasik makine öğrenmesi algoritması için gereklidir.
$i$ örneğine karşılık gelen özellik vektörünü $\mathbf{x}_i$ olarak göstermeye devam edeceğiz ve tüm özellik vektörlerini içeren tam tabloya $X$ olarak kastedeceğiz.

Bir problemi *bağlanım* yapan aslında çıktılardır.
Yeni bir ev için pazarda olduğunuzu varsayalım.
Bunun gibi bazı öznitelikler göz önüne alındığında, bir evin adil piyasa değerini tahmin etmek isteyebilirsiniz.
Hedef değer, satış fiyatı, bir *gerçel sayıdır*.
Gerçel sayıların resmi tanımını hatırlarsanız, şimdi başınızı kaşıyor olabilirsiniz.
Fiyatların irrasyonel sayılar olarak ifade edilmesini birakin, evler muhtemelen asla bir cent'in kesirine satmazlar.
Bu gibi durumlarda, hedefin aslında ayrık olduğunda, ancak yuvarlamanın yeterince iyi bir ölçekte gerçekleştiği durumlarda, dili biraz kötüye kullanacağız ve çıktılarımızı ve hedeflerimizi gerçel-değerli sayılar olarak tanımlamaya devam edeceğiz.

Tek hedefi $y_i$ (mesela $\mathbf{x}_i $ örneğine karşılık gelir) ve tüm hedefler kümesini $\mathbf{y}$ ($X$ tüm örneklerine karşılık gelir) olarak belirtiriz.
Hedeflerimiz belirli bir aralıkta keyfi değerler aldığında buna bir bağlanım problenim diyoruz.
Hedefimiz, tahminleri gerçek hedef değerlerine çok yakın olan bir model üretmektir.
Bir örnekler için öngörülen hedefi $\hat{y}_i$ diye belirtiriz.
Gösterim sizi rahatsız ediyorsa endişelenmeyin.
Sonraki bölümlerde daha ayrıntılı olarak açacağız.

A good rule of thumb is that any *How much?* or *How many?* problem
should suggest regression.

Birçok pratik problem iyi tanımlanmış bağlanım problemleridir.
Bir kullanıcının bir filme atayacağı puanı tahmin etmek bir bağlanım sorunu olarak düşünülebilir ve 2009'da bu özelliği gerçekleştirmek için harika bir algoritma tasarlasaydınız, [1 milyon dolarlık Netflix ödülünü](https: //en.wikipedia.org/wiki/Netflix_Prize) kazanmış olabilirsiniz .
Hastanedeki hastalar için kalış süresinin öngörülmesi de bir bağlanım sorunudur.
Pratik bir kural; herhangi bir *Ne kadar?* veya *Kaç tane?* problemi bağlanım içerir.

* "Bu ameliyat kaç saat sürecek?": *Bağlanım*
* "Bu fotoğrafta kaç köpek var?": *Bağlanım*.

Ancak, sorununuzu "Bu bir _ mi?" olarak kolayca ortaya koyabiliyorsanız, muhtemelen, daha sonra ele alacağımız, farklı bir tür gözetimli problem olabilir.
Daha önce hiç makine öğrenmesi ile çalışmamış olsanız bile, muhtemelen gayri ihtiyari olarak bir bağlanım problemi ile çalışmışsınızdır.
Örneğin, giderlerinizin onarıldığını ve personelin kanalizasyon borularınızdan pisliği temizlemek için $x_1=3$ saat harcadığını düşünün.
Sonra size $y_1=350\$$ tutarında bir fatura gönderdi.
Şimdi arkadaşınızın aynı personelini $x_2=2$ saat kiraladığını ve $y_2=250\$$ fatura aldığını düşünün.
Birisi size yaklaşan pislik temizleme faturasında ne kadar bekleyeceğinizi sorarsa, bazı makul varsayımlar yapabilirsiniz,
daha fazla çalışma saati daha fazla dolar maliyeti gibi.
Ayrıca bir baz ücretin olduğunu ve personelin saatlik ücret aldığını varsayabilirsiniz.
Bu varsayımlar geçerliyse, bu iki veri noktası göz önüne alındığında, personelin fiyatlandırma yapısını zaten tanımlayabilirsiniz: Saat başı 100\$ artı evinizde görünmesi için 50\$.
Eğer buraya kadar izleyebildiyseniz, doğrusal bağlanım arkasındaki üst-kademe fikri zaten anlıyorsunuz (ve dolaylı olarak bir ekdeğer (bias) terimli doğrusal bir model tasarladınız).

Bu durumda, personelin fiyatlarına tam olarak uyan parametreleri üretebiliriz.
Bazen bu mümkün olmayabilir; örneğin, varyansın bir kısmı iki özniteliğinizin yanı sıra bazı diğer faktörlere borçluysa.
Bu durumlarda, tahminlerimiz ile gözlenen değerler arasındaki mesafeyi en aza indiren modelleri öğrenmeye çalışacağız.
Bölümlerimizin çoğunda, iki yaygın kayıptan birine, L1 kaybına

$$l(y, y') = \sum_i |y_i-y_i'|$$

ve en küçük ortalama kareler kaybı veya L2 kaybına

$$l(y, y') = \sum_i (y_i - y_i')^2.$$

odaklanacağız.

Daha sonra göreceğimiz gibi, $L_2$ kaybı (yitimi), verilerimizin Gauss gürültüsü tarafından bozulduğu varsayımına karşılık gelirken, $L_1$ kaybı, Laplace dağılımından kaynaklanan bir gürültü varsayımına karşılık gelir.

#### Sınıflandırma

Bağlanım modelleri *kaç tane?* sorusunu ele almak için mükemmel olsa da, birçok sorun bu şablona rahatça uymaz.
Örneğin, bir banka mobil uygulamasına çek taraması eklemek istiyor.
Bu, müşterinin akıllı telefonunun kamerasıyla bir çekin fotoğrafını çekmesini içerir ve makine öğrenmesi modelinin görüntüde görülen metni otomatik olarak anlaması gerektirir.
Daha dirençli olması için elle yazılmış metni de anlaması gerekir.
Bu tür sisteme optik karakter tanıma (OKT) denir ve ele aldığı sorunun türüne *sınıflandırma* denir.
Bağlanım için kullanılanlardan farklı bir algoritma seti ile işlenir (birçok teknik buraya taşınacak olsa da).

Sınıflandırmada, modelimizin bir öznitelik vektörüne, örneğin bir görüntüdeki piksel değerlerine bakmasını ve ardından bazı (ayrık) seçenekler kümesi arasından hangi kategoriye (aslen *sınıflar* olarak adlandırılırlar) ait olduğunu tahmin etmesini istiyoruz.
Elle yazılmış rakamlar için, 0 ile 9 arasındaki rakamlara karşılık gelen 10 sınıfımız olabilir.
Sınıflandırmanın en basit şekli, sadece iki sınıf olduğunda, ikili sınıflandırma dediğimiz bir problemdir.
Örneğin, $X$ veri kümemiz hayvanların görüntülerinden oluşabilir ve *etiketlerimiz*, $Y$, $\mathrm{\{kedi, köpek\}}$ sınıfları olabilir.
Bağlanımdayken, gerçel bir değer, $\hat{y}$, çıkarmak için bir *bağlanımcı* aradık, sınıflandırmada, $\hat{y}$ çıkışı öngörülen sınıf ataması olan bir *sınıflandırıcı* arıyoruz.

Kitap daha teknik hale geldikçe gireceğimiz nedenlerden ötürü, yalnızca kategorik bir atama, örneğin *kedi* veya *köpek* çıktısı, alabilen bir modeli optimize etmek zor olabilir.
Bu tür durumlarda, modelimizi olasılıklar dilinde ifade etmek genellikle daha kolaydır.
Bir örnek, $x$, verildiğinde, modelimiz her bir $k$ etiketine $\hat{y}_k$ olasılığı atar. Bunlar olasılıklar olduğundan, pozitif sayılar olmalı ve $1$'e toplanabilmeliler ve bu nedenle $K$ kategorinin olasılıklarını atamak için sadece $K-1$ tane değere ihtiyacımız var.
Bunu ikili sınıflandırma için görmek kolaydır.
Eğer hileli bir madalyonun $0.6$ ($\%60$) tura çıkma olasılığı varsa, o zaman yazı ortaya çıkma olasılığı $0.4$ ($\%40 $) olabilir.
Hayvan sınıflandırma örneğimize dönersek, bir sınıflandırıcı bir görüntü görebilir ve görüntünün bir kedi olma olasılığını $P(y=\text{kedi} \mid x) = 0.9$ çıkarabilir.
Bu sayıyı, sınıflandırıcının görüntünün bir kediyi gösterdiğinden $\%90$ emin olduğunu söyleyerek yorumlayabiliriz.
Öngörülen sınıf için olasılığın büyüklüğü bir çeşit belirsizlik taşır.
Bu tek mevcut belirsizlik kavramı değildir ve diğerlerini de daha ileri bölümlerde tartışacağız.

İkiden fazla olası sınıfımız olduğunda, soruna *çok sınıflı sınıflandırma* diyoruz.
Yaygın örnekler arasında elle yazılmış karakter tanıma, `[0, 1, 2, 3 ... 9, a, b, c, ...]`, yer alır.
Bağlanım sorunlarına saldırırken L1 veya L2 yitim işlevlerini en aza indirmeye çalışırız; sınıflandırma sorunları için genel olan kayıp işlevine de çapraz düzensizlik (entropi) deriz.

En olası sınıfın kararınız için kullanacağınız esas sınıf olmak zorunda olmadığını unutmayın.
Bu güzel mantarı arka bahçenizde :numref:`fig_death_cap`de gösterildiği gibi bulduğunuzu varsayın .

![Ölüm tehlikesi --- yemeyin!](../img/death_cap.jpg)
:width:`200px`
:label:`fig_death_cap`


Şimdi, bir sınıflandırıcı oluşturduğunuzu ve bir mantarın bir fotoğrafa göre zehirli olup olmadığını tahmin etmek için eğittiğinizi varsayın.
Zehir tespit sınıflandırıcısının $P(y=\mathrm{ölüm tehlikesi}|\mathrm{image}) = 0.2$ sonucunu verdiğini varsayalım.
Başka bir deyişle, sınıflandırıcı, mantarımızın ölüm sınırında *olmadığından* $\%80$ emindir.
Yine de, yemek için aptal olmalısın.
Çünkü lezzetli bir akşam yemeğinin belirli bir yararı, ondan ölme riski olan $\%20$ değerine değmez.
Başka bir deyişle, *belirsiz riskin* etkisi faydadan çok daha fazladır. Buna daha kurallı bakabiliriz.
Temel olarak, maruz kaldığımız beklenen riski hesaplamamız gerekir, yani sonucun olasılığını, bununla ilişkili fayda (veya zarar) ile çarpmamız gerekir:

$$L(\mathrm{action}| x) = E_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)].$$

Bu nedenle, mantar yiyerek meydana gelen $L$ kaybı $L(a=\mathrm{ye}| x) = 0.2 * \infty + 0.8 * 0 = \infty$, oysa atılma maliyeti $L(a=\mathrm{at}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$.

Dikkatimiz haklıydı: herhangi bir mantarbilimcinin bize söyleyeceği gibi, yukarıdaki mantar aslında *ölümcüldür*.
Sınıflandırma sadece ikili sınıflandırmadan çok daha karmaşık hale gelebilir; çok sınıflı ve hatta çoklu etiketli.
Örneğin, hiyerarşilere yönelik bazı değişik sınıflandırmalar vardır.
Hiyerarşiler birçok sınıf arasında bazı ilişkilerin olduğunu varsayar.
Bu yüzden tüm hatalar eşit değildir - eğer hata yapacaksak, uzak bir sınıf yerine ilgili bir sınıfa yanlış sınıflamayı tercih ederiz.
Genellikle buna *hiyerarşik sınıflandırma* denir.
İlk örneklerden biri, hayvanları bir hiyerarşide düzenleyen [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus)'tır.

Hayvan sınıflandırması durumunda, bir kanişi bir schnauzer (bir tür Alman köpeği) ile karıştırmak o kadar kötü olmayabilir, ancak modelimiz bir dinozor ile bir fino köpeğini karıştırırsa büyük bir ceza ödeyecektir.
Hangi hiyerarşinin alakalı olduğu, modeli nasıl kullanmayı planladığınıza bağlı olabilir.
Örneğin, çıngıraklı yılanlar ve garter yılanları filogenetik ağaçta yakın olabilir, ancak bir garter yılanını bir çıngıraklı ile karıştırmak ölümcül olabilir.

#### Etiketleme (Tagging)

Bazı sınıflandırma sorunları, ikili veya çok sınıflı sınıflandırma ayarlarına tam olarak uymaz.
Örneğin, kedileri köpeklerden ayırmak için normal bir ikili sınıflandırıcı eğitebiliriz.
Bilgisayarlı görmenin mevcut durumu göz önüne alındığında, bunu hali-hazırda araçlarla kolayca yapabiliriz.
Bununla birlikte, modelimiz ne kadar doğru olursa olsun, sınıflandırıcı Bremen Mızıkacılarının bir görüntüsüyle karşılaştığında kendimizi ufak bir belada bulabiliriz.

![Bir kedi, bir horoz, bir köpek ve bir eşek](../img/stackedanimals.jpg)
:width:`300px`

Gördüğünüz gibi, resimde bir kedi ve bir horoz, bir köpek, bir eşek ve bir kuş, arka planda bazı ağaçlar var.
Nihayetinde modelimizle ne yapmak istediğimize bağlı olarak, bunu ikili bir sınıflandırma problemi olarak ele almak pek anlamlı olmayabilir.
Bunun yerine, modele görüntünün bir kediyi *ve* bir köpeği *ve* bir eşeği *ve* bir horozu *ve* bir kuşu tasvir ettiğini söyleme seçeneği vermek isteyebiliriz.

*Karşılıklı olarak münhasır olmayan* sınıfları tahmin etmeyi öğrenme problemine çoklu etiket sınıflandırması denir.
Otomatik etiketleme sorunları genellikle en iyi çoklu etiket sınıflandırma sorunları olarak tanımlanır.
Kullanıcıların bir teknoloji blogundaki yayınlara uygulayabilecekleri etiketleri, örneğin "makine öğrenmesi", "teknoloji", "araçlar", "programlama dilleri", "linux", "bulut bilişim", "AWS" gibi, düşünün.
Tipik bir makalede 5-10 etiket uygulanabilir, çünkü bu kavramlar birbiriyle ilişkilidir.
"Bulut bilişim" hakkındaki gönderilerin "AWS"den bahsetmesi muhtemeldir ve "makine öğrenmesi" ile ilgili gönderiler de "programlama dilleri" ile ilgili olabilir.

Ayrıca, makalelerin doğru etiketlenmesinin önemli olduğu biyomedikal literatürle uğraşırken bu tür bir sorunla uğraşmak zorundayız, çünkü bu araştırmacıların literatürde kapsamlı incelemeler yapmasına izin veriyor.
(Amerikan) Ulusal Tıp Kütüphanesi'nde, bir dizi profesyonel yorumlayıcı, PubMed'de endekslenen her makaleyi, kabaca 28 bin etiketlik bir koleksiyon olan MeSH'den ilgili terimlerle ilişkilendirmek için gözden geçiriyor.
Bu zaman alıcı bir süreçtir ve yorumlayıcıların genellikle arşivleme ve etiketleme arasında bir yıllık bir gecikmesi vardır.
Makine öğrenimi burada, her makaleye uygun bir manuel (elle) incelemeye sahip oluncaya kadar geçici etiketler sağlamak için kullanılabilir.
Gerçekten de, birkaç yıl boyunca, BioASQ organizasyonu tam olarak bunu yapmak için [bir yarışma düzenledi](http://bioasq.org/).

#### Arama ve sıralama

Bazen her örneği bir kovaya veya gerçek bir değere atamak istemiyoruz. Bilgi geri çağırma alanında, bir dizi maddeye bir sıralama uygulamak istiyoruz.
Örneğin, web aramasını ele alalım, hedef belirli bir sayfanın bir sorgu için alakalı olup olmadığını belirlemekten daha ziyade, birçok arama sonuçlarından hangisinin belirli bir kullanıcı için *en alakalı* olduğunu belirlemektir.
Alakalı arama sonuçlarının sırasına gerçekten önem veriyoruz ve öğrenme algoritmamızın daha geniş bir gruptan sıralanmış alt kümeleri üretmesi gerekiyor.
Başka bir deyişle, alfabeden ilk 5 harfi üretmemiz istenirse, `` A B C D E`` ve `` C A B E D`` döndürme arasında bir fark vardır.
Sonuç kümesi aynı olsa bile, küme içindeki sıralama önemlidir.

Bu soruna olası bir çözüm, önce kümedeki her bir öğeye, ona karşılık gelen bir uygunluk puanı atamak ve daha sonra en yüksek dereceli öğeleri almaktır.
[PageRank](https://en.wikipedia.org/wiki/PageRank), Google arama motorunun arkasındaki esas gizli bileşen, böyle bir puanlama sisteminin erken bir örneğiydi, fakat tuhaf tarafı gerçek sorguya bağlı değildi.
Burada, ilgili öğelerin kümesini tanımlamak için basit bir alaka filtresine ve ardından sorgu terimini içeren sonuçları sıralamak için PageRank'e güveniyorlardı.
Günümüzde arama motorları, sorguya bağlı alaka düzeyi puanlarını belirlemek için makine öğrenmesi ve davranışsal modeller kullanmaktadır.
Sadece bu konuyla ilgili akademik konferanslar vardır.

#### Tavsiye sistemleri
:label:`subsec_recommender_systems`

Tavsiye sistemleri, arama ve sıralama ile ilgili başka bir problem ailesidir.
Amaç, kullanıcıya ilgili bir dizi öğeyi görüntülemek olduğu sürece benzer problemlerdir.
Temel fark, tavsiye sistemleri bağlamında, belirli kullanıcılara *kişiselleştirme* vurgusu yapılmasıdır.
Mesela, film önerilerinde, bir SciFi hayranı için sonuçlar sayfası ile Peter Sellers komedileri uzmanı için sonuçlar sayfası önemli ölçüde farklılıklar gösterebilir.
Perakende satış ürünleri, müzik veya haber önerileri gibi diğer öneri gruplarında da benzer sorunlar ortaya çıkar.

Bazı durumlarda, müşteriler belirli bir ürünü ne kadar sevdiklerini bildiren açık geri bildirimler sağlar (ör. Amazon, IMDB, GoodReads, vb. Üzerindeki ürün puanları ve incelemeleri).
Diğer bazı durumlarda, örneğin bir müzik çalma listesindeki başlıkları atlama, memnuniyetsizliği de, şarkının o anki bağlamında uygunsuz olduğunu da gösterebilecek (gizli) örtük geri bildirim sağlarlar.
En basit formülasyonlarda, bu sistemler $u_i$ kullanıcısı ve $p_j$ ürünü göz önüne alındığında tahmini bir derecelendirme veya satın alma olasılığını, $y_ {ij}$, tahmin etmek üzere eğitilir.

Böyle bir model göz önüne alındığında, herhangi bir kullanıcı için, en yüksek puanları $y_ {ij}$ olan ve daha sonra müşteriye önerilebilecek nesneler kümesini bulabiliriz.
Üretim sistemleri oldukça ileri düzeydedir ve bu puanları hesaplarken ayrıntılı kullanıcı etkinliği ve öğenin özelliklerini dikkate alır. :numref:`fig_deeplearning_amazon` imgesi, yazarın tercihlerini yakalamak için ayarlanan kişiselleştirme algoritmalarına dayanarak Amazon tarafından önerilen derin öğrenme kitaplarına bir örnektir.

![Amazon tarafından önerilen derin öğrenme kitapları.](../img/deeplearning_amazon.png)
:label:`fig_deeplearning_amazon`

Muazzam ekonomik değerlerine rağmen, tahminci modeller üzerine saf olarak inşa edilmiş tavsiye sistemleri bazı ciddi kavramsal kusurlara maruz kalmaktadırlar.
Öncelikle sadece *sansürlü geri bildirim* gözlemliyoruz.
Kullanıcılar tercih ettikleri filmleri özellikle güçlü bir şekilde hissettiklerine göre derecelendirir: Öğelerin çok sayıda 5 ve 1 yıldız derecelendirmesi aldığını, ancak dikkat çekici derecede az 3 yıldızlı derecelendirme olduğunu fark edebilirsiniz.
Ayrıca, mevcut satın alma alışkanlıkları genellikle şu anda mevcut olan tavsiye algoritmasının bir sonucudur, ancak öğrenme algoritmaları bu ayrıntıyı her zaman dikkate almazlar.
Bu nedenle, bir geri bildirim döngüsünün oluşması mümkündür: Bir tavsiye sistemi, daha sonra daha iyi olması için (daha büyük satın alımlar nedeniyle), alınan bir öğeyi tercihli olarak yukarı iter ve daha da sık tavsiye edilmesine neden olur.
Sansür, teşvikler ve geri bildirim döngüleri ile nasıl başa çıkılacağı gibi ilgili bu tarz sorunların birçoğu önemli açık araştırma konularıdır.

#### Dizi Öğrenimi

Şimdiye kadar, sabit sayıda girdimiz olan ve sabit sayıda çıktı üreten sorunlara baktık.
Öncesinde ev fiyatlarını sabit bir dizi özellikten tahmin ettik: Metrekare alanları, yatak odası sayısı, banyo sayısı, şehir merkezine yürüme süresi.
Ayrıca, bir görüntüyü (sabit boyutlu), sabit sayıda sınıfın hangi birine ait olduğu tahmin eden olasılıklarla eşlemeyi veya bir kullanıcı kimliği ve ürün kimliği alarak bir yıldız derecelendirmesi tahmin etmeyi tartıştık. Bu durumlarda, sabit uzunluklu girdimizi bir çıktı üretmek için modele beslediğimizde, model hemen gördüklerini hemen unutur.

Girdilerimizin hepsi aynı boyutlara sahipse ve birbirini takip eden girdilerin birbirleriyle hiçbir ilgisi yoksa, bu iyi olabilir.
Ancak video parçalarıyla nasıl başa çıkardık?
Bu durumda, her parça farklı sayıda çerçeveden oluşabilir.
Ayrıca önceki veya sonraki kareleri dikkate alırsak, her karede neler olup bittiğine dair tahminimiz çok daha güçlü olabilir. Aynı şey dil için de geçerli. Popüler bir derin öğrenme sorunu, makine çevirisidir: Bazı kaynak dilde cümleleri alma ve başka bir dilde çevirilerini tahmin etme görevidir.

Bu problemler tıpta da görülür.
Yoğun bakım ünitesindeki hastaları izlemek ve önümüzdeki 24 saat içinde ölüm riskleri belli bir eşiği aşarsa, uyarıcıları tetiklemek için bir model isteyebiliriz.
Bu modelin her saatteki hasta geçmişi hakkında bildiği her şeyi atmasını ve sadece en son ölçümlere dayanarak tahminlerini yapmasını kesinlikle istemeyiz.

Bu problemler makine öğrenmesinin en heyecan verici uygulamaları arasındadır ve *dizi öğrenmenin* örnekleridir.
Girdilerin dizilerini almak veya çıkış dizilerini (veya her ikisini!) saçmak için bir modele ihtiyaç duyarlar.
Bu ikinci tip problemlere bazen ``seq2seq`` problemleri denir. Dil çevirisi bir ``seq2seq`` problemidir.
Sözlü konuşmadan metine kopyalama da bir ``seq2seq`` problemidir.
Her türlü dizi dönüşümünü anlatmak mümkün olmasa da, bir dizi özel durumdan bahsetmeye değer:

**Etiketleme ve Ayrıştırma**. Bu, nitelikleri olan bir metin dizisine açıklama eklemeyi içerir.
Başka bir deyişle, girdi ve çıktıların sayısı aslında aynıdır.
Örneğin, fiillerin ve öznelerin nerede olduğunu bilmek isteyebiliriz.
Alternatif olarak, hangi kelimelerin adlandırılmış varlıklar olduğunu bilmek isteyebiliriz.
Genel olarak amaç, bir açıklama almak için yapısal ve dilbilgisel varsayımlara dayalı olarak metni ayrıştırmak ve açıklama eklemektir.
Bu aslında olduğundan daha karmaşıkmış gibi geliyor.
Aşağıdaki çok basit bir örnek, hangi kelimelerin adlandırılmış varlıkları ifade ettiğini belirten etiketleri bir cümleye açıklama olarak eklemeyi gösterir.

```text
Tom'un Washington'da Sally ile akşam yemeği var.
Var      Var          Var  -    -     -      -
```

**Otomatik Konuşma Tanıma**. Konuşma tanımada, $x$ girdi dizisi bir hoparlörün ses kaydıdır (:numref:`fig_speech`da gösterilen) ve $y$ çıktısı konuşmacının söylediklerinin metne dökümüdür.
Buradaki zorluk, metinden çok daha fazla ses karesi çerçevesi olması (ses genellikle 8kHz veya 16kHz'de örneklenmiştir), yani ses ve metin arasında 1:1 karşılık olmamasıdır, çünkü binlerce sesli örnek tek bir sözlü kelimeye karşılık gelir.
Bunlar, çıktının girdiden çok daha kısa olduğu ``seq2seq`` problemleridir.

![`-D-e-e-p- L-ea-r-ni-ng-`](../img/speech.png)
:width:`700px`
:label:`fig_speech`

**Metinden Konuşmaya**. Metinden-Konuşmaya, konuşma tanımanın tersidir.
Başka bir deyişle, $x$ girdisi metindir ve $y$ çıktısı bir ses dosyasıdır.
Bu durumda, çıktı girdiden *çok daha uzun* olur.
*İnsanların* kötü bir ses dosyasını tanıması kolay olsa da, bu bilgisayarlar için o kadar da bariz değildir.

**Makine Çevirisi**. Karşılık gelen girdi ve çıktıların aynı sırada (hizalamadan sonra) gerçekleştiği konuşma tanıma durumundan farklı olarak, makine çevirisinde, sırayı ters çevirme hayati önem taşıyabilir.
Başka bir deyişle, bir diziyi diğerine dönüştürürken, ne girdi ve çıktıların sayısı ne de karşılık gelen veri noktalarının sırası aynı kabul edilmektedir.
Almanların fiilleri cümle sonuna yerleştirme eğiliminin aşağıdaki açıklayıcı örneğini düşünün.

```text
Almanca:          Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
İngilizce:        Did you already check out this excellent tutorial?
Wrong alignment:  Did you yourself already this excellent tutorial looked-at?
```
İlgili birçok sorun diğer öğrenme görevlerinde ortaya çıkar.
Örneğin, bir kullanıcının bir Web sayfasını okuma sırasını belirlemek iki boyutlu bir düzen analizi sorunudur.
Diyalog sorunları her türlü ek komplikasyon ortaya çıkarır: Bir sonrasında ne söyleneceğini belirlemede, gerçek dünya bilgisini ve uzun zamansal mesafelerde konuşmanın önceki durumunu dikkate almayı gerektirmek gibi. Bu aktif bir araştırma alanıdır.


### Gözetimsiz öğrenme

Şimdiye kadarki tüm örnekler *Gözetimli Öğrenme*, yani, modeli hem öznitelikleri hem de karşılık gelen hedef değerleri içeren dev bir veri kümesi ile beslediğimiz durumlarla ilgilidir.
Gözetimli öğreniciyi son derece uzmanlaşmış bir işe ve son derece konuşkan bir patrona sahip olmak gibi düşünebilirsiniz.
Patron omzunuzun üzerinden bakar ve siz durumlardan eylemlere eşlemeyi öğrenene kadar her durumda tam olarak ne yapacağınızı söyler.
Böyle bir patron için çalışmak oldukça tatsızdır.
Öte yandan, bu patronu memnun etmek kolaydır.
Deseni mümkün olduğunca çabuk tanır ve eylemlerini taklit edersiniz.

Tamamen zıt bir şekilde, ne yapmanızı istediğini bilmeyen bir patron için çalışmak sinir bozucu olabilir.
Ancak, bir veri bilimcisi olmayı planlıyorsanız, buna alışsanız iyi olur.
Patron size sadece dev bir veri dökümü verebilir ve *onunla veri bilimi yapmanızı söyleyebilir!* Bu kulağa belirsiz geliyor çünkü öyle.
Bu sorun sınıfına *gözetimsiz öğrenme* diyoruz ve sorabileceğimiz soruların türü ve sayısı yalnızca yaratıcılığımızla sınırlıdır.
Daha sonraki bölümlerde bir dizi denetimsiz öğrenme tekniğini ele alacağız. Şimdilik iştahınızı hafifletmek için sormak isteyebileceğiniz birkaç sorudan bahsediyoruz:

* Verileri doğru bir şekilde özetleyen az sayıda ilk örnek (prototip) bulabilir miyiz?
Bir dizi fotoğraf verildiğinde, onları manzara fotoğrafları, köpek resimleri, bebekler, kediler, dağ zirveleri vb. olarak gruplandırabilir miyiz?
Benzer şekilde, kullanıcıların göz atma etkinliği koleksiyonu göz önüne alındığında, onları benzer davranışa sahip kullanıcılara ayırabilir miyiz?
Bu sorun genellikle *kümeleme* olarak bilinir.
* Verilerin ilgili özelliklerini doğru bir şekilde yakalayan az sayıda parametre bulabilir miyiz?
Bir topun yörüngeleri, topun hızı, çapı ve kütlesi ile oldukça iyi tanımlanmıştır.
Terziler, kıyafetlerin uyması amacıyla insan vücudunun şeklini oldukça doğru bir şekilde tanımlayan az sayıda parametre geliştirmiştir.
Bu problemlere *altuzay tahmini* problemleri denir.
Bağımlılık doğrusal ise, buna *ana bileşen analizi* denir.
* (Keyfi olarak yapılandırılmış) Nesnelerin Öklid uzayında (yani, $\mathbb{R}^n$ vektör uzayında) sembolik özelliklerinin iyi eşleştirilebileceği bir temsili var mı?
Buna *temsil öğrenme* denir ve varlıkları ve onların ilişkilerini, Roma $-$ İtalya $+$ Fransa $=$ Paris gibi, tanımlamak için kullanılır.
* Gözlemlediğimiz verilerin çoğunun temel nedenlerinin bir açıklaması var mı?
Örneğin, konut fiyatları, kirlilik, suç, yer, eğitim, maaşlar vb. ile ilgili demografik verilerimiz varsa, bunların deneysel verilerine dayanarak nasıl ilişkili olduğunu bulabilir miyiz? *Nedensellik* ve *olasılıksal grafik modeller* ile ilgili alanlar bu sorunu ele almaktadır.
* Gözetimsiz öğrenmedeki bir diğer önemli ve heyecan verici gelişme, *üretici çekişmeli ağların* (GAN'lar) ortaya çıkmasıdır.
Bunlar bize verileri, görüntüler ve ses gibi karmaşık yapılandırılmış verileri bile, sentezlemek için yöntemsel bir yol sunar.
Temel istatistiksel mekanizmalar, gerçek ve sahte verilerin aynı olup olmadığını kontrol etmek için kullanılan testlerdir.
Onlara birkaç not defteri ayıracağız.

### Bir Ortamla Etkileşim

Şimdiye kadar, verilerin gerçekte nereden geldiğini veya bir makine öğrenmesi modeli bir çıktı oluşturduğunda gerçekte *ne olduğunu* tartışmadık.
Çünkü gözetimli öğrenme ve gözetimsiz öğrenme bu konuları çok karmaşık bir şekilde ele almaz.
Her iki durumda da, büyük bir veri yığınını önceden alıyoruz, ardından bir daha çevre ile etkileşime girmeden desen tanıma makinelerimizi harekete geçiriyoruz.
Tüm öğrenme, algoritma ortamdan ayrıldıktan sonra gerçekleştiği için, buna bazen *çevrimdışı öğrenme* denir.
Gözetimli öğrenme için süreç şuna benzer :numref:`fig_data_collection`.

![Bir ortamdan gözetimli öğrenme için veri toplama.](../img/data-collection.svg)
:label:`fig_data_collection`

Çevrimdışı öğrenmenin bu basitliğinin cazibesi vardır.
Bunun olumu kısmı, bu diğer sorunlardan herhangi bir dikkat dağılmadan, sadece örüntü tanıma konusu ile tek başına ilgilenebiliriz.
Ancak olumsuz tarafı, formülasyonun oldukça kısıtlayıcı olmasıdır.
Daha hırslıysanız ya da Asimov'un Robot Serisi'ni okuyarak büyüdüyseniz, sadece tahminler yapmakla kalmayıp, dünyada hareket edebilecek yapay zeka botları hayal edebilirsiniz.
Sadece *modelleri* değil, akıllı *etmenleri (ajanları)* de düşünmek istiyoruz.
Bu, sadece *tahminler* yapmakla kalmayıp, *eylemleri* seçmeyi düşünmemiz gerektiği anlamına gelir. Dahası, öngörülerin aksine, eylemler aslında çevreyi etkiler.
Akıllı bir ajanı eğitmek istiyorsak, eylemlerinin ajanın gelecekteki gözlemlerini nasıl etkileyebileceğini hesaba katmalıyız.

Bir çevre ile etkileşimi dikkate almak, bir dizi yeni modelleme sorusu açar.
Çevre:

* Daha önceden ne yaptığımızı hatırlıyor mu?
* Bize bir konuşma tanıyıcıya metin okuyan bir kullanıcı gibi yardım etmek ister mi?
* Bizi yenmek mi istiyor, yani spam filtreleme (spam göndericilere karşı) veya oyun oynama (rakiplere karşı) gibi rakip bir ortam mı?
* Umursumuyor mu (birçok durumda olduğu gibi)?
* Değişen dinamiklere sahip mi (gelecekteki veriler her zaman geçmişe benziyor mu doğal olarak veya otomatik araçlarımıza yanıt olarak zaman içinde değişiyor mu)?

Bu son soru *dağılım kayması* sorununu gündeme getirmektedir (eğitim ve test verileri farklı olduğunda).
Bu bir öğretim üyesi tarafından hazırlanan yazılıya girerken yaşadığımız bir problemdir, çünkü ödevler asistanlar tarafından oluşturulmuştur.
Bir çevreyle etkileşimi açıkça dikkate alan iki ortam olan pekiştirmeli öğrenmeyi ve çekişmeli öğrenmeyi kısaca anlatacağız.

### Pekiştirmeli öğrenme

Bir ortamla etkileşime giren ve eylemler yapan bir ajan geliştirmek için makine öğrenmesini kullanmakla ilgileniyorsanız, muhtemelen *pekiştirmeli öğrenimi* (PÖ) konusuna odaklanacaksınız.
Bu, robotik, diyalog sistemleri ve hatta video oyunları için YZ geliştirme uygulamalarını içerebilir.
Derin sinir ağlarını PÖ problemlerine uygulayan *derin pekiştirmel öğrenme* (DPÖ) popülerlik kazanmıştır.
Bu atılımda [yalnızca görsel girdileri kullanarak Atari oyunlarında insanları yenen derin Q-ağ](https://www.wired.com/2015/02/google-ai-plays-atari-like-pros/) ve [Go oyunu dünya şampiyonunu tahtından indiren AlphaGo programı](https://www.wired.com/2017/05/googles-alphago-trounces-humans-also-gives-boost/) iki önemli örnektir.

Pekiştirmeli öğrenmede, bir ajanın bir dizi *zaman adımı* üzerinde bir çevre ile etkileşime girdiği çok genel bir sorun ifade edilir.
$T$ her bir zaman adımında, etmen ortamdan $o_t$ gözlemini alır ve daha sonra bir mekanizma (bazen çalıştırıcı (aktüatör) olarak da adlandırılır) aracılığıyla çevreye geri iletilecek bir $a_t$ eylemi seçmelidir.
Son olarak, temsilci ortamdan bir ödül, $r_t$, alır.
Etmen daha sonra bir gözlem alır ve bir sonraki eylemi seçer, vb.
Bir PÖ etmenin davranışı bir *politika* tarafından yönetilir.
Kısacası, bir *politika*, sadece, gözlemlerden (çevrenin) eylemlere eşlenen bir fonksiyondur.
Pekiştirmeli öğrenmenin amacı iyi bir politika üretmektir.

![Pekiştirmeli öğrenme ve çevre arasındaki etkileşim.](../img/rl-environment.svg)

PÖ çerçevesinin genelliğini abartmak zordur.
Örneğin, herhangi bir gözetimli öğrenme problemini bir PÖ problemine dönüştürebiliriz.
Diyelim ki bir sınıflandırma problemimiz var.
Her sınıfa karşılık gelen bir *eylem* ile bir PÖ etmeni oluşturabiliriz.
Daha sonra, orijinal gözetimli problemin yitim fonksiyonuna tam olarak eşit olan bir ödül veren bir ortam yaratabiliriz.

Bununla birlikte, PÖ, gözetimli öğrenmenin yapamadığı birçok sorunu da ele alabilir.
Örneğin, gözetimli öğrenmede her zaman eğitim girdisinin doğru etiketle ilişkilendirilmesini bekleriz.
Ancak PÖ'de, her gözlem için çevrenin bize en uygun eylemi söylediğini varsaymıyoruz.
Genel olarak, sadece bir ödül alırız.
Dahası, çevre bize hangi eylemlerin ödüle yol açtığını bile söylemeyebilir.

Örneğin satranç oyununu düşünün.
Tek gerçek ödül sinyali, oyunun sonunda ya kazandığımızda 1, ya da kaybettiğimizde -1 diye gelir.
Bu yüzden pekiştirmeli öğreniciler *kredi atama problemi* ile ilgilenmelidir: Bir sonuç için hangi eylemlerin beğeni toplayacağını veya suçlanacağını belirleme.
Aynı şey 11 Ekim'de terfi alan bir çalışan için de geçerli.
Bu terfi büyük olasılıkla bir önceki yılda itibaren çok sayıda iyi seçilmiş eylemi yansıtmaktadır.
Gelecekte daha fazla terfi almak için zaman boyunca hangi eylemlerin terfiye yol açtığını bulmak gerekir.

Pekiştirmeli öğreniciler de kısmi gözlenebilirlik sorunuyla uğraşmak zorunda kalabilirler.
Yani, mevcut gözlem size mevcut durumunuz hakkında her şeyi söylemeyebilir.
Diyelim ki bir temizlik robotu kendini bir evdeki birçok aynı dolaptan birinde sıkışmış buldu.
Robotun kesin yerini (ve dolayısıyla durumunu) bulmak, dolaba girmeden önce önceki gözlemlerini dikkate almayı gerektirebilir.

Son olarak, herhangi bir noktada, pekiştirmeli öğreniciler iyi bir politika biliyor olabilir, ancak etmenin hiç denemediği daha iyi politikalar olabilir.
Pekiştirmeli öğrenici ya sürekli olarak politika olarak şu anda bilinen en iyi stratejiyi *sömürmeyi* veya stratejiler alanını *keşfetmeyi*, yani potansiyel olarak bilgi karşılığında kısa vadede ödül vermeyi, seçmelidir.

#### MKS'ler, haydutlar ve arkadaşlar

Genel pekiştirme öğrenme sorunu çok genel bir ortamdır.
Eylemler sonraki gözlemleri etkiler.
Ödüller yalnızca seçilen eylemlere karşılık gelir.
Ortam tamamen veya kısmen gözlenebilir.
Tüm bu karmaşıklığı bir kerede hesaplamak çok fazla araştırmacı isteyebilir.
Dahası, her pratik sorun tüm bu karmaşıklığı sergilemez.
Sonuç olarak, araştırmacılar pekiştirmeli öğrenme sorunlarının bir dizi *özel vakasını* incelemişlerdir.

Ortam tam olarak gözlemlendiğinde, PÖ sorununa *Markov Karar Süreci* (MKS) diyoruz.
Durum önceki eylemlere bağlı olmadığında, soruna *bağlamsal bir kollu kumar makinesi sorunu* diyoruz.
Durum yoksa, sadece başlangıçta bilinmeyen ödülleri olan bir dizi kullanılabilir eylem, bu sorun klasik *çok kollu kumar makinesi problemidir*.

## Kökenler

Birçok derin öğrenme yöntemi yeni icatlar olmasına rağmen, yüzyıllar boyunca insanlar verileri analiz etme ve gelecekteki sonuçları tahmin etme arzusundaydılar.
Aslında, doğa bilimlerinin çoğunun kökenleri budur.
Örneğin, Bernoulli dağılımı [Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli) ve Gaussian dağılımı [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) tarafından keşfedildi.
Örneğin, bugün hala sigorta hesaplamalarından tıbbi teşhislere kadar sayısız problem için kullanılan en düşük kareler ortalaması algoritmasını icat etti.
Bu araçlar doğa bilimlerinde deneysel bir yaklaşıma yol açmıştır - örneğin, Ohm'un bir dirençteki akım ve voltajla ilgili yasası doğrusal bir modelle mükemmel bir şekilde tanımlanmıştır.

Orta çağlarda bile, matematikçilerin tahminlerde keskin bir sezgileri vardı.
Örneğin, [Jacob Köbel (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)'ün geometri kitabı ortalama ayak uzunluğunu elde etmek 16 erkek yetişkinin ayak uzunluğunu ortalamayı göstermektedir.

![Ayak uzunluğunu tahmin etme.](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`

:numref:`fig_koebel` bu tahmincinin nasıl çalıştığını gösterir.
16 yetişkin erkekten kiliseden ayrılırken üst üste dizilmeleri istendi.
Daha sonra toplam uzunlukları günümüzdeki 1 ayak (foot) birimine ilişkin bir tahmin elde etmek için 16'ya bölündü.
Bu "algoritma" daha sonra biçimsiz ayaklarla başa çıkmak için de düzenlendi - sırasıyla en kısa ve en uzun ayakları olan 2 adam gönderildi, sadece geri kalanların ortalaması alındı.
Bu, kırpılmış ortalama tahminin en eski örneklerinden biridir.

İstatistikler gerçekten verilerin toplanması ve kullanılabilirliği ile başladı.
Dev isimlerden biri [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), istatistik teorisine ve aynı zamanda genetikteki uygulamalarına önemli katkıda bulundu.
Algoritmalarının çoğu (Doğrusal Ayırtaç Analizi gibi) ve formülü (Fisher Information Matrix gibi) günümüzde hala sık kullanılmaktadır (1936'da piyasaya sürdüğü İris veri kümesi bile, bazen makine öğrenme algoritmalarını göstermek için hala kullanılıyor).
Fisher aynı zamanda, veri biliminin ahlaki olarak şüpheli kullanımının, endüstride ve doğa bilimlerinde verimli kullanımı kadar uzun ve kalıcı bir geçmişi olduğunu hatırlatan bir öjeni (doğum ile kalıtımsal olarak istenen özelliklere sahip bireylerin üremesine çalışan bilim dalı) savunucusuydu.

Makine öğrenmesi için ikinci bir etki, [(Claude Shannon, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) aracılığıyla Bilgi Teorisi ve [Alan Turing (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing) aracılığıyla Hesaplama Teorisi'nden geldi. .
Turing, ünlü makalesinde "makineler düşünebilir mi?" diye sordu [Computing machinery and intelligence](https://en.wikipedia.org/wiki/Computing_Machinery_and_Intelligence) (Mind, Ekim 1950).
Turing testi olarak tanımladığı şeyde, bir insan değerlendiricinin metin etkileşimlerine dayanarak cevapların bir makineden mi ve bir insan mı geldiğini arasında ayırt etmesinin zor olması durumunda, bir makine akıllı kabul edilebilir.

Nörobilim ve psikolojide de başka bir etki bulunabilir.
Sonuçta, insanlar açıkça akıllı davranış sergilerler.
Bu nedenle, sadece bu beceriyi açıklayıp tersine mühendislik yapıp yapamayacağını sormak mantıklıdır.
Bu şekilde esinlenen en eski algoritmalardan biri [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb) tarafından formüle edildi.
Çığır Açan "Davranış Örgütlenmesi :cite:`Hebb.Hebb.1949` adlı kitabında, nöronların pozitif pekiştirme ile öğrendiklerini ileri sürdü.
Bu Hebbian öğrenme kuralı olarak biliniyordu.
Rosenblatt'ın algılayıcı öğrenme algoritmasının ilk örneğidir ve bugün derin öğrenmeyi destekleyen birçok rassal eğim inişi (stochastic gradient descent) algoritmasının temellerini atmıştır: sinir ağındaki parametrelerin iyi ayarlarını elde etmek için arzu edilen davranışı güçlendirmek ve istenmeyen davranışı zayıflatmak.

*Sinir ağlarına* adını veren şey biyolojik ilhamdir.
Yüzyılı aşkın bir süredir (Alexander Bain, 1873 ve James Sherrington, 1890 modellerine kadar geri gider) araştırmacılar, etkileşen nöron ağlarına benzeyen hesaplama devreleri oluşturmaya çalıştılar.
Zamanla, biyolojinin yorumu daha az gerçek hale geldi, ancak isim yapıştı. Özünde, bugün çoğu ağda bulunabilecek birkaç temel ilke yatmaktadır:

* Genellikle *katmanlar* olarak adlandırılan doğrusal ve doğrusal olmayan işlem birimlerinin değişimi.
* Tüm ağdaki parametreleri bir kerede ayarlamak için zincir kuralının (*geri yayma (backpropagation)* olarak da bilinir) kullanımı.

İlk hızlı ilerlemeden sonra, sinir ağlarındaki araştırmalar 1995'ten 2005'e kadar yavaşladı.
Bunun birkaç nedeni vardır.
Bir ağın eğitimi hesaplamaya göre çok pahalıdır.
RAM (Rasgele Erişim Belleği) geçen yüzyılın sonunda bol miktarda bulunurken, hesaplama gücü azdı.
İkincisi, veri kümeleri nispeten küçüktü.
Aslında, Fisher'in 1932'deki Iris veri kümesi algoritmaların etkinliğini test etmek için popüler bir araçtır.
MNIST, 60.000 el yazısı rakam ile devasa sayılırdı.

Veri ve hesaplama kıtlığı göz önüne alındığında, Çekirdek (Kernel) Yöntemleri, Karar Ağaçları ve Grafik Modeller gibi güçlü istatistiksel araçlar deneysel olarak daha üstün oldu.
Sinir ağlarından farklı olarak, eğitim için haftalar gerektirmediler ve güçlü teorik garantilerle öngörülebilir sonuçlar verdiler.

## Derin Öğrenmeye Giden Yol

Bunların çoğu, yüz milyonlarca kullanıcıya çevrimiçi hizmet veren şirketlerin gelişi, ucuz ve yüksek kaliteli sensörlerin yayılması, ucuz veri depolama (Kryder yasası) ve özellikle bilgisayar oyunları için tasarlanan GPU'ları kullanan ucuz hesaplama (Moore yasası) maliyeti ile değişti.
Aniden, hesaplamaya elverişli görünmeyen algoritmalar ve modeller bariz hale geldi (ve tersi).
Bu en iyi şekilde :numref:`tab_intro_decade`de gösterilmiştir .

: Veri kümesi ve bilgisayar belleği ve hesaplama gücü

|On Yıl|Veri Kümesi|Bellek|Saniyede Yüzer (Floating) Sayı Hesaplaması|
|:--|:-|:-|:-|
| 1970 | 100 (İris) | 1 KB | 100 KF (Intel 8080) |
1980 | 1 K (Boston'daki ev fiyatları) | 100 KB | 1 MF (Intel 80186) |
| 1990 | 10 K (optik karakter tanıma) | 10 MB | 10 MF (Intel 80486) |
| 2000 | 10 M (web sayfaları) | 100 MB | 1 GF (Intel Core) |
| 2010 | 10 G (reklam) | 1 GB | 1 TF (Nvidia C2050) |
| 2020 | 1 T (sosyal ağ) | 100 GB | 1 PF (Nvidia DGX-2) |
:label:`tab_intro_decade`

RAM'in veri büyümesine ayak uyduramadığı açıktır.
Aynı zamanda, hesaplama gücündeki artış mevcut verilerinkinden daha fazladır.
Bu, istatistiksel işlemlerin bellekte daha verimli hale gelmesi (bu genellikle doğrusal olmayan özellikler ekleyerek elde edilir) ve aynı zamanda, artan bir hesaplama bütçesi nedeniyle bu parametreleri optimize etmek için daha fazla zaman harcanması gerektiği anlamına gelir.
Sonuç olarak, makine öğrenmesi ve istatistikteki tatlı nokta (genelleştirilmiş) doğrusal modellerden ve çekirdek yöntemlerinden derin ağlara taşındı.
Bu aynı zamanda derin öğrenmenin dayanak noktalarının, çok katmanlı algılayıcılar :cite:`McCulloch.Pitts.1943`, evrişimli sinir ağları :cite:`LeCun.Bottou.Bengio.ea.1998`, Uzun Kısa Süreli Bellek :cite:`Hochreiter.Schmidhuber.1997` ve Q-Öğrenme :cite:` Watkins.Dayan.1992` gibi, oldukça uzun bir süre nispeten uykuda kaldıktan sonra, esasen "yeniden keşfedilme"sindeki birçok nedenden biridir.

İstatistiksel modeller, uygulamalar ve algoritmalardaki son gelişmeler bazen Kambriyen (Cambrian) Patlaması'na benzetildi: Türlerin evriminde hızlı bir ilerleme anı.
Gerçekten de, en son teknoloji, sadece, onlarca yıllık algoritmaların mevcut kaynaklara uygulanmasının bir sonucu değildir.
Aşağıdaki listen, araştırmacıların son on yılda muazzam bir ilerleme kaydetmesine yardımcı olan fikirlerin sadece yüzeyine ışık tutmaktadir.

* Bırakma (dropout) gibi kapasite kontrolüne yönelik yeni yöntemler :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`, aşırı öğrenme tehlikesini azaltmaya yardımcı oldu.
  Bu, ağ boyunca gürültü zerk edilerek (enjeksiyon) sağlandı, :cite:`Bishop.1995`, eğitim amaçlı ağırlıkları rastgele değişkenlerle değiştirdi.
* Dikkat mekanizmaları, yüzyılı aşkın bir süredir istatistikleri rahatsız eden ikinci bir sorunu çözdü: Öğrenilebilir parametre sayısını artırmadan bir sistemin belleğini ve karmaşıklığını nasıl artırabiliriz. :cite:`Bahdanau.Cho.Bengio.2014` sadece öğrenilebilir bir işaretçi yapısı olarak görülebilecek zarif bir çözüm buldu.
  Bir cümlenin tamamını hatırlamak yerine, örneğin, sabit boyutlu bir gösterimdeki makine çevirisi için, depolanması gereken tek şey, çeviri işleminin ara durumunu gösteren bir işaretçiydi. Bu, modelin artık yeni bir cümle oluşturulmadan önce tüm cümleyi hatırlaması gerekmediğinden, uzun cümleler için önemli ölçüde artırılmış doğruluğa izin verdi.
* Çok aşamalı tasarımlar, örneğin, Bellek Ağları (MemNets) aracılığıyla :cite:`Sukhbaatar.Weston.Fergus.ea.2015` ve Sinir Programcısı-Tercüman (Neural Programmer-Interpreter) :cite:`Reed.De-Freitas.2015`  istatistiksel modelcilerin yinelemeli yaklaşımlar ile akıl yürütme tanımlamasına izin verdi. Bu araçlar, derin ağın dahili bir durumunun tekrar tekrar değiştirilmesine izin verir; bir işlemcinin bir hesaplama için belleği değiştirmesine benzer şekilde, böylece bir akıl yürütme zincirinde sonraki adımlar gerçekleştirilebilir.
* Bir başka önemli gelişme de (GAN) ÜÇA'ların icadıdır :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`. Geleneksel olarak, yoğunluk tahmini için istatistiksel yöntemler ve üretici modeller, uygun olasılık dağılımlarını ve bunlardan örnekleme için (genellikle yaklaşık) algoritmaları bulmaya odaklanmıştır. Sonuç olarak, bu algoritmalar büyük ölçüde istatistiksel modellerin doğasında var olan esneklik eksikliği ile sınırlıydı. ÜÇA'lardaki en önemli yenilik, örnekleyiciyi türevlenebilir parametrelere sahip rastgele bir algoritma ile değiştirmekti. Bunlar daha sonra, ayırıcın (aslen ikili-örneklem testi) sahte verileri gerçek verilerden ayırt edemeyeceği şekilde ayarlanır. Veri üretmek için rasgele algoritmalar kullanma yeteneği sayesinde yoğunluk tahminini çok çeşitli tekniklere açmıştır. Dörtnala Zebralar :cite:`Zhu.Park.Isola.ea.2017` ve sahte ünlü yüzler :cite:`Karras.Aila.Laine.ea.2017` örnekleri bu ilerlemenin kanıtıdır. Amatör karalamacılar bile, bir sahnenin düzeninin nasıl göründüğünü açıklayan eskizlere dayanan fotogerçekçi görüntüler üretebilir :cite:`Park.Liu.Wang.ea.2019`.
* Çoğu durumda, tek bir GPU eğitim için mevcut olan büyük miktarda veriyi işlemek için yetersizdir. Son on yılda, paralel dağıtılmış eğitim algoritmaları oluşturma yeteneği önemli ölçüde gelişmiştir. Ölçeklenebilir algoritmaların tasarlanmasındaki temel zorluklardan biri, derin öğrenme optimizasyonunun ana öğesinin, rassal eğim inişinin, işlenecek verilerin nispeten küçük mini-grup'larına (minibatch) dayanmasıdır. Aynı zamanda, küçük gruplar GPU'ların verimliliğini sınırlar. Bu nedenle, 1024 GPU'nun eğitimindeki mini-grup büyüklüğü, örneğin toplu iş başına 32 resim diyelim, toplam 32 bin resim anlamına gelir. Son çalışmalarda, önce Li :cite:`Li.2017` ve ardından :cite:`You.Gitman.Ginsburg.2017` ve :cite:`Jia.Song.He.ea.2018` boyutu 64 bin gözleme yükselterek, ImageNet'teki ResNet50 için eğitim süresini 7 dakikadan daha az bir sürede azalttılar. Karşılaştırma için - başlangıçta eğitim süreleri günlere göre ölçülmüştü.
* Hesaplamayı paralel hale getirme yeteneği, en azından simülasyon (benzetim) bir seçenek olduğunda, pekiştirmeli öğrenmedeki ilerlemeye oldukça önemli bir katkıda bulunmuştur. Bu önemli ilerlemelerle Go, Atari oyunları, Starcraft ve fizik simülasyonlarında (örn. MuJoCo kullanarak) insanüstü performans elde eden bilgisayarlara yol açtı. AlphaGo'da bunun nasıl yapılacağına ilişkin açıklama için bakınız :cite:`Silver.Huang.Maddison.ea.2016`. Özetle, pek çok (durum, eylem, ödül) üçlük mevcutsa, yani birbirleriyle nasıl ilişkilendiklerini öğrenmek için birçok şeyi denemek mümkün olduğunda pekiştirmeli öğrenme en iyi sonucu verir. Benzetim böyle bir yol sağlar.
* Derin Öğrenme çerçeveleri fikirlerin yayılmasında önemli bir rol oynamıştır. Kapsamlı modellemeyi kolaylaştıran ilk nesil çerçeveler: [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch) ve [Theano](https: / /github.com/Theano/Theano). Bu araçlar kullanılarak birçok yeni ufuklar açan makale yazılmıştır. Şimdiye kadar yerlerini [TensorFlow](https://github.com/tensorflow/tensorflow) ve onu da genellikle yüksek düzey API aracılığıyla kullanan [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2) ve [Apache MxNet](https://github.com/ apache'nin / inkübatör-mxnet) aldı. Üçüncü nesil araçlara, yani derin öğrenme için zorunlu araçlar, modelleri tanımlamak için Python NumPy'ye benzer bir sözdizimi kullanan [Chainer](https://github.com/chainer/chainer) öncülük etti. Bu fikir hem [PyTorch](https://github.com/pytorch/pytorch), hem [Gluon API](https://github.com/apache/incubator-mxnet) MXNet ve [Jax](https://github.com/google/jax) tarafından benimsenmiştir. Bu derste derin öğrenmeyi öğretmek için kullanılan ikinci gruptur.

Daha iyi araçlar üreten sistem araştırmacıları ve daha iyi ağlar inşa eden istatistiksel modelciler arasındaki işbölümü, işleri basitleştirdi. Örneğin, doğrusal bir lojistik regresyon modelinin eğitilmesi, 2014 yılında Carnegie Mellon Üniversitesi'nde yeni makine öğrenmesi doktora öğrencilerine bir ödev problemi vermeğe değer bariz olmayan bir problemdi. Şimdilerde, sıkı bir programcı kavrayışını içine katarak, bu görev 10'dan az kod satırı ile gerçekleştirilebilir.

## Başarı Öyküleri

Yapay Zeka, aksi takdirde başarılması zor olacak, sonuçları dağıtmak için uzun bir geçmişe sahiptir.
Örneğin, postalar optik karakter tanıma kullanılarak sıralanır.
Bu sistemler 90'lardan beri kullanılmaktadır (bu, sonuçta, ünlü MNIST ve USPS el yazısı rakam kümelerinin kaynağıdır).
Aynı şey banka mevduatları için çek okuma ve başvuru sahiplerinin kredi değerliliğini puanlanma için de geçerlidir.
Finansal işlemler otomatik olarak sahtekarlığa karşı kontrol edilir.
Bu, PayPal, Stripe, AliPay, WeChat, Apple, Visa, MasterCard gibi birçok e-ticaret ödeme sisteminin bel kemiğini oluşturur.
Bilgisayar satranç programları onlarca yıldır rekabetçidir.
Makine öğrenimi, internette arama, öneri, kişiselleştirme ve sıralamayı besler. Başka bir deyişle, yapay zeka ve makine öğrenmesi, çoğu zaman gözden gizli olsa da, yaygındır.

Sadece son zamanlarda YZ, çoğunlukla daha önce zorlu olarak kabul edilen sorunlara çözümlerinden dolayı ilgi odağı olmuştur.

* Apple'ın Siri, Amazon'un Alexa veya Google asistanı gibi akıllı asistanlar sözlü soruları makul bir doğrulukla cevaplayabilir. Bu, ışık anahtarlarını (devre dışı bırakılan bir nimet) açmadan, berber randevuları ayarlamaya ve telefon destek iletişim diyalogu sunmaya kadar önemli görevleri içerir. Bu muhtemelen YZ'nın hayatlarımızı etkilediğinin en belirgin işaretidir.
* Dijital asistanların önemli bir bileşeni, konuşmayı doğru bir şekilde tanıma yeteneğidir. Yavaş yavaş bu tür sistemlerin doğruluğu, belirli uygulamalar için insan paritesine ulaştığı noktaya kadar artmıştır :cite:`Xiong.Wu.Alleva.ea.2018`.
* Nesne tanıma da aynı şekilde uzun bir yol kat etti. Bir resimdeki nesneyi tahmin etmek 2010 yılında oldukça zor bir işti. ImageNet karşılaştırmalı değerlendirmesinde :cite:`Lin.Lv.Zhu.ea.2010` %28'lik bir ilk 5 hata oranına ulaştı. 2017 itibariyle :cite:`Hu.Shen.Sun.2018` bu hata oranını %2.25'e düşürdü. Benzer şekilde, kuşları tanımlamada veya cilt kanserini teşhis etmede çarpıcı sonuçlar elde edilmiştir.
* Oyunlar eskiden insan zekasının kalesi idi. TDGammon'dan [23] başlayarak, Tavla oynamak için zamansal fark (ZF) pekiştirmeli öğrenme kullanan bir program, algoritmik ve hesaplamalı ilerleme, çok çeşitli uygulamalar için algoritmalara yol açmıştır. Tavla'nın aksine, satranç çok daha karmaşık bir durum uzayına ve bir dizi eyleme sahiptir. DeepBlue, Garry Kasparov'u, Campbell ve ark. :cite:`Campbell.Hoane-Jr.Hsu.2002`, büyük paralellik, özel amaçlı donanım ve oyun ağacında verimli arama kullanarak yendi. Büyük durum uzayı nedeniyle Go hala daha zor. AlphaGo, 2015 yılında insan paritesine ulaştı :cite:`Silver.Huang.Maddison.ea.2016`, derin öğrenmeyi Monte Carlo ağaç örneklemesi ile birlikte kullandı. Poker'deki zorluk, durum uzayının geniş olması ve tam olarak gözlenmemesidir (rakiplerin kartlarını bilmiyoruz). Libratus, etkin bir şekilde yapılandırılmış stratejiler kullanarak Poker'deki insan performansını aştı :cite:`Brown.Sandholm.2017`. Bu, oyunlardaki etkileyici ilerlemeyi ve gelişmiş algoritmaların oyunlarda önemli bir rol oynadığını göstermektedir.
* Yapay zekadaki ilerlemenin bir başka göstergesi, kendi kendine giden otomobil ve kamyonların ortaya çıkışıdır. Tam özerklik henüz tam olarak ulaşılamamasına rağmen, Tesla, NVIDIA ve Waymo nakliye ürünleri gibi şirketlerle en azından kısmi özerkliğe olanak tanıyan mükemmel bir ilerleme kaydedildi. Tam özerkliği bu kadar zorlaştıran şey, uygun sürüşün, kuralları algılama, akıl yürütme ve kuralları bir sisteme dahil etme yeteneğini gerektirmesidir. Günümüzde, derin öğrenme bu sorunların öncelikle bilgisayarlı görme alanında kullanılmaktadır. Geri kalanlar mühendisler tarafından yoğun bir şekilde ince ayarlanmıştır.

Yine, yukarıdaki liste, makine öğreniminin pratik uygulamaları etkilediği yüzeylere çok az ışık tutmaktadır. Örneğin, robotik, lojistik, hesaplamalı biyoloji, parçacık fiziği ve astronomi, en etkileyici son gelişmelerinden bazılarını en azından kısmen makine öğrenimine borçludur. MÖ böylece mühendisler ve bilim insanları için her yerde mevcut bir araç haline geliyor.

YZ ile ilgili teknik olmayan makalelerde, YZ kıyameti veya YZ tekilliği sorunu sıkça gündeme gelmiştir.
Korku, bir şekilde makine öğrenme sistemlerinin, insanların geçimini doğrudan etkileyen şeyler hakkında programcılarından (ve ustalarından) bağımsız bir şekilde  duyarlı (akıllı) olacağına ve karar vereceğinedir.
Bir dereceye kadar, YZ zaten insanların geçimini şimdiden etkiliyor - kredibilite otomatik olarak değerlendiriliyor, otomatik pilotlar çoğunlukla taşıtları yönlendiriyor, kefalet vermeye istatistiksel veri kullanarak karar veriyor.
Daha anlamsızcası, Alexa'dan kahve makinesini açmasını isteyebiliriz.

Neyse ki, insan yaratıcılarını manipüle etmeye (veya kahvelerini yakmaya) hazır, duyarlı bir YZ sisteminden çok uzaktayız. İlk olarak, YZ sistemleri belirli, hedefe yönelik bir şekilde tasarlanır, eğitilir ve devreye alınır. Davranışları genel zeka yanılsamasını verebilse de, tasarımın altında yatan kuralların, sezgisel ve istatistiksel modellerin birleşimleridirler.
İkincisi, şu anda, *yapay genel zeka* için, kendilerini geliştirebilen, kendileriyle ilgili akıl yürüten ve genel görevleri çözmeye çalışırken kendi mimarilerini değiştirebilen, genişletebilen ve geliştirebilen araçlar yoktur.

Çok daha acil bir endişe YZ'nın günlük yaşamımızda nasıl kullanıldığıdır.
Kamyon şoförleri ve mağaza asistanları tarafından yerine getirilen birçok önemli görevin otomatikleştirilebileceği ve otomatikleştirileceği muhtemeldir.
Çiftlik robotları büyük olasılıkla organik tarım maliyetini düşürecek, ayrıca hasat işlemlerini de otomatikleştirecek.
Sanayi devriminin bu aşamasının toplumun büyük kesimlerinde derin sonuçları olabilir (kamyon şoförleri ve mağaza asistanları birçok ülkedeki en yaygın işlerden ikisi).
Ayrıca, istatistiksel modeller, dikkatsizce uygulandığında ırksal, cinsiyet veya yaş yanlılığına yol açabilir ve sonuç kararları otomatik hale getirildiklerinde usul adaleti konusunda makul endişeler doğurabilir.
Bu algoritmaların dikkatle kullanılmasını sağlamak önemlidir.
Bugün bildiklerimizle, bu bize, kötü niyetli bir süper zekanın insanlığı yok etme potansiyelinden çok daha acil bir endişe gibi getiriyor.

## Özet

* Makine öğrenmesi, belirli görevlerde performansı artırmak için bilgisayar sistemlerinin *deneyiminden* (genellikle veri) nasıl yararlanabileceğini inceler. İstatistik, veri madenciliği, yapay zeka ve optimizasyon fikirlerini birleştirir. Genellikle, yapay olarak zeki çözümlerinin uygulanmasında bir araç olarak kullanılır.
* Bir makine öğrenmesi sınıfı olarak, temsili öğrenme, verileri temsil etmek için uygun yolu otomatik olarak nasıl bulacağınıza odaklanır. Bu genellikle öğrenilen dönüşümlerin ilerlemesi ile gerçekleştirilir.
* Derin öğrenmedeki son ilerlemenin çoğu, ucuz sensörler ve İnternet ölçekli uygulamalardan kaynaklanan çok sayıda veri ve çoğunlukla GPU'lar aracılığıyla hesaplamadaki önemli ilerleme ile tetiklenmiştir.
* Tüm sistem optimizasyonu, iyi performans elde etmede önemli bir ana bileşendir. Etkili derin öğrenme çerçevelerinin mevcudiyeti, bunun tasarımını ve uygulamasını önemli ölçüde kolaylaştırmıştır.

## Alıştırmalar

1. Şu anda yazdığınız kodun hangi bölümleri "öğrenilebilir", yani kodunuzda yapılan tasarım seçimleri öğrenilerek ve otomatik olarak belirlenerek geliştirilebilir? Kodunuzda sezgisel (heuristic) tasarım seçenekleri var mı?
1. Hangi karşılaştığınız sorunlarda nasıl çözüleceğine dair birçok örnek var, ancak bunları otomatikleştirmenin belirli bir yolu yok? Bunlar derin öğrenmeyi kullanmaya aday olabilirler.
1. Yapay zekanın gelişimini yeni bir sanayi devrimi olarak görürsek algoritmalar ve veriler arasındaki ilişki nedir? Buhar motorlarına ve kömüre benzer mi (temel fark nedir)?
1. Uçtan uca eğitim yaklaşımını başka nerede uygulayabilirsiniz? Fizik? Mühendislik? Ekonometri?

[Tartışmalar](https://discuss.d2l.ai/t/22)
