# Ortamı Düşünmek

Önceki bölümlerde, modelleri çeşitli veri kümelerine oturtarak bir dizi uygulamalı makine öğrenmesi üzerinde çalıştık. Yine de, verilerin ilk etapta nereden geldiğini veya modellerimizin çıktılarıyla sonuçta *yapmayı* planladığımız şeyi düşünmeyi asla bırakmadık. Sıklıkla, verilere sahip olan makine öğrenmesi geliştiricileri, bu temel sorunları dikkate alıp duraklamadan modeller geliştirmek için acele ederler.

Başarısız olan birçok makine öğrenimi dağıtımı bu desen ile köklendirilebilir. Bazen modeller, test kümesi doğruluğu ile ölçüldüğünde harika bir performans sergiliyor gibi görünebilir, ancak veri dağılımı aniden değiştiğinde ürün olarak konuşlandırıldığında felaket bir şekilde başarısız olur. Daha sinsi bir şekilde, bazen bir modelin konuşlandırılması, veri dağılımını bozan katalizör olabilir. Örneğin, bir krediyi kimin geri ödeyeceğini tahmin etmek için bir model eğittiğimizi ve başvuranın ayakkabı seçiminin temerrüt (geri ödememe) riskiyle ilişkili olduğunu tespit ettiğimizi varsayalım (Oxfords geri ödemeyi gösterir, spor ayakkabılar temerrüdü gösterir). Daha sonra Oxfords giyen tüm başvuru sahiplerine kredi verme ve spor ayakkabı giyen tüm başvuru sahiplerini reddetme eğiliminde olabiliriz.

Bu durumda, örüntü tanımadan karar vermeye yanlış düşünülmüş bir sıçrayışımız ve ortamı eleştirel bir şekilde dikkate almadaki başarısızlığımız feci sonuçlar doğurabilir. Başlangıç olarak, ayakkabılara dayalı kararlar almaya başlar başlamaz, müşteriler farkına varır ve davranışlarını değiştirirdi. Çok geçmeden, tüm başvuru sahipleri, kredi güvenirliliğinde herhangi bir tesadüfi gelişme olmaksızın Oxfords giyeceklerdi. Bunu sindirmek için bir dakikanızı ayırın çünkü makine öğrenmesinin pek çok uygulamasında benzer sorunlar bolca bulunur: Modele dayalı kararlarımızı ortama sunarak modeli bozabiliriz.

Bu konuları tek bir bölümde tam olarak ele alamasak da, burada bazı ortak endişeleri ortaya çıkarmayı ve bu durumları erken tespit etmek, hasarı azaltmak ve makine öğrenmesini sorumlu bir şekilde kullanmak için gereken eleştirel düşünmeyi teşvik etmeyi amaçlıyoruz. Çözümlerden bazıları basittir ("doğru" veriyi isteyin), bazıları teknik olarak zordur (pekiştirmeli öğrenme sistemi uygulamak) ve diğerleri, istatistiksel tahmin alanının tamamen dışına çıkmamızı ve algoritmaların etik uygulanması gibi konularla ilgili zor felsefi sorularla boğuşmamızı gerektirir.


## Dağılım Kayması

Başlangıç olarak, veri dağılımlarının değişebileceği çeşitli yolları ve model performansını kurtarmak için neler yapılabileceğini göz önünde bulundurarak pasif tahminler ayarına sadık kalıyoruz. Klasik bir kurulumda, eğitim verilerimizin herhangi bir $p_S(\mathbf{x},y)$ dağılımından örneklendiğini, ancak etiketsiz test verilerimizin farklı bir $p_T(\mathbf{x},y)$ dağılımından çekildiğini varsayalım. Şimdiden ciddi bir gerçekle yüzleşmeliyiz. $p_S$ ve $p_T$'nin birbiriyle nasıl ilişkili olduğuna dair herhangi bir varsayım olmadığından, gürbüz bir sınıflandırıcı öğrenmek imkansızdır.

Köpekler ve kediler arasında ayrım yapmak istediğimiz bir ikili sınıflandırma problemini düşünün. Dağılım keyfi şekillerde kayabiliyorsa, kurulumumuz girdiler üzerinden dağılımın sabit kaldığı patolojik bir duruma izin verir: $p_S(\mathbf{x}) = p_T(\mathbf{x})$ ancak etiketlerin tümü ters çevrilmiştir $ p_S (y | \ mathbf {x}) = 1 - p_T(y | \mathbf{x})$. Başka bir deyişle, eğer Tanrı aniden gelecekte tüm "kedilerin" artık köpek olduğuna ve daha önce "köpek" dediğimiz şeyin artık kedi olduğuna karar verebilirse---$p(\mathbf{x})$ girdilerinin dağılımında herhangi bir değişiklik olmaksızın, bu kurulumu dağılımın hiç değişmediği bir kurulumdan ayırt edemeyiz.

Neyse ki, verilerimizin gelecekte nasıl değişebileceğine dair bazı kısıtlı varsayımlar altında, ilkeli algoritmalar kaymayı algılayabilir ve hatta bazen anında kendilerini uyarlayarak orijinal sınıflandırıcının doğruluğunu iyileştirebilir.


### Ortak Değişken Kayması

Dağılım kayması kategorileri arasında, *ortak değişken kayması* en yaygın olarak çalışılmışı olabilir. Burada, girdilerin dağılımının zamanla değişebileceğini varsayıyoruz, etiketleme fonksiyonu, yani koşullu dağılım $P(y \mid \mathbf{x})$ değişmez. İstatistikçiler buna *ortak değişken kayması* diyorlar çünkü problem *ortak değişkenlerin* (öznitelikler) dağılımındaki bir kayma nedeniyle ortaya çıkıyor. Bazen nedenselliğe başvurmadan dağıtım kayması hakkında akıl yürütebiliyor olsak da, ortak değişken kaymasının $\mathbf{x}$'in $y$'ye neden olduğuna inandığımız durumlarda çağrılacak doğal varsayım olduğuna dikkat ediyoruz.

Kedileri ve köpekleri ayırt etmenin zorluğunu düşünün. Eğitim verilerimiz aşağıdaki türden resimlerden oluşabilir:

|kedi|kedi|köpek|köpek|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|

Test zamanında aşağıdaki resimleri sınıflandırmamız istenrbilir:

|kedi|kedi|köpek|köpek|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|

Eğitim kümesi fotoğraflardan oluşurken, test kümesi sadece çizimlerden oluşuyor. Test kümesinden büyük ölçüde farklı özelliklere sahip bir veri kümesi üzerinde eğitim, yeni etki alanına nasıl adapte olacağına dair tutarlı bir planın olmaması sorununu yaratabilir.

### Etiket Kayması

*Etiket kayması* ters problemi tanımlar. Burada, etiketin marjinali $P(y)$'nin değişebileceğini varsayıyoruz ($P(\mathbf{x})$'de bir değişikliğe neden olur) ancak sınıf koşullu dağılım $P(\mathbf{x} \mid y)$ etki alanları arasında sabit kalır. Etiket kayması, $y$'nin $\mathbf{x}$'e neden olduğuna inandığımızda yaptığımız makul bir varsayımdır. Örneğin, tanıların göreceli yaygınlığı zamanla değişse bile, semptomları (veya diğer belirtileri) verilen tanıları tahmin etmek isteyebiliriz. Etiket kayması burada uygun varsayımdır çünkü hastalıklar semptomlara neden olur. Bazı yozlaşmış durumlarda, etiket kayması ve ortak değişken kayma varsayımları aynı anda geçerli olabilir. Örneğin, etiket deterministik olduğunda, $y$ $\mathbf{x}$'e neden olsa bile, ortak değişken kayma varsayımı karşılanacaktır. İlginç bir şekilde, bu durumlarda, etiket kayması varsayımından kaynaklanan yöntemlerle çalışmak genellikle avantajlıdır. Bunun nedeni, (derin öğrenmede) yüksek boyutlu olma eğiliminde olan girdiye benzeyen nesnelerin aksine, bu yöntemlerin etikete benzeyen (genellikle düşük boyutlu olan) nesnelerde oynama yapmak eğiliminde olmasıdır.


### Kavram Kayması

Ayrıca etiketlerin tanımları değiştiğinde ortaya çıkan ilgili *kavram kayması* sorunuyla da karşılaşabiliriz. Kulağa garip geliyor---bir *kedi* bir *kedi*dir, değil mi? Ancak, diğer kategoriler zaman içinde kullanımda değişikliklere tabidir. Akıl hastalığı için tanı kriterleri, modaya uygun olanlar ve iş unvanları önemli miktarda *kavram kaymasına* tabidir. Amerika Birleşik Devletleri çevresinde dolaşırsak, verilerimizin kaynağını coğrafyaya göre değiştirirsek, *meşrubat* adlarının dağılımıyla ilgili olarak, :numref:`fig_popvssoda`'da gösterildiği gibi önemli bir kavram kayması bulacağımız ortaya çıkar.

![Amerika Birleşik Devletleri'nde meşrubat isimlerinde kavram değişikliği.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Bir makine çeviri sistemi kuracak olsaydık, $P(y \mid x)$ dağılımı konumumuza bağlı olarak farklı olabilirdi. Bu sorunu tespit etmek zor olabilir. Değişimin yalnızca kademeli olarak gerçekleştiği bilgiden (hem zamansal hem de coğrafi anlamda) yararlanmayı umabiliriz.


### Örnekler

Biçimselliğe ve algoritmalara girmeden önce, ortak değişken veya kavram kaymasının bariz olmayabileceği bazı somut durumları tartışabiliriz.


#### Tıbbi Teşhis

Kanseri tespit etmek için bir algoritma tasarlamak istediğinizi hayal edin. Sağlıklı ve hasta insanlardan veri topluyorsunuz ve algoritmanızı geliştiriyorsunuz. İyi çalışıyor, size yüksek doğruluk sağlıyor ve tıbbi teşhis alanında başarılı bir kariyere hazır olduğunuz sonucuna varıyorsunuz. *O kadar da hızlı değil.*

Eğitim verilerini ortaya çıkaranlar ile vahşi doğada karşılaşacağınız dağılımlar önemli ölçüde farklılık gösterebilir. Bu, bizden birinin yıllar önce çalıştığı talihsiz bir girişimin başına geldi. Çoğunlukla yaşlı erkekleri etkileyen bir hastalık için bir kan testi geliştiriyorlardı ve hastalardan topladıkları kan örneklerini kullanarak bu hastalığı incelemeyi umuyorlardı. Bununla birlikte, sağlıklı erkeklerden kan örnekleri almak, sistemdeki mevcut hastalardan almaktan çok daha zordur. Girişim, bununla başa çıkmak için, bir üniversite kampüsündeki öğrencilerden testlerini geliştirmede sağlıklı kontrol grubu olmaları amacıyla kan bağışı istedi. Ardından, onlara hastalığı tespit etmek için bir sınıflandırıcı oluşturmalarına yardım edip edemeyeceğimiz soruldu.

Onlara açıkladığımız gibi, sağlıklı ve hasta grupları neredeyse mükemmel bir doğrulukla ayırt etmek gerçekten kolay olurdu. Çünkü, bunun nedeni, deneklerin yaş, hormon seviyeleri, fiziksel aktivite, diyet, alkol tüketimi ve hastalıkla ilgisi olmayan daha birçok faktör açısından farklılık göstermesidir. Gerçek hastalarda durum böyle değildi. Örneklem prosedürleri nedeniyle, aşırı ortak değişken kayması ile karşılaşmayı bekleyebiliriz. Dahası, bu durumun geleneksel yöntemlerle düzeltilmesi pek olası değildir. Kısacası, önemli miktarda para israf ettiler.

#### Kendi Kendine Süren Arabalar

Bir şirketin sürücüsüz otomobiller geliştirmek için makine öğrenmesinden yararlanmak istediğini varsayalım. Buradaki temel bileşenlerden biri yol kenarı detektörüdür. Gerçek açıklamalı verilerin elde edilmesi pahalı olduğu için, bir oyun oluşturma motorundan gelen sentetik verileri ek eğitim verileri olarak kullanma (zekice ve şüpheli) fikirleri vardı. Bu, işleme motorundan alınan "test verileri" üzerinde gerçekten iyi çalıştı. Ne yazık ki, gerçek bir arabanın içinde tam bir felaketti. Görünüşe göre oluşturulan yol kenarı çok basit bir dokuya sahipti. Daha da önemlisi, *tüm* yol kenarı *aynı* dokuya sahipti ve yol kenarı dedektörü bu "özniteliği" çok çabuk öğrendi.

ABD Ordusu ormandaki tankları ilk defa tespit etmeye çalıştıklarında da benzer bir şey oldu. Ormanın tanksız hava fotoğraflarını çektiler, ardından tankları ormana sürdüler ve bir dizi fotoğraf daha çektiler. Sınıflandırıcının *mükemmel* çalıştığı görüldü. Ne yazık ki, o sadece gölgeli ağaçları gölgesiz ağaçlardan nasıl ayırt edeceğini öğrenmişti---ilk fotoğraf kümesi sabah erken, ikincisi öğlen çekilmişti.

#### Durağan Olmayan Dağılımlar

Dağılım yavaş değiştiğinde ve model yeterince güncellenmediğinde çok daha hassas bir durum ortaya çıkar. İşte bazı tipik durumlar:

* Bir hesaplamalı reklamcılık modelini eğitiyor ve ardından onu sık sık güncellemekte başarısız oluyoruz (örneğin, iPad adı verilen belirsiz yeni bir cihazın henüz piyasaya sürüldüğünü dahil etmeyi unuttuk)
* Bir yaramaz posta filtresi oluşturuyoruz. Şimdiye kadar gördüğümüz tüm yaramaz postaları tespit etmede iyi çalışıyor. Ancak daha sonra, yaramaz posta gönderenler akıllanıyor ve daha önce gördüğümüz hiçbir şeye benzemeyen yeni mesajlar oluşturuyorlar.
* Ürün öneri sistemi oluşturuyoruz. Kış boyunca işe yarıyor, ancak Noel'den sonra da Noel Baba şapkalarını önermeye devam ediyor.

#### Daha Fazla Kısa Hikaye

* Yüz dedektörü yapıyoruz. Tüm kıyaslamalarda iyi çalışıyor. Ne yazık ki test verilerinde başarısız oluyor---rahatsız edici örnekler, yüzün tüm resmi doldurduğu yakın çekimlerdir (eğitim kümesinde böyle bir veri yoktu).
* ABD pazarı için bir web arama motoru oluşturuyoruz ve bunu Birleşik Krallık'ta kullanmak istiyoruz.
* Büyük bir sınıf kümesinin her birinin veri kümesinde eşit olarak temsil edildiği büyük bir veri kümesi derleyerek bir imge sınıflandırıcı eğitiyoruz, örneğin 1000 kategori var ve her biri 1000 görüntü ile temsil ediliyor. Ardından sistemi, fotoğrafların gerçek etiket dağılımının kesinlikle tekdüze olmadığı gerçek dünyada konuşlandırıyoruz.

Kısacası, $p(\mathbf {x}, y)$ eğitim ve test dağılımlarının farklı olduğu birçok durum vardır. Bazı durumlarda şanslıyız ve modeller ortak değişken, etiket veya kavram kaymasına rağmen çalışıyor. Diğer durumlarda, kaymalarla başa çıkmak için ilkeli stratejiler kullanarak daha iyisini yapabiliriz. Bu bölümün geri kalanı önemli ölçüde daha teknik hale geliyor. Sabırsız okuyucu bir sonraki bölüme geçebilir çünkü bu içerik sonraki kavramlar için ön koşul değildir.

### Ortak Değişken Kaymasını Düzeltme

Verileri $(\mathbf{x}_i, y_i)$ olarak etiketlediğimiz $P(y \mid \mathbf{x})$ bağımlılığını tahmin etmek istediğimizi varsayalım. Ne yazık ki, $x_i$ gözlemleri, *kaynak* dağılımı $p(\mathbf{x})$ yerine *hedef* dağılım $q(\mathbf{x})$'dan alınmıştır. İlerleme yapmak için, eğitim sırasında tam olarak neler olduğunu düşünmemiz gerekiyor: Eğitim verilerini ve ilişkili etiketleri $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ yineleriz ve her minigruptan sonra modelin ağırlık vektörlerini güncelleriz. Bazen ek olarak parametrelere ağırlık sönümü, hattan düşürme veya başka bir ilgili teknik kullanarak bazı cezalar uygularız. Bu, eğitimdeki kaybı büyük ölçüde en aza indirdiğimiz anlamına gelir.

$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w).
$$

İstatistikçiler ilk terime *deneysel ortalama* diyorlar, yani $P(x) P(y \mid x)$'den alınan veriler üzerinden hesaplanan bir ortalama. Veriler "yanlış" dağıtımdan ($q$) alınmışsa, aşağıdaki basit eşitliği kullanarak bunu düzeltebiliriz:

$$
\begin{aligned}
\int p(\mathbf{x}) f(\mathbf{x}) dx
& = \int q(\mathbf{x}) f(\mathbf{x}) \frac{p(\mathbf{x})}{q(\mathbf{x})} dx.
\end{aligned}
$$

Başka bir deyişle, her bir örneği, doğru dağılımdan, $\beta(\mathbf{x}) := p(\mathbf{x})/q(\mathbf{x})$, elde edilecek olasılık oranına göre yeniden ağırlıklandırmamız gerekir. Ne yazık ki, bu oranı bilmiyoruz, bu yüzden yararlı bir şey yapmadan önce onu tahmin etmemiz gerekiyor. Beklenti operatörünü doğrudan bir minimum norm veya maksimum entropi ilkesi kullanarak yeniden ayarlamaya çalışan bazı süslü operatör-teorik yaklaşımlar da dahil olmak üzere birçok yöntem mevcuttur. Bu tür bir yaklaşım için, her iki dağılımdan da alınan örneklere ihtiyacımız olduğuna dikkat edin---örneğin eğitim verilerine erişen "gerçek" $p$ ve eğitim kümesi oluşturmak için kullanılan $q$ (ikincisi zaten mevcut). Ancak, sadece $\mathbf{x} \sim q(\mathbf{x})$ örneklerine ihtiyacımız olduğuna dikkat edin; $y \sim q(y)$ etiketlerine erişmiyoruz.

Bu durumda, neredeyse o kadar iyi sonuçlar verecek çok etkili bir yaklaşım vardır: Lojistik regresyon. Olasılık oranlarını hesaplamak için gereken tek şey budur. $p(\mathbf{x})$'den alınan veriler ile $q(\mathbf{x})$'den alınan verileri ayırt etmek için bir sınıflandırıcı öğreniyoruz. İki dağılım arasında ayrım yapmak imkansızsa, bu, ilişkili örneklerin iki dağılımdan birinden gelme olasılığının eşit olduğu anlamına gelir. Öte yandan, iyi ayırt edilebilen herhangi bir örnek, buna göre önemli ölçüde yüksek veya düşük ağırlıklı olmalıdır. Basit olması açısından, her iki dağılımdan da eşit sayıda örneğe sahip olduğumuzu ve sırasıyla $\mathbf{x}_i \sim p(\mathbf{x})$ ve $\mathbf{x}_i' \sim q(\mathbf{x})$ diye gösterildiklerini varsayalım. Şimdi, $p$'den alınan veriler için 1 ve $q$'dan alınan veriler için -1 olan $z_i$ etiketlerini belirtelim. Daha sonra, karışık bir veri kümesindeki olasılık şu şekilde verilir:

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Dolayısıyla, lojistik regresyon yaklaşımı kulanırsak, $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-f(\mathbf{x}))}$, şuna varırız:

$$
\beta(\mathbf{x}) = \frac{1/(1 + \exp(-f(\mathbf{x})))}{\exp(-f(\mathbf{x}))/(1 + \exp(-f(\mathbf{x})))} = \exp(f(\mathbf{x})).
$$

Sonuç olarak, iki sorunu çözmemiz gerekiyor: Birincisi her iki dağılımdan alınan verileri ayırt etmek ve ardından terimleri $\beta$ ile yeniden ağırlıklandırılmış bir küçültme problemini çözmek, örneğin ana gradyanlar aracılığıyla. İşte bu amaç için etiketlenmemiş bir eğitim kümesi $X$ ve test kümesi $Z$ kullanan prototip bir algoritma:

1. $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$ ile eğitim kümesi oluşturun.
1. $f$ fonksiyonunu elde etmek için lojistik regresyon kullanarak ikili sınıflandırıcıyı eğitin.
1. $\beta_i = \exp(f(\mathbf{x}_i))$ veya daha iyisi $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$ kullanarak eğitim verilerini ağırlıklandırın.
1. $X$ üzerinde eğitim için $Y$ etiketleriyle $\beta_i$ ağırlıkları kullanın.

Bu yöntemin önemli bir varsayıma dayandığını unutmayın. Bu düzenin çalışması için, hedef (test zamanı) dağılımındaki her veri noktasının eğitim zamanında meydana gelme olasılığının sıfır olmayan bir şekilde olması gerekir. $q(\mathbf{x}) > 0$ ama $p(\mathbf{x}) = 0$ olan bir nokta bulursak, buna karşılık gelen önem ağırlığı sonsuz olmalıdır.

*Çekişmeli Üretici Ağlar*, bir referans veri kümesinden örneklenen örneklerden ayırt edilemeyen verileri çıkaran bir *veri üretici* oluşturmak için yukarıda tarif edilene çok benzer bir fikir kullanır. Bu yaklaşımlarda, gerçek ve sahte verileri ayırt etmek için bir $f$ ağı ve sahte verileri gerçek olarak kabul etmesi için $f$ ayrımcısını kandırmaya çalışan ikinci bir $g$ ağı kullanıyoruz. Bunu daha sonra çok daha detaylı tartışacağız.

### Etiket Kaymasını Düzeltme

$k$-çeşit çok sınıflı bir sınıflandırma görevi ile uğraştığımızı varsayalım. Etiketlerin dağılımı zaman içinde değişir, $p(y) \neq q(y)$, ancak sınıf-koşullu dağılımları aynı $p(\mathbf{x})=q(\mathbf{x})$ olarak kalır. Burada, önem ağırlıklarımız etiket olabilirlik oranlarına $q(y)/p(y)$ karşılık gelecektir. Etiket kayması ile ilgili güzel bir şey, makul derecede iyi bir modelimiz varsa (kaynak dağılımında), ortam boyutuyla uğraşmak zorunda kalmadan bu ağırlıkların tutarlı tahminlerini elde edebiliriz. Derin öğrenmede, girdiler resimler gibi yüksek boyutlu nesneler olma eğilimindedir, etiketler ise genellikle kategoriler gibi daha basit nesnelerdir.

Hedef etiket dağılımını tahmin etmek için, önce makul ölçüde iyi olan raftaki mevcut sınıflandırıcımızı (tipik olarak eğitim verileri üzerinde eğitilmiştir) alıp geçerleme kümesini kullanarak (o da eğitim dağılımından) hata matrisini hesaplıyoruz. Hata matrisi, C, basitçe bir $k \times k$ matrisidir, burada her sütun *gerçek* etikete ve her satır, modelimizce tahmin edilen etikete karşılık gelir. Her bir hücrenin değeri $c_ {ij}$, gerçek etiketin $j$ olduğu *ve* modelimizin $i$ tahmin ettiği tahminlerin oranıdır.

Şimdi, hedef verilerdeki hata matrisini doğrudan hesaplayamayız, çünkü karmaşık bir gerçek zamanlı açıklama veri işleme hattına yatırım yapmazsak, gerçek hayatta gördüğümüz örneklerin etiketlerini göremeyiz. Bununla birlikte, yapabileceğimiz şey, test zamanında tüm model tahminlerimizin ortalamasını almak ve ortalama model çıktısını $\mu_y$ vermek.

Bazı hafif koşullar altında---eğer sınıflandırıcımız ilk etapta makul ölçüde doğruysa ve hedef veriler yalnızca daha önce gördüğümüz imge sınıflarını içeriyorsa ve etiket kayması varsayımı ilk etapta geçerli ise (buradaki en güçlü varsayım), o zaman basit bir doğrusal sistemi $C \cdot q(y) = \mu_y$ çözerek test kümesi etiket dağılımını yeniden elde edebiliriz. Sınıflandırıcımız başlamak için yeterince doğruysa, o zaman hata matrisimiz $C$ tersine çevrilebilir ve $q(y) = C^{-1} \mu_y$ çözümünü elde ederiz. Burada, etiket frekanslarının vektörünü belirtmek için $q(y)$'yi kullanarak gösterimi biraz kötüye kullanıyoruz. Kaynak verilerdeki etiketleri gözlemlediğimiz için, $p(y)$ dağılımını tahmin etmek kolaydır. Daha sonra, $y$ etiketli herhangi bir eğitim örneği $i$'ye ait $w_i$ ağırlığını hesaplamak için tahminlerimizin oranını, $\hat{q}(y)/\hat{p}(y)$, alabiliriz ve bunu yukarıdaki ağırlıklı risk azaltma algoritmasına ekleriz.

### Kavram Kayması Düzeltmesi

Kavram kaymasını ilkeli bir şekilde düzeltmek çok daha zordur. Örneğin, sorunun birdenbire kedileri köpeklerden ayırt etmekten beyaz siyah hayvanları ayırmaya dönüştüğü bir durumda, yeni etiketler toplamadan ve sıfırdan eğitmeden çok daha iyisini yapabileceğimizi varsaymak mantıksız olacaktır. Neyse ki pratikte bu tür aşırı değişimler nadirdir. Bunun yerine, genellikle olan şey, görevin yavaş yavaş değişmeye devam etmesidir. İşleri daha somut hale getirmek için işte bazı örnekler:

* Hesaplamalı reklamcılıkta yeni ürünler piyasaya sürülür, eski ürünler daha az popüler hale gelir. Bu, reklamlar üzerindeki dağılımın ve popülerliğinin kademeli olarak değiştiği ve herhangi bir tıklama oranı tahmincisinin bununla birlikte kademeli olarak değişmesi gerektiği anlamına gelir.
* Trafik kamerası lensleri, çevresel aşınma nedeniyle kademeli olarak bozulur ve görüntü kalitesini aşamalı olarak etkiler.
* Haber içeriği kademeli olarak değişir (yani, haberlerin çoğu değişmeden kalır, ancak yeni hikayeler ortaya çıkar).

Bu gibi durumlarda, ağları verilerdeki değişime adapte etmek için eğitim ağlarında kullandığımız yaklaşımı kullanabiliriz. Başka bir deyişle, mevcut ağ ağırlıklarını kullanıyoruz ve sıfırdan eğitim yerine yeni verilerle birkaç güncelleme adımı gerçekleştiriyoruz.

## Öğrenme Sorunlarının Sınıflandırması

$p(x)$ ve $P(y \mid x)$'teki değişikliklerle nasıl başa çıkılacağı hakkında bilgi sahibi olarak, şimdi makine öğrenmesi problem formülasyonunun diğer bazı yönlerini ele alabiliriz.

* **Toplu Öğrenme.** Burada eğitim verilerine ve $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ etiketlerine erişebiliyoruz, onları $f(x, w)$ ağını eğitme de kullanıyoruz. Daha sonra, aynı dağılımdan çekilen yeni verileri $(x, y)$ puanlamak için bu ağı konuşlandırıyoruz. Bu, burada tartıştığımız sorunlardan herhangi biri için varsayılan varsayımdır. Örneğin, çok sayıda kedi ve köpek resmine dayanarak bir kedi dedektörü eğitebiliriz. Eğittikten sonra, onu yalnızca kedilerin içeri girmesine izin veren akıllı bir kedi kapısı bilgisayar görüş sisteminin parçası olarak gönderiyoruz. Bu daha sonra bir müşterinin evine kurulur ve bir daha asla güncellenmez (aşırı koşullar hariç).
* **Çevrimiçi Öğrenme.** Şimdi $(x_i, y_i)$ verisinin her seferinde bir örnek olarak geldiğini hayal edin. Daha belirleyici olarak, önce $x_i $'i gözlemlediğimizi, ardından bir $f(x_i, w)$ tahmini bulmamız gerektiğini ve yalnızca bunu yaptığımızda $y_i$'yi gözlemlediğimizi ve bununla bizim kararımıza göre bir ödül aldığımızı (veya bir zarar da olabilir) varsayalım. Birçok gerçek sorun bu kategoriye girer. Örneğin, yarınki hisse senedi fiyatını tahmin etmemiz gerekir, bu, bu tahmine dayalı olarak işlem yapmamızı sağlar ve günün sonunda tahminimizin kâr elde etmemize izin verip vermediğini öğreniriz. Başka bir deyişle, yeni gözlemlerle modelimizi sürekli iyileştirdiğimiz aşağıdaki döngüye sahibiz.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ x_t \longrightarrow
\mathrm{estimate} ~ f_t(x_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

* **Kollu Kumar Makinesi.** Yukarıdaki problemin *özel bir durumu*. Çoğu öğrenme probleminde parametrelerini öğrenmek istediğimiz yerde (örneğin derin bir ağ) sürekli değerlerle parametrize edilmiş bir $f$ fonksiyonumuz varken, bir kumar makinesi probleminde çekebileceğimiz sınırlı sayıda kolumuz var (yani, sonlu yapabileceğimiz eylem sayısı). Bu basit problem için optimallik açısından daha güçlü teorik garantilerin elde edilebilmesi çok şaşırtıcı değildir. Onu temel olarak listeliyoruz çünkü bu problem genellikle (kafa karıştırıcı bir şekilde) farklı bir öğrenim ortamı gibi ele alınır.
* **Kontrol (ve çekişmesiz Pekiştirmeli Öğrenme).** Çoğu durumda çevre ne yaptığımızı hatırlar. Mutlaka düşmanca bir şekilde değil, ancak sadece hatırlayacak ve yanıt daha önce ne olduğuna bağlı olacaktır. Örneğin, bir kahve kaynatıcı kontrolörü, kaynatıcıyı önceden ısıtıp ısıtmadığına bağlı olarak farklı sıcaklıklar gözlemleyecektir. PID (orantılı integral türev) kontrolcü algoritmaları burada popüler bir seçimdir. Benzer şekilde, bir kullanıcının bir haber sitesindeki davranışı, ona daha önce ne gösterdiğimize bağlı olacaktır (örneğin, çoğu haberi yalnızca bir kez okuyacaktır). Bu tür birçok algoritma, kararlarının daha az rastgele görünmesini sağlamak (yani, varyansı azaltmak için) gibi hareket ettikleri ortamın bir modelini oluşturur.
* **Pekiştirmeli Öğrenme.** Hafızalı bir ortamın daha genel durumu olarak çevrenin bizimle *işbirliği yapmaya* çalıştığı durumlarla karşılaşabiliriz (özellikle sıfır toplamlı olmayan oyunlar için işbirlikçi oyunlar) veya çevrenin *kazanmaya* çalışacağı diğerler durumlarla. Satranç, Go, Tavla veya StarCraft bu durumlardan bazılarıdır. Aynı şekilde, otonom arabalar için iyi bir kontrolör inşa etmek isteyebiliriz. Diğer arabaların otonom arabanın sürüş tarzına önemsiz şekillerde tepki vermesi muhtemeldir; örneğin, ondan kaçınmaya çalışmak, bir kazaya neden olmaya çalışmak, onunla işbirliği yapmaya çalışmak, vb.

Yukarıdaki farklı durumlar arasındaki önemli bir ayrım, sabit bir ortam durumunda baştan sona işe yaramış olabilecek aynı stratejinin, ortam uyum sağlayabildiğinde baştan sona çalışmayabileceğidir. Örneğin, bir tüccar tarafından keşfedilen bir borsada kar fırsatı, onu kullanmaya başladığında muhtemelen ortadan kalkacaktır. Ortamın değiştiği hız ve tarz, büyük ölçüde uygulayabileceğimiz algoritma türlerini belirler. Örneğin, şeylerin yalnızca yavaş değişebileceğini *bilirsek*, herhangi bir tahmini de yalnızca yavaşça değişmeye zorlayabiliriz. Ortamın aniden değişebileceğini bilirsek, ancak çok seyrek olarak, buna izin verebiliriz. Bu tür bilgiler, hevesli veri bilimcilerinin kavram kayması, yani çözmeye çalıştığı problem zamanla değişmesi, ile başa çıkması için çok önemlidir.

## Makine Öğrenmesinde Adillik, Hesap Verebilirlik ve Şeffaflık

Son olarak, makine öğrenmesi sistemlerini devreye aldığınızda, yalnızca bir tahmine dayalı modeli optimize etmediğinizi, genellikle kararları (kısmen veya tamamen) otomatikleştirmek için kullanılacak bir araç sağladığınızı hatırlamak önemlidir. Bu teknik sistemler, ortaya çıkan kararlara tabi bireylerin yaşamlarını etkileyebilir. Tahminleri değerlendirmekten kararlara sıçrama, yalnızca yeni teknik soruları değil, aynı zamanda dikkatle değerlendirilmesi gereken bir dizi etik soruyu da gündeme getirir. Tıbbi bir teşhis sistemi kuruyorsak, hangi topluluklar için işe yarayıp hangilerinin işe yaramayacağını bilmemiz gerekir. Bir nüfus altgrubunun refahına yönelik öngörülebilir riskleri gözden kaçırmak, daha düşük düzeyde bakım vermemize neden olabilir. Dahası, karar verme sistemlerini düşündüğümüzde geri adım atmalı ve teknolojimizi nasıl değerlendirdiğimizi yeniden düşünmeliyiz. Bu kapsam değişikliğinin diğer sonuçlarının yanı sıra, *doğruluğun* nadiren doğru ölçüm olduğunu göreceğiz. Örneğin, tahminleri eyleme dönüştürürken, genellikle çeşitli şekillerde hataların olası maliyet hassaslığını hesaba katmak isteriz. Bir imgeyi yanlış sınıflandırmanın bir yolu ırkçı bir aldatmaca olarak algılanabilirken, farklı bir kategoriye yanlış sınıflandırma zararsızsa, o zaman eşiklerimizi buna göre ayarlamak ve karar verme protokolünü tasarlarken toplumsal değerleri hesaba katmak isteyebiliriz. Ayrıca tahmin sistemlerinin nasıl geri bildirim döngülerine yol açabileceği konusunda dikkatli olmak istiyoruz. Örneğin, devriye görevlilerini suç oranı yüksek olan alanlara tahsis eden tahmine dayalı polisiye sistemleri düşünün. Endişe verici bir modelin nasıl ortaya çıkacağını kolayca görebiliriz:

 1. Suçun daha fazla olduğu mahallelerde daha fazla devriye gezer.
 1. Sonuç olarak, bu mahallelerde daha fazla suç keşfedilir ve gelecekteki yinelemeler için mevcut eğitim verilerine eklenir.
 1. Daha fazla pozitif örneklere maruz kalan model, bu mahallelerde daha fazla suç öngörür.
 1. Bir sonraki yinelemede, güncellenmiş model aynı mahalleyi daha da yoğun bir şekilde hedef alır ve daha fazla suç keşfedilmesine neden olur vb.

Çoğunlukla, bir modelin tahminlerinin eğitim verileriyle birleştirildiği çeşitli mekanizmalar, modelleme sürecinde hesaba katılmaz. Bu, araştırmacıların "kaçak geri bildirim döngüleri" dediği şeye yol açabilir. Ek olarak, ilk etapta doğru sorunu ele alıp almadığımıza dikkat etmek istiyoruz. Tahmine dayalı algoritmalar artık bilginin yayılmasına aracılık etmede büyük bir rol oynuyor. Bir bireyin karşılaşacağı haberler, *Beğendikleri (Liked)* Facebook sayfalarına göre mi belirlenmeli? Bunlar, makine öğrenmesindeki bir kariyerde karşılaşabileceğiniz birçok baskın etik ikilemden sadece birkaçı.

## Özet

* Çoğu durumda eğitim ve test kümeleri aynı dağıtımdan gelmez. Buna ortak değişken kayması denir.
* İlgili varsayımlar altında, *ortak değişken* ve *etiket* kayması tespit edilebilir ve test zamanında düzeltilebilir. Bu yanlılığın hesaba katılmaması, test zamanında sorunlu hale gelebilir.
* Bazı durumlarda, ortam otomatik eylemleri *hatırlayabilir* ve şaşırtıcı şekillerde yanıt verebilir. Modeller oluştururken bu olasılığı hesaba katmalı ve canlı sistemleri izlemeye devam etmeliyiz, modellerimizin ve çevremizin beklenmedik şekillerde dolaşması olasılığına açık olmalıyız.

## Alıştırmalar

1. Bir arama motorunun davranışını değiştirdiğimizde ne olabilir? Kullanıcılar ne yapabilir? Peki ya reklamverenler?
1. Bir ortak değişken kayması detektörü uygulayınız. İpucu: Bir sınıflandırıcı oluşturunuz.
1. Bir ortak değişken kayması düzelticisi uygulayınız.
1. Eğitim ve test kümeleri çok farklıysa ne ters gidebilir? Örneklem ağırlıklarına ne olur?

[Tartışmalar](https://discuss.d2l.ai/t/105)
