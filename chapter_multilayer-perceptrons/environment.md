# Ortam ve Dağılım Kayması

Önceki bölümlerde, modelleri çeşitli veri kümelerine oturtarak bir dizi uygulamalı makine öğrenmesi üzerinde çalıştık. Yine de, verilerin ilk etapta nereden geldiğini veya modellerimizin çıktılarıyla sonuçta yapmayı planladığımız şeyi düşünmeyi asla bırakmadık. Sıklıkla, verilere sahip olan makine öğrenmesi geliştiricileri, bu temel sorunları dikkate alıp üstlerinde duraklamadan modeller geliştirmek için acele ederler.

Başarısız olan birçok makine öğrenmesi konuşlandırılması bu örüntü ile köklendirilebilir. Bazen modeller, test kümesi doğruluğu ile ölçüldüğünde harika bir performans sergiliyor gibi görünebilir, ancak veri dağılımı aniden değiştiğinde, ürün olarak konuşlandırılınca felaket bir şekilde başarısız olur. Daha sinsi bir şekilde, bazen bir modelin konuşlandırılması, veri dağılımını bozan katalizör olabilir. Örneğin, bir krediyi kimin geri ödeyeceğini tahmin etmek için bir model eğittiğimizi ve başvuranın ayakkabı seçiminin temerrüt (geri ödememe) riskiyle ilişkili olduğunu tespit ettiğimizi varsayalım (Oxfords türü ayakkabılar geri ödemeyi gösterir, spor ayakkabılar temerrüdü gösterir). Daha sonra Oxfords giyen tüm başvuru sahiplerine kredi verme ve spor ayakkabı giyen tüm başvuru sahiplerini reddetme eğiliminde olabiliriz.

Bu durumda, örüntü tanımadan karar vermeye yanlış düşünülmüş bir sıçrayışımız ve ortamı eleştirel bir şekilde dikkate almadaki başarısızlığımız feci sonuçlar doğurabilir. Başlangıç olarak, ayakkabı seçimlerine dayalı kararlar almaya başlar başlamaz, müşteriler farkına varır ve davranışlarını değiştirirdi. Çok geçmeden, tüm başvuru sahipleri, kredi güvenirliliğinde herhangi bir tesadüfi gelişme olmaksızın Oxfords giyeceklerdi. Bunu sindirmek için bir dakikanızı ayırın çünkü makine öğrenmesinin pek çok uygulamasında benzer sorunlar bolca bulunur: Modele dayalı kararlarımızı ortama sunarak modeli bozabiliriz.

Bu konuları tek bir bölümde tam olarak ele alamasak da, burada bazı ortak endişeleri ortaya çıkarmayı ve bu durumları erken tespit etmek, hasarı azaltmak ve makine öğrenmesini sorumlu bir şekilde kullanmak için gereken eleştirel düşünmeyi teşvik etmeyi amaçlıyoruz. Çözümlerden bazıları basittir ("doğru" veriyi istemek), bazıları teknik olarak zordur (pekiştirmeli öğrenme sistemi uygulamak) ve diğerleri, istatistiksel tahmin alanının tamamen dışına çıkmamızı ve algoritmaların etik uygulanması gibi konularla ilgili zor felsefi sorularla boğuşmamızı gerektirir.


## Dağılım Kayması Türleri

Başlangıç olarak, veri dağılımlarının değişebileceği çeşitli yolları ve model performansını kurtarmak için neler yapılabileceğini göz önünde bulundurarak pasif tahminleme ayarına sadık kalıyoruz. Klasik bir kurulumda, eğitim verilerimizin herhangi bir $p_S(\mathbf{x},y)$ dağılımından örneklendiğini, ancak etiketsiz test verilerimizin farklı bir $p_T(\mathbf{x},y)$ dağılımından çekildiğini varsayalım. Şimdiden ciddi bir gerçekle yüzleşmeliyiz. $p_S$ ve $p_T$'nin birbiriyle nasıl ilişkili olduğuna dair herhangi bir varsayım olmadığından, gürbüz bir sınıflandırıcı öğrenmek imkansızdır.

Köpekler ve kediler arasında ayrım yapmak istediğimiz bir ikili sınıflandırma problemini düşünün. Dağılım keyfi şekillerde kayabiliyorsa, kurulumumuz girdiler üzerinden dağılımın sabit kaldığı patolojik bir duruma izin verir: $p_S(\mathbf{x}) = p_T(\mathbf{x})$ ancak etiketlerin tümü ters çevrilmiştir $p_S(y | \mathbf {x}) = 1 - p_T(y | \mathbf{x})$. Başka bir deyişle, eğer Tanrı aniden gelecekte tüm "kedilerin" artık köpek olduğuna ve daha önce "köpek" dediğimiz şeyin artık kedi olduğuna karar verebilirse---$p(\mathbf{x})$ girdilerinin dağılımında herhangi bir değişiklik olmaksızın, bu kurulumu dağılımın hiç değişmediği bir kurulumdan ayırt edemeyiz.

Neyse ki, verilerimizin gelecekte nasıl değişebileceğine dair bazı kısıtlı varsayımlar altında, ilkeli algoritmalar kaymayı algılayabilir ve hatta bazen anında kendilerini uyarlayarak orijinal sınıflandırıcının doğruluğunu iyileştirebilir.


### Ortak Değişken Kayması

Dağılım kayması kategorileri arasında, ortak değişken kayması en yaygın olarak çalışılmışı olabilir. Burada, girdilerin dağılımının zamanla değişebileceğini varsayıyoruz, etiketleme fonksiyonu, yani koşullu dağılım $P(y \mid \mathbf{x})$ değişmez. İstatistikçiler buna *ortak değişken kayması* diyorlar çünkü problem ortak değişkenlerin (öznitelikler) dağılımındaki bir kayma nedeniyle ortaya çıkıyor. Bazen nedenselliğe başvurmadan dağılım kayması hakkında akıl yürütebiliyor olsak da, ortak değişken kaymasının $\mathbf{x}$'in $y$'ye neden olduğuna inandığımız durumlarda çağrılacak doğal varsayım olduğuna dikkat ediyoruz.

Kedileri ve köpekleri ayırt etmenin zorluğunu düşünün. Eğitim verilerimiz :numref:`fig_cat-dog-train` içindeki türden resimlerden oluşabilir:

![Kedi ve köpek ayrımında eğitim kümesi.](../img/cat-dog-train.svg)
:label:`fig_cat-dog-train`

Test zamanında :numref:`fig_cat-dog-test` içindeki resimleri sınıflandırmamız istenebilir:

![Kedi ve köpek ayrımında test kümesi.](../img/cat-dog-test.svg)
:label:`fig_cat-dog-test`

Eğitim kümesi fotoğraflardan oluşurken, test kümesi sadece çizimlerden oluşuyor. Test kümesinden büyük ölçüde farklı özelliklere sahip bir veri kümesi üzerinde eğitim, yeni etki alanına nasıl adapte olacağına dair tutarlı bir planın olmaması sorununu yaratabilir.

### Etiket Kayması

*Etiket kayması* ters problemi tanımlar. Burada, etiketin marjinali $P(y)$'nin değişebileceğini varsayıyoruz ancak sınıf koşullu dağılım $P(\mathbf{x} \mid y)$ etki alanları arasında sabit kalır. Etiket kayması, $y$'nin $\mathbf{x}$'e neden olduğuna inandığımızda yaptığımız makul bir varsayımdır. Örneğin, tanıların göreceli yaygınlığı zamanla değişse bile, semptomları (veya diğer belirtileri) verilen tanıları tahmin etmek isteyebiliriz. Etiket kayması burada uygun varsayımdır çünkü hastalıklar semptomlara neden olur. Bazı yozlaşmış durumlarda, etiket kayması ve ortak değişken kayma varsayımları aynı anda geçerli olabilir. Örneğin, etiket belirlenimci olduğunda, $y$ $\mathbf{x}$'e neden olsa bile, ortak değişken kayma varsayımı karşılanacaktır. İlginç bir şekilde, bu durumlarda, etiket kayması varsayımından kaynaklanan yöntemlerle çalışmak genellikle avantajlıdır. Bunun nedeni, derin öğrenmede yüksek boyutlu olma eğiliminde olan girdiye benzeyen nesnelerin aksine, bu yöntemlerin etiketlere benzeyen (genellikle düşük boyutlu) nesnelerde oynama yapmak eğiliminde olmasıdır.


### Kavram Kayması

Ayrıca etiketlerin tanımları değiştiğinde ortaya çıkan ilgili *kavram kayması* sorunuyla da karşılaşabiliriz. Kulağa garip geliyor---bir *kedi* bir *kedi*dir, değil mi? Ancak, diğer kategoriler zaman içinde kullanımda değişikliklere tabidir. Akıl hastalığı için tanı kriterleri, modaya uygun olanlar ve iş unvanları önemli miktarda kavram kaymasına tabidir. Amerika Birleşik Devletleri çevresinde dolaşırsak, verilerimizin kaynağını coğrafyaya göre değiştirirsek, *meşrubat* adlarının dağılımıyla ilgili olarak, :numref:`fig_popvssoda` içinde gösterildiği gibi önemli bir kavram kayması bulacağımız ortaya çıkar.

![Amerika Birleşik Devletleri'nde meşrubat isimlerinde kavram değişikliği.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Bir makine çeviri sistemi kuracak olsaydık, $P(y \mid \mathbf{x})$ dağılımı konumumuza bağlı olarak farklı olabilirdi. Bu sorunu tespit etmek zor olabilir. Değişimin yalnızca kademeli olarak gerçekleştiği bilgiden hem zamansal hem de coğrafi anlamda yararlanmayı umabiliriz.


### Dağılım Kayması Örnekleri

Biçimselliğe ve algoritmalara girmeden önce, ortak değişken veya kavram kaymasının bariz olmayabileceği bazı somut durumları tartışabiliriz.


### Tıbbi Teşhis

Kanseri tespit etmek için bir algoritma tasarlamak istediğinizi hayal edin. Sağlıklı ve hasta insanlardan veri topluyorsunuz ve algoritmanızı geliştiriyorsunuz. İyi çalışıyor, size yüksek doğruluk sağlıyor ve tıbbi teşhis alanında başarılı bir kariyere hazır olduğunuz sonucuna varıyorsunuz. *O kadar da hızlı değil.*

Eğitim verilerini ortaya çıkaranlar ile gerçek hayatta karşılaşacağınız dağılımlar önemli ölçüde farklılık gösterebilir. Bu, bizden bazılarının (biz yazarlar) yıllar önce çalıştığı talihsiz bir girişimin başına geldi. Çoğunlukla yaşlı erkekleri etkileyen bir hastalık için bir kan testi geliştiriyorlardı ve hastalardan topladıkları kan örneklerini kullanarak bu hastalığı incelemeyi umuyorlardı. Bununla birlikte, sağlıklı erkeklerden kan örnekleri almak, sistemdeki mevcut hastalardan almaktan çok daha zordur. Girişim, bununla başa çıkmak için, bir üniversite kampüsündeki öğrencilerden testlerini geliştirmede sağlıklı kontrol grubu olmaları amacıyla kan bağışı istedi. Ardından, onlara hastalığı tespit etmek için bir sınıflandırıcı oluşturmalarına yardım edip edemeyeceğimiz soruldu.

Onlara açıkladığımız gibi, sağlıklı ve hasta grupları neredeyse mükemmel bir doğrulukla ayırt etmek gerçekten kolay olurdu. Çünkü, bunun nedeni, deneklerin yaş, hormon seviyeleri, fiziksel aktivite, diyet, alkol tüketimi ve hastalıkla ilgisi olmayan daha birçok faktör açısından farklılık göstermesidir. Gerçek hastalarda durum böyle değildi. Örneklem prosedürleri nedeniyle, aşırı ortak değişken kayması ile karşılaşmayı bekleyebiliriz. Dahası, bu durumun geleneksel yöntemlerle düzeltilmesi pek olası değildir. Kısacası, önemli miktarda para israf ettiler.

### Kendi Kendine Süren Arabalar

Bir şirketin sürücüsüz otomobiller geliştirmek için makine öğrenmesinden yararlanmak istediğini varsayalım. Buradaki temel bileşenlerden biri yol kenarı detektörüdür. Gerçek açıklamalı verilerin elde edilmesi pahalı olduğu için, bir oyun oluşturma motorundan gelen sentetik verileri ek eğitim verileri olarak kullanma (zekice ve şüpheli) fikirleri vardı. Bu, işleme motorundan alınan "test verileri" üzerinde gerçekten iyi çalıştı. Ne yazık ki, gerçek bir arabanın içinde tam bir felaketti. Görünüşe göre oluşturulan yol kenarı çok basit bir dokuya sahipti. Daha da önemlisi, *tüm* yol kenarı *aynı* dokuya sahipti ve yol kenarı dedektörü bu "özniteliği" çok çabuk öğrendi.

ABD Ordusu ormandaki tankları ilk defa tespit etmeye çalıştıklarında da benzer bir şey oldu. Ormanın tanksız hava fotoğraflarını çektiler, ardından tankları ormana sürdüler ve bir dizi fotoğraf daha çektiler. Sınıflandırıcının *mükemmel* çalıştığı görüldü. Ne yazık ki, o sadece gölgeli ağaçları gölgesiz ağaçlardan nasıl ayırt edeceğini öğrenmişti---ilk fotoğraf kümesi sabah erken, ikinci küme öğlen çekilmişti.

### Durağan Olmayan Dağılımlar

Dağılım yavaş değiştiğinde ve model yeterince güncellenmediğinde çok daha hassas bir durum ortaya çıkar (*durağan olmayan dağılımlar* da diye de bilinir). Bazı tipik durumlar aşağıdadır:

* Bir hesaplamalı reklamcılık modelini eğitiyor ve ardından onu sık sık güncellemekte başarısız oluyoruz (örneğin, iPad adı verilen belirsiz yeni bir cihazın henüz piyasaya sürüldüğünü dahil etmeyi unuttuk).
* Bir yaramaz posta filtresi oluşturuyoruz. Şimdiye kadar gördüğümüz tüm yaramaz postaları tespit etmede iyi çalışıyor. Ancak daha sonra, yaramaz posta gönderenler akıllanıyor ve daha önce gördüğümüz hiçbir şeye benzemeyen yeni mesajlar oluşturuyorlar.
* Ürün öneri sistemi oluşturuyoruz. Kış boyunca işe yarıyor, ancak Noel'den sonra da Noel Baba şapkalarını önermeye devam ediyor.

### Birkaç Benzer Tecrübe

* Yüz dedektörü yapıyoruz. Tüm kıyaslamalarda iyi çalışıyor. Ne yazık ki test verilerinde başarısız oluyor---zorlayıcı örnekler, yüzün tüm resmi doldurduğu yakın çekimlerdir (eğitim kümesinde böyle bir veri yoktu).
* ABD pazarı için bir web arama motoru oluşturuyoruz ve bunu Birleşik Krallık'ta kullanmak istiyoruz.
* Büyük bir sınıf kümesinin her birinin veri kümesinde eşit olarak temsil edildiği büyük bir veri kümesi derleyerek bir imge sınıflandırıcı eğitiyoruz, örneğin 1000 kategori var ve her biri 1000 görüntü ile temsil ediliyor. Ardından sistemi, fotoğrafların gerçek etiket dağılımının kesinlikle tekdüze olmadığı gerçek dünyada konuşlandırıyoruz.

## Dağılım Kaymasını Düzeltme

Daha önce tartıştığımız gibi, $P(\mathbf {x}, y)$ eğitim ve test dağılımlarının farklı olduğu birçok durum vardır. Bazı durumlarda şanslıyız ve modeller ortak değişken, etiket veya kavram kaymasına rağmen çalışıyor. Diğer durumlarda, kaymalarla başa çıkmak için ilkeli stratejiler kullanarak daha iyisini yapabiliriz. Bu bölümün geri kalanı önemli ölçüde daha teknik hale geliyor. Sabırsız okuyucu bir sonraki bölüme geçebilir çünkü bu içerik sonraki kavramlar için ön koşul değildir.

### Deneysel Risk ve Risk
:label:`subsec_empirical-risk-and-risk`

Önce model eğitimi sırasında tam olarak ne olduğunu düşünelim: Eğitim verilerinin öznitelikleri ve ilişkili etiketleri $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ üzerinde yineleniriz ve her minigruptan sonra $f$ modelinin parametrelerini güncelleriz. Basit olması için düzenlileştirmeyi düşünmüyoruz, böylece eğitimdeki kaybı büyük ölçüde en aza indiriyoruz:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

burada $l$, ilişkili $y_i$ etiketi ile verilen $f(\mathbf{x}_i)$ tahmininin "ne kadar kötü" olduğunu ölçen kayıp işlevidir. İstatistikçiler :eqref:`eq_empirical-risk-min` içindeki bu terimi  *deneysel risk* olarak adlandırır. *Deneysel risk*,  $p(\mathbf{x},y)$ gerçek dağılımından elde edilen tüm veri popülasyonu üzerindeki kaybın beklentisi olan *riske* yaklaştıran eğitim verileri üzerindeki ortalama kayıptır:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

Bununla birlikte, pratikte tipik olarak tüm veri popülasyonunu elde edemeyiz. Bu nedenle, :eqref:`eq_empirical-risk-min` içindeki deneysel riski en aza indiren *deneysel risk minimizasyonu*, riski yaklaşık olarak en aza indirmeyi uman makine öğrenmesi için pratik bir stratejidir.

### Ortak Değişken Kaymasını Düzeltme
:label:`subsec_covariate-shift-correction`

Verileri $(\mathbf{x}_i, y_i)$ olarak etiketlediğimiz $P(y \mid \mathbf{x})$ bağımlılığını tahmin etmek istediğimizi varsayalım. Ne yazık ki, $\mathbf{x}_i$ gözlemleri, *hedef dağılımı* $p(\mathbf{x})$ yerine bazı *kaynak dağılımı* $q(\mathbf{x})$'den alınmıştır.
Neyse ki, bağımlılık varsayımı koşullu dağılımın değişmediği anlamına gelir: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$. $q(\mathbf{x})$ kaynak dağılımı "yanlış" ise, riskte aşağıdaki basit özdeşlik kullanarak bunu düzeltebiliriz:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Başka bir deyişle, her bir veri örneğini, doğru dağılımdan alınmış olma olasılığının yanlış dağılımdan alınmış olma olasılığına oranıyla yeniden ağırlıklandırmamız gerekir:

$$\beta_i \stackrel{\mathrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Her veri örneği $(\mathbf{x}_i, y_i)$ için $\beta_i$ ağırlığını ekleyerek modelimizi *ağırlıklı deneysel risk minimizasyonu* kullanarak eğitebiliriz:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`



Ne yazık ki, bu oranı bilmiyoruz, bu yüzden faydalı bir şey yapmadan önce onu tahmin etmemiz gerekiyor. Beklenti işlemini bir minimum norm veya bir maksimum entropi ilkesi kullanarak doğrudan yeniden ayar etmeye çalışan bazı süslü operatör-teorik yaklaşımlar dahil olmak üzere birçok yöntem mevcuttur. Bu tür herhangi bir yaklaşım için, her iki dağılımdan da alınan örneklere ihtiyacımız olduğuna dikkat edin: "Doğru" $p$, örneğin, test verilerine erişim yoluyla elde edilen ve $q$ eğitim kümesini oluşturmak için kullanılan (sonraki bariz mevcuttur). Ancak, yalnızca $\mathbf{x} \sim p(\mathbf{x})$ özniteliklerine ihtiyacımız olduğunu unutmayın; $y \sim p(y)$ etiketlerine erişmemize gerek yok.

Bu durumda, neredeyse orijinali kadar iyi sonuçlar verecek çok etkili bir yaklaşım vardır: İkili sınıflandırmada softmaks regresyonun özel bir durumu olan (bkz. :numref:`sec_softmax`) lojistik regresyon. Tahmini olasılık oranlarını hesaplamak için gereken tek şey budur. $p(\mathbf{x})$'den alınan veriler ile $q(\mathbf{x})$'den alınan verileri ayırt etmek için bir sınıflandırıcı öğreniyoruz. İki dağılım arasında ayrım yapmak imkansızsa, bu, ilişkili örneklerin iki dağılımdan birinden gelme olasılığının eşit olduğu anlamına gelir. Öte yandan, iyi ayırt edilebilen herhangi bir örnek, buna göre önemli ölçüde yüksek veya düşük ağırlıklı olmalıdır. 

Basit olması açısından, her iki dağılımdan da eşit sayıda örneğe sahip olduğumuzu ve sırasıyla $p(\mathbf{x})$ ve $q(\mathbf{x})$ diye gösterildiklerini varsayalım. Şimdi, $p$'den alınan veriler için $1$ ve $q$'dan alınan veriler için $-1$ olan $z$ etiketlerini belirtelim. Daha sonra, karışık bir veri kümesindeki olasılık şu şekilde verilir:

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ ve bu yüzden } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Bu nedenle, lojistik regresyon yaklaşımını kullanırsak, öyleki $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ parametreli bir fonksiyondur), aşağıdaki sonuca varırız:

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

Sonuç olarak, iki sorunu çözmemiz gerekiyor: İlk olarak her iki dağılımdan alınan verileri ayırt etme ve ardından terimleri $\beta_i$ ile ağırlıklandırdığımız :eqref:`eq_weighted-empirical-risk-min` içindeki ağırlıklı deneysel risk minimizasyon problemi. 

Artık bir düzeltme algoritması tanımlamaya hazırız. $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ eğitim kümemiz ve etiketlenmemiş bir $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$ test kümemiz olduğunu varsayalım. Ortak değişken kaydırma için, tüm $1 \leq i \leq n$ için $\mathbf{x}_i$'nin bir kaynak dağılımından ve tüm $1 \leq i \leq m$ için $\mathbf{u}_i$'in hedef dağılımdan çekildiğini varsayıyoruz. İşte ortak değişken kaymasını düzeltmek için prototipik bir algoritma:

1. Bir ikili sınıflandırma eğitim kümesi oluşturun: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1) , \ldots, (\mathbf{u}_m, 1)\}$.
1. $h$ fonksiyonunu elde etmek için lojistik regresyon kullanarak bir ikili sınıflandırıcı eğitin.
1. $\beta_i = \exp(h(\mathbf{x}_i))$ veya daha iyisi $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$, $c$ herhangi bir sabittir, kullanarak eğitim verilerini ağırlıklandırın.
1. $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ üzerinde eğitim için :eqref:`eq_weighted-empirical-risk-min` içindeki $\beta_i$ ağırlıklarını kullanın.

Yukarıdaki algoritmanın önemli bir varsayıma dayandığını unutmayın. Bu düzenin çalışması için, hedef (örn. test zamanı) dağılımındaki her veri örneğinin eğitim zamanında meydana gelme olasılığının sıfır olmayan bir şekilde olması gerekir. $p(\mathbf{x}) > 0$ ama $q(\mathbf{x}) = 0$ olan bir nokta bulursak, buna karşılık gelen önem ağırlığı sonsuz olmalıdır.


### Etiket Kaymasını Düzeltme

$k$ kategorili bir sınıflandırma göreviyle uğraştığımızı varsayalım. :numref:`subsec_covariate-shift-correction` içindeki aynı gösterimi kullanarak, $q$ ve $p$ sırasıyla kaynak dağılımı (ör. eğitim zamanı) ve hedef dağılımıdır (ör. test zamanı).
Etiketlerin dağılımının zaman içinde değiştiğini varsayın: $q(y) \neq p(y)$, ancak sınıf-koşullu dağılım aynı kalır: $q(\mathbf{x} \mid y)=p(\mathbf {x} \mid y)$.
$q(y)$ kaynak dağılımı "yanlış" ise, bunu :eqref:`eq_true-risk` içinde tanımlanan riskte aşağıdaki özdeşliğe göre düzeltebiliriz:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$



Burada önem ağırlıklarımız etiket olabilirlik oranlarına karşılık gelecektir.

$$\beta_i \stackrel{\mathrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

Etiket kayması ile ilgili güzel bir şey, kaynak dağılımı üzerinde oldukça iyi bir modelimiz varsa, ortam boyutuyla hiç uğraşmadan bu ağırlıkların tutarlı tahminlerini elde edebilmemizdir.
Derin öğrenmede girdiler, imgeler gibi yüksek boyutlu nesneler olma eğilimindeyken, etiketler genellikle kategoriler gibi daha basit nesnelerdir.

Hedef etiket dağılımını tahmin etmek için, önce makul ölçüde iyi olan kullanıma hazır mevcut sınıflandırıcımızı (tipik olarak eğitim verileri üzerinde eğitilmiştir) alıp geçerleme kümesini kullanarak (o da eğitim dağılımından) hata matrisini hesaplıyoruz. *Hata matrisi*, $\mathbf{C}$, basitçe bir $k \times k$ matrisidir, burada her sütun etiket sınıfına (temel doğru) ve her satır, modelimizce tahmin edilen sınıfa karşılık gelir. Her bir hücrenin değeri $c_ {ij}$, geçerleme kümesinde gerçek etiketin $j$ olduğu ve modelimizin $i$ tahmin ettiği toplam tahminlerin oranıdır.

Şimdi, hedef verilerdeki hata matrisini doğrudan hesaplayamayız, çünkü karmaşık bir gerçek zamanlı açıklama veri işleme hattına yatırım yapmazsak, gerçek hayatta gördüğümüz örneklerin etiketlerini göremeyiz. Ancak yapabileceğimiz şey, birlikte test zamanında tüm model tahminlerimizin ortalamasıdır ve ortalama model çıktılarını $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$ verir, ki $i.$ öğesi $\mu(\hat{y}_i)$, modelimizin $i$ tahmin ettiği test kümesindeki toplam tahminlerin oranıdır.

Bazı ılımlı koşullar altında --- eğer sınıflandırıcımız ilk etapta makul ölçüde doğruysa ve hedef veriler yalnızca daha önce gördüğümüz kategorileri içeriyorsa ve ilk etapta etiket kayması varsayımı geçerliyse (en güçlü varsayım), o zaman basit bir doğrusal sistemi çözerek test kümesi etiket dağılımını tahmin edebiliriz.

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

çünkü bir tahmin olarak $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ tüm $1 \leq i \leq k$ için geçerlidir, burada $p(y_j)$, $k$ boyutlu etiket dağılım vektörü $p(\mathbf{y})$'nin $j.$ öğesidir. Sınıflandırıcımız başlangıç için yeterince doğruysa, o zaman $\mathbf{C}$ hata matrisi tersine çevrilebilir ve $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$ çözümünü elde ederiz.

Kaynak verilerdeki etiketleri gözlemlediğimiz için $q(y)$ dağılımını tahmin etmek kolaydır. Ardından, $y_i$ etiketli herhangi bir eğitim örneği $i$ için, $\beta_i$ ağırlığını hesaplamak için tahmini $p(y_i)/q(y_i)$ oranını alabilir ve bunu :eqref:`eq_weighted-empirical-risk-min` içindeki ağırlıklı deneysel risk minimizasyonuna bağlayabiliriz.


### Kavram Kayması Düzeltmesi

Kavram kaymasını ilkeli bir şekilde düzeltmek çok daha zordur. Örneğin, sorunun birdenbire kedileri köpeklerden ayırt etmekten beyaz siyah hayvanları ayırmaya dönüştüğü bir durumda, yeni etiketler toplamadan ve sıfırdan eğitmeden çok daha iyisini yapabileceğimizi varsaymak mantıksız olacaktır. Neyse ki pratikte bu tür aşırı değişimler nadirdir. Bunun yerine, genellikle olan şey, görevin yavaş yavaş değişmeye devam etmesidir. İşleri daha somut hale getirmek için işte bazı örnekler:

* Hesaplamalı reklamcılıkta yeni ürünler piyasaya sürülür, eski ürünler daha az popüler hale gelir. Bu, reklamlar üzerindeki dağılımın ve popülerliğinin kademeli olarak değiştiği ve herhangi bir tıklama oranı tahmincisinin bununla birlikte kademeli olarak değişmesi gerektiği anlamına gelir.
* Trafik kamerası lensleri, çevresel aşınma nedeniyle kademeli olarak bozulur ve görüntü kalitesini aşamalı olarak etkiler.
* Haber içeriği kademeli olarak değişir (yani, haberlerin çoğu değişmeden kalır, ancak yeni hikayeler ortaya çıkar).

Bu gibi durumlarda, ağları verilerdeki değişime adapte etmek için eğitim ağlarında kullandığımız yaklaşımı kullanabiliriz. Başka bir deyişle, mevcut ağ ağırlıklarını kullanıyoruz ve sıfırdan eğitim yerine yeni verilerle birkaç güncelleme adımı gerçekleştiriyoruz.

## Öğrenme Sorunlarının Sınıflandırması

Dağılımlardaki değişikliklerle nasıl başa çıkılacağı hakkında bilgi sahibi olarak, şimdi makine öğrenmesi problem formülasyonunun diğer bazı yönlerini ele alabiliriz.

### Toplu Öğrenme

*Toplu öğrenmede*, bir model, $f(\mathbf{x})$, eğitmek için kullandığımız $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ eğitim özniteliklerine ve etiketlerine erişimimiz var. Daha sonra, aynı dağılımdan çekilen yeni verileri $(\mathbf{x}, y)$ değerlendirmek için bu modeli konuşlandırabiliriz. Bu, burada tartıştığımız sorunların herhangi biri için varsayılan varsayımdır. Örneğin, birçok kedi ve köpek resmine dayalı bir kedi dedektörü eğitebiliriz. Onu eğittikten sonra, sadece kedilerin içeri girmesine izin veren akıllı bir kedi kapısı bilgisayarla görme sisteminin parçası olarak göndeririz. Bu daha sonra bir müşterinin evine kurulur ve bir daha asla güncellenmez (aşırı durumlar hariç).

### Çevrimiçi Öğrenme

Şimdi $(\mathbf{x}_i, y_i)$ verisinin her seferinde bir örnek olarak geldiğini hayal edin. Daha belirleyici olarak, önce $\mathbf{x}_i$'i gözlemlediğimizi, ardından bir $f(\mathbf{x}_i)$ tahmini bulmamız gerektiğini ve yalnızca bunu yaptığımızda $y_i$'yi gözlemlediğimizi ve bununla bizim kararımıza göre bir ödül veya bir ceza aldığımızı varsayalım. 
Birçok gerçek sorun bu kategoriye girer. Örneğin, yarınki hisse senedi fiyatını tahmin etmemiz gerekir, bu, bu tahmine dayalı olarak işlem yapmamızı sağlar ve günün sonunda tahminimizin kâr elde etmemize izin verip vermediğini öğreniriz. Başka bir deyişle, *çevrimiçi öğrenme*de, yeni gözlemlerle modelimizi sürekli iyileştirdiğimiz aşağıdaki döngüye sahibiz.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{veri} ~ \mathbf{x}_t \longrightarrow
\mathrm{tahmin} ~ f_t(\mathbf{x}_t) \longrightarrow
\mathrm{gözlem} ~ y_t \longrightarrow
\mathrm{kayıp} ~ l(y_t, f_t(\mathbf{x}_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

### Kollu Kumar Makinesi

*Kollu kumar makinesi* yukarıdaki problemin özel bir durumudur. Çoğu öğrenme probleminde parametrelerini öğrenmek istediğimiz yerde (örneğin derin bir ağ) sürekli değerlerle parametrize edilmiş bir $f$ fonksiyonumuz varken, bir *kollu kumar makinesi* probleminde çekebileceğimiz sınırlı sayıda kolumuz var, yani, sonlu yapabileceğimiz eylem sayısı. Bu basit problem için optimallik açısından daha güçlü teorik garantilerin elde edilebilmesi çok şaşırtıcı değildir. Onu temel olarak listeliyoruz çünkü bu problem genellikle (kafa karıştırıcı bir şekilde) farklı bir öğrenim ortamı gibi ele alınır.

### Kontrol

Çoğu durumda ortam ne yaptığımızı hatırlar. Mutlaka düşmanca bir şekilde değil, ancak sadece hatırlayacak ve yanıt daha önce ne olduğuna bağlı olacaktır. Örneğin, bir kahve kaynatıcı kontrolörü, kaynatıcıyı önceden ısıtıp ısıtmadığına bağlı olarak farklı sıcaklıklar gözlemleyecektir. PID (orantılı integral türev) kontrolcü algoritmaları burada popüler bir seçimdir. Benzer şekilde, bir kullanıcının bir haber sitesindeki davranışı, ona daha önce ne gösterdiğimize bağlı olacaktır (örneğin, çoğu haberi yalnızca bir kez okuyacaktır). Bu tür birçok algoritma, kararlarının daha az rastgele görünmesini sağlamak gibi hareket ettikleri ortamın bir modelini oluşturur.
Son zamanlarda, kontrol teorisi (örneğin, PID varyantları), daha iyi çözme ve geri çatma kalitesi elde etmek ve oluşturulan metnin çeşitliliğini ve oluşturulan görüntülerin geri çatma kalitesini iyileştirmek için hiper parametreleri otomatik olarak ayarlamak için de kullanıldı :cite:`Shao.Yao.Sun.ea.2020`.

### Pekiştirmeli Öğrenme

Hafızalı bir ortamın daha genel durumu olarak ortamın bizimle işbirliği yapmaya çalıştığı durumlarla karşılaşabiliriz (özellikle sıfır toplamlı olmayan oyunlar için işbirlikçi oyunlar) veya çevrenin kazanmaya çalışacağı diğerler durumlarla. Satranç, Go, Tavla veya StarCraft pekiştirmeli öğrenme vakalardan bazılarıdır. Aynı şekilde, otonom arabalar için iyi bir kontrolör inşa etmek isteyebiliriz. Diğer arabaların otonom arabanın sürüş tarzına önemsiz şekillerde tepki vermesi muhtemeldir; örneğin, ondan kaçınmaya çalışmak, bir kazaya neden olmamaya çalışmak ve onunla işbirliği yapmaya çalışmak.

### Ortamı Düşünmek

Yukarıdaki farklı durumlar arasındaki önemli bir ayrım, sabit bir ortam durumunda baştan sona işe yaramış olabilecek aynı stratejinin, ortam uyum sağlayabildiğinde baştan sona çalışmayabileceğidir. Örneğin, bir tüccar tarafından keşfedilen bir borsada kar fırsatı, onu kullanmaya başladığında muhtemelen ortadan kalkacaktır. Ortamın değiştiği hız ve tarz, büyük ölçüde uygulayabileceğimiz algoritma türlerini belirler. Örneğin, nesnelerin yalnızca yavaş değişebileceğini bilirsek, herhangi bir tahmini de yalnızca yavaşça değişmeye zorlayabiliriz. Ortamın aniden değişebileceğini bilirsek, ancak çok seyrek olarak, buna izin verebiliriz. Bu tür bilgiler, hevesli veri bilimcilerinin kavram kayması, yani çözmeye çalıştığı problem zamanla değişmesi, ile başa çıkması için çok önemlidir.

## Makine Öğrenmesinde Adillik, Hesap Verebilirlik ve Şeffaflık

Son olarak, makine öğrenmesi sistemlerini devreye aldığınızda, yalnızca bir tahmine dayalı modeli optimize etmediğinizi, genellikle kararları (kısmen veya tamamen) otomatikleştirmek için kullanılacak bir araç sağladığınızı hatırlamak önemlidir. Bu teknik sistemler, ortaya çıkan kararlara tabi bireylerin yaşamlarını etkileyebilir. Tahminleri değerlendirmekten kararlara sıçrama, yalnızca yeni teknik soruları değil, aynı zamanda dikkatle değerlendirilmesi gereken bir dizi etik soruyu da gündeme getirir. Tıbbi bir teşhis sistemi kuruyorsak, hangi topluluklar için işe yarayıp hangilerinin işe yaramayacağını bilmemiz gerekir. Bir nüfus altgrubunun refahına yönelik öngörülebilir riskleri gözden kaçırmak, daha düşük düzeyde bakım vermemize neden olabilir. Dahası, karar verme sistemlerini düşündüğümüzde geri adım atmalı ve teknolojimizi nasıl değerlendirdiğimizi yeniden düşünmeliyiz. Bu kapsam değişikliğinin diğer sonuçlarının yanı sıra, *doğruluğun* nadiren doğru ölçü olduğunu göreceğiz. Örneğin, tahminleri eyleme dönüştürürken, genellikle çeşitli şekillerde hataların olası maliyet hassaslığını hesaba katmak isteriz. Bir imgeyi yanlış sınıflandırmanın bir yolu ırkçı bir aldatmaca olarak algılanabilirken, farklı bir kategoriye yanlış sınıflandırma zararsızsa, o zaman eşiklerimizi buna göre ayarlamak ve karar verme protokolünü tasarlarken toplumsal değerleri hesaba katmak isteyebiliriz. Ayrıca tahmin sistemlerinin nasıl geri bildirim döngülerine yol açabileceği konusunda dikkatli olmak istiyoruz. Örneğin, devriye görevlilerini suç oranı yüksek olan alanlara tahsis eden tahmine dayalı polisiye sistemleri düşünün. Endişe verici bir modelin nasıl ortaya çıkacağını kolayca görebiliriz:

 1. Suçun daha fazla olduğu mahallelerde daha fazla devriye gezer.
 1. Sonuç olarak, bu mahallelerde daha fazla suç keşfedilir ve gelecekteki yinelemeler için mevcut eğitim verilerine eklenir.
 1. Daha fazla pozitif örneklere maruz kalan model, bu mahallelerde daha fazla suç öngörür.
 1. Bir sonraki yinelemede, güncellenmiş model aynı mahalleyi daha da yoğun bir şekilde hedef alır ve daha fazla suç keşfedilmesine neden olur vb.

Çoğunlukla, bir modelin tahminlerinin eğitim verileriyle birleştirildiği çeşitli mekanizmalar, modelleme sürecinde hesaba katılmaz. Bu, araştırmacıların *kaçak geri bildirim döngüleri* dediği şeye yol açabilir. Ek olarak, ilk etapta doğru sorunu ele alıp almadığımıza dikkat etmek istiyoruz. Tahmine dayalı algoritmalar artık bilginin yayılmasına aracılık etmede büyük bir rol oynuyor. Bir bireyin karşılaşacağı haberler, *Beğendikleri (Liked)* Facebook sayfalarına göre mi belirlenmeli? Bunlar, makine öğrenmesindeki bir kariyerde karşılaşabileceğiniz birçok baskın etik ikilemden sadece birkaçı.

## Özet

* Çoğu durumda eğitim ve test kümeleri aynı dağılımdan gelmez. Buna dağılım kayması denir.
* Risk, gerçek dağılımlarından elde edilen tüm veri popülasyonu üzerindeki kayıp beklentisidir. Ancak, bu popülasyonun tamamı genellikle mevcut değildir. Deneysel risk, riski yaklaşık olarak tahmin etmek için eğitim verileri üzerinden ortalama bir kayıptır. Uygulamada, deneysel risk minimizasyonu gerçekleştiriyoruz.
* İlgili varsayımlar altında, ortak değişken ve etiket kayması tespit edilebilir ve test zamanında düzeltilebilir. Bu yanlılığın hesaba katılmaması, test zamanında sorunlu hale gelebilir.
* Bazı durumlarda, ortam otomatik eylemleri hatırlayabilir ve şaşırtıcı şekillerde yanıt verebilir. Modeller oluştururken bu olasılığı hesaba katmalı ve canlı sistemleri izlemeye devam etmeliyiz, modellerimizin ve ortamın beklenmedik şekillerde dolaşması olasılığına açık olmalıyız.

## Alıştırmalar

1. Bir arama motorunun davranışını değiştirdiğimizde ne olabilir? Kullanıcılar ne yapabilir? Peki ya reklamverenler?
1. Bir ortak değişken kayması detektörü uygulayınız. İpucu: Bir sınıflandırıcı oluşturunuz.
1. Bir ortak değişken kayması düzelticisi uygulayınız.
1. Dağılım kaymasının yanı sıra, deneysel riskin riske yaklaşmasını başka ne etkileyebilir?

[Tartışmalar](https://discuss.d2l.ai/t/105)
