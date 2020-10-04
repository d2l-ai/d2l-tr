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


### Örenkler

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

### Label Shift Correction

Assume that we are dealing with a $k$-way multiclass classification task. When the distribution of labels shifts over time, $p(y) \neq q(y)$ but the class-conditional distributions stay the same $p(\mathbf{x})=q(\mathbf{x})$. Here, our importance weights will correspond to the label likelihood ratios $q(y)/p(y)$. One nice thing about label shift is that if we have a reasonably good model (on the source distribution) then we can get consistent estimates of these weights without ever having to deal with the ambient dimension In deep learning, the inputs tend to be high-dimensional objects like images, while the labels are often simpler objects like categories.

To estimate the target label distribution, we first take our reasonably good off the shelf classifier (typically trained on the training data) and compute its confusion matrix using the validation set (also from the training distribution). The confusion matrix, C, is simply a $k \times k$ matrix, where each column corresponds to the *actual* label and each row corresponds to our model's predicted label. Each cell's value $c_{ij}$ is the fraction of predictions where the true label was $j$ *and* our model predicted $i$.

Now, we cannot calculate the confusion matrix on the target data directly, because we do not get to see the labels for the examples that we see in the wild, unless we invest in a complex real-time annotation pipeline. What we can do, however, is average all of our models predictions at test time together, yielding the mean model output $\mu_y$.

It turns out that under some mild conditions---if our classifier was reasonably accurate in the first place, and if the target data contains only classes of images that we have seen before, and if the label shift assumption holds in the first place (the strongest assumption here), then we can recover the test set label distribution by solving a simple linear system $C \cdot q(y) = \mu_y$. If our classifier is sufficiently accurate to begin with, then the confusion $C$ will be invertible, and we get a solution $q(y) = C^{-1} \mu_y$. Here we abuse notation a bit, using $q(y)$ to denote the vector of label frequencies. Because we observe the labels on the source data, it is easy to estimate the distribution $p(y)$. Then for any training example $i$ with label $y$, we can take the ratio of our estimates $\hat{q}(y)/\hat{p}(y)$ to calculate the weight $w_i$, and plug this into the weighted risk minimization algorithm above.


### Concept Shift Correction

Concept shift is much harder to fix in a principled manner. For instance, in a situation where suddenly the problem changes from distinguishing cats from dogs to one of distinguishing white from black animals, it will be unreasonable to assume that we can do much better than just collecting new labels and training from scratch. Fortunately, in practice, such extreme shifts are rare. Instead, what usually happens is that the task keeps on changing slowly. To make things more concrete, here are some examples:

* In computational advertising, new products are launched, old products become less popular. This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic camera lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).

In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.

## A Taxonomy of Learning Problems

Armed with knowledge about how to deal with changes in $p(x)$ and in $P(y \mid x)$, we can now consider some other aspects of machine learning problem formulation.


* **Batch Learning.** Here we have access to training data and labels $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, which we use to train a network $f(x, w)$. Later on, we deploy this network to score new data $(x, y)$ drawn from the same distribution. This is the default assumption for any of the problems that we discuss here. For instance, we might train a cat detector based on lots of pictures of cats and dogs. Once we trained it, we ship it as part of a smart catdoor computer vision system that lets only cats in. This is then installed in a customer's home and is never updated again (barring extreme circumstances).
* **Online Learning.** Now imagine that the data $(x_i, y_i)$ arrives one sample at a time. More specifically, assume that we first observe $x_i$, then we need to come up with an estimate $f(x_i, w)$ and only once we have done this, we observe $y_i$ and with it, we receive a reward (or incur a loss), given our decision. Many real problems fall into this category. E.g., we need to predict tomorrow's stock price, this allows us to trade based on that estimate and at the end of the day we find out whether our estimate allowed us to make a profit. In other words, we have the following cycle where we are continuously improving our model given new observations.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ x_t \longrightarrow
\mathrm{estimate} ~ f_t(x_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

* **Bandits.** They are a *special case* of the problem above. While in most learning problems we have a continuously parametrized function $f$ where we want to learn its parameters (e.g., a deep network), in a bandit problem we only have a finite number of arms that we can pull (i.e., a finite number of actions that we can take). It is not very surprising that for this simpler problem stronger theoretical guarantees in terms of optimality can be obtained. We list it mainly since this problem is often (confusingly) treated as if it were a distinct learning setting.
* **Control (and nonadversarial Reinforcement Learning).** In many cases the environment remembers what we did. Not necessarily in an adversarial manner but it will just remember and the response will depend on what happened before. E.g., a coffee boiler controller will observe different temperatures depending on whether it was heating the boiler previously. PID (proportional integral derivative) controller algorithms are a popular choice there. Likewise, a user's behavior on a news site will depend on what we showed him previously (e.g., he will read most news only once). Many such algorithms form a model of the environment in which they act such as to make their decisions appear less random (i.e., to reduce variance).
* **Reinforcement Learning.** In the more general case of an environment with memory, we may encounter situations where the environment is trying to *cooperate* with us (cooperative games, in particular for non-zero-sum games), or others where the environment will try to *win*. Chess, Go, Backgammon or StarCraft are some of the cases. Likewise, we might want to build a good controller for autonomous cars. The other cars are likely to respond to the autonomous car's driving style in nontrivial ways, e.g., trying to avoid it, trying to cause an accident, trying to cooperate with it, etc.

One key distinction between the different situations above is that the same strategy that might have worked throughout in the case of a stationary environment, might not work throughout when the environment can adapt. For instance, an arbitrage opportunity discovered by a trader is likely to disappear once he starts exploiting it. The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. For instance, if we *know* that things may only change slowly, we can force any estimate to change only slowly, too. If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. These types of knowledge are crucial for the aspiring data scientist to deal with concept shift, i.e., when the problem that he is trying to solve changes over time.


## Fairness, Accountability, and Transparency in Machine Learning

Finally, it is important to remember that when you deploy machine learning systems you are not merely optimizing a predictive model---you are typically providing a tool that will be used to (partially or fully) automate decisions. These technical systems can impact the lives of individuals subject to the resulting decisions. The leap from considering predictions to decisions raises not only new technical questions, but also a slew of ethical questions that must be carefully considered. If we are deploying a medical diagnostic system, we need to know for which populations it may work and which it may not. Overlooking foreseeable risks to the welfare of a subpopulation could cause us to administer inferior care. Moreover, once we contemplate decision-making systems, we must step back and reconsider how we evaluate our technology. Among other consequences of this change of scope, we will find that *accuracy* is seldom the right metric. For instance, when translating predictions into actions, we will often want to take into account the potential cost sensitivity of erring in various ways. If one way of misclassifying an image could be perceived as a racial sleight, while misclassification to a different category would be harmless, then we might want to adjust our thresholds accordingly, accounting for societal values in designing the decision-making protocol. We also want to be careful about how prediction systems can lead to feedback loops. For example, consider predictive policing systems, which allocate patrol officers to areas with high forecasted crime. It is easy to see how a worrying pattern can emerge:

 1. Neighborhoods with more crime get more patrols.
 1. Consequently, more crimes are discovered in these neighborhoods, entering the training data available for future iterations.
 1. Exposed to more positives, the model predicts yet more crime in these neighborhoods.
 1. In the next iteration, the updated model targets the same neighborhood even more heavily leading to yet more crimes discovered, etc.

Often, the various mechanisms by which a model's predictions become coupled to its training data are unaccounted for in the modeling process. This can lead to what researchers call "runaway feedback loops." Additionally, we want to be careful about whether we are addressing the right problem in the first place. Predictive algorithms now play an outsize role in mediating the dissemination of information. Should the news that an individual encounters be determined by the set of Facebook pages they have *Liked*? These are just a few among the many pressing ethical dilemmas that you might encounter in a career in machine learning.

## Summary

* In many cases training and test sets do not come from the same distribution. This is called covariate shift.
* Under the corresponding assumptions, *covariate* and *label* shift can be detected and corrected for at test time. Failure to account for this bias can become problematic at test time.
* In some cases, the environment may *remember* automated actions and respond in surprising ways. We must account for this possibility when building models and continue to monitor live systems, open to the possibility that our models and the environment will become entangled in unanticipated ways.

## Exercises

1. What could happen when we change the behavior of a search engine? What might the users do? What about the advertisers?
1. Implement a covariate shift detector. Hint: build a classifier.
1. Implement a covariate shift corrector.
1. What could go wrong if training and test sets are very different? What would happen to the sample weights?

[Tartışmalar](https://discuss.d2l.ai/t/105)


