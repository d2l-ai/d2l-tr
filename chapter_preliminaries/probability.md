# Olasılık
:label:`sec_prob`

Bir şekilde, makine öğrenmesi tamamen tahminlerde bulunmakla ilgilidir.
Klinik geçmişi göz önüne alındığında, bir hastanın önümüzdeki yıl kalp krizi geçirme *olasılığını* tahmin etmek isteyebiliriz. Anormallik tespitinde, bir uçağın jet motorundan bir dizi okumanın uçak normal çalışıyor olsaydı ne kadar *muhtemel* olacağını değerlendirmek isteyebiliriz. Pekiştirmeli öğrenmede, bir etmenin bir ortamda akıllıca hareket etmesini istiyoruz. Bu, mevcut eylemlerin her birinin altında yüksek bir ödül alma olasılığını düşünmemiz gerektiği anlamına gelir. Ayrıca tavsiye sistemleri oluşturduğumuzda, olasılık hakkında da düşünmemiz gerekir. Örneğin, *varsayımsal olarak* büyük bir çevrimiçi kitapçı için çalıştığımızı söyleyelim. Belirli bir kullanıcının belirli bir kitabı satın alma olasılığını tahmin etmek isteyebiliriz. Bunun için olasılık dilini kullanmamız gerekiyor.
Birçok ders, anadal, tez, kariyer ve hatta bölüm olasılığa ayrılmıştır. Doğal olarak, bu bölümdeki amacımız konunun tamamını öğretmek değildir. Bunun yerine sizi ayaklarınızın üzerine kaldırmayı, ilk derin öğrenme modellerinizi oluşturmaya başlayabileceğiniz kadarını öğretmeyi ve dilerseniz konuyu kendi başınıza keşfetmeye başlayabilmeniz için bir tutam bilgi vermeyi umuyoruz.

Daha önceki bölümlerde, tam olarak ne olduklarını açıklamadan veya somut bir örnek vermeden olasılıkları zaten çağırmıştık. Şimdi ilk vakayı ele alarak daha ciddileşelim: Fotoğraflardan kedi ve köpekleri ayırmak. Bu basit gelebilir ama aslında zorlu bir görevdir. Başlangıç ​​olarak, sorunun zorluğu imgenin çözünürlüğüne bağlı olabilir.

![Farklı çözünürlükteki imgeler ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ piksel).](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

Gösterildiği gibi :numref:`fig_cat_dog`, insanlar için kedileri ve köpekleri $160 \times 16$ piksel çözünürlükte tanımak kolayken, $40 \times 40$ pikselde zorlayıcı ve $10 \times 10$ pikselde imkansıza yakın hale geliyor. Başka bir deyişle, kedi ve köpekleri büyük bir mesafeden (ve dolayısıyla düşük çözünürlükten) ayırma yeteneğimiz bilgisiz (cahilce) tahminlere yaklaşabilir. Olasılık, bize kesinlik seviyemiz hakkında resmi (kurallı) bir mantık yürütme yöntemi verir.
İmgenin bir kediyi gösterdiğinden tamamen eminsek, karşılık gelen $y$ etiketinin "kedi" olma *olasılığının*, $P(y=$ "kedi"$)$ , $1$'e eşit olduğunu söyleriz.
$y =$ "kedi" veya $y =$ "köpek" olduğunu önerecek hiçbir kanıtımız yoksa, iki olasılığın eşit derecede *muhtemelen* olduğunu $P(y=$ "kedi"$) = P(y=$ "köpek"$) = 0.5$ diye ifade ederek söyleyebiliriz. Makul derecede emin olsaydık, ancak imgenin bir kediyi gösterdiğinden kesin emin olamasaydık, $0.5  < P(y=$ "kedi"$) < 1$ bir olasılık atayabilirdik.

Şimdi ikinci bir durumu düşünün: Bazı hava durumu izleme verilerini göz önüne alarak yarın Taipei'de yağmur yağma olasılığını tahmin etmek istiyoruz. Yaz mevsimindeyse, yağmur 0.5 olasılıkla gelebilir.

Her iki durumda da, bir miktar ilgi değerimiz var. Her iki durumda da sonuç hakkında emin değiliz.
Ancak iki durum arasında temel bir fark var. Bu ilk durumda, imge aslında ya bir köpektir ya da bir kedidir ve  hangisi olduğunu bilmiyoruz. İkinci durumda, eğer bu tür şeylere inanıyorsanız (ve çoğu fizikçi bunu yapıyor), sonuç aslında rastgele bir olay olabilir. Dolayısıyla olasılık, kesinlik seviyemiz hakkında akıl yürütmek için esnek bir dildir ve geniş bir bağlam kümesinde etkili bir şekilde uygulanabilir.

## Temel Olasılık Kuramı

Bir zar attığımızı ve başka bir rakam yerine 1 rakamını görme şansının ne olduğunu bilmek istediğimizi düşünelim. Eğer zar adilse, altı sonucun tümü $\{1, \ldots, 6\}$ eşit derecede meydana gelir ve bu nedenle altı durumdan birinde $1$ görürüz. Resmi olarak $1$'in $\frac{1}{6}$ olasılıkla oluştuğunu belirtiyoruz.

Bir fabrikadan aldığımız gerçek bir zar için bu oranları bilmeyebiliriz ve hileli olup olmadığını kontrol etmemiz gerekir. Zarı araştırmanın tek yolu, onu birçok kez atmak ve sonuçları kaydetmektir. Zarın her atımında, $\{1, \ldots, 6\}$'den bir değer gözlemleyeceğiz. Bu sonuçlar göz önüne alındığında, her bir sonucu gözlemleme olasılığını araştırmak istiyoruz.

Doğal bir yaklaşım her değer için, o değerin bireysel sayımını almak ve bunu toplam atış sayısına bölmektir.
Bu bize belirli bir *olayın* olasılığının *tahminini* verir. *Büyük sayılar yasası* bize, atışların sayısı arttıkça bu tahminin gerçek temel olasılığa gittikçe yaklaşacağını söyler. Burada neler olup bittiğinin ayrıntılarına girmeden önce bunu deneyelim.

Başlamak için gerekli paketleri içeri aktaralım.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

Sonra, zarı atabilmeyi isteyeceğiz. İstatistikte bu olasılık dağılımlarından örnek alma sürecini *örnekleme* olarak adlandırıyoruz.
Olasılıkları bir dizi ayrık seçime atayan dağılıma *katlıterimli (multinomial) dağılım* denir. Daha sonra *dağılımın* daha resmi bir tanımını vereceğiz, ancak üst seviyede, bunu sadece olaylara olasılıkların atanması olarak düşünün.

Tek bir örneklem çekmek için, basitçe bir olasılık vektörü aktarırız.
Çıktı aynı uzunlukta başka bir vektördür: $i$ indisindeki değeri, örnekleme sonucunun $i$'ye karşılık gelme sayısıdır.

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

Örnekleyiciyi birkaç kez çalıştırırsanız, her seferinde rastgele değerler çıkardığını göreceksiniz. Bir zarın adilliğini tahmin ederken olduğu gibi, genellikle aynı dağılımdan birçok örneklem oluşturmak isteriz. Bunu bir Python `for` döngüsüyle yapmak dayanılmaz derecede yavaş olacaktır, bu nedenle kullandığımız işlev, aynı anda birden fazla örneklem çekmeyi destekler ve arzu edebileceğimiz herhangi bir şekile sahip bağımsız örneklemler dizisi döndürür.

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

Artık zar atışlarını nasıl örnekleyeceğimizi bildiğimize göre, 1000 atış benzetimi yaparız. Daha sonra, 1000 atışın her birinden sonra, her bir sayının kaç kez atıldığını inceleyebilir ve sayabiliriz.
Özellikle belirtirsek, göreceli frekansı gerçek olasılığın tahmini olarak hesaplarız.

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# Bölme için sonuçları 32-bitlik virgüllü kazan sayı olarak depolarız
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

Verileri adil bir zardan oluşturduğumuz için, her sonucun gerçek olasılığının $\frac{1}{6}$ olduğunu, yani kabaca $0.167$ olduğunu biliyoruz, bu nedenle yukarıdaki çıktı tahminleri iyi görünüyor.

Ayrıca bu olasılıkların zaman içinde gerçek olasılığa doğru nasıl yakınsadığını da görselleştirebiliyoruz.
Her grubun 10 örnek çektiği 500 adet deney yapalım.

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Deney Gruplari')
d2l.plt.gca().set_ylabel('Tahmini Olasilik')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Deney Gruplari')
d2l.plt.gca().set_ylabel('Tahmini Olasilik')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Deney Gruplari')
d2l.plt.gca().set_ylabel('Tahmini Olasilik')
d2l.plt.legend();
```

Her katı eğri, zarın altı değerinden birine karşılık gelir ve her deney grubundan sonra değerlendirildiğinde zarın bu değerleri göstermesinde tahmin ettiğimiz olasılığı verir.
Kesikli siyah çizgi, gerçek temel olasılığı verir.
Daha fazla deney yaparak daha fazla veri elde ettikçe, $6$ katı eğri gerçek olasılığa doğru yaklaşıyor.

### Olasılık Teorisinin Aksiyomları

Bir zarın atışları ile uğraşırken, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ kümesine *örnek uzay* veya *sonuç uzayı* diyoruz, burada her eleman bir *sonuçtur*.
Bir *olay*, belirli bir örnek uzaydan alınan bir dizi sonuçtur.
Örneğin, "$5$ görmek" ($\{5\}$) ve "tek sayı görmek" ($\{1, 3, 5\}$) zar atmanın geçerli olaylarıdır.
Rastgele bir denemenin sonucu $\mathcal{A}$ olayındaysa, $\mathcal{A}$ olayının gerçekleştiğine dikkat edin.
Yani, bir zar atıldıktan sonra $3$ nokta üstte gelirse, $3 \in \{1, 3, 5 \}$ olduğundan, "tek bir sayı görme" olayı gerçekleşti diyebiliriz.

Biçimsel olarak, *olasılık*, bir kümeyi gerçek bir değere eşleyen bir işlev olarak düşünülebilir.
$P(\mathcal{A})$ olarak gösterilen, verilen $\mathcal{S}$ örnek uzayında bir $\mathcal{A}$ olayının olasılığı aşağıdaki özellikleri karşılar:

* Herhangi bir $\mathcal{A}$ olayı için, olasılığı asla negatif değildir, yani, $P(\mathcal{A}) \geq 0$;
* Tüm örnek alanın olasılığı $1$'dir, yani $P(\mathcal{S}) = 1$;
* Birbirini dışlayan sayılabilir herhangi bir olay dizisi için ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ bütün $i \neq j$) için, herhangi bir şeyin olma olasılığı kendi olasılıklarının toplamına eşittir, yani, $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Bunlar aynı zamanda 1933'te Kolmogorov tarafından önerilen olasılık teorisinin aksiyomlarıdır.
Bu aksiyom sistemi sayesinde, rastlantısallıkla ilgili herhangi bir felsefi tartışmayı önleyebiliriz; bunun yerine matematiksel bir dille titiz bir şekilde akıl yürütebiliriz.
Örneğin, $\mathcal{A}_1$ olayının tüm örnek uzay olmasına ve $\mathcal{A}_i = \emptyset$'a bütün $i > 1$ için izin vererek, $P(\emptyset) = 0$, yani imkansız bir olayın olasılığı $0$'dır.

### Rastgele Değişkenler

Bir zar atma rastgele deneyimizde, *rastgele değişken* kavramını tanıttık. Rastgele bir değişken hemen hemen herhangi bir miktar olabilir ve deterministik (belirlenimci) değildir. Değişken rastgele bir deneyde bir dizi olasılık arasından bir değer alabilir.
Değeri bir zar atmanın $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ örnek uzayında olan $X$ rastgele değişkenini düşünün. "Bir $5$ görme" olayını $\{X = 5\}$ veya $X = 5$ ve olasılığını $P(\{X = 5\})$ veya $P(X = 5)$ diye belirtiriz.
$P(X = a)$ ile, $X$ rastgele değişkeni ile $X$'in alabileceği değerler (örneğin, $a$) arasında bir ayrım yaparız.
Bununla birlikte, bu tür bilgiçlik, hantal bir gösterimle sonuçlanır.
Kısa bir gösterim için, bir yandan, $P(X)$'i, $X$ rasgele değişkeni üzerindeki *dağılım* olarak gösterebiliriz: Dağılım bize $X$'in herhangi bir değeri alma olasılığını söyler.
Öte yandan, rastgele bir değişkenin $a$ değerini alma olasılığını belirtmek için $P(a)$ yazabiliriz.
Olasılık teorisindeki bir olay, örnek uzaydan bir küme sonuç olduğu için, rastgele bir değişkenin alması için bir dizi değer belirleyebiliriz.
Örneğin, $P(1 \leq X \leq 3)$, $\{1 \leq X \leq 3\}$ olayının olasılığını belirtir, yani $\{X = 1, 2, \text{veya} 3\}$ anlamına gelir. Aynı şekilde, $P(1 \leq X \leq 3)$, $X$ rasgele değişkeninin $\{1, 2, 3\}$'ten bir değer alabilme olasılığını temsil eder.

Bir zarın yüzleri gibi *kesikli* rastgele değişkenler ile bir kişinin ağırlığı ve boyu gibi *sürekli* olanlar arasında ince bir fark olduğunu unutmayın. İki kişinin tam olarak aynı boyda olup olmadığını sormanın pek bir anlamı yok. Yeterince hassas ölçümler alırsak, gezegendeki hiçbir insanın aynı boyda olmadığını göreceksiniz. Aslında, yeterince ince bir ölçüm yaparsak, uyandığınızda ve uyuduğunuzda da boyunuz aynı olmayacaktır. Dolayısıyla, birinin 1.80139278291028719210196740527486202 metre boyunda olma olasılığını sormanın hiçbir amacı yoktur. Dünya insan nüfusu göz önüne alındığında, olasılık neredeyse 0'dır. Bu durumda, birinin boyunun belirli bir aralıkta, örneğin 1.79 ile 1.81 metre arasında olup olmadığını sormak daha mantıklıdır. Bu durumlarda, bir değeri *yoğunluk* olarak görme olasılığımızı ölçüyoruz. Tam olarak 1.80 metrelik boyun olasılığı yoktur, ancak yoğunluğu sıfır değildir. Herhangi iki farklı boy arasındaki aralıkta sıfır olmayan bir olasılığa sahibiz.
Bu bölümün geri kalanında, olasılığı ayrık uzayda ele alıyoruz.
Sürekli rastgele değişkenler üzerindeki olasılık için, şuraya başvurabilirsiniz :numref:`sec_random_variables`.

## Çoklu Rastgele Değişkenlerle Başa Çıkma

Çok sık olarak, bir seferde birden fazla rastgele değişkeni dikkate almak isteyeceğiz.
Örneğin, hastalıklar ile belirtiler arasındaki ilişkiyi modellemek isteyebiliriz. Bir hastalık ve bir belirti verildiğinde, örneğin "grip" ve "öksürük", bir olasılıkla bir hastada ortaya çıkabilirler veya çıkmayabilirler. Her ikisinin de olasılığının sıfıra yakın olacağını umarken, bu olasılıkları ve bunların birbirleriyle olan ilişkilerini tahmin etmek isteyebiliriz, böylece daha iyi tıbbi bakım sağlamak için çıkarımlarımızı uygulayabiliriz.

Daha karmaşık bir örnek olarak, imgeler milyonlarca piksel, dolayısıyla milyonlarca rastgele değişken içerir. Ve çoğu durumda imgeler, imgedeki nesneleri tanımlayan bir etiketle birlikte gelir. Etiketi rastgele bir değişken olarak da düşünebiliriz. Tüm meta (üst) verileri konum, zaman, diyafram, odak uzaklığı, ISO, odak mesafesi ve kamera türü gibi, rastgele değişkenler olarak bile düşünebiliriz.
Bunların hepsi birlikte oluşan rastgele değişkenlerdir. Birden çok rastgele değişkenle uğraştığımızda, ilgilendiğimiz birkaç miktar vardır.

### Bileşik Olasılık

İlki, *bileşik olasılık* $P(A=a, B=b)$ olarak adlandırılır. Herhangi $a$ ve $b$ değerleri verildiğinde, bileşik olasılık şu cevabı vermemizi sağlar: $A = a$ ve $B = b$ olaylarının aynı anda olma olasılığı nedir?
Tüm $a$ ve $b$ değerleri için, $P(A=a, B=b) \leq P (A=a)$ olduğuna dikkat edin.
Durum böyle olmalıdır, çünkü $A=a$ ve $B=b$ olması için $A=a$ olması gerekir *ve* $B=b$ de gerçekleşmelidir (ve bunun tersi de geçerlidir). Bu nedenle, $A=a$ ve $B=b$, tek tek $A=a$ veya $B=b$ değerinden daha büyük olamaz.

### Koşullu Olasılık

Bu bizi ilginç bir orana getiriyor: $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$. Bu oranı bir *koşullu olasılık* olarak adlandırıyoruz ve bunu $P(B=b \mid A=a)$ ile gösteriyoruz: $A=a$ olması koşuluyla, $B=b$ olasılığıdır.

### Bayes Kuramı (Teoremi)

Koşullu olasılıkların tanımını kullanarak, istatistikteki en kullanışlı ve ünlü denklemlerden birini türetebiliriz: *Bayes teoremi*.
Aşağıdaki gibidir.
Yapısı gereği, $P(A, B) = P(B \mid A) P(A)$ şeklindeki *çarpma kuralına* sahibiz. Simetriye göre, bu aynı zamanda $P(A, B) = P(A \mid B) P(B)$ için de geçerlidir. $P(B) > 0$ olduğunu varsayalım. Koşullu değişkenlerden birini çözerek şunu elde ederiz: 

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

Burada, $P(A, B)$'nin *bileşik dağılım* ve $P(A \mid B)$'nin *koşullu dağılım* olduğu, daha sıkıştırılmış gösterimi kullandığımıza dikkat edin. Bu tür dağılımlar belirli $A = a, B=b$ değerleri için hesaplanabilir.


### Marjinalleştirme

Bayes teoremi, bir şeyi diğerinden çıkarmak istiyorsak, neden ve sonuç mesela, çok kullanışlıdır, ancak bu bölümde daha sonra göreceğimiz gibi, yalnızca özellikleri ters yönde biliyoruz. Bunun işe yaraması için ihtiyacımız olan önemli bir işlem, *marjinalleştirme*dir.
$P(A, B)$'den $P(B)$ belirleme işlemidir. $B$ olasılığının, tüm olası $A$ seçeneklerini hesaba katma ve bunların hepsinde bileşik olasılıkları bir araya toplama olduğunu görebiliriz:

$$P(B) = \sum_{A} P(A, B),$$

bu aynı zamanda *toplam kuralı* olarak da bilinir. Tümleştirmenin bir sonucu olan olasılık veya dağılım, *marjinal (tümsel) olasılık* veya *marjinal (tümsel) dağılım* olarak adlandırılır.

### Bağımsızlık

Kontrol edilmesi gereken diğer bir yararlı özellik, *bağımlılık* ve *bağımsızlık*tır.
İki rastgele değişken olan $A$ ve $B$'nin birbirinden bağımsızlığı, $A$ olayının ortaya çıkmasının, $B$ olayının oluşumu hakkında herhangi bir bilgi vermediği anlamına gelir.
Bu durumda $P(B \mid A) = P(B)$'dir. İstatistikçiler bunu genellikle $A \perp B$ olarak ifade ederler. Bayes teoreminden, bunu aynı zamanda $P(A \mid B) = P(A)$ olduğunu da izler.
Diğer tüm durumlarda $A$ ve $B$'ye bağımlı diyoruz. Örneğin, bir zarın iki ardışık atışı bağımsızdır. Aksine, bir ışık anahtarının konumu ve odadaki parlaklık değildir (her zaman kırılmış bir ampulümüz, elektrik kesintisi veya kırık bir anahtarımız olabileceğinden, tam olarak belirlenimci (deterministik) değildirler).

$P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ ve $P(A, B) = P(A)P(B)$ eşit olduğundan, iki rastgele değişken ancak ve ancak bileşik dağılımları kendi bireysel dağılımlarının çarpımına eşit ise bağımsızdır.
Benzer şekilde, iki rastgele değişken $A$ ve $B$ başka bir rasgele değişken $C$ verildiğinde, ancak ve ancak $P(A, B \mid C) = P(A \mid C)P(B \mid C)$ ise *koşullu olarak bağımsızdır*. Bu, $A \perp B \mid C$ olarak ifade edilir.

### Uygulama
:label:`subsec_probability_hiv_app`

Becerilerimizi test edelim. Bir doktorun bir hastaya HIV testi uyguladığını varsayalım. Bu test oldukça doğrudur ve yalnızca %1 olasılıkla hasta sağlıklı olduğu halde hasta olarak bildirme hatası yapar. Dahası, eğer hastada gerçekten varsa HIV'i asla tespit etmemezlik yapmaz. Teşhisi belirtmek için $D_1$ (pozitifse $1$ ve negatifse $0$) ve HIV durumunu belirtmek için $H$ (pozitifse $1$ ve negatifse $0$) kullanırız.
:numref:`conditional_prob_D1` bu tür koşullu olasılıkları listeler.

:$P(D_1 \mid H)$ koşullu olasılığı

| Koşullu olasılık | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

Koşullu olasılığın, olasılık gibi 1'e toplanması gerektiğinden, sütun toplamlarının hepsinin 1 olduğuna dikkat edin (ancak satır toplamları değildir). Test pozitif çıkarsa hastanın HIV'li olma olasılığını hesaplayalım, yani $P(H = 1 \mid D_1 = 1)$. Açıkçası bu, yanlış alarmların sayısını etkilediği için hastalığın ne kadar yaygın olduğuna bağlı olacaktır. Nüfusun oldukça sağlıklı olduğunu varsayalım, örneğin $P(H=1) = 0.0015$. Bayes teoremini uygulamak için, marjinalleştirmeyi ve çarpım kuralını uygulamalıyız.

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Böylece bunu elde ederiz,

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

Diğer bir deyişle, çok doğru bir test kullanmasına rağmen, hastanın gerçekten HIVli olma şansı yalnızca %13.06'dır.
Gördüğümüz gibi, olasılık sezgilere ters olabilir.

Böylesine korkunç bir haber alan hasta ne yapmalıdır? Muhtemelen hasta, netlik elde etmek için hekimden başka bir test yapmasını isteyecektir. İkinci testin farklı özellikleri vardır ve şu şekilde gösterildiği gibi birincisi kadar iyi değildir :numref:`conditional_prob_D2`.

:$P(D_2 \mid H)$ koşullu olasılığı

| Koşullu olasılık | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

Maalesef ikinci test de pozitif çıkıyor.
Koşullu bağımsızlığı varsayıp Bayes teoremini çağırarak gerekli olasılıkları bulalım:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

Şimdi marjinalleştirme ve çarpım kuralını uygulayabiliriz:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

En sonunda, her iki pozitif testin de verildiği hastanın HIV'li olma olasılığı

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

Böylece ikinci test, her şeyin yolunda olmadığına dair çok daha yüksek bir güven kazanmamızı sağladı. İkinci test, birincisinden çok daha az doğru olmasına rağmen, tahminimizi önemli ölçüde iyileştirdi.


## Beklenti ve Varyans (Değişinti)

Olasılık dağılımlarının temel özelliklerini özetlemek için bazı ölçülere ihtiyacımız var.
$X$ rastgele değişkeninin *beklentisi* (veya ortalaması) şu şekilde belirtilir:

$$E[X] = \sum_{x} x P(X = x).$$


$f(x)$ fonksiyonunun girdisi $P$ dağılımından farklı $x$ değerleriyle elde edilen rastgele bir değişken olduğunda, $f(x)$ beklentisi şu şekilde hesaplanır:

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$


Çoğu durumda $X$ rastgele değişkeninin beklentisinden ne kadar saptığını ölçmek isteriz. Bu, varyans (değişinti) ile ölçülebilir

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

Varyansın kareköküne *standart sapma* denir.
Rastgele değişkenin bir fonksiyonunun varyansı, fonksiyonun beklentisinden ne kadar saptığını ölçer, çünkü rastgele değişkenin farklı değerleri $x$, dağılımından örneklenir:

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$


## Özet

* Olasılık dağılımlarından örnekleme yapabiliriz.
* Bileşik dağılım, koşullu dağılım, Bayes teoremi, marjinalleştirme ve bağımsızlık varsayımlarını kullanarak birden çok rastgele değişkeni analiz edebiliriz.
* Beklenti ve varyans, olasılık dağılımlarının temel özelliklerini özetlemek için yararlı ölçüler sunar.

## Alıştırmalar

1. Her grubun $n = 10$ örnek çektiği $m = 500$ deney grubu yürüttük. $m$ ve $n$ değerlerini değiştirin. Deneysel sonuçları gözlemleyin ve analiz edin.
1. $P(\mathcal{A})$ ve $P(\mathcal{B})$ olasılığına sahip iki olay verildiğinde, $P(\mathcal{A} \cup \mathcal{B})$ ve $P(\mathcal{A} \cap \mathcal{B})$ için üst ve alt sınırları hesaplayın (İpucu: Durumu bir [Venn Şeması](https://en.wikipedia.org/wiki/Venn_diagram) kullanarak görüntüleyin.)
1. $A$, $B$ ve $C$ gibi rastgele değişkenlerden oluşan bir dizimiz olduğunu varsayalım, burada $B$ yalnızca $A$'ya bağlıdır ve $C$ yalnızca $B$'ye bağlıdır, $P(A, B, C)$ birleşik olasılığı basitleştirebilir misiniz? (İpucu: Bu bir [Markov Zinciri](https://en.wikipedia.org/wiki/Markov_chain)'dir.)
1. :numref:`subsec_probability_hiv_app` içinde, ilk test daha doğrudur. Hem birinci hem de ikinci testleri yapmak yerine neden ilk testi iki kez yapmıyorsunuz? 


:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/198)
:end_tab:
