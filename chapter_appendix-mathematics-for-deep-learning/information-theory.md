# Bilgi Teorisi
:label:`sec_information_theory`

Evren bilgi ile dolup taşıyor. Bilgi, disiplinler arası açıklıklarda ortak bir dil sağlar: Shakespeare'in Sone'sinden Cornell ArXiv'deki araştırmacıların makalesine, Van Gogh'un baskısı Yıldızlı Gece'den Beethoven'in 5. Senfonisi'ne, ilk programlama dili Plankalkül'den son teknoloji makine öğrenmesi algoritmalarına. Biçimi ne olursa olsun, her şey bilgi teorisinin kurallarını izlemelidir. Bilgi teorisi ile, farklı sinyallerde ne kadar bilgi bulunduğunu ölçebilir ve karşılaştırabiliriz. Bu bölümde, bilgi teorisinin temel kavramlarını ve bilgi teorisinin makine öğrenmesindeki uygulamalarını inceleyeceğiz.

Başlamadan önce, makine öğrenmesi ve bilgi teorisi arasındaki ilişkiyi özetleyelim. Makine öğrenmesi, verilerden ilginç sinyaller çıkarmayı ve kritik tahminlerde bulunmayı amaçlar. Öte yandan, bilgi teorisi, bilgiyi kodlama, kod çözme, iletme ve üstünde oynama yapmayı inceler. Sonuç olarak, bilgi teorisi, makine öğrenmesi sistemlerinde bilgi işlemeyi tartışmak için temel bir dil sağlar. Örneğin, birçok makine öğrenmesi uygulaması çapraz entropi kaybını şurada açıklandığı gibi kullanır : numref:`sec_softmax`. Bu kayıp, doğrudan bilgi teorisel değerlendirmelerinden türetilebilir.

## Bilgi
 
Bilgi teorisinin "ruhu" ile başlayalım: Bilgi. *Bilgi*, bir veya daha fazla kodlama biçimli belirli bir dizi ile kodlanabilir. Kendimizi bir bilgi kavramını tanımlamaya çalışmakla görevlendirdiğimizi varsayalım. Başlangıç ​​noktası ne olabilir?

Aşağıdaki düşünce deneyini düşünün. Kart destesi olan bir arkadaşımız var. Desteyi karıştıracaklar, bazı kartları ters çevirecekler ve bize kartlar hakkında açıklamalar yapacaklar. Her ifadenin bilgi içeriğini değerlendirmeye çalışacağız.

Önce, bir kartı çevirip bize "Bir kart görüyorum" diyorlar. Bu bize hiçbir bilgi sağlamaz. Durumun bu olduğundan zaten emindik, bu yüzden bilginin sıfır olduğu umuyoruz.

Sonra bir kartı çevirip "Bir kalp görüyorum" diyorlar. Bu bize biraz bilgi sağlar, ancak gerçekte, her biri eşit olasılıkla mümkün olan yalnızca $4$ farklı takım vardır, bu nedenle bu sonuca şaşırmadık. Bilginin ölçüsü ne olursa olsun, bu olayın düşük bilgi içeriğine sahip olmasını umuyoruz.

Sonra, bir kartı çevirip "Bu maça $3$" diyorlar. Bu daha fazla bilgidir. Gerçekten de $52$ eşit olasılıklı sonuçlar vardı ve arkadaşımız bize bunun hangisi olduğunu söyledi. Bu orta miktarda bilgi olmalıdır.

Bunu mantıksal uç noktaya götürelim. Sonunda destedeki her kartı çevirdiklerini ve karışık destenin tüm dizisini okuduklarını varsayalım. Destede $52!$ farklı dizilim var, gene hepsi aynı olasılığa sahip, bu yüzden hangisinin olduğunu bilmek için çok fazla bilgiye ihtiyacımız var.

Geliştirdiğimiz herhangi bir bilgi kavramı bu sezgiye uygun olmalıdır. Aslında, sonraki bölümlerde bu olayların $0\text{ bit}$, $2\text{ bit}$, $~5.7\text{ bit}$, and $~225.6\text{ bit}$ bilgiye sahip olduğunu nasıl hesaplayacağımızı öğreneceğiz.

Bu düşünce deneylerini okursak, doğal bir fikir görürüz. Başlangıç noktası olarak, bilgiyi önemsemekten ziyade, bilginin olayın sürpriz derecesini veya soyut olasılığını temsil ettiği fikrini geliştirebiliriz. Örneğin, alışılmadık bir olayı tanımlamak istiyorsak, çok fazla bilgiye ihtiyacımız var. Genel (sıradan) bir olay için fazla bilgiye ihtiyacımız olmayabilir.

1948'de Claude E. Shannon bilgi teorisini oluşturan *İletişimin Bir Matematiksel Teorisi (A Mathematical Theory of Communication)* kitabını yayınladı :cite:`Shannon.1948`. Shannon kitabında ilk kez bilgi entropisi kavramını tanıttı. Yolculuğumuza buradan başlayacağız.

### Öz-Bilgi

Bilgi bir olayın soyut olasılığını içerdiğinden, olasılığı bit adeti ile nasıl eşleştirebiliriz? Shannon, başlangıçta John Tukey tarafından oluşturulmuş olan *bit* terimini bilgi birimi olarak tanıttı. Öyleyse "bit" nedir ve bilgiyi ölçmek için neden onu kullanıyoruz? Tarihsel olarak, antika bir verici yalnızca iki tür kod gönderebilir veya alabilir: $0$ ve $1$. Aslında, ikili kodlama hala tüm modern dijital bilgisayarlarda yaygın olarak kullanılmaktadır. Bu şekilde, herhangi bir bilgi bir dizi $0$ ve $1$ ile kodlanır. Ve bu nedenle, $n$ uzunluğundaki bir dizi ikili rakam, $n$ bit bilgi içerir.

Şimdi, herhangi bir kod dizisi için, her $0$ veya $1$'in $\frac{1}{2}$ olasılıkla gerçekleştiğini varsayalım. Bu nedenle, $n$ uzunluğunda bir dizi koda sahip bir $X$ olayı, $\frac{1}{2^n}$ olasılıkla gerçekleşir. Aynı zamanda, daha önce bahsettiğimiz gibi, bu seri $n$ bit bilgi içerir. Öyleyse, $p$ olasılığını bit sayısına aktarabilen bir matematik fonksiyonuna genelleme yapabilir miyiz? Shannon, *öz-bilgi*yi tanımlayarak cevabı verdi,

$$I(X) = - \log_2 (p),$$

yani bu $X$ etkinliği için aldığımız bilginin *bitleri* olarak. Bu bölümde her zaman 2 tabanlı logaritma kullanacağımızı unutmayın. Basitlik adına, bu bölümün geri kalanı logaritma gösteriminde 2 altindisini göstermeyecektir, yani $\log(.)$ her zaman $\log_2(.)$ anlamına gelir. Örneğin, "0010" kodu şu öz-bilgiyi içerir:

$$I(\text{``0010"}) = - \log (p(\text{``0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \text{ bits}.$$

MXNet'te öz-bilgiyi aşağıda gösterildiği gibi hesaplayabiliriz. Ondan önce, önce bu bölümdeki gerekli tüm paketleri içe aktaralım.

```{.python .input}
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # Define nansum, as pytorch doesn't offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

## Entropi

Öz-bilgi yalnızca tek bir ayrık olayın bilgisini ölçtüğü için, ayrık veya sürekli dağılımın herhangi bir rastgele değişkeni için daha genelleştirilmiş bir ölçüme ihtiyacımız var.

### Motive Edici Entropi

Ne istediğimiz konusunda belirli olmaya çalışalım. Bu, *Shannon entropisinin aksiyomları* olarak bilinenlerin gayri resmi bir ifadesi olacaktır. Aşağıdaki sağduyu beyanları topluluğunun bizi benzersiz bir bilgi tanımına zorladığı ortaya çıkacaktır. Bu aksiyomların usule uygun bir versiyonu, diğer birçoklarıyla birlikte şu adreste bulunabilir :cite:`Csiszar.2008`.

1. Rastgele bir değişkeni gözlemleyerek kazandığımız bilgi, elemanlar dediğimiz şeye veya olasılığı sıfır olan ek elemanların varlığına bağlı değildir.
2. İki rastgele değişkeni gözlemleyerek elde ettiğimiz bilgi, onları ayrı ayrı gözlemleyerek elde ettiğimiz bilgilerin toplamından fazlası değildir. Bağımsız iseler, o zaman tam toplamdır.
3. (Neredeyse) Kesin olayları gözlemlerken kazanılan bilgi (neredeyse) sıfırdır.

Bu gerçeğin kanıtlanması kitabımızın kapsamı dışında olduğu halde, bunun entropinin alması gereken şekli benzersiz bir şekilde belirlediğini bilmek önemlidir. Bunların izin verdiği tek belirsizlik, daha önce gördüğümüz seçimi yaparak normalize edilen temel birimlerin seçimidir; tek bir adil yazı tura ile sağlanan bilginin bir bit olması gibi.

### Tanım

Olasılık yoğunluk fonksiyonu (pdf/oyf) veya olasılık kütle fonksiyonu (pmf/okf) $p(x)$ ile olasılık dağılımı $P$'yi takip eden rastgele değişken $X$ için, beklenen bilgi miktarını *entropi* ( veya *Shannon entropisi*) ile ölçebiliriz:

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`

Özel olmak gerekirse, $X$ ayrıksa, $$H(X)= - \sum_i p_i \log p_i \text {, burada } p_i = P (X_i)$$.

Aksi takdirde, $X$ sürekli ise, entropiyi *diferansiyel (farksal) entropi* olarak da adlandırırız

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

MXNet'te entropiyi aşağıdaki gibi tanımlayabiliriz.

```{.python .input}
def entropy(p):
    entropy = - p * np.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

### Yorumlama

Merak ediyor olabilirsiniz: Entropi tanımında :eqref:`eq_ent_def`, neden negatif bir logaritma ortalaması kullanıyoruz? Burada bazı sezgileri verelim.

İlk olarak, neden *logaritma* işlevi $\log$ kullanıyoruz? $p(x) = f_1(x) f_2(x) \ldots, f_n(x)$ olduğunu varsayalım, burada her bileşen işlevi $f_i(x)$ birbirinden bağımsızdır. Bu, her bir $f_i(x)$'in $p(x)$'den elde edilen toplam bilgiye bağımsız olarak katkıda bulunduğu anlamına gelir. Yukarıda tartışıldığı gibi, entropi formülünün bağımsız rastgele değişkenler üzerinde toplamsal olmasını istiyoruz. Neyse ki, $\log$ doğal olarak olasılık dağılımlarının çarpımını bireysel terimlerin toplamına dönüştürebilir.

Sonra, neden *negatif* $\log$ kullanıyoruz? Sezgisel olarak, alışılmadık bir vakadan genellikle sıradan olandan daha fazla bilgi aldığımız için, daha sık olaylar, daha az yaygın olaylardan daha az bilgi içermelidir. Bununla birlikte, $\log$ olasılıklarla birlikte monoton bir şekilde artıyor ve aslında $[0, 1]$ içindeki tüm değerler için negatif. Olayların olasılığı ile entropileri arasında monoton olarak azalan bir ilişki kurmamız gerekir ki bu ideal olarak her zaman pozitif olacaktır (çünkü gözlemlediğimiz hiçbir şey bizi bildiklerimizi unutmaya zorlamamalıdır). Bu nedenle, $\log$ fonksiyonunun önüne bir eksi işareti ekliyoruz.

Son olarak, *beklenti (ortalama)* işlevi nereden geliyor? Rastgele bir değişken olan $X'i$ düşünün. Öz-bilgiyi ($-\log(p)$), belirli bir sonucu gördüğümüzde sahip olduğumuz *sürpriz* miktarı olarak yorumlayabiliriz. Nitekim, olasılık sıfıra yaklaştıkça sürpriz sonsuz olur. Benzer şekilde, entropiyi $X$'i gözlemlemekten kaynaklanan ortalama sürpriz miktarı olarak yorumlayabiliriz. Örneğin, bir slot makinesi sisteminin ${p_1, \ldots, p_k}$ olasılıklarıyla ${s_1, \ldots, s_k}$ sembollerini istatistiksel olarak bağımsız yaydığını düşünün. O zaman bu sistemin entropisi, her bir çıktının gözlemlenmesinden elde edilen ortalama öz-bilgiye eşittir, yani,

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$

### Entropinin Özellikleri

Yukarıdaki örnekler ve yorumlarla, entropinin şu özelliklerini türetebiliriz :eqref:`eq_ent_def`. Burada, X'i bir olay ve P'yi X'in olasılık dağılımı olarak adlandırıyoruz.

* Entropi negatif değildir, yani $H(X) \geq 0, \forall X$.

* Bir o.y.f veya o.k.f. $p(x)$ ile $X \sim P$ ise ve o.y.f veya o.k.f. $q(x)$'ya sahip yeni bir olasılık dağılımı $Q$ ile $P$'yi tahmin etmeye çalışıyoruz,  o zaman $$H(X) = - E_{x \sim P} [\log p(x)] \leq - E_{x \sim P} [\log q(x)], \text {eşitlikle ancak ve ancak eğer} P = Q.$$ Alternatif olarak, $H(X)$, $P$'den çekilen sembolleri kodlamak için gereken ortalama bit sayısının alt sınırını verir.

* $X \sim P$ ise, $x$ tüm olası sonuçlar arasında eşit olarak yayılırsa maksimum bilgi miktarını iletir. Özel olarak, $P$  $k$-sınıflı ayrık olasılık dağılımı $\{p_1, \ldots, p_k \} ise, o halde $$H(X) \leq \log(k), \text {eşitlikle ancak ve ancak eğer} p_i = \frac{1}{k}, \forall x_i.$$ Eğer $P$ sürekli bir rastgele değişkene, öykü çok daha karmaşık hale gelir. Bununla birlikte, ek olarak $P$'nin sonlu bir aralıkta (tüm değerler $0$ ile $1$ arasında) desteklenmesini zorlarsak, bu aralıkta tekdüze dağılım varsa $P$ en yüksek entropiye sahip olur.

## Ortak Bilgi

Daha önce tek bir rastgele değişken $X$ entropisini tanımlamıştık, bir çift rastgele değişken $(X, Y)$ entropisine ne dersiniz? Bu teknikleri şu soru tipini yanıtlamaya çalışırken düşünebiliriz: "$X$ ve $Y$'de her biri ayrı ayrı olması bir arada olmalarıyla karşılaştırıldığında ne tür bilgi bulunur? Gereksiz bilgi var mı, yoksa hepsi tek mi?"

Aşağıdaki tartışma için, her zaman $(X, Y)$'yi, bir o.y.f veya o.k.f. olan $p_{X, Y}(x, y)$ ile bileşik olasılık dağılımı $P$'yi izleyen bir çift rastgele değişken olarak kullanıyoruz, aynı zamanda da $X$ ve $Y$ sırasıyla $p_X(x) $ ve $p_Y(y) $ olasılık dağılımlarını takip eder.

### Bileşik Entropi 

Tek bir rastgele değişkenin entropisine benzer şekilde :eqref:`eq_ent_def`, rastgele değişken çiftinin, $(X, Y)$, *bileşik entropisini* $H (X, Y)$ olarak tanımlarız.

$$H(X, Y) = −E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

Tam olarak, bir yandan $(X, Y)$ bir çift ayrık rastgele değişkense, o zaman

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

Öte yandan, $(X, Y)$ bir çift sürekli rastgele değişken ise, o zaman *diferansiyel (farksal) bileşik entropiyi* tanımlarız. 

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

Şunu düşünebiliriz :eqref:`eq_joint_ent_def` bize rastgele değişkenler çiftindeki toplam rastgeleliği anlatıyor. Bir çift uç vaka olarak, eğer $X = Y$ iki özdeş rastgele değişken ise, o zaman çiftteki bilgi tam olarak bir tanedeki bilgidir ve $H(X, Y) = H(X) = H(Y)$'dir. Diğer uçta, $X$ ve $Y$ bağımsızsa, $H(X, Y) = H(X) + H(Y)$'dir. Aslında, her zaman bir çift rasgele değişkenin içerdiği bilginin her iki rasgele değişkenin entropisinden daha küçük ve her ikisinin toplamından daha fazla olmadığını bilecegiz.

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

MXNet'te ortak entropiyi en başından uygulayalım.

```{.python .input}
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator nansum will sum up the non-nan number
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # nansum will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

Bunun öncekiyle aynı *kod* olduğuna dikkat edin, ancak şimdi onu iki rastgele değişkenin bileşik dağılımı üzerinde çalışırken farklı bir şekilde yorumluyoruz.

### Koşullu Entropi

Bileşik entropi bir çift rastgele değişkende bulunan bilgi miktarının üzerinde tanımlıdır. Bu yararlıdır, ancak çoğu zaman umursadığımız şey değildir. Makine öğrenmesinin ayarlarını düşünün. Bir görüntünün piksel değerlerini tanımlayan rastgele değişken (veya rastgele değişkenlerin vektörü) olarak $X$'i ve sınıf etiketi olan rastgele değişken olarak $Y$'yi alalım. $X$ önemli bilgi içermelidir---doğal bir görüntü karmaşık bir şeydir. Ancak, görüntü gösterildikten sonra $Y$ içindeki bilgi düşük olmalıdır. Aslında, bir rakamın görüntüsü, rakam okunaksız olmadıkça, hangi rakam olduğu hakkında bilgiyi zaten içermelidir. Bu nedenle, bilgi teorisi kelime dağarcığımızı genişletmeye devam etmek için, rastgele bir değişkenin diğerine koşullu bağlı olarak bilgi içeriği hakkında mantık yürütebilmeliyiz.

Olasılık teorisinde, değişkenler arasındaki ilişkiyi ölçmek için *koşullu olasılığın* tanımını gördük. Şimdi, *koşullu entropiyi*, $H(Y \mid X)$, benzer şekilde tanımlamak istiyoruz. Bunu şu şekilde yazabiliriz:

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

Burada $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$ koşullu olasılıktır. Özellikle, $(X, Y)$ bir çift ayrık rastgele değişken ise, o zaman

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

$(X, Y)$ bir çift sürekli rastgele değişkense, *diferansiyel bileşik entropi* benzer şekilde şöyle tanımlanır:

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$

Şimdi bunu sormak doğaldır, *koşullu entropi* $H(Y \mid X)$, $H(X)$ entropisi ve bileşik entropi $H(X, Y)$ ile nasıl ilişkilidir? Yukarıdaki tanımları kullanarak bunu net bir şekilde ifade edebiliriz:

$$H(Y \mid X) = H(X, Y) - H(X).$$

Bunun sezgisel bir yorumu vardır: $X$ verildiğinde ($H(Y \mid X)$) $Y$'deki bilgi, hem $X$ hem de $Y$ ($H(X, Y)$) birlikteyken olan bilgi eksi $X$ içinde zaten bulunan bilgidir. Bu bize $Y$'de olup da aynı zamanda $X$ ile temsil edilmeyen bilgiyi verir.

Şimdi, koşullu entropiyi, :eqref:`eq_cond_ent_def`, MXNet’te sıfırdan uygulayalım.

```{.python .input}
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator nansum will sum up the non-nan number
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # nansum will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]), 
                    torch.tensor([0.2, 0.8]))
```

### Mutual Information

Given the previous setting of random variables $(X, Y)$, you may wonder: "Now that we know how much information is contained in $Y$ but not in $X$, can we similarly ask how much information is shared between $X$ and $Y$?" The answer will be the *mutual information* of $(X, Y)$, which we will write as $I(X, Y)$.  

Rather than diving straight into the formal definition, let us practice our intuition by first trying to derive an expression for the mutual information entirely based on terms we have constructed before.  We wish to find the information shared between two random variables.  One way we could try to do this is to start with all the information contained in both $X$ and $Y$ together, and then we take off the parts that are not shared.  The information contained in both $X$ and $Y$ together is written as $H(X, Y)$.  We want to subtract from this the information contained in $X$ but not in $Y$, and the information contained in $Y$ but not in $X$.  As we saw in the previous section, this is given by $H(X \mid Y)$ and $H(Y \mid X)$ respectively.  Thus, we have that the mutual information should be

$$
I(X, Y) = H(X, Y) - H(Y \mid X) − H(X \mid Y).
$$

Indeed, this is a valid definition for the mutual information.  If we expand out the definitions of these terms and combine them, a little algebra shows that this is the same as

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def` 


We can summarize all of these relationships in image :numref:`fig_mutual_information`.  It is an excellent test of intuition to see why the following statements are all also equivalent to $I(X, Y)$.

* $H(X) − H(X \mid Y)$
* $H(Y) − H(Y \mid X)$
* $H(X) + H(Y) − H(X, Y)$

![Mutual information's relationship with joint entropy and conditional entropy.](../img/mutual_information.svg)
:label:`fig_mutual_information`


In many ways we can think of the mutual information :eqref:`eq_mut_ent_def` as principled extension of correlation coefficient we saw in :numref:`sec_random_variables`.  This allows us to ask not only for linear relationships between variables, but for the maximum information shared between the two random variables of any kind.

Now, let us implement mutual information from scratch.

```{.python .input}
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

### Properties of Mutual Information

Rather than memorizing the definition of mutual information :eqref:`eq_mut_ent_def`, you only need to keep in mind its notable properties:

* Mutual information is symmetric, i.e., $I(X, Y) = I(Y, X)$.
* Mutual information is non-negative, i.e., $I(X, Y) \geq 0$.
* $I(X, Y) = 0$ if and only if $X$ and $Y$ are independent. For example, if $X$ and $Y$ are independent, then knowing $Y$ does not give any information about $X$ and vice versa, so their mutual information is zero.
* Alternatively, if $X$ is an invertible function of $Y$, then $Y$ and $X$ share all information and $$I(X, Y) = H(Y) = H(X).$$

### Pointwise Mutual Information

When we worked with entropy at the beginning of this chapter, we were able to provide an interpretation of $-\log(p_X(x))$ as how *surprised* we were with the particular outcome.  We may give a similar interpretation to the logarithmic term in the mutual information, which is often referred to as the *pointwise mutual information*:

$$\mathrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`

We can think of :eqref:`eq_pmi_def` as measuring how much more or less likely the specific combination of outcomes $x$ and $y$ are compared to what we would expect for independent random outcomes.  If it is large and positive, then these two specific outcomes occur much more frequently than they would compared to random chance (*note*: the denominator is $p_X(x) p_Y(y)$ which is the probability of the two outcomes were independent), whereas if it is large and negative it represents the two outcomes happening far less than we would expect by random chance.  

This allows us to interpret the mutual information :eqref:`eq_mut_ent_def` as the average amount that we were surprised to see two outcomes occurring together compared to what we would expect if they were independent.

### Applications of Mutual Information

Mutual information may be a little abstract in it pure definition, so how does it related to machine learning? In natural language processing, one of the most difficult problems is the *ambiguity resolution*, or the issue of the meaning of a word being unclear from context. For example, recently a headline in the news reported that "Amazon is on fire". You may wonder whether the company Amazon has a building on fire, or the Amazon rain forest is on fire. 

In this case, mutual information can help us resolve this ambiguity. We first find the group of words that each has a relatively large mutual information with the company Amazon, such as e-commerce, technology, and online. Second, we find another group of words that each has a relatively large mutual information with the Amazon rain forest, such as rain, forest, and tropical. When we need to disambiguate "Amazon", we can compare which group has more occurrence in the context of the word Amazon.  In this case the article would go on to describe the forest, and make the context clear.


## Kullback–Leibler Divergence

As what we have discussed in :numref:`sec_linear-algebra`, we can use norms to measure distance between two points in space of any dimensionality.  We would like to be able to do a similar task with probability distributions.  There are many ways to go about this, but information theory provides one of the nicest.  We now explore the *Kullback–Leibler (KL) divergence*, which provides a way to measure if two distributions are close together or not. 


### Definition

Given a random variable $X$ that follows the probability distribution $P$ with a p.d.f. or a p.m.f. $p(x)$, and we estimate $P$ by another probability distribution $Q$ with a p.d.f. or a p.m.f. $q(x)$. Then the *Kullback–Leibler (KL) divergence* (or *relative entropy*) between $P$ and $Q$ is

$$D_{\mathrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`

As with the pointwise mutual information :eqref:`eq_pmi_def`, we can again provide an interpretation of the logarithmic term:  $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$ will be large and positive if we see $x$ far more often under $P$ than we would expect for $Q$, and large and negative if we see the outcome far less than expected.  In this way, we can interpret it as our *relative* surprise at observing the outcome compared to how surprised we would be observing it from our reference distribution.

In MXNet, let us implement the KL divergence from Scratch.

```{.python .input}
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

### KL Divergence Properties

Let us take a look at some properties of the KL divergence :eqref:`eq_kl_def`.

* KL divergence is non-symmetric, i.e., $$D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P), \text{ if } P \neq Q.$$
* KL divergence is non-negative, i.e., $$D_{\mathrm{KL}}(P\|Q) \geq 0.$$ Note that the equality holds only when $P = Q$.
* If there exists an $x$ such that $p(x) > 0$ and $q(x) = 0$, then $D_{\mathrm{KL}}(P\|Q) = \infty$.
* There is a close relationship between KL divergence and mutual information. Besides the relationship shown in :numref:`fig_mutual_information`, $I(X, Y)$ is also numerically equivalent with the following terms:
    1. $D_{\mathrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\mathrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\mathrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$.
    
  For the first term, we interpret mutual information as the KL divergence between $P(X, Y)$ and the product of $P(X)$ and $P(Y)$, and thus is a measure of how different the joint distribution is from the distribution if they were independent. For the second term, mutual information tells us the average reduction in uncertainty about $Y$ that results from learning the value of the $X$'s distribution. Similarly to the third term.


### Example

Let us go through a toy example to see the non-symmetry explicitly. 

First, let us generate and sort three tensors of length $10,000$: an objective tensor $p$ which follows a normal distribution $N(0, 1)$, and two candidate tensors $q_1$ and $q_2$ which follow normal distributions $N(-1, 1)$ and $N(1, 1)$ respectively.

```{.python .input}
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

Since $q_1$ and $q_2$ are symmetric with respect to the y-axis (i.e., $x=0$), we expect a similar value of KL divergence between $D_{\mathrm{KL}}(p\|q_1)$ and $D_{\mathrm{KL}}(p\|q_2)$. As you can see below, there is only a 1% off between $D_{\mathrm{KL}}(p\|q_1)$ and $D_{\mathrm{KL}}(p\|q_2)$.

```{.python .input}
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

```{.python .input}
#@tab pytorch
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

In contrast, you may find that $D_{\mathrm{KL}}(q_2 \|p)$ and $D_{\mathrm{KL}}(p \| q_2)$ are off a lot, with around 40% off as shown below.

```{.python .input}
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

```{.python .input}
#@tab pytorch
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## Cross Entropy

If you are curious about applications of information theory in deep learning, here is a quick example. We define the true distribution $P$ with probability distribution $p(x)$, and the estimated distribution $Q$ with probability distribution $q(x)$, and we will use them in the rest of this section.

Say we need to solve a binary classification problem based on given $n$ data points {$x_1, \ldots, x_n$}. Assume that we encode $1$ and $0$ as the positive and negative class label $y_i$ respectively, and our neural network is parameterized by $\theta$. If we aim to find a best $\theta$ so that $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, it is natural to apply the maximum log-likelihood approach as was seen in :numref:`sec_maximum_likelihood`. To be specific, for true labels $y_i$ and predictions $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, the probability to be classified as positive is $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$. Hence, the log-likelihood function would be

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$

Maximizing the log-likelihood function $l(\theta)$ is identical to minimizing $- l(\theta)$, and hence we can find the best $\theta$ from here. To generalize the above loss to any distributions, we also called $-l(\theta)$ the *cross entropy loss* $\mathrm{CE}(y, \hat{y})$, where $y$ follows the true distribution $P$ and $\hat{y}$ follows the estimated distribution $Q$. 

This was all derived by working from the maximum likelihood point of view.  However, if we look closely we can see that terms like $\log(\pi_i)$ have entered into our computation which is a solid indication that we can understand the expression from an information theoretic point of view.    


### Formal Definition

Like KL divergence, for a random variable $X$, we can also measure the divergence between the estimating distribution $Q$ and the true distribution $P$ via *cross entropy*,

$$\mathrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

By using properties of entropy discussed above, we can also interpret it as the summation of the entropy $H(P)$ and the KL divergence between $P$ and $Q$, i.e.,

$$\mathrm{CE} (P, Q) = H(P) + D_{\mathrm{KL}}(P\|Q).$$


In MXNet, we can implement the cross entropy loss as below.

```{.python .input}
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

Now define two tensors for the labels and predictions, and calculate the cross entropy loss of them.

```{.python .input}
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### Properties

As alluded in the beginning of this section, cross entropy :eqref:`eq_ce_def` can be used to define a loss function in the optimization problem. It turns out that the following are equivalent:

1. Maximizing predictive probability of $Q$ for distribution $P$, (i.e., $E_{x 
\sim P} [\log (q(x))]$);
1. Minimizing cross entropy $\mathrm{CE} (P, Q)$;
1. Minimizing the KL divergence $D_{\mathrm{KL}}(P\|Q)$.

The definition of cross entropy indirectly proves the equivalent relationship between objective 2 and objective 3, as long as the entropy of true data $H(P)$ is constant.


### Cross Entropy as An Objective Function of Multi-class Classification

If we dive deep into the classification objective function with cross entropy loss $\mathrm{CE}$, we will find minimizing $\mathrm{CE}$ is equivalent to maximizing the log-likelihood function $L$.

To begin with, suppose that we are given a dataset with $n$ samples, and it can be classified into $k$-classes. For each data point $i$, we represent any $k$-class label $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$ by *one-hot encoding*. To be specific, if the data point $i$ belongs to class $j$, then we set the $j$-th entry to $1$, and all other components to $0$, i.e., 

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\text{otherwise.}\end{cases}$$

For instance, if a multi-class classification problem contains three classes $A$, $B$, and $C$, then the labels $\mathbf{y}_i$ can be encoded in {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$}.


Assume that our neural network is parameterized by $\theta$. For true label vectors $\mathbf{y}_i$ and predictions $$\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$$

Hence, the *cross entropy loss* would be

$$
\mathrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

On the other side, we can also approach the problem through maximum likelihood estimation. To begin with, let us quickly introduce a $k$-class multinoulli distribution. It is an extension of the Bernoulli distribution from binary class to multi-class. If a random variable $\mathbf{z} = (z_{1}, \ldots, z_{k})$ follows a $k$-class *multinoulli distribution* with probabilities $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$), i.e., $$p(\mathbf{z}) = p(z_1, \ldots, z_k) = \mathrm{Multi} (p_1, \ldots, p_k), \text{ where } \sum_{i=1}^k p_i = 1,$$ then the joint probability mass function(p.m.f.) of $\mathbf{z}$ is
$$\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$$


It can be seen that each data point, $\mathbf{y}_i$, is following a $k$-class multinoulli distribution with probabilities $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$). Therefore, the joint p.m.f. of each data point $\mathbf{y}_i$ is  $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$
Hence, the log-likelihood function would be

$$
\begin{aligned}
l(\theta) 
 = \log L(\theta) 
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$

Since in maximum likelihood estimation, we maximizing the objective function $l(\theta)$ by having $\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$. Therefore, for any multi-class classification, maximizing the above log-likelihood function $l(\theta)$ is equivalent to minimizing the CE loss $\mathrm{CE}(y, \hat{y})$.


To test the above proof, let us apply the built-in measure `NegativeLogLikelihood` in MXNet. Using the same `labels` and `preds` as in the earlier example, we will get the same numerical loss as the previous example up to the 5 decimal place.

```{.python .input}
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# Implementation of CrossEntropy loss in pytorch combines nn.LogSoftmax() and
# nn.NLLLoss()
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

## Summary

* Information theory is a field of study about encoding, decoding, transmitting, and manipulating information.
* Entropy is the unit to measure how much information is presented in different signals.
* KL divergence can also measure the divergence between two distributions.
* Cross Entropy can be viewed as an objective function of multi-class classification. Minimizing cross entropy loss is equivalent to maximizing the log-likelihood function.


## Exercises

1. Verify that the card examples from the first section indeed have the claimed entropy.
1. Show that the KL divergence $D(p\|q)$ is nonnegative for all distributions $p$ and $q$. Hint: use Jensen's inequality, i.e., use the fact that $-\log x$ is a convex function.
1. Let us compute the entropy from a few data sources:
    * Assume that you are watching the output generated by a monkey at a typewriter. The monkey presses any of the $44$ keys of the typewriter at random (you can assume that it has not discovered any special keys or the shift key yet). How many bits of randomness per character do you observe?
    * Being unhappy with the monkey, you replaced it by a drunk typesetter. It is able to generate words, albeit not coherently. Instead, it picks a random word out of a vocabulary of $2,000$ words. Moreover, assume that the average length of a word is $4.5$ letters in English. How many bits of randomness do you observe now?
    * Still being unhappy with the result, you replace the typesetter by a high quality language model. These can currently obtain perplexity numbers as low as $15$ points per character. The perplexity is defined as a length normalized probability, i.e., $$PPL(x) = \left[p(x)\right]^{1 / \text{length(x)} }.$$ How many bits of randomness do you observe now?
1. Explain intuitively why $I(X, Y) = H(X) - H(X|Y)$.  Then, show this is true by expressing both sides as an expectation with respect to the joint distribution.
1. What is the KL Divergence between the two Gaussian distributions $\mathcal{N}(\mu_1, \sigma_1^2)$ and $\mathcal{N}(\mu_2, \sigma_2^2)$?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/420)
:end_tab:
