# Küresel Vektörler ile Sözcük Gömme (Eldiven)
:label:`sec_glove`

Bağlam pencerelerindeki kelime-kelime birlikte oluşumlar zengin anlamsal bilgiler taşıyabilir. Örneğin, büyük bir korpus kelime “katı” “buhar” daha “buz” ile birlikte meydana olasılığı daha yüksektir, ancak kelime “gaz” muhtemelen “buhar” ile birlikte oluşur “buz” daha sık. Ayrıca, bu tür ortak olayların küresel korpus istatistikleri önceden hesaplanabilir: bu daha verimli eğitime yol açabilir. Kelime gömme için tüm corpus içindeki istatistiksel bilgileri kullanmak için, önce atlama gram modelini :numref:`subsec_skip-gram`'te tekrar gözden geçirelim, ancak ortak oluşum sayıları gibi küresel korpus istatistiklerini kullanarak yorumlayalım. 

## Global Corpus İstatistikleri ile Skip-Gram
:label:`subsec_skipgram-global`

$q_{ij}$ tarafından belirtilen koşullu olasılık $P(w_j\mid w_i)$ kelimesinin $P(w_j\mid w_i)$ kelimesinin $w_j$ kelimesi verilen atlama gram modelinde $w_i$ kelimesi verildi, 

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

$i$ vektörleri $\mathbf{v}_i$ ve $\mathbf{u}_i$ için $\mathbf{u}_i$, sırasıyla orta kelime ve bağlam sözcük olarak $w_i$ kelimesini temsil eder ve $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ sözcük dağarcığının dizin kümesidir. 

Corpus birden çok kez oluşabilecek $w_i$ kelimesini düşünün. Tüm korpusta, $w_i$'ün merkez kelimesi olarak alındığı tüm bağlam kelimeleri, *çoklu set* $\mathcal{C}_i$'i, aynı elementin birden çok örneğini sağlayan sözcük indekslerinin bir *multiset* $\mathcal{C}_i$'ini oluşturur. Herhangi bir öğe için, örnek sayısı*çokluğu* olarak adlandırılır. Bir örnekle göstermek için, $w_i$ sözcüğünün iki kez corpus içinde gerçekleştiğini ve iki bağlam penceresinde merkez sözcük olarak $w_i$ alan bağlam sözcüklerinin indekslerinin $k, j, m, k$ ve $k, l, k, j$ olduğunu varsayalım. Böylece, $j, k, l, m$ elemanlarının çokluğu sırasıyla 2, 4, 1, 1 olan $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$ multiset $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$. 

Şimdi $x_{ij}$ olarak multiset $\mathcal{C}_i$'da $\mathcal{C}_i$ elemanının çokluğunu gösterelim. Bu, tüm corpus içindeki aynı bağlam penceresinde $w_j$ (bağlam sözcüğü olarak) ve $w_i$ kelimesinin (orta kelime olarak) sözcüğünün küresel ortak oluşum sayısıdır. Bu tür küresel korpus istatistiklerini kullanarak, atlama gram modelinin kayıp fonksiyonu 

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

Ayrıca $x_i$ ile $x_i$'e eşdeğer olan $w_i$'nin merkez sözcük olarak gerçekleştiği bağlam pencerelerindeki tüm bağlam kelimelerinin sayısını belirtiyoruz. $p_{ij}$ koşullu olasılık olması $x_{ij}/x_i$ bağlam sözcük oluşturmak için $w_j$ verilen merkez sözcük $w_i$, :eqref:`eq_skipgram-x_ij` olarak yeniden yazılabilir 

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij`'te $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$, küresel korpus istatistiklerinin $p_{ij}$'nin koşullu dağılımının çapraz entropisi ve model tahminlerinin $q_{ij}$'in koşullu dağılımını hesaplar. Yukarıda açıklandığı gibi bu kayıp da $x_i$ tarafından ağırlıklandırılmıştır. :eqref:`eq_skipgram-p_ij`'te kayıp fonksiyonunun en aza indirilmesi, tahmini koşullu dağılımın küresel korpus istatistiklerinden koşullu dağılıma yaklaşmasına olanak tanır. 

Olasılık dağılımları arasındaki mesafeyi ölçmek için yaygın olarak kullanılsa da, çapraz entropi kayıp fonksiyonu burada iyi bir seçim olmayabilir. Bir yandan, :numref:`sec_approx_train`'te belirttiğimiz gibi, $q_{ij}$'i düzgün bir şekilde normalleştirme maliyeti, hesaplama açısından pahalı olabilecek tüm kelime dağarcığının toplamına neden olur. Öte yandan, büyük bir korpustan gelen çok sayıda nadir olay genellikle çapraz entropi kaybı ile çok fazla kilo verilecek şekilde modellenir. 

## Eldiven Modeli

Bunun ışığında, *GloVe* modeli, atlama gram modelinde :cite:`Pennington.Socher.Manning.2014`'ün kare kaybına dayanarak üç değişiklik yapar: 

1. $p'_{ij}=x_{ij}$ ve $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ değişkenlerini kullan 
Bu olasılık dağılımları değildir ve her ikisinin de logaritmasını alır, bu nedenle kareli kayıp terimi $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$'tür.
2. Her kelime için iki skaler model parametresi ekleyin $w_i$: orta kelime önyargı $b_i$ ve bağlam sözcüğü önyargı $c_i$.
3. Her kayıp teriminin ağırlığını $h(x_{ij})$ ağırlık fonksiyonuyla değiştirin, burada $h(x)$'in $[0, 1]$ aralığında arttığı $h(x_{ij})$.

Eldiven eğitimi, her şeyi bir araya getirerek aşağıdaki kayıp fonksiyonunu en aza indirmektir: 

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

Ağırlık fonksiyonu için önerilen bir seçim şu şekildedir: $h(x) = (x/c) ^\alpha$ (örn. $\alpha = 0.75$) $x < c$ (örn. $c = 100$); aksi takdirde $h(x) = 1$. Bu durumda, $h(0)=0$, herhangi bir $x_{ij}=0$ için kareli kayıp terimi hesaplama verimliliği için atlanabilir. Örneğin, eğitim için mini batch stokastik degrade iniş kullanırken, her yinelemede degradeleri hesaplamak ve model parametrelerini güncellemek için rasgele *sıfır* $x_{ij}$ mini batch örnekleriz. Bu sıfır olmayan $x_{ij}$'ün önceden hesaplanmış küresel korpus istatistikleri olduğunu unutmayın; bu nedenle, modele *Global Vectors* için Eldiven denir. 

$w_i$ sözcüğü $w_j$ sözcüğünün bağlam penceresinde görünürse, *tersi*. Bu nedenle, $x_{ij}=x_{ji}$. Asimetrik koşullu olasılık $p_{ij}$'ye uyan word2vec aksine, Eldiven simetrik $\log \, x_{ij}$'e uyuyor. Bu nedenle, GloVe modelinde herhangi bir kelimenin orta kelime vektörü ve bağlam sözcük vektörü matematiksel olarak eşdeğerdir. Ancak pratikte, farklı başlatma değerleri nedeniyle, aynı kelime eğitimden sonra bu iki vektörde yine de farklı değerler alabilir: GloVe bunları çıktı vektörü olarak özetliyor. 

## Eş-Oluşum Olasılıklarının Oranından Eldivenin Yorumlanması

Eldiven modelini başka bir perspektiften de yorumlayabiliriz. :numref:`subsec_skipgram-global`'te aynı gösterimi kullanarak, $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ verilmiş $w_j$ verilmiş $w_i$ corpus merkez kelime olarak $w_i$ verilen bağlam kelimesini üretme koşullu olasılık olsun. :numref:`tab_glove`, “buz” ve “buhar” kelimeleri verilen çeşitli ortak oluşum olasılıkları listeler ve bunların oranlarını büyük bir korpus istatistiklerine dayanarak. 

:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:) 

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

Aşağıdakileri :numref:`tab_glove`'ten gözlemleyebiliriz: 

* $w_k$ “Buz” ile ilgili ancak $w_k=\text{solid}$ gibi “buhar” ile ilgisiz bir kelime için, 8.9 gibi birlikte oluşma olasılıklarının daha büyük bir oranda bekleriz.
* $w_k$ “buhar” ile ilgili ancak $w_k=\text{gas}$ gibi “buz” ile ilgisiz bir kelime için, 0.085 gibi birlikte oluşma olasılıklarının daha küçük bir oranını bekliyoruz.
* $w_k=\text{water}$ gibi “buz” ve “buhar” ile ilgili bir kelime $w_k$ için, 1.36 gibi 1'e yakın bir arada oluşma olasılıkları oranı bekliyoruz.
* $w_k=\text{fashion}$ gibi “buz” ve “buhar” ile alakasız olan $w_k$ kelimesi için, 0.96 gibi 1'e yakın bir eş oluşma olasılığının bir oranı bekleriz.

Birlikte oluşum olasılıklarının oranının kelimeler arasındaki ilişkiyi sezgisel olarak ifade edebileceği görülebilir. Böylece, bu orana uyacak şekilde üç kelime vektöründen oluşan bir fonksiyon tasarlayabiliriz. $w_i$ orta kelime olan $w_i$ ile birlikte oluşum olasılıklarının oranı için $w_j$ ve $w_k$'in bağlam sözcükleri olması için, bazı işlevler $f$ kullanarak bu oranı sığdırmak istiyoruz: 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

$f$ için birçok olası tasarım arasında, sadece aşağıdakilerden makul bir seçim seçiyoruz. Birlikte oluşum olasılıklarının oranı bir skaler olduğundan, $f$'un $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$ gibi skaler bir fonksiyon olmasını isteriz. :eqref:`eq_glove-f`'te $j$ ve $k$ sözcük endeksleri değiştirilmesi, $f(x)f(-x)=1$'yı tutması gerekir, bu nedenle bir olasılık $f(x)=\exp(x)$, yani  

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Şimdi $\alpha$'in sabit olduğu $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$'i seçelim. $p_{ij}=x_{ij}/x_i$'den beri, her iki tarafta logaritmayı aldıktan sonra $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$ alıyoruz. $- \log\, \alpha + \log\, x_i$'a uyacak ek önyargı terimleri kullanabiliriz, örneğin merkez sözcük önyargısı $b_i$ ve bağlam sözcüğü önyargısı $c_j$: 

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

:eqref:`eq_glove-square`'ün kare hatasının ağırlıklarla ölçülmesi, :eqref:`eq_glove-loss`'teki GloVe kaybı fonksiyonu elde edilir. 

## Özet

* Skip-gram modeli, sözcük-kelime birlikte oluşum sayıları gibi küresel corpus istatistikleri kullanılarak yorumlanabilir.
* Çapraz entropi kaybı, özellikle büyük bir korpus için iki olasılık dağılımının farkını ölçmek için iyi bir seçim olmayabilir. Eldiven, önceden hesaplanmış küresel korpus istatistiklerine uymak için kare kaybı kullanır.
* Orta kelime vektörü ve bağlam sözcük vektörü, Glove'daki herhangi bir kelime için matematiksel olarak eşdeğerdir.
* Eldiven kelime-kelime birlikte oluşum olasılıklarının oranından yorumlanabilir.

## Egzersizler

1. $w_i$ ve $w_j$ kelimeleri aynı bağlam penceresinde birlikte ortaya çıkarsa, koşullu olasılık $p_{ij}$ hesaplama yöntemini yeniden tasarlamak için metin dizisindeki mesafelerini nasıl kullanabiliriz? Hint: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`.
1. Herhangi bir kelime için, ortadaki kelime önyargılı ve bağlam sözcük önyargılı Glove'da matematiksel olarak eşdeğer mi? Neden?

[Discussions](https://discuss.d2l.ai/t/385)
