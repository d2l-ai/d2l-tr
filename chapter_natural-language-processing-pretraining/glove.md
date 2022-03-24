# Küresel Vektörler ile Sözcük Gömme (GloVe)
:label:`sec_glove`

Bağlam pencerelerindeki sözcük-sözcük birlikte oluşumları zengin anlamsal bilgiler taşıyabilir. Örneğin, büyük bir külliyatta "katı" sözcüğünün "buhar" yerine "buz" ile birlikte ortaya çıkması daha olasıdır, ancak "gaz" sözcüğü muhtemelen "buhar" ile birlikte bulunur. Ayrıca, bu tür ortak olayların küresel külliyat istatistikleri önceden hesaplanabilir: Bu daha verimli eğitime yol açabilir. Sözcük gömme için tüm külliyat içindeki istatistiksel bilgileri kullanmak için, önce :numref:`subsec_skip-gram`'teki skip-gram modelini tekrar gözden geçirelim, ancak ortak oluşum sayıları gibi küresel külliyat istatistiklerini kullanarak yorumlayalım. 

## Küresel Külliyat İstatistikleri ile Skip-Gram
:label:`subsec_skipgram-global`

$q_{ij}$ tarafından belirtilen koşullu olasılık $P(w_j\mid w_i)$ sözcüksinin $P(w_j\mid w_i)$ sözcüksinin $w_j$ sözcüksi verilen atlama gram modelinde $w_i$ sözcüksi verildi, 

Skip-gram modelinde $w_i$ sözcüğü verildiğinde $w_j$ sözcüğünün $P(w_j\mid w_i)$ koşullu olasılığını $q_{ij}$ ile ifade edersek, elimizde şu vardır:

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

burada herhangi $i$ dizini için $\mathbf{v}_i$ ve $\mathbf{u}_i$ için $\mathbf{u}_i$, sırasıyla merkez sözcük ve bağlam sözcüğü olarak $w_i$ sözcüğünü temsil eden vektörlerdir ve $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ sözcük dağarcığının dizin kümesidir. 

Külliyatta birden çok kez oluşabilecek $w_i$ sözcüğünü düşünün. Tüm külliyatta, $w_i$'nin merkez kelime olarak alındığı tüm bağlam kelimeleri, *aynı elemanın birden çok örneğine izin veren* bir *çoklu küme* $\mathcal{C}_i$ kelime indeksi oluşturur.  Herhangi bir öğe için, örnek sayısı onun *çokluğu* olarak adlandırılır. Bir örnekle göstermek için, $w_i$ sözcüğünün iki kez külliyat içinde gerçekleştiğini ve iki bağlam penceresinde merkez sözcük olarak $w_i$ alan bağlam sözcüklerinin indekslerinin $k, j, m, k$ ve $k, l, k, j$ olduğunu varsayalım. Böylece, $j, k, l, m$ elemanlarının çokluğu sırasıyla 2, 4, 1, 1 olan $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$ çoklu kümelidir. 

Şimdi $x_{ij}$ olarak çoklu küme $\mathcal{C}_i$'de $j$ elemanının çokluğunu gösterelim. Bu, tüm külliyattaki aynı bağlam penceresindeki $w_j$ (bağlam sözcüğü olarak) ve $w_i$ (orta sözcük olarak) sözcüğünün küresel birlikte bulunma sayısıdır. Bu tür küresel külliyat istatistiklerini kullanarak, skip-gram modelinin kayıp fonksiyonu aşağıdaki ifadeye eşdeğerdir:

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

Ayrıca $x_i$ ile $|\mathcal{C}_i|$'e eşdeğer olan $w_i$'nin merkez sözcük olarak gerçekleştiği bağlam pencerelerindeki tüm bağlam sözcüklerinin sayısını belirtiyoruz. $p_{ij}$, merkez kelime $w_i$ verildiğinde bağlam kelimesi $w_j$'yi oluşturmak için koşullu olasılık $x_{ij}/x_i$ olsun, :eqref:`eq_skipgram-x_ij` aşağıdaki gibi yeniden yazılabilir:

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij`'te $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$, küresel külliyat istatistiklerinin $p_{ij}$'nin koşullu dağılımının çapraz entropisi ve model tahminlerinin $q_{ij}$'in koşullu dağılımını hesaplar. Yukarıda açıklandığı gibi bu kayıp da $x_i$ tarafından ağırlıklandırılmıştır. :eqref:`eq_skipgram-p_ij`'te kayıp fonksiyonunun en aza indirilmesi, tahmini koşullu dağılımın küresel külliyat istatistiklerinden koşullu dağılıma yaklaşmasına olanak tanır. 

Olasılık dağılımları arasındaki mesafeyi ölçmek için yaygın olarak kullanılsa da, çapraz entropi kayıp fonksiyonu burada iyi bir seçim olmayabilir. Bir yandan, :numref:`sec_approx_train`'te belirttiğimiz gibi, $q_{ij}$'i düzgün bir şekilde normalleştirme maliyeti, hesaplama açısından pahalı olabilecek tüm sözcük dağarcığının toplamına neden olur. Öte yandan, büyük bir külliyattan gelen çok sayıda nadir olay genellikle çapraz entropi kaybı ile çok fazla ağırlıkla modellenir. 

## GloVe Modeli

Bunun ışığında, *GloVe* modeli, skip-gram modelinde :cite:`Pennington.Socher.Manning.2014` kare kaybına dayanarak üç değişiklik yapar: 

1. $p'_{ij}=x_{ij}$ ve $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ değişkenlerini kullanın, bunlar olasılık dağılımları değildir ve her ikisinin de logaritmasını alın, bu da kare kayıp terimidir: $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Her $w_i$ sözcüğü için iki sayıl model parametresi ekleyin: Merkez sözcük önyargısı $b_i$'dir ve bağlam sözcüğü önyargısı $c_i$'dir.
3. Her kayıp teriminin ağırlığını $h(x_{ij})$ ağırlık fonksiyonu ile değiştirin, burada $h(x)$ $[0, 1]$ aralığında artıyor.

Her şeyi bir araya getirirsek, GloVe eğitimi, aşağıdaki kayıp fonksiyonunu en aza indirmektir: 

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

Ağırlık fonksiyonu için önerilen bir seçim şu şekildedir: eğer $x < c$ (örn. $c = 100$) ise $h(x) = (x/c) ^\alpha$ (örn. $\alpha = 0.75$); aksi takdirde $h(x) = 1$. Bu durumda, $h(0)=0$ olduğundan, herhangi bir $x_{ij}=0$ için kare kayıp terimi hesaplama verimliliği için atlanabilir. Örneğin, eğitim için minigrup rasgele gradyan inişi kullanırken, her yinelemede gradyanları hesaplamak ve model parametrelerini güncellemek için rasgele *sıfır olmayan* $x_{ij}$ minigrup örnekleriz. Bu sıfır olmayan $x_{ij}$'lerin önceden hesaplanmış küresel külliyat istatistikleri olduğunu unutmayın; bu nedenle, modele *küresel vektörler (Global Vectors)* için Glove denir. 

$w_i$ sözcüğü $w_j$ sözcüğünün bağlam penceresinde görünürse, *tersi de geçerlidir*. Bu nedenle, $x_{ij}=x_{ji}$ olur. $p_{ij}$ asimetrik koşullu olasılığa uyan word2vec'in aksine, GloVe simetrik $\log \, x_{ij}$'a uyar. Bu nedenle, GloVe modelinde herhangi bir sözcüğün merkez sözcük vektörü ve bağlam sözcük vektörü matematiksel olarak eşdeğerdir. Ancak pratikte, farklı ilkleme değerleri nedeniyle, aynı sözcük eğitimden sonra bu iki vektörde yine de farklı değerler alabilir: GloVe bunları çıktı vektörü olarak toplar. 

## Eş-Oluşum Olasılıklarının Oranından Eldivenin Yorumlanması

Eldiven modelini başka bir perspektiften de yorumlayabiliriz. :numref:`subsec_skipgram-global`'te aynı gösterimi kullanarak, $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ verilmiş $w_j$ verilmiş $w_i$ corpus merkez sözcük olarak $w_i$ verilen bağlam sözcüksini üretme koşullu olasılık olsun. :numref:`tab_glove`, “buz” ve “buhar” sözcükleri verilen çeşitli ortak oluşum olasılıkları listeler ve bunların oranlarını büyük bir külliyat istatistiklerine dayanarak. 

:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:) 

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

Aşağıdakileri :numref:`tab_glove`'ten gözlemleyebiliriz: 

* $w_k$ “Buz” ile ilgili ancak $w_k=\text{solid}$ gibi “buhar” ile ilgisiz bir sözcük için, 8.9 gibi birlikte oluşma olasılıklarının daha büyük bir oranda bekleriz.
* $w_k$ “buhar” ile ilgili ancak $w_k=\text{gas}$ gibi “buz” ile ilgisiz bir sözcük için, 0.085 gibi birlikte oluşma olasılıklarının daha küçük bir oranını bekliyoruz.
* $w_k=\text{water}$ gibi “buz” ve “buhar” ile ilgili bir sözcük $w_k$ için, 1.36 gibi 1'e yakın bir arada oluşma olasılıkları oranı bekliyoruz.
* $w_k=\text{fashion}$ gibi “buz” ve “buhar” ile alakasız olan $w_k$ sözcüksi için, 0.96 gibi 1'e yakın bir eş oluşma olasılığının bir oranı bekleriz.

Birlikte oluşum olasılıklarının oranının sözcükler arasındaki ilişkiyi sezgisel olarak ifade edebileceği görülebilir. Böylece, bu orana uyacak şekilde üç sözcük vektöründen oluşan bir fonksiyon tasarlayabiliriz. $w_i$ orta sözcük olan $w_i$ ile birlikte oluşum olasılıklarının oranı için $w_j$ ve $w_k$'in bağlam sözcükleri olması için, bazı işlevler $f$ kullanarak bu oranı sığdırmak istiyoruz: 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

$f$ için birçok olası tasarım arasında, sadece aşağıdakilerden makul bir seçim seçiyoruz. Birlikte oluşum olasılıklarının oranı bir skaler olduğundan, $f$'un $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$ gibi skaler bir fonksiyon olmasını isteriz. :eqref:`eq_glove-f`'te $j$ ve $k$ sözcük endeksleri değiştirilmesi, $f(x)f(-x)=1$'yı tutması gerekir, bu nedenle bir olasılık $f(x)=\exp(x)$, yani  

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Şimdi $\alpha$'in sabit olduğu $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$'i seçelim. $p_{ij}=x_{ij}/x_i$'den beri, her iki tarafta logaritmayı aldıktan sonra $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$ alıyoruz. $- \log\, \alpha + \log\, x_i$'a uyacak ek önyargı terimleri kullanabiliriz, örneğin merkez sözcük önyargısı $b_i$ ve bağlam sözcüğü önyargısı $c_j$: 

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

:eqref:`eq_glove-square`'ün kare hatasının ağırlıklarla ölçülmesi, :eqref:`eq_glove-loss`'teki GloVe kaybı fonksiyonu elde edilir. 

## Özet

* Skip-gram modeli, sözcük-sözcük birlikte oluşum sayıları gibi küresel corpus istatistikleri kullanılarak yorumlanabilir.
* Çapraz entropi kaybı, özellikle büyük bir külliyat için iki olasılık dağılımının farkını ölçmek için iyi bir seçim olmayabilir. Eldiven, önceden hesaplanmış küresel külliyat istatistiklerine uymak için kare kaybı kullanır.
* Orta sözcük vektörü ve bağlam sözcük vektörü, Glove'daki herhangi bir sözcük için matematiksel olarak eşdeğerdir.
* Eldiven sözcük-sözcük birlikte oluşum olasılıklarının oranından yorumlanabilir.

## Egzersizler

1. $w_i$ ve $w_j$ sözcükleri aynı bağlam penceresinde birlikte ortaya çıkarsa, koşullu olasılık $p_{ij}$ hesaplama yöntemini yeniden tasarlamak için metin dizisindeki mesafelerini nasıl kullanabiliriz? Hint: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`.
1. Herhangi bir sözcük için, ortadaki sözcük önyargılı ve bağlam sözcük önyargılı Glove'da matematiksel olarak eşdeğer mi? Neden?

[Discussions](https://discuss.d2l.ai/t/385)
