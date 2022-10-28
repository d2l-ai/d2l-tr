# Yaklaşık Eğitim
:label:`sec_approx_train`

:numref:`sec_word2vec` içindeki tartışmalarımızı hatırlayın. Skip-gram modelinin ana fikri, logaritmik kaybın :eqref:`eq_skip-gram-log` tersine karşılık gelen :eqref:`eq_skip-gram-softmax` içinde verilen $w_c$ merkez sözcüğü bazında bir $w_o$ bağlam sözcüğü oluşturmanın koşullu olasılığını hesaplamak için softmaks işlemlerini kullanmaktır.

Softmaks işleminin doğası gereği, bir bağlam sözcüğü $\mathcal{V}$ sözlüğündeki herhangi biri olabileceğinden, :eqref:`eq_skip-gram-log` denkleminin tersine, sözcük dağarcığının tüm boyutu kadar olan öğelerin toplamını içerir. Sonuç olarak, :eqref:`eq_skip-gram-grad` denklemindeki skip-gram modeli için gradyan hesaplamasının ve :eqref:`eq_cbow-gradient` denklemindeki sürekli sözcük torba modelinin ikisi de toplamı içerir. Ne yazık ki, büyük bir sözlük üzerinde (genellikle yüz binlerce veya milyonlarca sözcükle) toplamı bu tür gradyanlar için hesaplama maliyeti çok büyüktür! 

Yukarıda bahsedilen hesaplama karmaşıklığını azaltmak için, bu bölüm iki yaklaşıklama eğitim yöntemi sunacaktır: *Negatif örnekleme* ve *hiyerarşik softmaks*.
Skip-gram modeli ile sürekli sözcük torbası modeli arasındaki benzerlik nedeniyle, bu iki yaklaşıklama eğitim yöntemini tanımlamak için skip-gram modelini örnek olarak alacağız. 

## Negatif Örnekleme
:label:`subsec_negative-sampling`

Negatif örnekleme, orijinal amaç işlevini değiştirir. Bir $w_c$ merkez sözcüğünün bağlam penceresi göz önüne alındığında, herhangi bir $w_o$ (bağlam) sözcüğünün bu bağlam penceresinden gelmesi olayı olasılığı olarak modellenir: 

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

burada $\sigma$ ile sigmoid etkinleştirme fonksiyonunun tanımı kullanılır: 

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

Sözcük gömmeyi eğitmek için metin dizilerindeki tüm bu olayların bileşik olasılığını en üst düzeye çıkararak başlayalım. Özellikle, $T$ uzunluğunda bir metin dizisi göz önüne alındığında, $t$ zaman adımındaki sözcüğü $w^{(t)}$ ile belirtin ve bağlam penceresi boyutunun $m$ olmasına izin verin, bileşik olasılığın en üst değere çıkarılmasını düşünün: 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

Ancak, :eqref:`eq-negative-sample-pos` yalnızca olumlu örnekler içeren olayları dikkate alır. Sonuç olarak, :eqref:`eq-negative-sample-pos` içindeki bileşik olasılık, yalnızca tüm kelime vektörleri sonsuza eşitse 1'e maksimize edilir. Tabii ki, bu tür sonuçlar anlamsızdır. Amaç işlevini daha anlamlı hale getirmek için *negatif örnekleme*, önceden tanımlanmış bir dağılımdan örneklenen negatif örnekler ekler. 

$w_o$ bağlam kelimesinin bir merkez $w_c$ kelimesinin bağlam penceresinden gelmesi olayını $S$ ile belirtin. $w_o$ içeren bu olay için, önceden tanımlanmış bir $P(w)$ dağılımından, bu bağlam penceresinden olmayan $K$ *gürültü sözcükleri*ni örnekler. $w_k$ ($k=1, \ldots, K$) gürültü sözcüğünün $w_c$ bağlam penceresinden gelmeme olayını $N_k$ ile belirtin. Hem olumlu hem de olumsuz örnekleri içeren bu olayların, $S, N_1, \ldots, N_K$, karşılıklı dışlanan olaylar olduğunu varsayın. :eqref:`eq-negative-sample-pos` denklemindeki negatif örnekleme, bileşik olasılığı (sadece olumlu örnekler içeren) aşağıdaki gibi yeniden yazar:

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

$S, N_1, \ldots, N_K$ olayları aracılığıyla koşullu olasılık şöyle yaklaşıklanır: 

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

Sırasıyla, bir metin dizisinin $t$ adımında $w^{(t)}$ sözcüğünün ve bir gürültü $w_k$ sözcüğünün indekslerini $i_t$ ve $h_k$ ile gösterin. :eqref:`eq-negative-sample-conditional-prob`'teki koşullu olasılıklara göre logaritmik kayıp şöyledir:
$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

Artık her eğitim adımındaki gradyanlar için hesaplama maliyetinin sözlük boyutuyla ilgisi olmadığını, ancak doğrusal olarak $K$'ya bağlı olduğunu görebiliyoruz. Hiper parametre $K$'yı daha küçük bir değere ayarlarken, negatif örnekleme ile her eğitim adımındaki gradyanlar için hesaplama maliyeti daha küçüktür. 

## Hiyerarşik Softmaks

Alternatif bir yaklaşıklama eğitimi yöntemi olarak, *hiyerarşik softmaks* ikili ağacı kullanır ve bu veri yapısı :numref:`fig_hi_softmax` içinde gösterilen bir veri yapısıdır; burada ağacın her bir yaprak düğümü $\mathcal{V}$ sözlüğünde bir sözcüğü temsil eder.

![Ağacın her yaprak düğümünün sözlükte bir kelimeyi temsil ettiği yaklaşıklama eğitimi için hiyerarşik softmaks.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

İkili ağaçtaki $w$ kelimesini temsil eden kök düğümden yaprak düğüme giden yoldaki düğümlerin sayısını (her iki uç dahil) $L(w)$ ile belirtin. $n(w,j)$, bağlam kelime vektörü $\mathbf{u}_{n(w, j)}$ olacak şekilde, bu yoldaki $j.$ düğüm olsun. Örneğin, :numref:`fig_hi_softmax` içinde $L(w_3) = 4$ olsun. :eqref:`eq_skip-gram-softmax` denklemindeki koşullu olasılık hiyerarşik softmaks ile yaklaşıklanır: 

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

$\sigma$ işlevi :eqref:`eq_sigma-f` denleminde tanımlanır ve $\text{leftChild}(n)$, $n$ düğümünün sol alt düğümüdür: $x$ doğruysa, $[\![x]\!] = 1$; aksi halde $[\![x]\!] = -1$. 

Göstermek için, :numref:`fig_hi_softmax` içinde $w_c$ sözcüğü verildiğinde $w_3$ sözcüğünü üretme koşullu olasılığını hesaplayalım. Bu, $w_c$ arasında $\mathbf{v}_c$ sözcük vektörü ve yaprak olmayan düğüm vektörleri arasında nokta çarpımlarını gerektirir (:numref:`fig_hi_softmax` içindeki kalın yol), kökten $w_3$'e kadar sola, sağa, sonra sola ilerler: 

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

$\sigma(x)+\sigma(-x) = 1$ olduğundan, $\mathcal{V}$ sözlüğündeki tüm sözcükleri herhangi bir $w_c$ sözcüğüne dayalı olarak üretmenin koşullu olasılıklarının toplamının 1 olduğunu tutar: 

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

Neyse ki, $L(w_o)-1$ ikili ağaç yapısı nedeniyle $\mathcal{O}(\text{log}_2|\mathcal{V}|)$ düzeyinde olduğundan, sözlük boyutu $\mathcal{V}$ çok büyüktür, hiyerarşik softmaks kullanan her eğitim adımının hesaplama maliyeti, yaklaşıklama eğitimi olmadan yapılana kıyasla önemli ölçüde azalır.

## Özet

* Negatif örnekleme, hem pozitif hem de negatif örnekler içeren karşılıklı dışlanan olayları göz önünde bulundurarak kayıp işlevini oluşturur. Eğitim için hesaplama maliyeti doğrusal olarak her adımdaki gürültü sözcüklerinin sayısına bağlıdır.
* Hiyerarşik softmaks, kök düğümünden ikili ağacın yaprak düğümüne giden yolu kullanarak kayıp işlevini oluşturur. Eğitim için hesaplama maliyeti, her adımdaki sözlük boyutunun logaritmasına bağlıdır.

## Alıştırmalar

1. Negatif örneklemede gürültü sözcüklerini nasıl örnekleyebiliriz?
1. :eqref:`eq_hi-softmax-sum-one` denkleminin tuttuğunu doğrulayın.
1. Sırasıyla negatif örnekleme ve hiyerarşik softmaks kullanarak sürekli kelime torbası modeli nasıl eğitilir?

[Tartışmalar](https://discuss.d2l.ai/t/382)
