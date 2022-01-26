# Yaklaşık Eğitim
:label:`sec_approx_train`

:numref:`sec_word2vec`'teki tartışmalarımızı hatırlayın. Skip-gram modelinin ana fikri, karşılık gelen logaritmik kaybı :eqref:`eq_skip-gram-log`'in tersi tarafından verilen :eqref:`eq_skip-gram-softmax`'da :eqref:`eq_skip-gram-softmax`'da $w_c$, verilen merkez kelimesine dayanan bir bağlam kelimesi $w_o$ oluşturma koşullu olasılığını hesaplamak için softmax işlemlerini kullanmaktır. 

Softmax işleminin doğası gereği, bir bağlam sözcüğü $\mathcal{V}$ sözlüğünde herhangi biri olabileceğinden, :eqref:`eq_skip-gram-log`'ün tersi, kelime dağarcığının tüm boyutu kadar olan öğelerin toplamını içerir. Sonuç olarak, :eqref:`eq_skip-gram-grad`'te atlama gram modeli için degrade hesaplaması ve :eqref:`eq_cbow-gradient`'daki sürekli torba kelime modeli için her ikisi de toplamı içerir. Ne yazık ki, büyük bir sözlük üzerinde (genellikle yüz binlerce veya milyonlarca kelimeyle) toplamı bu tür degradeler için hesaplama maliyeti çok büyüktür! 

Yukarıda bahsedilen hesaplama karmaşıklığını azaltmak için, bu bölüm yaklaşık iki eğitim yöntemi sunacaktır:
*negatif örnekleme* ve*hiyerarşik softmax*.
Skip-gram modeli ile sürekli kelime çantası modeli arasındaki benzerlik nedeniyle, bu iki yaklaşık eğitim yöntemini tanımlamak için atlama gram modelini örnek olarak alacağız. 

## Negatif Örnekleme
:label:`subsec_negative-sampling`

Negatif örnekleme, orijinal objektif işlevini değiştirir. Bir merkez kelimesinin bağlam penceresi göz önüne alındığında $w_c$, herhangi bir (bağlam) sözcüğünün $w_o$ bu bağlamda pencereden gelmesi olasılığı ile bir olay olarak kabul edilir. 

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

burada $\sigma$ sigmoid aktivasyon fonksiyonunun tanımını kullanır: 

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

Kelime gömme eğitmek için metin dizilerindeki tüm bu olayların ortak olasılığını en üst düzeye çıkararak başlayalım. Özellikle, $T$ uzunluğunda bir metin dizisi göz önüne alındığında, $w^{(t)}$ ile zaman adım $t$ kelimesini belirtin ve bağlam penceresi boyutunun $m$ olmasına izin verin, ortak olasılığın en üst düzeye çıkarılmasını düşünün 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

Ancak, :eqref:`eq-negative-sample-pos` yalnızca olumlu örnekler içeren olayları dikkate alır. Sonuç olarak, :eqref:`eq-negative-sample-pos`'teki ortak olasılık, yalnızca tüm kelime vektörleri sonsuzluğa eşitse 1'e en üst düzeye çıkarılır. Tabii ki, bu tür sonuçlar anlamsızdır. Objektif işlevi daha anlamlı hale getirmek için
*negatif örnekleme*
, önceden tanımlanmış bir dağıtımdan örneklenen negatif örnekler ekler. 

$S$ ile bir bağlam sözcüğünün $w_o$ $w_c$ merkez sözcüğünün bağlam penceresinden geldiği olayını belirtin. $w_o$ içeren bu olay için, önceden tanımlanmış bir dağıtımdan $P(w)$ örnek $K$*gürültü sözcükleri* bu bağlam penceresinden olmayan. $N_k$ ile $w_k$ ($k=1, \ldots, K$) gürültü sözcüğünün $w_c$ bağlam penceresinden gelmediği olayını belirtin. Olumlu örnek ve negatif örnekler $S, N_1, \ldots, N_K$ içeren bu olayların karşılıklı olarak bağımsız olduğunu varsayalım. Negatif örnekleme, :eqref:`eq-negative-sample-pos`'te ortak olasılık (sadece olumlu örnekler içeren) yeniden yazar 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

koşullu olasılık olayları $S, N_1, \ldots, N_K$ aracılığıyla yaklaşık nerede: 

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

$i_t$ ve $h_k$ ile bir sözcük $h_k$, sırasıyla bir metin dizisinin $t$ adımında $t$ ve bir gürültü sözcüğü $w_k$ olarak belirtin. :eqref:`eq-negative-sample-conditional-prob`'teki koşullu olasılıklara göre logaritmik kayıp 

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

Artık her eğitim adımındaki degradeler için hesaplama maliyetinin sözlük boyutuyla ilgisi olmadığını, ancak doğrusal olarak $K$'e bağlı olduğunu görebiliyoruz. Hiperparametre $K$'ü daha küçük bir değere ayarlarken, negatif örnekleme ile her eğitim adımındaki degradeler için hesaplama maliyeti daha küçüktür. 

## Hiyerarşik Softmax

Alternatif bir yaklaşık eğitim yöntemi olarak,
*hiyerarşik softmax*
, ağacın her yaprak düğümünün $\mathcal{V}$ sözlüğünde bir sözcüğü temsil ettiği :numref:`fig_hi_softmax`'te gösterilen bir veri yapısı olan ikili ağacı kullanır. 

![Hierarchical softmax for approximate training, where each leaf node of the tree represents a word in the dictionary.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

Kök düğümünden ikili ağacın $w$ sözcüğünü temsil eden yaprak düğümüne giden yol üzerindeki düğüm sayısını $L(w)$ ile belirtin. $n(w,j)$, bağlam sözcük vektörü $\mathbf{u}_{n(w, j)}$ olan bu yoldaki $j^\mathrm{th}$ düğümü olsun. Örneğin, :numref:`fig_hi_softmax`'te $L(w_3) = 4$. Hiyerarşik softmax koşullu olasılık yaklaşır :eqref:`eq_skip-gram-softmax` olarak 

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

$\sigma$ işlevi :eqref:`eq_sigma-f`'te tanımlanır ve $\text{leftChild}(n)$ düğümün sol alt düğümdür $n$:$x$ doğruysa, $ [\! [x]\!] = 1$; otherwise $ [\! [x]\!] = -1$. 

Göstermek için, :numref:`fig_hi_softmax`'te :numref:`fig_hi_softmax`'te $w_c$ kelimesi verilen $w_3$ kelimesini üretme koşullu olasılığını hesaplayalım. Bu, $w_c$ arasında $\mathbf{v}_c$ sözcük vektörü ve yapraksız düğüm vektörleri arasında nokta ürünleri gerektirir (:numref:`fig_hi_softmax`'te kalın yol) kökten $w_3$'e kadar sola, sağa, sonra sola geçilir: 

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

$\sigma(x)+\sigma(-x) = 1$ yılından bu yana, $w_c$ herhangi bir sözcüğe dayanan sözlük $\mathcal{V}$'daki tüm kelimeleri üretme koşullu olasılıklarının birine kadar topladığını tutar: 

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

Neyse ki, $L(w_o)-1$ ikili ağaç yapısı nedeniyle $\mathcal{O}(\text{log}_2|\mathcal{V}|)$ sırasına göre $\mathcal{O}(\text{log}_2|\mathcal{V}|)$ sırasına göre, $\mathcal{V}$ sözlük boyutu büyük olduğunda, hiyerarşik softmax kullanarak her eğitim adımı için hesaplama maliyeti yaklaşık eğitim olmadan karşılaştırıldığında önemli ölçüde azaltılır. 

## Özet

* Negatif örnekleme, hem pozitif hem de olumsuz örnekler içeren karşılıklı bağımsız olayları göz önünde bulundurarak kayıp işlevini oluşturur. Eğitim için hesaplama maliyeti doğrusal olarak her adımdaki gürültü kelimelerinin sayısına bağlıdır.
* Hiyerarşik softmax, kök düğümünden ikili ağacın yaprak düğümüne giden yolu kullanarak kayıp işlevini oluşturur. Eğitim için hesaplama maliyeti, her adımdaki sözlük boyutunun logaritmasına bağlıdır.

## Egzersizler

1. Negatif örneklemede gürültü kelimelerini nasıl örnekleyebiliriz?
1. :eqref:`eq_hi-softmax-sum-one`'ün tuttuğunu doğrulayın.
1. Nasıl sırasıyla negatif örnekleme ve hiyerarşik softmax kullanarak kelime modeli sürekli çanta eğitmek için?

[Discussions](https://discuss.d2l.ai/t/382)
