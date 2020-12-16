# Yinelemeli Sinir Ağları
:label:`sec_rnn`

:numref:`sec_language_model`'te $n$-gramlık modelleri tanıttık; burada $x_t$ kelimesinin $x_t$'in zaman adımındaki koşullu olasılığının sadece $n-1$ önceki kelimelere bağlı olduğu $n$-gramlık modeller. Biz $x_t$ üzerinde $t-(n-1)$ zaman adımından daha erken kelimelerin olası etkisini dahil etmek istiyorsanız, biz $n$ artırmak gerekir. Bununla birlikte, model parametrelerinin sayısı da katlanarak artacaktır, çünkü $|\mathcal{V}|^n$ bir kelime seti için $\mathcal{V}$ numaralarını depolamamız gerekir. Bu nedenle, $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$'yı modellemek yerine gizli bir değişken model kullanılması tercih edilir:

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

burada $h_{t-1}$, $t-1$ adıma kadar sıra bilgilerini depolayan bir*gizli durum* (gizli değişken olarak da bilinir) dir. Genel olarak, $t$ adımındaki herhangi bir zamanda gizli durum geçerli giriş $x_{t}$ ve önceki gizli durum $h_{t-1}$ temel alınarak hesaplanabilir:

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

:eqref:`eq_ht_xt`'te $f$ yeterince güçlü bir işlev için, latent değişken modeli bir yaklaşım değildir. Sonuçta, $h_t$ şimdiye kadar gözlemlediği tüm verileri saklayabilir. Ancak, potansiyel olarak hem hesaplama hem de depolama pahalı hale getirebilir.

:numref:`chap_perceptrons`'te gizli birimlerle gizli katmanları tartıştığımızı hatırlayın. Gizli katmanların ve gizli durumların iki çok farklı konsepte işaret etmeleri dikkat çekicidir. Gizli katmanlar, açıklandığı gibi, girdiden çıktıya giden yolda görünümden gizlenen katmanlardır. Gizli devletler teknik olarak belirli bir adımda yaptığımız her şeye *girişler* konuşmaktadır ve yalnızca önceki zaman adımlarında verilere bakarak hesaplanabilirler.

*Tekrarlayan sinir ağları* (RNN) gizli durumlara sahip sinir ağlarıdır. RNN modelini tanıtmadan önce, ilk olarak :numref:`sec_mlp`'te tanıtılan MLP modelini tekrar ziyaret ediyoruz.

## Gizli Devletler Olmayan Sinir Ağları

Tek bir gizli katmana sahip bir MLP'ye bir göz atalım. Gizli katmanın etkinleştirme işlevinin $\phi$ olmasına izin verin. $n$ ve $d$ girişleri toplu boyutu ile örnek $\mathbf{X} \in \mathbb{R}^{n \times d}$ bir minibatch göz önüne alındığında, gizli katmanın çıkış $\mathbf{H} \in \mathbb{R}^{n \times h}$ olarak hesaplanır

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

:eqref:`rnn_h_without_state`'te, $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ ağırlık parametresi, $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ önyargı parametresi ve gizli katman için $h$ numaralı gizli birimlerin sayısına sahibiz. Böylece, yayın (bkz. :numref:`subsec_broadcasting`) toplamı sırasında uygulanır. Ardından, çıkış katmanının girişi olarak $\mathbf{H}$ gizli değişken kullanılır. Çıktı katmanı tarafından verilir

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

burada $\mathbf{O} \in \mathbb{R}^{n \times q}$ çıktı değişkeni, $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ ağırlık parametresi ve $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ çıktı katmanının önyargı parametresidir. Eğer bir sınıflandırma problemi ise, çıktı kategorilerinin olasılık dağılımını hesaplamak için $\text{softmax}(\mathbf{O})$'i kullanabiliriz.

Bu, :numref:`sec_sequence`'te daha önce çözdüğümüz regresyon problemine tamamen benzer, dolayısıyla ayrıntıları atlıyoruz. Özellik etiketi çiftlerini rastgele seçebileceğimizi ve ağımızın parametrelerini otomatik farklılaşma ve stokastik degrade iniş yoluyla öğrenebileceğimizi söylemek yeterli.

## Gizli Devletler ile Tekrarlayan Sinir Ağları
:label:`subsec_rnn_w_hidden_states`

Gizli devletlerimiz olduğunda işler tamamen farklıdır. Yapıyı biraz daha ayrıntılı olarak inceleyelim.

Biz girişleri bir minibatch olduğunu varsayalım $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ zaman adım $t$. Başka bir deyişle, $n$ dizi örneklerinden oluşan bir minibatch için, $\mathbf{X}_t$'ün her satırı, diziden $t$ adımındaki bir örneğe karşılık gelir. Ardından, $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ ile $t$ zaman adımının gizli değişkeni belirtin. MLP'den farklı olarak, burada gizli değişkeni $\mathbf{H}_{t-1}$'i önceki zaman adımından kaydediyoruz ve geçerli zaman adımında önceki zaman adımının gizli değişkeni nasıl kullanılacağını açıklamak için $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ yeni bir ağırlık parametresi tanıtıyoruz. Özellikle, geçerli zaman adımının gizli değişkeninin hesaplanması, önceki zaman adımının gizli değişkeni ile birlikte geçerli zaman adımının girdisi tarafından belirlenir:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

:eqref:`rnn_h_without_state` ile karşılaştırıldığında, :eqref:`rnn_h_with_state` bir terim daha $\mathbf{H}_{t-1} \mathbf{W}_{hh}$ ekler ve böylece :eqref:`eq_ht_xt`'ü başlatır. Bitişik zaman adımlarının $\mathbf{H}_t$ ve $\mathbf{H}_{t-1}$ gizli değişkenleri arasındaki ilişkiden, bu değişkenlerin sıranın tarihsel bilgilerini güncel zaman adımına kadar yakaladığını ve sakladığını biliyoruz, tıpkı sinir ağının şimdiki zaman adımının durumu veya hafızası gibi. Bu nedenle, böyle bir gizli değişkeni *gizli durum* olarak adlandırılır. Gizli durum geçerli zaman adımında önceki zaman adımının aynı tanımını kullandığından, :eqref:`rnn_h_with_state` hesaplama*yineleme* olur. Bu nedenle, tekrarlayan hesaplamalara dayalı gizli durumlara sahip sinir ağları
*tekrarlayan sinir ağları*.
RNN'lerde :eqref:`rnn_h_with_state`'ün hesaplanmasını gerçekleştiren katmanlar*tekrarlayan katmanlar* olarak adlandırılır.

RNN oluşturmak için birçok farklı yol vardır. :eqref:`rnn_h_with_state` tarafından tanımlanan gizli bir duruma sahip RNN'ler çok yaygındır. Zaman adımı $t$ için çıktı katmanının çıktısı MLP'deki hesaplamaya benzer:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

RNN parametreleri $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ ağırlıkları ve $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ çıkış katmanının $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ ağırlıkları ve $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ önyargı ile birlikte gizli katmanın $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ önyargılarını içerir. Farklı zaman adımlarında bile, RNN'lerin her zaman bu model parametrelerini kullandığını belirtmek gerekir. Bu nedenle, bir RNN parameterizasyon maliyeti zaman adım sayısı arttıkça büyümez.

:numref:`fig_rnn`, bitişik üç zaman adımında bir RNN'nin hesaplama mantığını göstermektedir. Herhangi bir zamanda adım $t$, gizli durumun hesaplanması şu şekilde kabul edilebilir: i) $\mathbf{X}_t$ giriş $t$ geçerli zaman adımında ve $\mathbf{H}_{t-1}$ gizli durum $\mathbf{H}_{t-1}$ önceki zaman adımı $t-1$; ii) birleştirme sonucunu etkinleştirme ile tam bağlı bir katmana beslemek fonksiyon $\phi$. Böyle tam bağlı bir katmanın çıktısı, $t$ geçerli zaman adımının $\mathbf{H}_t$'inin gizli durumudur. Bu durumda, model parametreleri $\mathbf{W}_{xh}$ ve $\mathbf{W}_{hh}$'ün birleştirilmesi ve $\mathbf{b}_h$'lik bir önyargı olup, hepsi :eqref:`rnn_h_with_state`'ten itibaren $\mathbf{b}_h$'dir. Geçerli zaman adımının $t$, $\mathbf{H}_t$ gizli durumu, $t+1$ sonraki adımın $\mathbf{H}_{t+1}$ gizli durumunun hesaplanmasına katılacak. Dahası, $\mathbf{H}_t$, $t$ geçerli zaman adımının $\mathbf{O}_t$ çıkışını hesaplamak için tam bağlı çıktı katmanına da beslenir.

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

Biz sadece gizli durum için $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ hesaplama $\mathbf{X}_t$ ve $\mathbf{H}_{t-1}$ ve $\mathbf{W}_{xh}$ ve $\mathbf{W}_{hh}$ birleştirme matris çarpma eşdeğer olduğunu belirtti. Bu matematikte kanıtlanmış olsa da, aşağıda sadece bunu göstermek için basit bir kod parçacığı kullanıyoruz. Başlangıç olarak, şekilleri (3, 1), (1, 4), (3, 4), (3, 4), (3, 4), `W_xh`, `H` ve `W_hh` matrisleri tanımlarız. `X` ile `W_xh` ve `H` ile sırasıyla `W_hh` çarparak ve daha sonra bu iki çarpımı ekleyerek, bir şekil matrisi elde ederiz (3, 4).

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Şimdi matrisleri `X` ve `H` sütunlar boyunca (eksen 1) ve `W_xh` ve `W_hh` matrisleri satırlar boyunca (eksen 0) birleştiririz. Bu iki birleştirme, sırasıyla şekil matrisleri (3, 5) ve şekil (5, 4) ile sonuçlanır. Bu iki birleştirilmiş matrisi çarparak, yukarıdaki gibi aynı çıkış matrisini (3, 4) elde ederiz.

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## RNN Tabanlı Karakter Düzeyinde Dil Modelleri

:numref:`sec_language_model`'teki dil modellemesi için, mevcut ve geçmiş belirteçlere dayalı bir sonraki simgeyi tahmin etmeyi amaçladığımızı hatırlayın, böylece orijinal diziyi etiketler olarak bir belirteç ile kaydırıyoruz. Şimdi RNN'lerin bir dil modeli oluşturmak için nasıl kullanılabileceğini gösteriyoruz. Minibatch boyutunun 1 olmasına ve metnin sırasının “makine” olmasına izin verin. Sonraki bölümlerdeki eğitimi basitleştirmek için, metni sözcükler yerine karakterler haline getiririz ve *karakter düzeyinde bir dil modeli* göz önünde bulundururuz. :numref:`fig_rnn_train`, karakter düzeyinde dil modellemesi için bir RNN aracılığıyla geçerli ve önceki karakterlere dayalı bir sonraki karakterin nasıl tahmin edileceğini gösterir.

![A character-level language model based on the RNN. The input and label sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

Eğitim işlemi sırasında, çıkış katmanından çıktıda her zaman adım için bir softmax işlemi çalıştırır ve daha sonra model çıktısı ile etiket arasındaki hatayı hesaplamak için çapraz entropi kaybını kullanırız. Gizli katmandaki gizli durumun tekrarlayan hesaplanması nedeniyle, :numref:`fig_rnn_train`, $\mathbf{O}_3$'teki zaman adım 3'ün çıkışı, “m”, “a” ve “c” metin dizisi ile belirlenir. Eğitim verilerindeki dizinin bir sonraki karakteri “h” olduğu için, zaman adım 3 kaybı, “m”, “a”, “c” ve bu zaman adımının “h” etiketine göre oluşturulan bir sonraki karakterin olasılık dağılımına bağlı olacaktır.

Uygulamada, her belirteç bir $d$ boyutlu vektör ile temsil edilir ve bir toplu boyutu $n>1$ kullanıyoruz. Bu nedenle, $t$'deki $\mathbf X_t$ giriş :numref:`subsec_rnn_w_hidden_states`'te tartıştığımız şeyle aynı olan $n\times d$ matrisi olacaktır.

## Şaşksızlık
:label:`subsec_perplexity`

Son olarak, sonraki bölümlerde RNN tabanlı modellerimizi değerlendirmek için kullanılacak dil modeli kalitesini nasıl ölçeceğimizi tartışalım. Bir yol, metnin ne kadar şaşırtıcı olduğunu kontrol etmektir. İyi bir dil modeli, daha sonra ne göreceğimizi yüksek hassasiyetli belirteçlerle tahmin edebilir. Farklı dil modelleri tarafından önerilen “Yağmur yağıyor” ifadesinin aşağıdaki devamlarını göz önünde bulundurun:

1. “Dışarıda yağmur yağıyor”
1. “Muz ağacı yağıyor”
1. “Piouw yağıyor; kcj pwepoiut”

Kalite açısından, örnek 1 açıkça en iyisidir. Sözcükler mantıklı ve mantıksal olarak tutarlı. Hangi kelimenin anlamsal olarak takip ettiğini tam olarak doğru bir şekilde yansıtmayabilir (“San Francisco'da” ve “kışın” mükemmel şekilde makul uzantıları olurdu), model hangi kelimeyi takip ettiğini yakalayabilir. Örnek 2, mantıksız bir uzantı üreterek oldukça kötüdür. Yine de, en azından model kelimelerin nasıl yazılacağını ve kelimeler arasındaki korelasyon derecesini öğrendi. Son olarak, örnek 3, verileri düzgün şekilde uymayan kötü eğitilmiş bir modeli gösterir.

Dizinin olasılığını hesaplayarak modelin kalitesini ölçebiliriz. Ne yazık ki bu, anlaşılması zor ve karşılaştırılması zor bir sayıdır. Sonuçta, daha kısa dizilerin daha uzun olanlardan daha fazla gerçekleşme olasılığı daha yüksektir, bu nedenle modeli Tolstoy'un magnum opus'unda değerlendirir
*Savaş ve Barış* kaçınılmaz olarak Saint-Exupery'nin “Küçük Prens” romanından çok daha küçük bir olasılık üretecektir. Ekip olan bir ortalamanın eşdeğeridir.

Bilgi teorisi burada işe yarar. softmax regresyonunu (:numref:`subsec_info_theory_basics`) tanıttığımızda entropi, sürpriz ve çapraz entropi tanımladık ve [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html)'da daha fazla bilgi teorisi tartışıldı. Metni sıkıştırmak istiyorsak, geçerli belirteç kümesi verilen bir sonraki belirteci tahmin etmeyi sorabiliriz. Daha iyi bir dil modeli, bir sonraki belirteci daha doğru tahmin etmemizi sağlamalıdır. Böylece, diziyi sıkıştırmak için daha az bit harcamamıza izin vermelidir. Böylece, bir dizinin tüm $n$ belirteçleri üzerinden ortalaması yapılan çapraz entropi kaybıyla ölçebiliriz:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

Burada $P$ bir dil modeli tarafından verilir ve $x_t$, diziden $t$ adımında gözlenen gerçek simgedir. Bu, farklı uzunluklardaki belgelerdeki performansı karşılaştırılabilir hale getirir. Tarihsel nedenlerden dolayı, doğal dil işlemede bilim adamları *şaşkınlık* adı verilen bir miktar kullanmayı tercih ederler. Kısacası, :eqref:`eq_avg_ce_for_lm`'ün üssü:

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

Şaşkınlık, hangi simgeyi seçeceğimize karar verirken sahip olduğumuz gerçek seçeneklerin sayısının harmonik ortalaması olarak en iyi anlaşılabilir. Bize vakaların bir dizi bakalım:

* En iyi senaryoda, model her zaman etiket belirteci olasılığını 1 olarak mükemmel şekilde tahmin eder. Bu durumda modelin şaşkınlığı 1'dir.
* En kötü senaryoda, model her zaman etiket belirteci olasılığını 0 olarak öngörür. Bu durumda şaşkınlık pozitif sonsuzdur.
* Taban çizgisinde, model, sözcük dağarcığının tüm kullanılabilir belirteçleri üzerinde tekdüze bir dağılım öngörür. Bu durumda, şaşkınlık, kelime dağarcığının benzersiz belirteçlerinin sayısına eşittir. Aslında, diziyi herhangi bir sıkıştırma olmadan saklarsak, kodlamak için yapabileceğimiz en iyi şey bu olurdu. Bu nedenle, bu, herhangi bir yararlı modelin yenmesi gereken önemsiz bir üst sınır sağlar.

Aşağıdaki bölümlerde, karakter düzeyi dil modelleri için RNN'leri uygulayacağız ve bu modelleri değerlendirmek için şaşkınlığı kullanacağız.

## Özet

* Gizli durumlar için tekrarlayan hesaplama kullanan bir sinir ağı, tekrarlayan bir sinir ağı (RNN) olarak adlandırılır.
* Bir RNN'nin gizli durumu, dizinin geçmiş bilgilerini geçerli zaman adımına kadar yakalayabilir.
* Zaman adımlarının sayısı arttıkça RNN model parametrelerinin sayısı artmaz.
* Bir RNN kullanarak karakter düzeyinde dil modelleri oluşturabiliriz.
* Dil modellerinin kalitesini değerlendirmek için şaşkınlığı kullanabiliriz.

## Egzersizler

1. Bir metin dizisindeki bir sonraki karakteri tahmin etmek için bir RNN kullanırsak, herhangi bir çıktı için gerekli boyut nedir?
1. Neden RNN'ler metin dizisindeki önceki tüm belirteçlere dayalı bir zaman adımında bir belirteç koşullu olasılığını ifade edebilir?
1. Uzun bir sekansta geriye yayılırsa degradeye ne olur?
1. Bu bölümde açıklanan dil modeliyle ilgili bazı sorunlar nelerdir?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
