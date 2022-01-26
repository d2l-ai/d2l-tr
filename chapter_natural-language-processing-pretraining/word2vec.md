# Sözcük Gömme (word2vec)
:label:`sec_word2vec`

Doğal dil anlamları ifade etmek için kullanılan karmaşık bir sistemdir. Bu sistemde, kelimeler anlamın temel birimidir. Adından da anlaşılacağı gibi,
*sözcük vektörleri*, sözcükleri temsil etmek için kullanılan vektörlerdir
ve ayrıca özellik vektörleri veya sözcüklerin temsilleri olarak da düşünülebilir. Kelimeleri gerçek vektörlere eşleme tekniği*kelime gömme* olarak adlandırılır. Son yıllarda, kelime gömme yavaş yavaş doğal dil işleme temel bilgi haline gelmiştir. 

## One-Hot Vektörler Kötü Bir Seçimdir

:numref:`sec_rnn_scratch`'te kelimeleri temsil etmek için tek sıcak vektörler kullandık (karakterler kelimelerdir). Sözlükteki farklı kelimelerin sayısının (sözlük boyutu) $N$ olduğunu ve her kelimenin $0$ ila $N−1$ arasında farklı bir tamsayıya (dizin) karşılık geldiğini varsayalım. İndeksi $i$ olan herhangi bir sözcük için tek sıcak vektör temsilini elde etmek için, tüm 0s ile bir uzunluk-$N$ vektör oluşturuyoruz ve öğeyi $i$ için 1 konumuna ayarlıyoruz. Bu şekilde, her kelime $N$ uzunluğunda bir vektör olarak temsil edilir ve doğrudan sinir ağları tarafından kullanılabilir. 

Tek sıcak kelime vektörlerinin oluşturulması kolay olsa da, genellikle iyi bir seçim değildir. Ana nedeni, tek sıcak kelime vektörlerinin, sık sık kullandığımız *kosinüs benzerliği* gibi farklı kelimeler arasındaki benzerliği doğru bir şekilde ifade edememesidir. Vektörler $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ için kosinüs benzerliği aralarındaki açının kosinüsüdür: 

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

İki farklı kelimenin tek sıcak vektörleri arasındaki kosinüs benzerliği 0 olduğundan, tek sıcak vektörler kelimeler arasındaki benzerlikleri kodlayamaz. 

## Kendiliğinden süpervisli word2vec

Yukarıdaki sorunu gidermek için [word2vec](https://code.google.com/archive/p/word2vec/) aracı önerildi. Her kelimeyi sabit uzunlukta bir vektöre eşler ve bu vektörler farklı kelimeler arasındaki benzerlik ve benzerlik ilişkisini daha iyi ifade edebilir. Word2vec aracı iki model içerir, yani *skip-gram* :cite:`Mikolov.Sutskever.Chen.ea.2013` ve*sürekli kelime torbası* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`. Anlamsal anlamlı temsiller için, eğitimleri, bazı kelimeleri corpora'da çevreleyen kelimeleri kullanarak tahmin etmek olarak görülebilecek koşullu olasılıklara dayanır. Denetim, etiketsiz verilerden geldiğinden, hem atlama gram hem de sürekli kelime çantası kendi kendini denetleyen modellerdir. 

Aşağıda, bu iki modeli ve eğitim yöntemlerini tanıtacağız. 

## Skip-Gram Modeli
:label:`subsec_skip-gram`

*skip-gram* modeli, bir kelimenin etrafındaki sözcükleri bir metin dizisinde oluşturmak için kullanılabileceğini varsayar. Örnek olarak “the”, “adam”, “seviyor”, “onun”, “oğlu” metin sırasını alın. *center kelime* olarak “sevgiler” i seçelim ve içerik penceresi boyutunu 2'ye ayarlayalım. :numref:`fig_skip_gram`'te gösterildiği gibi, “seviyor” merkez kelimesi göz önüne alındığında, atlama gramı modeli, *bağlam kelimelerini* üretmek için koşullu olasılığı göz önünde bulundurur: “the”, “man”, “onun” ve “oğul”, bunlar orta kelimeden en fazla 2 kelimeden uzak değildir: 

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

Bağlam sözcüklerinin bağımsız olarak orta kelime (yani koşullu bağımsızlık) verilen oluşturulduğunu varsayalım. Bu durumda, yukarıdaki koşullu olasılık olarak yeniden yazılabilir 

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![The skip-gram model considers the conditional probability of generating the surrounding context words given a center word.](../img/skip-gram.svg)
:label:`fig_skip_gram`

Skip-gram modelinde, her kelimenin koşullu olasılıkları hesaplamak için iki adet $d$ boyut-vektör gösterimi vardır. Daha somut bir şekilde, sözlükte indeks $i$ olan herhangi bir kelime için, sırasıyla $\mathbf{v}_i\in\mathbb{R}^d$ ve $\mathbf{u}_i\in\mathbb{R}^d$'e göre iki vektörünü, *center* kelimesi ve bir *context* kelimesi olarak kullanıldığında ifade eder. $c$ (sözlükte indeks $c$ ile) orta kelime verilen $w_o$ (sözlükte indeks $o$ ile) herhangi bir bağlam sözcüğü oluşturma koşullu olasılık vektör nokta ürünlerinde bir softmax işlemi ile modellenebilir: 

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

burada kelime indeksi $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ ayarlanır. $T$ uzunluğunda bir metin dizisi göz önüne alındığında, burada adım $t$ kelimesi $w^{(t)}$ olarak gösterilir. Bağlam sözcükleri bağımsız olarak herhangi bir merkez sözcük verilen oluşturulur varsayalım. Bağlam penceresi boyutu $m$ için atlama grafiği modelinin olasılık işlevi, herhangi bir merkez sözcük verilen tüm bağlam sözcükleri oluşturma olasılığıdır: 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

burada $1$ veya $T$ daha büyük olan herhangi bir zaman adımı atlanabilir. 

### Eğitim

Skip-gram modeli parametreleri, kelime dağarcığındaki her kelime için merkez sözcük vektörü ve bağlam sözcük vektörüdür. Eğitimde, olasılık fonksiyonunu (yani maksimum olasılık tahmini) maksimize ederek model parametrelerini öğreniriz. Bu, aşağıdaki kayıp işlevini en aza indirmeye eşdeğerdir: 

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

Kaybı en aza indirmek için stokastik degrade iniş kullanırken, her yinelemede rastgele model parametrelerini güncellemek için bu sonrakinin (stokastik) degrade hesaplamak için daha kısa bir sonrasını örnekleyebiliriz. Bu (stokastik) degradeyi hesaplamak için, merkez sözcük vektörü ve bağlam sözcük vektörüne göre günlük koşullu olasılığının degradelerini elde etmemiz gerekir. Genel olarak, :eqref:`eq_skip-gram-softmax`'e göre $w_c$ ve $w_o$ bağlam sözcüğünün herhangi bir çiftini içeren günlük koşullu olasılık 

$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

Farklılaşma sayesinde, merkez kelime vektörü $\mathbf{v}_c$ ile ilgili olarak gradyanı elde edebiliriz 

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`

:eqref:`eq_skip-gram-grad`'teki hesaplamanın, sözlükteki tüm kelimelerin orta kelime olarak $w_c$ ile koşullu olasılıklarını gerektirdiğini unutmayın. Diğer sözcük vektörlerinin degradeleri aynı şekilde elde edilebilir. 

Eğitimden sonra, sözlükte indeks $i$ olan herhangi bir kelime için, $\mathbf{v}_i$ (orta kelime olarak) ve $\mathbf{u}_i$ (bağlam kelimesi olarak) hem de $\mathbf{u}_i$ kelimesini elde ederiz. Doğal dil işleme uygulamalarında, atlama grafiği modelinin orta kelime vektörleri genellikle kelime temsilleri olarak kullanılır. 

## Sürekli Kelime Çantası (CBOW) Modeli

*Sürekli kelime torbası* (CBOW) modeli, atlama gram modeline benzer. Skip-gram modelinden en büyük fark, kelimelerin sürekli torba modeli, bir merkez sözcüğün metin dizisindeki çevreleyen bağlam sözcüklerine dayanarak oluşturulduğunu varsaymasıdır. Örneğin, aynı metin dizisi “the”, “adam”, “seviyor”, “onun” ve “oğul”, orta kelime ve bağlam penceresi boyutu 2 olarak “seviyor” ile, kelime sürekli çanta modeli “seviyor” bağlam kelimeleri dayalı “seviyor” orta kelime üretme koşullu olasılığını dikkate alır “the”, “adam”, “onun” ve “oğlu “(:numref:`fig_cbow` gösterildiği gibi), hangi 

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![The continuous bag of words model considers the conditional probability of generating the center word given its surrounding context words.](../img/cbow.svg)
:eqlabel:`fig_cbow`

Kelimelerin sürekli çanta modelinde birden fazla bağlam kelimeleri bulunduğundan, bu bağlam kelime vektörleri koşullu olasılığın hesaplanmasında ortalamalanır. Özellikle, sözlükte indeks $i$ olan herhangi bir kelime için, sırasıyla $\mathbf{v}_i\in\mathbb{R}^d$ ve $\mathbf{u}_i\in\mathbb{R}^d$ ile iki vektörünü, *context* kelimesi ve bir *center* sözcük olarak kullanıldığında (anlamlar atlama gram modelinde değiştirilir). Çevredeki bağlam kelimeleri $w_{o_1}, \ldots, w_{o_{2m}}$ (sözlükte indeks $c$ ile) $w_{o_1}, \ldots, w_{o_{2m}}$ (dizin $o_1, \ldots, o_{2m}$ sözlükte indeks) verilen herhangi bir merkez sözcük üretme koşullu olasılık 

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`

Kısalık için $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ ve $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$'e izin verin. Daha sonra :eqref:`fig_cbow-full` olarak basitleştirilebilir 

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

$T$ uzunluğunda bir metin dizisi göz önüne alındığında, burada adım $t$ kelimesi $w^{(t)}$ olarak gösterilir. Bağlam penceresi boyutu $m$ için, sürekli kelime torbasının olasılık fonksiyonu, bağlam kelimeleri verilen tüm merkez kelimeleri oluşturma olasılığıdır: 

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### Eğitim

Kelime modellerinin sürekli çanta eğitimi atlama gram modellerini eğitmekle hemen hemen aynıdır. Kelimelerin sürekli torba modelinin maksimum olasılık tahmini aşağıdaki kayıp fonksiyonunu en aza indirmeye eşdeğerdir: 

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Dikkat et 

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

Farklılaşma sayesinde, herhangi bir bağlam kelime vektörü ile ilgili gradyanı elde edebilirsiniz $\mathbf{v}_{o_i}$ ($i = 1, \ldots, 2m$) olarak 

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`

Diğer sözcük vektörlerinin degradeleri aynı şekilde elde edilebilir. Skip-gram modelinin aksine, kelimelerin sürekli çanta modeli genellikle sözcük temsilleri olarak bağlam sözcük vektörlerini kullanır. 

## Özet

* Kelime vektörleri sözcükleri temsil etmek için kullanılan vektörlerdir ve ayrıca özellik vektörleri veya sözcüklerin temsilleri olarak da düşünülebilir. Kelimeleri gerçek vektörlere eşleme tekniğine kelime gömme denir.
* Word2vec aracı hem atlama gram hem de sürekli kelime modellerini içerir.
* Skip-gram modeli, bir kelimenin etrafındaki sözcükleri bir metin dizisinde üretmek için kullanılabileceğini varsayar; Sürekli kelime çantası modeli, bir merkez sözcüğün çevreleyen bağlam sözcüklerine dayanarak oluşturulduğunu varsayar.

## Egzersizler

1. Her degradeyi hesaplamak için hesaplama karmaşıklığı nedir? Sözlük boyutu büyükse sorun ne olabilir?
1. İngilizce bazı sabit ifadeler “new york” gibi birden fazla kelimeden oluşur. Nasıl kendi kelime vektörleri eğitmek için? Hint: see Section 4 in the word2vec paper :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. Skip-gram modelini örnek olarak alarak word2vec tasarımını düşünelim. Skip-gram modelindeki iki kelime vektörünün nokta çarpımıyla kosinüs benzerliği arasındaki ilişki nedir? Benzer semantiği olan bir çift kelime için, kelime vektörlerinin kosinüs benzerliği (atlama gram modeli tarafından eğitilmiştir) neden yüksek olabilir?

[Discussions](https://discuss.d2l.ai/t/381)
