# Dil Modelleri ve Veri Kümesi
:label:`sec_language_model`

:numref:`sec_text_preprocessing` içinde, metin verilerini andıçlara nasıl eşleyeceğimizi görüyoruz; burada bu andıçlar, sözcükler veya karakterler gibi ayrık gözlemler dizisi olarak görülebiliyor. $T$ uzunluğunda bir metin dizisinde andıçların sırayla $x_1, x_2, \ldots, x_T$ olduğunu varsayalım. Daha sonra, metin dizisinde, $x_t$ ($1 \leq t \leq T$), $t$ adımındaki gözlem veya etiket olarak kabul edilebilir. Böyle bir metin dizisi göz önüne alındığında, *dil modelinin amacı* dizinin bileşik olasılığını tahmin etmektir

$$P(x_1, x_2, \ldots, x_T).$$

Dil modelleri inanılmaz derecede kullanışlıdır. Örneğin, ideal bir dil modeli, sadece bir seferde bir andıç çekerek tek başına doğal metin oluşturabilir: $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$. Daktilo kullanan maymunun aksine, böyle bir modelden çıkan tüm metinler doğal dil, örneğin İngilizce metin olarak geçer. Ayrıca, sadece metni önceki diyalog parçalarına koşullandırarak anlamlı bir diyalog oluşturmak için yeterli olacaktır. Açıkçası, böyle bir sistemi tasarlamaktan çok uzaktayız, çünkü sadece dilbilgisi olarak mantıklı içerik üretmek yerine metni *anlamak* gerekir.

Bununla birlikte, dil modelleri sınırlı biçimlerde bile mükemmel hizmet vermektedir. Örneğin, “konuşmayı tanımak” ve “konuşma tınmak” ifadeleri çok benzerdir. Bu konuşma tanımada belirsizliğe neden olabilir, ki ikinci çeviriyi tuhaf görüp reddeden bir dil modeli aracılığıyla kolayca çözülebilir. Aynı şekilde, bir belge özetleme algoritmasında “köpek adamı ısırdı” ifadesinin “adam köpeği ısırdı” ifadesinden çok daha sık ya da “Ben büyükanne yemek istiyorum” oldukça rahatsız edici bir ifade iken, “Ben yemek istiyorum, büyükanne” cümlesinin çok daha anlamlı olduğunu bilmek önemlidir.

## Dil Modeli Öğrenme

Bariz soru, bir belgeyi, hatta bir dizi andıcı nasıl modellememiz gerektiğidir. Metin verilerini kelime düzeyinde andıçladığımızı varsayalım. :numref:`sec_sequence` içindeki dizi modellerine uyguladığımız analize başvuruda bulunabiliriz. Temel olasılık kurallarını uygulayarak başlayalım:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Örneğin, dört kelime içeren bir metin dizisinin olasılığı şu şekilde verilecektir:

$$\begin{aligned}
P(\text{derin}, \text{öğrenme}, \text{çok}, \text{eğlencelidir}) 
=& P(\text{derin}) P(\text{öğrenme}  \mid  \text{derin}) P(\text{çok})  \mid  \text{derin}, \text{öğrenme}) \\
 & P(\text{eğlencelidir}  \mid  \text{derin}, \text{öğrenme}, \text{çok})
\end{aligned}
$$

Dil modelini hesaplamak için, kelimelerin olasılığını ve bir kelimenin önceki birkaç kelimeye koşullu olasılığını hesaplamamız gerekir. Bu olasılıklar esasen dil modeli parametreleridir.

Burada, eğitim veri kümelerinin tüm Vikipedi girdileri, [Gutenberg Projesi](https://en.wikipedia.org/wiki/Project_Gutenberg) ve Web'de yayınlanan tüm metinler gibi büyük bir metin külliyatı olduğunu varsayıyoruz. Kelimelerin olasılığı, eğitim veri kümesindeki belirli bir kelimenin göreceli kelime frekansından hesaplanabilir. Örneğin, $\hat{P}(\text{deep})$ tahmini, “derin” kelimesiyle başlayan herhangi bir cümlenin olasılığı olarak hesaplanabilir. Biraz daha az doğru bir yaklaşım, “derin” kelimesinin tüm oluşlarını saymak ve onu külliyat içindeki toplam kelime sayısına bölmek olacaktır. Bu, özellikle sık kullanılan kelimeler için oldukça iyi çalışır. Devam edersek, tahmin etmeyi deneyebiliriz:

$$\hat{P}(\text{öğrenme} \mid \text{derin}) = \frac{n(\text{derin, öğrenme})}{n(\text{derin})},$$

burada $n(x)$ ve $n(x, x')$, sırasıyla tekli ve ardışık kelime çiftlerinin oluşlarının sayısıdır. Ne yazık ki, bir kelime çiftinin olasılığını tahmin etmek biraz daha zordur, çünkü “derin öğrenme” oluşları çok daha az sıklıktadır. Özellikle, bazı olağandışı kelime birleşimleri için, doğru tahminler elde ederken yeterli oluş bulmak zor olabilir. İşler üç kelimelik birleşimler ve ötesi için daha da kötüsü bir hal alır. Veri kümemizde muhtemelen göremeyeceğimiz birçok makul üç kelimelik birleşim olacaktır. Bu tür sözcük birleşimlerine sıfır olmayan sayım atamak için bazı çözümler sağlamadığımız sürece, bunları bir dil modelinde kullanamayacağız. Veri kümesi küçükse veya kelimeler çok nadirse, bunlardan bir tanesini bile bulamayabiliriz.

Genel bir strateji, bir çeşit *Laplace düzleştirme*si uygulamaktır. Çözüm, tüm sayımlara küçük bir sabit eklemektir. $n$ ile eğitim kümesindeki toplam kelime sayısını ve $m$ ile benzersiz kelimelerin sayısını gösterelim. Bu çözüm, teklilerde yardımcı olur, örn.

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Burada $\epsilon_1,\epsilon_2$ ve $\epsilon_3$ hiper parametrelerdir. Örnek olarak $\epsilon_1$'yi alın: $\epsilon_1 = 0$ olduğunda, düzleştirme uygulanmaz; $\epsilon_1$ pozitif sonsuzluğa yaklaştığında, $\hat{P}(x)$ tekdüze $1/m$ olasılığına yaklaşır. Yukarıdakiler, diğer tekniklerin başarabileceklerinin oldukça ilkel bir türüdür :cite:`Wood.Gasthaus.Archambeau.ea.2011`.

Ne yazık ki, bunun gibi modeller aşağıdaki nedenlerden dolayı oldukça hızlı bir şekilde hantallaşır. İlk olarak, tüm sayımları saklamamız gerekir. İkincisi, kelimelerin anlamı tamamen görmezden gelinir. Örneğin, “kedi” ve “pisi” ilgili bağlamlarda ortaya çıkmalıdır. Bu tür modelleri ek bağlamlara ayarlamak oldukça zordur, oysa, derin öğrenme tabanlı dil modelleri bunu dikkate almak için çok uygundur. Son olarak, uzun kelime dizilerinin sıradışı olması neredeyse kesindir, bu nedenle sadece daha önce görülen kelime dizilerinin sıklığını sayan bir model orada kötü başarım göstermeye mahkumdur.

## Markov Modeller ve $n$-gram

Derin öğrenmeyi içeren çözümleri tartışmadan önce, daha fazla terime ve kavrama ihtiyacımız var. :numref:`sec_sequence` içindeki Markov Modelleri hakkındaki tartışmamızı hatırlayın. Bunu dil modellemesine uygulayalım. Eğer $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$ ise, diziler üzerinden bir dağılım birinci mertebeden Markov özelliğini karşılar. Daha yüksek mertebeler daha uzun bağımlılıklara karşılık gelir. Bu, bir diziyi modellemek için uygulayabileceğimiz birtakım yaklaşımlara yol açar:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Bir, iki ve üç değişken içeren olasılık formülleri genellikle sırasıyla *unigram*, *bigram* ve *trigram* modelleri olarak adlandırılır. Aşağıda, daha iyi modellerin nasıl tasarlanacağını öğreneceğiz.

## Doğal Dil İstatistikleri

Bunun gerçek veriler üzerinde nasıl çalıştığını görelim. :numref:`sec_text_preprocessing` içinde tanıtılan zaman makinesi veri kümesine dayanan bir kelime dağarcığı oluşturuyoruz ve en sık kullanılan 10 kelimeyi basıyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Her metin satırı mutlaka bir cümle veya paragraf olmadığı için tüm metin
# satırlarını bitiştiririz
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

Gördüğümüz gibi, (**en popüler kelimelere**) bakmak aslında oldukça sıkıcı. Genellikle (***duraklama kelimeleri***) olarak adlandırılır ve böylece filtrelenir. Yine de, hala anlam taşırlar ve biz de onları kullanacağız. Ayrıca, kelime frekansının oldukça hızlı bir şekilde sönümlendiği oldukça açıktır. $10$. en sık kullanılan kelime, en popüler olanının $1/5$'nden daha az yaygındır. Daha iyi bir fikir elde etmek için, [**kelime frekansının şeklini çizdiriyoruz**].

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Burada oldukça temel bir şey üzerindeyiz: Kelime frekansı iyi tanımlanmış bir şekilde hızla sönümleniyor. İlk birkaç kelime istisna olarak ele alındıktan sonra, kalan tüm kelimeler log-log figür üzerinde kabaca düz bir çizgi izler. Bu, kelimelerin en sık $i$. kelimesinin $n_i$ frekansının aşağıdaki gibi olduğunu belirten *Zipf yasası*nı tatmin ettiği anlamına gelir:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

ki o da aşağıdaki ifadeye eşdeğerdir

$$\log n_i = -\alpha \log i + c,$$

burada $\alpha$, dağılımı karakterize eden üstür ve $c$ bir sabittir. İstatistiklerine saymaya ve düzleştirmeye göre kelimeleri modellemek istiyorsak, zaten burada ara vermeliyiz. Sonuçta, nadir kelimeler olarak da bilinen kuyruk sıklığına önemli ölçüde saparak fazla değer vereceğiz. Peki ya [**diğer kelime birleşimleri, örneğin bigramlar, trigramlar**] ve ötesi? Bigram frekansının unigram frekansı ile aynı şekilde davranıp davranmadığını görelim.

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Burada dikkat çeken bir şey var. En sık görülen on kelime çiftinden dokuzunun üyeleri duraklama kelimelerinden oluşur ve yalnızca bir tanesi gerçek kitapla (“zaman”) ilgilidir. Ayrıca, trigram frekansının aynı şekilde davranıp davranmadığını görelim.

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Son olarak, bu üç model arasında [**andıç frekansını görselleştirmemize**] izin verin: Unigramlar, bigramlar ve trigramlar.

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Bu şekil birtakım nedenlerden dolayı oldukça heyecan verici. Birincisi, unigram kelimelerin ötesinde, kelime dizileri dizi uzunluğuna bağlı olarak :eqref:`eq_zipf_law` içinde $\alpha$'da daha küçük bir üs ile de olsa Zipf yasasını takip ediyor gibi görünmektedir. İkincisi, farklı $n$-gram sayısı o kadar büyük değildir. Bu bize dilde çok fazla yapı olduğuna dair umut veriyor. Üçüncü olarak, birçok $n$-gram çok nadiren ortaya çıkar, bu da Laplace düzleştirmeyi dil modellemesi için uygunsuz hale getirir. Bunun yerine, derin öğrenme tabanlı modeller kullanacağız.

## Uzun Dizi Verilerini Okuma

Dizi verileri doğası gereği dizili olduğundan, işleme konusunu ele almamız gerekiyor. Bunu :numref:`sec_sequence` içinde oldukça geçici bir şekilde yaptık. Diziler, modeller tarafından tek seferde işlenemeyecek kadar uzun olduğunda, bu tür dizileri okumak için bölmek isteyebiliriz. Şimdi genel stratejileri tanımlayalım. Modeli tanıtmadan önce, ağın önceden tanımlanmış uzunlukta, bir seferde $n$ zaman adımı mesela, bir minigrup dizisini işlediği bir dil modelini eğitmek için bir sinir ağı kullanacağımızı varsayalım. Şimdi asıl soru, [**özniteliklerin ve etiketlerin minigruplarının rastgele nasıl okunacağıdır.**]

Başlarken, bir metin dizisi keyfi uzunlukta olabileceğinden, *Zaman Makinesi* kitabının tamamı gibi, bu kadar uzun bir diziyi aynı sayıda zaman adımlı altdizilere parçalayabiliriz. Sinir ağımızı eğitirken, bu tür altdizilerden bir minigrup modele beslenecektir. Ağın bir seferde $n$ zaman adımlı bir altdizisini işlediğini varsayalım. :numref:`fig_timemachine_5gram`, orijinal bir metin dizisinden altdizi elde etmenin tüm farklı yollarını gösterir; burada $n=5$ ve her seferinde bir andıç bir karaktere karşılık gelir. Başlangıç pozisyonunu gösteren keyfi bir bağıl konum (offset) seçebileceğimizden oldukça özgür olduğumuzu unutmayın.

![Farklı bağıl konumlar, metni bölerken farklı altdizilere yol açar.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

Bu nedenle, :numref:`fig_timemachine_5gram` içinde gösterilenden hangisini seçmeliyiz? Aslında, hepsi eşit derecede iyidir. Ancak, sadece bir bağıl konum seçersek, ağımızı eğitmek için olası tüm altdizilerin sınırlı kapsamı vardır. Bu nedenle, hem *kapsama* hem de *rasgelelik* elde etmek için bir diziyi bölümlerken rastgele bir bağıl konum ile başlayabiliriz. Aşağıda, bunu hem *rastgele örnekleme* hem de *sıralı bölümleme* stratejileri için nasıl gerçekleştireceğimizi açıklıyoruz.

### Rastgele Örnekleme

(**Rastgele örneklemede, her örnek, orijinal uzun dizide keyfi olarak yakalanan bir altdizidir.**) Yineleme sırasında iki bitişik rasgele minigruptaki altdiziler mutlaka orijinal dizide bitişik olmak zorunda değildir. Dil modellemesi için hedef, şimdiye kadar gördüğümüz andıçlara dayanan bir sonraki andıcı tahmin etmektir, bu nedenle etiketler bir andıç kaydırılmış orijinal dizidir.

Aşağıdaki kod, her seferinde verilerden bir rasgele minigrup oluşturur. Burada, `batch_size` bağımsız değişkeni her minigruptaki altdizi örneklerinin sayısını belirtir ve `num_steps` her altdizideki zaman adımlarının önceden tanımlanmış sayısıdır.

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Rastgele örnekleme kullanarak küçük bir dizi alt dizi oluşturun."""
    # Bir diziyi bölmek için rastgele bir kayma ile başlayın 
    # (`num_steps - 1` dahil)
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Etiketleri hesaba katmamız gerektiğinden 1 çıkarın
    num_subseqs = (len(corpus) - 1) // num_steps
    # `num_steps` uzunluğundaki alt diziler için başlangıç indeksleri
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # Rastgele örneklemede, yineleme sırasında iki bitişik rastgele 
    # minigruptan gelen alt diziler, orijinal dizide mutlaka bitişik değildir
    random.shuffle(initial_indices)

    def data(pos):
        # `pos`'dan başlayarak `num_steps` uzunluğunda bir dizi döndür
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Burada, `initial_indices`, alt diziler için rastgele başlangıç
        #  dizinlerini içerir
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

Elimiz ile [**0'dan 34'e kadar bir dizi oluşturalım.**] Biz grup boyutunun ve zaman adımlarının sayısının sırasıyla 2 ve 5 olduğunu varsayalım. Bu, $\lfloor (35 - 1) / 5 \rfloor= 6$ öznitelik-etiket altdizi çiftleri üretebileceğimiz anlamına gelir. 2 minigrup boyutu ile sadece 3 minigrup elde ediyoruz.

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### Sıralı Bölümleme

Orijinal dizinin rastgele örneklemesine ek olarak, [**yineleme sırasında iki bitişik minigrubun altdizilerinin orijinal dizide bitişik olmasını da sağlayabiliriz.**] Bu strateji, minigruplar üzerinde yineleme yaparken bölünmüş altdizilerin sırasını korur, dolayısıyla sıralı bölümleme olarak adlandırılır.

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Sıralı bölümlemeyi kullanarak küçük bir alt dizi dizisi oluşturun."""
    # Bir diziyi bölmek için rastgele bir bağıl konum ile başlayın
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Sıralı bölümlemeyi kullanarak küçük bir alt dizi dizisi oluşturun."""
    # Bir diziyi bölmek için rastgele bir bağıl konum ile başlayın
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

Aynı ayarları kullanarak, sıralı bölümleme tarafından okunan her altdizi [**minigrubu için `X` özniteliklerini ve `Y` etiketleri yazdıracağız.**] Yineleme sırasında iki bitişik minigrubun altdizilerinin orijinal dizisi üzerinde gerçekten bitişik olduğunu unutmayın.

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

Şimdi yukarıdaki iki örnekleme işlevini bir sınıf ile sarmalıyoruz, böylece daha sonra bir veri yineleyici olarak kullanabiliriz.

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """Sıra verilerini yüklemek için bir yineleyici."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

[**Son olarak, hem veri yineleyiciyi hem de kelime dağarcığını döndüren bir `load_data_time_machine` işlevi tanımlarız**], böylece :numref:`sec_fashion_mnist` içinde tanımlanan `d2l.load_data_fashion_mnist` gibi `load_data` öneki ile diğer işlevlerle benzer şekilde kullanabiliriz.

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Zaman makinesi veri kümesinin yineleyicisini ve sözcük dağarcığını döndür."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## Özet

* Dil modelleri doğal dil işlemenin anahtarıdır.
* $n$-gram, bağımlılığı budayarak uzun dizilerle başa çıkmak için uygun bir model sağlar.
* Uzun diziler, çok nadiren olma veya neredeyse hiç olmama problemden muzdariptir.
* Zipf yasası sadece unigram değil, aynı zamanda diğer $n$-gramlar için de kelime dağılımını yönetir.
* Çok fazla yapı var, ancak Laplace düzleştirmesi yoluyla nadir kelime birleşimleri ile verimli bir şekilde başa çıkmak için yeterli sıklıkta kullanım yok.
* Uzun dizileri okumak için ana seçenekler rastgele örnekleme ve sıralı bölümlemedir. İkincisi, yineleme sırasında iki bitişik minigruptaki altdizilerin orijinal dizide de bitişik olmasını sağlar.

## Alıştırmalar

1. Eğitim veri kümesinde $100000$ kelime olduğunu varsayalım. Dört-gramın ne kadar kelime frekansı ve çok kelimeli bitişik frekansı depolaması gerekiyor?
1. Bir diyaloğu nasıl modellersiniz?
1. Unigramlar, bigramlar ve trigramlar için Zipf yasasının üssünü tahmin edin.
1. Uzun dizi verilerini okumak için başka hangi yöntemleri düşünebilirsiniz?
1. Uzun dizileri okumak için kullandığımız rastgele bağıl konumu düşünün.
    1. Rastgele bir bağıl konum olması neden iyi bir fikir?
    1. Belgedeki diziler üzerinde gerçekten mükemmel bir şekilde tekdüze bir dağılıma yol açar mı?
    1. İşleri daha tekdüze hale getirmek için ne yapmalısınız?
1. Eğer bir dizi örneğinin tam bir cümle olmasını istiyorsak, bu minigrup örneklemede ne tür bir sorun ortaya çıkarır? Bu sorunu nasıl çözebiliriz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1049)
:end_tab:
