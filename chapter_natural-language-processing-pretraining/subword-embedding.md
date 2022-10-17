# Alt Sözcük Gömme
:label:`sec_fasttext`

İngilizce'de, “yardım eder”, “yardım etti” ve “yardım ediyor” gibi sözcükler aynı “yardım” sözcüğünün bükülmüş biçimleridir. “Köpek” ve “köpekler” arasındaki ilişki “kedi” ve “kediler” arasındaki ilişkiyle aynıdır ve “erkek” ve “erkek arkadaş” arasındaki ilişki “kız” ile “kız arkadaş” arasındaki ilişkiyle aynıdır. Fransızca ve İspanyolca gibi diğer dillerde, birçok fiil 40'tan fazla bükülmüş forma sahipken, Fince'de bir isim için 15 kadar vaka olabilir. Dilbilimde, şekilbilim sözcük oluşumunu ve sözcük ilişkilerini çalışır. Bununla birlikte, sözcüklerin iç yapısı ne word2vec ne de Glove'de araştırıldı. 

## fastText Modeli

Kelimelerin word2vec içinde nasıl temsil edildiğini hatırlayın. Hem skip-gram modelinde hem de sürekli sözcük torbası modelinde, aynı sözcüğün farklı bükülmüş biçimleri, paylaşılan parametreler olmadan farklı vektörler tarafından doğrudan temsil edilir. Şekilbilimsel bilgileri kullanmak için, *fastText* modeli, bir alt sözcüğün $n$-gram karakteri olduğu bir *alt sözcük gömme* yaklaşımı önerdi :cite:`Bojanowski.Grave.Joulin.ea.2017`. FastText, sözcük düzeyinde vektör temsillerini öğrenmek yerine, her *merkez sözcüğün* alt sözcük vektörlerinin toplamıyla temsil edildiği alt sözcük düzeyinde skip-gram modeli olarak düşünülebilir. 

“Where” (nerede) sözcüğünü kullanarak fastText'te her merkez sözcük için alt sözcüklerin nasıl elde edileceğini gösterelim. İlk olarak, diğer alt sözcüklerden önekleri ve sonekleri ayırt etmek için sözcüğün başına ve sonuna “&lt;” ve “&gt;” özel karakterlerini ekleyin. Daha sonra $n$-gram karakterini sözcükten ayıklayın. Örneğin, $n=3$ olduğunda, 3 uzunluğundaki tüm alt sözcükleri, "&lt;wh", "whe", "her", "ere", "re&gt;" ve "&lt;where&gt;" özel alt sözcüğünü elde ederiz. 

fastText'te, herhangi bir sözcük $w$ için, $\mathcal{G}_w$ ile 3 ve 6 arasında uzunluğa sahip tüm alt sözcüklerin ve özel alt sözcüğünün birleşimini belirtiriz. Sözcük dağarcığı, tüm sözcüklerin alt sözcüklerinin birleşimidir. Sözlükte $\mathbf{z}_g$ $g$ alt kelimesinin vektörü olsun, skip-gram modelinde merkez kelime olarak $w$ kelimesi için $\mathbf{v}_w$ vektörü onun alt kelime vektörlerinin toplamıdır: 

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

FastText'in geri kalanı skip-gram modeliyle aynıdır. Skip-gram modeliyle karşılaştırıldığında, fastText içindeki sözcük dağarcığı daha büyüktür ve daha fazla model parametresi ile sonuçlanır. Ayrıca, bir sözcüğün temsilini hesaplamak için, tüm alt sözcük vektörlerinin toplanması ve daha yüksek hesaplama karmaşıklığına yol açar. Ancak benzer yapıya sahip sözcükler arasında alt sözcüklerden paylaşılan parametreler sayesinde, fastText'te nadir sözcükler ve hatta sözcük hazinesinde olmayan sözcükler daha iyi vektör temsilleri elde edebilir. 

## Sekizli Çifti Kodlaması
:label:`subsec_Byte_Pair_Encoding`

fastText'te, ayıklanan tüm alt sözcüklerin belirtilen uzunluklarda olması gerekir ($3$ - $6$ gibi), bu nedenle sözcük dağarcığı boyutu önceden tanımlanamaz. Sabit boyutlu bir sözcük dağarcığında değişken uzunlukta alt sözcüklere izin vermek için :cite:`Sennrich.Haddow.Birch.2015` alt sözcüklerini ayıklarken *sekizli (bayt) çifti kodlama* (BPE) adlı bir sıkıştırma algoritması uygulayabiliriz. 

Sekizli çifti kodlaması, bir sözcük içindeki ortak sembolleri keşfetmek için eğitim veri kümesinin istatistiksel analizini gerçekleştirir (örneğin, rastgele uzunlukta ardışık karakterler). 1 uzunluklu sembollerden başlayarak, sekizli çifti kodlaması yinelemeli olarak en sık ardışık sembol çiftini birleştirerek yeni daha uzun semboller üretir. Verimlilik için, sözcük sınırlarını aşan çiftlerin dikkate alınmadığını unutmayın. Sonunda, sözcükleri parçalara ayırmak için alt sözcükler gibi sembolleri kullanabiliriz. Sekizli çifti kodlaması ve farklı biçimleri, GPT-2 :cite:`Radford.Wu.Child.ea.2019` ve RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` gibi, popüler doğal dil işleme ön eğitim modellerinde girdi temsilleri için kullanılmıştır. Aşağıda, sekizli çifti kodlamanın nasıl çalıştığını göstereceğiz. 

İlk olarak, sembollerin sözcük dağarcığını tüm İngilizce küçük harfli karakterler, özel bir sözcük sonu sembolü `'_'` ve özel bir bilinmeyen sembolü `'[UNK]'` olarak ilkliyoruz.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Kelimelerin sınırlarını aşan sembol çiftlerini dikkate almadığımızdan, bir veri kümelerinde sözcükleri frekanslarına (oluşum sayısı) eşleyen bir sözlüğe ihtiyacımız var. Her sözcüğe `'_'` özel sembolünün eklendiğini unutmayın, böylece bir sözcük dizisini (örneğin, “a taller man”) bir çıktı simgesi dizisinden kolayca elde edebiliriz (örneğin, “a_ tall er_ man”). Birleştirme işlemini yalnızca tek karakter ve özel sembollerden oluşan bir sözcük dağarcığından ilklediğimizden dolayı, her sözcük içindeki ardışık karakter çiftleri arasına boşluk eklenir (`token_freqs` sözlüğünün anahtarları). Başka bir deyişle, boşluk, bir sözcük içindeki semboller arasındaki sınırlayıcıdır.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Bir sözcük içinde ardışık sembollerin en sık çiftini döndüren aşağıdaki `get_max_freq_pair` işlevini tanımlıyoruz, burada sözcükler `token_freqs` girdi sözlüğünün anahtarlarından gelir.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # `pairs`'in anahtarı, iki ardışık sembolden oluşan bir çokludur
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Maksimum değere sahip `pairs` anahtarı
```

Ardışık sembollerin sıklığına dayanan açgözlü bir yaklaşım olarak, sekizli çifti kodlaması, yeni semboller üretmek için en sık ardışık semboller çiftini birleştirmek için aşağıdaki `merge_symbols` işlevini kullanır.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Şimdi yinelemeli olarak sekizli çifti kodlama algoritmasını `token_freqs` sözlüğünün anahtarları üzerinden gerçekleştiriyoruz. İlk yinelemede, ardışık sembollerin en sık çifti `'t'` ve `'a'`'dir, böylece sekizli çifti kodlaması yeni bir sembol, `'ta'`, üretmek için bunları birleştirir. İkinci yinelemede sekizli çifti kodlama `'ta'` ve `'l'` ile başka bir yeni simge, `'tal'`, yaratılacak şekilde birleştirmeye devam eder.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Sekizli çifti kodlamasının 10 yinelemesinden sonra, bu `symbols` listesinin artık diğer sembollerden birleştirilmiş fazladan 10 tane sembol içerdiğini görebiliriz.

```{.python .input}
#@tab all
print(symbols)
```

`raw_token_freqs` sözlüğünün anahtarlarından belirtilen aynı veri kümesi için, veri kümelerindeki her sözcük artık sekizli çifti kodlama algoritmasının bir sonucu olarak “fast_”, “hızlı”, “er_”, “tall_” ve “uzun boylu” alt sözcükleriyle bölümlenir. Örneğin, “faster_” ve “taller_” sözcükleri sırasıyla “fast er_” ve “tall er_” olarak bölümlenir.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Sekizli çifti kodlamasının sonucunun kullanılan veri kümesine bağlı olduğunu unutmayın. Bir veri kümesinden öğrenilen alt sözcükleri başka bir veri kümesinin sözcüklerini bölümlere ayırmak için de kullanabiliriz. Açgözlü bir yaklaşım olarak, aşağıdaki `segment_BPE` işlevi, sözcükleri girdi bağımsız değişkeni `symbols`'ten mümkün olan en uzun alt sözcüklere ayırmaya çalışır.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Sembollerden mümkün olan en uzun alt kelimelere sahip bölüm belirteci
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

Aşağıda, yukarıda belirtilen veri kümesinden öğrenilen `symbols` listesindeki alt sözcükleri, başka bir veri kümesini temsil eden `tokens` bölümlerine ayırmak için kullanıyoruz.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Özet

* fastText modeli bir alt sözcük gömme yaklaşımı önerir. Word2vec içindeki skip-gram modeline dayanarak, bir merkez sözcüğünü alt sözcük vektörlerinin toplamı olarak bir temsil eder.
* Sekizli çifti kodlaması, bir sözcük içindeki ortak sembolleri keşfetmek için eğitim veri kümesinin istatistiksel analizini gerçekleştirir. Açgözlü bir yaklaşım olarak, sekizli çifti kodlaması yinelemeli olarak en sık ardışık sembol çiftini birleştirir.
* Alt sözcük gömme nadir sözcüklerin ve sözlük dışı sözcüklerin temsillerinin kalitesini artırabilir.

## Alıştırmalar

1. Örnek olarak, İngilizcede yaklaşık $3\times 10^8$ olası $6$-gram vardır. Çok fazla alt sözcük olduğunda sorun ne olur? Sorunu nasıl ele alabilirsiniz? İpucu: fastText makalesinin :cite:`Bojanowski.Grave.Joulin.ea.2017` 3.2. bölümünün sonuna bakın.
1. Sürekli sözcük torbası modeline dayalı bir alt sözcük gömme modeli nasıl tasarlanır?
1. $m$ büyüklüğünde bir sözcük dağarcığı elde etmek için, ilk sembol sözcük dağarcığı boyutu $n$ olduğunda kaç birleştirme işlemi gereklidir?
1. İfadeleri çıkarmak için sekizli çifti kodlama fikri nasıl genişletilebilir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/386)
:end_tab:
