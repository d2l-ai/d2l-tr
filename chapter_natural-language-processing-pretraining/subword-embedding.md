# Alt Kelime Gömülme
:label:`sec_fasttext`

İngilizce'de, “yardım”, “yardım” ve “yardım” gibi kelimeler aynı “yardım” kelimesinin bükülmüş biçimleridir. “Köpek” ve “köpekler” arasındaki ilişki “kedi” ve “kediler” arasındaki ilişkiyle aynıdır ve “erkek” ve “erkek arkadaş” arasındaki ilişki “kız” ile “kız” arasındaki ilişkiyle aynıdır. Fransızca ve İspanyolca gibi diğer dillerde, birçok fiil 40'tan fazla bükülmüş forma sahipken, Fince bir isim 15 vakaya kadar olabilir. Dilbilimde, morfoloji çalışmaları kelime oluşumu ve kelime ilişkileri. Bununla birlikte, kelimelerin iç yapısı ne word2vec ne de Glove'de keşfedildi. 

## fastText Modeli

Kelimelerin word2vec içinde nasıl temsil edildiğini hatırlayın. Hem atlama gram modelinde hem de sürekli sözcük torba modelinde, aynı kelimenin farklı bükülmüş biçimleri, paylaşılan parametreler olmadan farklı vektörler tarafından doğrudan temsil edilir. Morfolojik bilgileri kullanmak için, *fastText* modeli, bir alt sözcüğün $n$-gram 73229363614 karakteri olduğu bir *alt kelime embedding* yaklaşımı önerdi. FastText, sözcük düzeyinde vektör temsillerini öğrenmek yerine, her *center kelimesi* alt kelime vektörlerinin toplamıyla temsil edildiği alt kelime düzeyinde atlama gramı olarak düşünülebilir. 

“Nerede” kelimesini kullanarak fastText'te her orta kelime için alt kelimelerin nasıl elde edileceğini gösterelim. İlk olarak, <” and “> diğer alt kelimelerden önekleri ve sonekleri ayırt etmek için kelimenin başında ve sonuna “” özel karakterler ekleyin. Daha sonra karakteri $n$-gram sözcüğünden ayıklayın. Örneğin, $n=3$ olduğunda, 3: “<wh”, “whe”, “her”, “ere”, “re>” uzunluğundaki tüm alt kelimeleri ve "<where>” özel alt kelimesini elde ederiz. 

fastText'te, herhangi bir kelime $w$ için, $\mathcal{G}_w$ ile 3 ve 6 arasındaki tüm alt kelimelerin ve özel alt sözcüğünün birleşimini belirtin. Kelime, tüm kelimelerin alt kelimelerinin birleşimidir. $\mathbf{z}_g$'nin sözlükte $g$ alt sözcüğünün vektörü olması, atlama grafiği modelinde bir merkez kelime olarak $w$ sözcük $w$ için vektör, alt kelime vektörlerinin toplamıdır: 

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

FastText öğesinin geri kalanı atlama gram modeliyle aynıdır. Skip-gram modeliyle karşılaştırıldığında, fastText içindeki kelime dağarcığı daha büyüktür ve daha fazla model parametresi elde edilir. Ayrıca, bir kelimenin temsilini hesaplamak için, tüm alt kelime vektörlerinin toplanması ve daha yüksek hesaplama karmaşıklığına yol açması gerekir. Bununla birlikte, benzer yapılara sahip sözcükler arasındaki alt kelimelerden paylaşılan parametreler sayesinde, nadir kelimeler ve hatta kelime dışı kelimeler fastText'te daha iyi vektör temsilleri elde edebilir. 

## Bayt Çifti Kodlaması
:label:`subsec_Byte_Pair_Encoding`

fastText'te, ayıklanan tüm alt sözcüklerin belirtilen uzunluklarda olması gerekir ($3$ - $6$ gibi), bu nedenle sözcük dağarcığı boyutu önceden tanımlanamaz. Sabit boyutlu bir sözcük dağarcığında değişken uzunlukta alt sözcüklere izin vermek için :cite:`Sennrich.Haddow.Birch.2015` alt sözcüklerini ayıklamak için*bayt çifti kodlama* (BPE) adlı bir sıkıştırma algoritması uygulayabiliriz. 

Bayt çifti kodlaması, bir sözcük içindeki ortak sembolleri keşfetmek için eğitim veri kümesinin istatistiksel analizini gerçekleştirir (örneğin, rastgele uzunlukta ardışık karakterler). Uzunluk 1'in sembollerinden başlayarak, bayt çifti kodlaması yinelemeli olarak en sık ardışık sembol çiftini birleştirerek yeni daha uzun semboller üretir. Verimlilik için, kelime sınırlarını aşan çiftlerin dikkate alınmadığını unutmayın. Sonunda, kelimeleri parçalara ayırmak için alt kelimeler gibi sembolleri kullanabiliriz. Bayt çifti kodlaması ve varyantları, GPT-2 :cite:`Radford.Wu.Child.ea.2019` ve RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` gibi popüler doğal dil işleme ön eğitim modellerinde giriş temsilleri için kullanılmıştır. Aşağıda, bayt çifti kodlamanın nasıl çalıştığını göstereceğiz. 

İlk olarak, sembollerin kelime dağarcığını tüm İngilizce küçük harfli karakterler, özel bir kelime sonu sembolü `'_'` ve özel bir bilinmeyen sembol `'[UNK]'` olarak başlatıyoruz.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Kelimelerin sınırlarını aşan sembol çiftlerini dikkate almadığımızdan, bir veri kümelerinde kelimeleri frekanslarına (oluşum sayısı) eşleyen bir sözlüğe ihtiyacımız var. Her kelimeye `'_'` özel sembolünün eklendiğini unutmayın, böylece bir sözcük dizisini (örneğin, “daha uzun bir adam”) bir çıktı simgesi dizisinden kolayca kurtarabiliriz (örneğin, “a_ tall er_ man”). Birleştirme işlemini yalnızca tek karakter ve özel sembollerden oluşan bir kelime dağarcığından başlattığımızdan beri, her kelime içindeki ardışık karakter çiftleri arasına boşluk eklenir (sözlüğün `token_freqs` tuşları). Başka bir deyişle, boşluk, bir sözcük içindeki semboller arasındaki sınırlayıcıdır.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Bir kelime içinde ardışık sembollerin en sık çiftini döndüren aşağıdaki `get_max_freq_pair` işlevini tanımlıyoruz, kelimelerin giriş sözlüğünün tuşlarından geldiği `token_freqs`.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

Ardışık sembollerin sıklığına dayanan açgözlü bir yaklaşım olarak, bayt çifti kodlaması, yeni semboller üretmek için en sık ardışık sembol çiftini birleştirmek için aşağıdaki `merge_symbols` işlevini kullanır.

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

Şimdi yinelemeli olarak bayt çifti kodlama algoritmasını `token_freqs` sözlüğünün tuşları üzerinden gerçekleştiriyoruz. İlk yinelemede, ardışık sembollerin en sık çifti `'t'` ve `'a'`'dir, böylece bayt çifti kodlaması yeni bir sembol `'ta'` üretmek için bunları birleştirir. İkinci yinelemede bayt çifti kodlama `'ta'` ve `'l'` başka bir yeni simge `'tal'` ile sonuçlanacak şekilde birleştirmeye devam eder.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Bayt çifti kodlamasının 10 yinelemesinden sonra, bu liste `symbols`'ün artık diğer sembollerden birleştirilmiş 10 daha fazla sembol içerdiğini görebiliriz.

```{.python .input}
#@tab all
print(symbols)
```

`raw_token_freqs` sözlüğünün tuşlarında belirtilen aynı veri kümesi için, veri kümelerindeki her kelime artık bayt çifti kodlama algoritmasının bir sonucu olarak “fast_”, “hızlı”, “er_”, “tall_” ve “uzun boylu” alt sözcükleriyle bölümlenir. Örneğin, “faster_” ve “taller_” kelimeleri sırasıyla “fast er_” ve “tall er_” olarak parçalanır.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Bayt çifti kodlamasının sonucunun kullanılan veri kümesine bağlı olduğunu unutmayın. Bir veri kümesinden öğrenilen alt kelimeleri başka bir veri kümesinin sözcüklerini segmentlere ayırmak için de kullanabiliriz. Açgözlü bir yaklaşım olarak, aşağıdaki `segment_BPE` işlevi, sözcükleri giriş bağımsız değişkeni `symbols`'ten mümkün olan en uzun alt sözcüklere ayırmaya çalışır.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
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

Aşağıda, yukarıda belirtilen veri kümesinde öğrenilen `symbols` listesindeki alt sözcükleri başka bir veri kümesini temsil eden `tokens`'i parçalara ayırmak için kullanıyoruz.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Özet

* fastText modeli bir alt kelime gömme yaklaşımı önerir. Word2vec içindeki atlama gram modeline dayanarak, alt kelime vektörlerinin toplamı olarak bir merkez kelimeyi temsil eder.
* Bayt çifti kodlaması, bir sözcük içindeki ortak sembolleri keşfetmek için eğitim veri kümesinin istatistiksel analizini gerçekleştirir. Açgözlü bir yaklaşım olarak, bayt çifti kodlaması yinelemeli olarak en sık ardışık sembol çiftini birleştirir.
* Alt kelime gömme nadir sözcüklerin ve sözlük dışı sözcüklerin temsillerinin kalitesini artırabilir.

## Egzersizler

1. Örnek olarak, İngilizce olarak $3\times 10^8$ olası $6$-gram hakkında vardır. Çok fazla alt kelime olduğunda sorun nedir? Sorunu nasıl ele alabilirim? Hint: refer to the end of Section 3.2 of the fastText paper :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. Sürekli torba kelime modeline dayalı bir alt kelime gömme modeli nasıl tasarlanır?
1. $m$ büyüklüğünde bir kelime haznesi elde etmek için, ilk sembol kelime dağarcığı boyutu $n$ olduğunda kaç birleştirme işlemi gereklidir?
1. Nasıl ifadeleri ayıklamak için bayt çifti kodlama fikrini genişletmek için?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:
