# Metin Ön 	İşleme
:label:`sec_text_preprocessing`

Dizi verileri için istatistiksel araçları ve tahmin zorluklarını inceledik ve değerlendirdik. Bu veriler birçok şekil alabilir. Özellikle, kitabın birçok bölümünde odaklanacağımız gibi, metinler dizi verilerinin en popüler örneklerindendir. Örneğin, bir makale basitçe bir sözcük dizisi veya hatta bir karakter dizisi olarak görülebilir. Dizi verileriyle gelecekteki deneylerimizi kolaylaştırmak amacıyla, bu bölümü metin için ortak ön işleme adımlarını açıklamaya adayacağız. Genellikle, aşağıdaki adımlar vardır:

1. Metni dizgi (string) olarak belleğe yükleyin.
1. Dizgileri andıçlara (token) ayırın (örn. kelimeler ve karakterler).
1. Bölünmüş andıçları sayısal indekslerle eşlemek için bir kelime tablosu oluşturun.
1. Metni sayısal indekslerin dizilerine dönüştürün, böylece modeller tarafından kolayca işlenebilirler.

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## Veri Kümesini Okuma

Başlamak için H. G. Wells'in [*Zaman Makinesi*](http://www.gutenberg.org/ebooks/35)'nden metin yüklüyoruz. Bu 30000 kelimenin biraz üzerinde oldukça küçük bir külliyat, ama göstermek istediğimiz şey için gayet iyi. Daha gerçekçi belge koleksiyonları milyarlarca kelime içerir. Aşağıdaki işlev, (**veri kümesini**) her satırın bir dizgi olduğu (**metin satırları listesine okur**). Basitlik için, burada noktalama işaretlerini ve büyük harfleri görmezden geliyoruz.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Zaman makinesi veri kümesini bir metin satırı listesine yükleyin."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## Andıçlama

Aşağıdaki `tokenize` işlevi, girdi olarak bir liste (`lines`) alır ve burada her eleman bir metin dizisidir (örneğin, bir metin satırı). [**Her metin dizisi bir andıç listesine bölünür**]. *Andıç* metindeki temel birimdir. Sonunda, her andıcın bir dizgi olduğu andıç listelerinin bir listesi döndürülür.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Metin satırlarını kelime veya karakter belirteçlerine ayırın."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## Kelime Dağarcığı

Andıcın dizgi tipi, sayısal girdiler alan modeller tarafından kullanılmak için elverişsizdir. Şimdi [**dizgi andıçlarını 0'dan başlayan sayısal indekslere eşlemek için, genellikle *kelime dağarcığı* olarak adlandırılan bir sözlük oluşturalım**]. Bunu yapmak için, önce eğitim kümesindeki tüm belgelerdeki benzersiz andıçları, yani bir *külliyat*ı, sayarız ve daha sonra her benzersiz andıca frekansına göre sayısal bir indeks atarız. Genellikle nadiren ortaya çıkan andıçlar karmaşıklığı azaltmak için kaldırılır. Külliyat içinde bulunmayan veya kaldırılan herhangi bir andıç, bilinmeyen özel bir andıç “&lt;unk&gt;” olarak eşleştirilir. İsteğe bağlı olarak, dolgu için “&lt;pad&gt;”, bir dizinin başlangıcını sunmak için “&lt;bos&gt;” ve bir dizinin sonu için “&lt;eos&gt;” gibi ayrılmış andıçların bir listesini ekleriz.

```{.python .input}
#@tab all
class Vocab:  #@save
    """Metin için kelime hazinesi."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # Frekanslara göre sırala
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # Bilinmeyen andıcın indeksi 0'dır
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Bilinmeyen andıç için dizin
        return 0

    @property
    def token_freqs(self):  # Bilinmeyen andıç için dizin
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Burada `tokens` bir 1B liste veya 2B listedir
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Bir belirteç listelerinin listesini belirteç listesine düzleştirin
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

Zaman makinesi veri kümesini külliyat olarak kullanarak [**bir kelime dağarcığı oluşturuyoruz**]. Daha sonra ilk birkaç sık andıcı dizinleriyle yazdırıyoruz.

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

Şimdi (**her metin satırını sayısal indekslerin bir listesine dönüştürebiliriz**).

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## Her Şeyi Bir Araya Koymak

Yukarıdaki işlevleri kullanarak, `corpus`'u (külliyat), andıç indekslerinin bir listesidir, ve `vocab`'ı döndüren [**`load_corpus_time_machine` işlevine her şeyi paketliyoruz**]. Burada yaptığımız değişiklikler şunlardır: (i) Metni daha sonraki bölümlerdeki eğitimi basitleştirmek için kelimelere değil, karakterlere andıçlıyoruz; (ii) `corpus` tek bir listedir, andıç listelerinin listesi değildir, çünkü zaman makinesi veri kümesindeki her metin satırı mutlaka bir cümle veya paragraf değildir.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Andıç indislerini ve zaman makinesi veri kümesinin kelime dağarcığını döndür."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Zaman makinesi veri kümesindeki her metin satırı mutlaka bir cümle veya 
    # paragraf olmadığından, tüm metin satırlarını tek bir listede düzleştirin
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Özet

* Metin, dizi verilerinin önemli bir biçimidir.
* Metni ön işlemek için genellikle metni andıçlara böleriz, andıç dizgilerini sayısal indekslere eşlemek için bir kelime dağarcığı oluştururuz ve modellerin işlenmesi için metin verilerini andıç dizinlerine dönüştürürüz.

## Alıştırmalar

1. Andıçlama önemli bir ön işleme adımıdır. Farklı diller için değişiktir. Metni andıçlamak için yaygın olarak kullanılan üç yöntem daha bulmaya çalışın.
1. Bu bölümün deneyinde, metni sözcüklere andıçlayın ve `Vocab` örneğinin `min_freq` argümanlarını değiştirin. Bu, kelime dağarcığı boyutunu nasıl etkiler?

[Tartışmalar](https://discuss.d2l.ai/t/115)
