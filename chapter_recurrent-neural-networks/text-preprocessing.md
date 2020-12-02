# Metin Önişleme
:label:`sec_text_preprocessing`

Dizi verileri için istatistiksel araçları ve tahmin zorluklarını inceledik ve değerlendirdik. Bu veriler birçok form alabilir. Özellikle, kitabın birçok bölümünde odaklanacağımız gibi, metin dizisi verilerinin en popüler örneklerinden biridir. Örneğin, bir makale basitçe bir sözcük dizisi veya hatta bir karakter dizisi olarak görülebilir. Dizi verileriyle gelecekteki deneylerimizi kolaylaştırmak için, bu bölümü metin için ortak ön işleme adımlarını açıklamak üzere adayacağız. Genellikle, aşağıdaki adımlar şunlardır:

1. Metni dize olarak belleğe yükleyin.
1. Dizeleri belirteçlere ayırın (örn. kelimeler ve karakterler).
1. Bölünmüş belirteçleri sayısal endekslerle eşlemek için bir kelime tablosu oluşturun.
1. Metni sayısal endekslerin dizilerine dönüştürün, böylece modeller tarafından kolayca manipüle edilebilir.

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

## Veri kümesini okuma

Başlamak için H. G. Wells'in [*The Time Machine*](http://www.gutenberg.org/ebooks/35)'ten metin yüklüyoruz. Bu biraz üzerinde 30000 kelime oldukça küçük bir korpus, ama göstermek istediğimiz şey için bu gayet iyi. Daha gerçekçi belge koleksiyonları milyarlarca kelime içerir. Aşağıdaki işlev, veri kümesini her satırın bir dize olduğu metin satırları listesine okur. Basitlik için, burada noktalama işaretlerini ve büyük harfleri görmezden geliyoruz.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## Tokenization

Aşağıdaki `tokenize` işlevi, giriş olarak bir liste (`lines`) alır ve burada her liste bir metin dizisi (örneğin, bir metin satırı). Her metin dizisi bir belirteç listesine bölünür. *token* metindeki temel birimdir. Sonunda, her belirteç bir dize olduğu belirteç listelerinin bir listesi döndürülür.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
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

## Kelime hazinesi

Simgenin dize tipi, sayısal girişleri alan modeller tarafından kullanılmak için elverişsizdir. Şimdi dize belirteçlerini 0'dan başlayan sayısal endekslere eşlemek için, genellikle *kelime dağarcığı* olarak adlandırılan bir sözlük oluşturalım. Bunu yapmak için, önce eğitim kümesindeki tüm belgelerdeki benzersiz belirteçleri, yani bir *corpus* sayarız ve daha sonra her benzersiz simgeye frekansına göre sayısal bir dizin atarız. Nadiren ortaya çıkan belirteçler karmaşıklığı azaltmak için genellikle kaldırılır. Korpus içinde bulunmayan veya kaldırılan herhangi bir belirteç, bilinmeyen özel bir belirteç “<unk>” olarak eşleştirilir. İsteğe bağlı olarak, “<pad>” dolgu için “”, bir dizinin <bos> başlangıcını sunmak için “” ve bir dizinin <eos> sonu için “” gibi ayrılmış belirteçlerin bir listesini ekleriz.

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
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

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

Zaman makinesi veri kümesini korpus olarak kullanarak bir kelime hazinesi oluşturuyoruz. Daha sonra ilk birkaç sık belirteci endeksleriyle yazdırıyoruz.

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

Şimdi her metin satırını sayısal endekslerin bir listesine dönüştürebiliriz.

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## Her şeyini bir araya koymak

Yukarıdaki işlevleri kullanarak, `corpus`, belirteç endekslerinin bir listesini ve `vocab`'yı döndüren `load_corpus_time_machine` işlevine her şeyi paketliyoruz. Burada yaptığımız değişiklikler şunlardır: i) Metni daha sonraki bölümlerdeki eğitimi basitleştirmek için kelimelere değil, karakterlere tokenize ediyoruz; ii) `corpus` tek bir listedir, belirteç listelerinin bir listesidir, çünkü zaman makinesi veri kümesindeki her metin satırı mutlaka bir cümle veya paragraf değildir.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Özet

* Metin, dizi verilerinin önemli bir şeklidir.
* Metni ön işlemek için genellikle metni belirteçlere böleriz, simge dizelerini sayısal endekslere eşlemek için bir kelime hazinesi oluştururuz ve modellerin işlenmesi için metin verilerini belirteç dizinlerine dönüştürürüz.

## Egzersizler

1. Tokenization önemli bir önişleme adıdır. Farklı diller için değişir. Metni tokenize etmek için yaygın olarak kullanılan üç yöntem daha bulmaya çalışın.
1. Bu bölümün denemesinde, metni sözcüklere tokenize edin ve `Vocab` örneğinin `min_freq` bağımsız değişkenlerini farklılık gösterin. Bu, kelime dağarcığı boyutunu nasıl etkiler?

[Discussions](https://discuss.d2l.ai/t/115)
