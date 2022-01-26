# Word Embeddings ön eğitim için veri kümesi
:label:`sec_word2vec_data`

Artık word2vec modellerinin teknik ayrıntılarını ve yaklaşık eğitim yöntemlerini bildiğimize göre, uygulamalarını inceleyelim. Özellikle, :numref:`sec_word2vec`'te atlama gram modelini ve :numref:`sec_approx_train`'te negatif örnekleme örneğini örnek olarak alacağız. Bu bölümde, kelime gömme modelini ön eğitim için veri kümesi ile başlıyoruz: verilerin orijinal formatı eğitim sırasında tekrarlanabilen minibüsler haline dönüştürülecektir.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
import os
import random
```

## Veri Kümesini Okuma

Burada kullandığımız veri kümesi [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42). Bu dergi, Wall Street Journal makalelerinden örneklenmiştir ve eğitim, doğrulama ve test setlerine bölünmüştür. Özgün biçimde, metin dosyasının her satırı boşluklarla ayrılmış sözcüklerin cümlesini temsil eder. Burada her kelimeyi bir jeton olarak ele alıyoruz.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

Eğitim setini okuduktan sonra, 10 kereden az görünen herhangi bir kelimenin "<unk>" belirteci ile değiştirildiği korpus için bir kelime dağarcığı oluşturuyoruz. Özgün veri kümesinin <unk>nadir (bilinmeyen) sözcükleri temsil eden "" belirteçleri de içerdiğini unutmayın.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Alt örnekleme

Metin verileri genellikle “the”, “a” ve “in” gibi yüksek frekanslı sözcüklere sahiptir: hatta çok büyük corpora'da milyarlarca kez ortaya çıkabilir. Ancak, bu kelimeler genellikle bağlam pencerelerinde birçok farklı kelimeyle birlikte ortaya çıkar ve çok az yararlı sinyaller sağlar. Örneğin, bir bağlam penceresinde “çip” kelimesini göz önünde bulundurun: Sezgisel olarak düşük frekanslı bir “intel” sözcüğüyle birlikte oluşması, eğitimde yüksek frekanslı bir kelime “a” ile birlikte oluşmaktan daha yararlıdır. Dahası, büyük miktarda (yüksek frekanslı) kelimelerle eğitim yavaştır. Böylece, kelime gömme modellerini eğitiyorken, yüksek frekanslı kelimeler*alt örneklenebilir* :cite:`Mikolov.Sutskever.Chen.ea.2013`. Özellikle, veri kümesinde $w_i$ dizinlenmiş her sözcük olasılık ile atılır 

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

burada $f(w_i)$, $w_i$ kelimelerinin sayısının veri kümelerindeki toplam kelime sayısına oranıdır ve $t$ sabit bir hiperparametre (denemede $10^{-4}$) 'dir. Sadece göreli frekans $f(w_i) > t$ (yüksek frekanslı) sözcük $w_i$ atılabilir ve kelimenin göreli frekansı ne kadar yüksek olursa, atılma olasılığı o kadar yüksektir.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Aşağıdaki kod parçacığı, alt örnekleme öncesi ve sonrasında cümle başına belirteç sayısının histogramını çizer. Beklendiği gibi, alt örnekleme, yüksek frekanslı kelimeleri bırakarak cümleleri önemli ölçüde kısaltır ve bu da eğitim hızlandırmasına yol açacaktır.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Bireysel belirteçler için, yüksek frekanslı “the” kelimesinin örnekleme oranı 1/20'den azdır.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Buna karşılık, düşük frekanslı kelimeler “katıl” tamamen tutulur.

```{.python .input}
#@tab all
compare_counts('join')
```

Alt örneklemeden sonra, belirteçleri korpus için endekslerine eşliyoruz.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Merkezi Kelimeler ve Bağlam Kelimeleri Ayıklamak

Aşağıdaki `get_centers_and_contexts` işlevi, `corpus`'ten tüm orta sözcükleri ve bağlam sözcüklerini ayıklar. Bağlam penceresi boyutu olarak rastgele olarak 1 ile `max_window_size` arasında bir tamsayıyı eşit olarak örnekler. Herhangi bir merkez sözcük için, uzaklığı örneklenen bağlam penceresi boyutunu aşmayan sözcükler, bağlam sözcükleridir.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Ardından, sırasıyla 7 ve 3 kelimeden oluşan iki cümle içeren yapay bir veri kümesi oluşturuyoruz. Maksimum bağlam penceresi boyutunun 2 olmasını sağlayın ve tüm orta sözcükleri ve bağlam sözcüklerini yazdırın.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

PTB veri kümesi üzerinde eğitim yaparken, maksimum bağlam penceresi boyutunu 5'e ayarladık. Aşağıdaki, veri kümelerindeki tüm orta sözcükleri ve bağlam sözcüklerini ayıklar.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Negatif Örnekleme

Yaklaşık eğitim için negatif örnekleme kullanıyoruz. Gürültülü kelimeleri önceden tanımlanmış bir dağıtıma göre örneklemek için aşağıdaki `RandomGenerator` sınıfını tanımlıyoruz; burada (muhtemelen normalleştirilmemiş) örnekleme dağılımının `sampling_weights` bağımsız değişkeni üzerinden geçirilir.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Örneğin, $P(X=1)=2/9, P(X=2)=3/9$ ve $P(X=3)=4/9$ örnekleme olasılıkları ile 1, 2 ve 3 indeksleri arasında 10 rasgele değişken $X$ çizebiliriz.

```{.python .input}
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Bir çift orta kelime ve bağlam kelimesi için, rastgele `K` (deneyde 5) gürültü kelimelerini örnekleriz. Word2vec kağıdındaki önerilere göre2vec, $w$ gürültü kelimesinin $P(w)$ örnekleme olasılığı, 0.75 :cite:`Mikolov.Sutskever.Chen.ea.2013`'ün gücüne yükseltilmiş sözlükteki göreli frekansına ayarlanır.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Minibatch'larda Eğitim Örnekleri Yükleme
:label:`subsec_word2vec-minibatch-loading`

Bağlam sözcükleri ve örneklenmiş gürültü sözcükleri ile birlikte tüm merkezi kelimeler çıkarıldıktan sonra, eğitim sırasında tekrarlı olarak yüklenebilecek örneklerin minibatch'larına dönüştürülecektir. 

Bir mini toplu işlemde, $i^\mathrm{th}$ örnek bir merkez sözcük ve $n_i$ bağlam sözcükleri ve $m_i$ parazit sözcükleri içerir. Değişen bağlam penceresi boyutları nedeniyle $n_i+m_i$ farklı $i$ için değişiklik gösterir. Böylece, her örnek için bağlam kelimelerini ve gürültü kelimelerini `contexts_negatives` değişkeninde birleştiririz ve birleştirme uzunluğu $\max_i n_i+m_i$'a (`max_len`) ulaşana kadar ped sıfırlarını birleştiririz. Kaybın hesaplanmasında yastıkları hariç tutmak için `masks` bir maske değişkeni tanımlıyoruz. `masks`'daki elemanlar ve `contexts_negatives`'teki `contexts_negatives`'teki elemanlar arasında bire bir yazışma vardır, burada `masks`'daki sıfırlar (aksi olanlar) `contexts_negatives`'teki pedlere karşılık gelir. 

Olumlu ve negatif örnekleri ayırt etmek için `contexts_negatives`'teki `contexts_negatives` içindeki parazit sözcüklerden `labels` değişkeni aracılığıyla bağlam sözcüklerini ayırırız. `masks`'e benzer şekilde, `labels`'daki elemanlar ve `contexts_negatives`'teki elemanlar arasında bire bir yazışma vardır ve `contexts_negatives`'teki `labels`'daki birlerin (aksi takdirde sıfırlar) `contexts_negatives`'teki bağlam sözcüklerine (olumlu örnekler) karşılık geldiği yerde. 

Yukarıdaki fikir aşağıdaki `batchify` işlevinde uygulanmaktadır. Girişi `data`, her öğe `center` merkez sözcüklerinden, `context` bağlam kelimelerinden ve `negative`'in gürültü kelimelerinden oluşan bir örnektir, toplu boyutuna eşit uzunlukta bir listedir. Bu işlev, maske değişkeni de dahil olmak üzere eğitim sırasında hesaplamalar için yüklenebilecek bir mini batch döndürür.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Bu işlevi iki örnekten oluşan bir mini batch kullanarak test edelim.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Her Şeyleri Bir Araya Getirmek

Son olarak, PTB veri kümesini okuyan ve veri yineleyicisini ve kelime dağarcığını döndüren `load_data_ptb` işlevini tanımlıyoruz.

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

Veri yineleyicisinin ilk mini batch yazdıralım.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Özet

* Yüksek frekanslı kelimeler eğitimde o kadar yararlı olmayabilir. Eğitimde hızlandırmak için onları alt deneyebiliriz.
* Hesaplama verimliliği için örnekler minibüsler halinde yükleriz. Paddingleri dolgulardan ayırmak için diğer değişkenleri ve negatif olanlardan olumlu örnekler tanımlayabiliriz.

## Egzersizler

1. Alt örnekleme kullanmıyorsa, bu bölümdeki kodun çalışma süresi nasıl değişir?
1. `RandomGenerator` sınıfı `k` rasgele örnekleme sonuçlarını önbelleğe alır. `k`'ü diğer değerlere ayarlayın ve veri yükleme hızını nasıl etkilediğini görün.
1. Bu bölümün kodundaki diğer hiperparametreler veri yükleme hızını etkileyebilir?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:
