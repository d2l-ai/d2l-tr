# Sözcük Gömme Ön Eğitimi İçin Veri Kümesi
:label:`sec_word2vec_data`

Artık word2vec modellerinin teknik ayrıntılarını ve yaklaşıklama eğitim yöntemlerini bildiğimize göre, uygulamalarını inceleyelim. Özellikle, :numref:`sec_word2vec` içinde skip-gram modelini ve :numref:`sec_approx_train` içinde negatif örneklemeyi örnek olarak alacağız. Bu bölümde, sözcük gömme modeli ön eğitimi için veri kümesi ile başlıyoruz: Verilerin orijinal biçimi eğitim sırasında yinelenebilen minigruplar haline dönüştürülecektir.

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

Burada kullandığımız veri kümesi [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42)'dir. Bu külliyat, Wall Street Journal makalelerinden örneklenmiştir ve eğitim, geçerleme ve test kümelerine bölünmüştür. Özgün biçimde, metin dosyasının her satırı boşluklarla ayrılmış bir sözcükler cümlesini temsil eder. Burada her sözcüğe bir belirteç gibi davranıyoruz.

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

Eğitim kümesini okuduktan sonra, 10 kereden az görünen herhangi bir sözcüğün "&lt;unk&gt;" belirteci ile değiştirildiği külliyat için bir sözcük dağarcığı oluşturuyoruz. Özgün veri kümesinin nadir (bilinmeyen) sözcükleri temsil eden "&lt;unk&gt;" belirteçleri de içerdiğini unutmayın.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Alt Örnekleme

Metin verileri genellikle "the", "a" ve "in" gibi yüksek frekanslı sözcüklere sahiptir: Hatta çok büyük külliyatta milyarlarca kez ortaya çıkabilirler. Ancak, bu sözcükler genellikle bağlam pencerelerinde birçok farklı sözcükle birlikte ortaya çıkar ve çok az yararlı sinyaller sağlar. Örneğin, bir bağlam penceresinde ("çip") "chip" sözcüğünü göz önünde bulundurun: Sezgisel olarak düşük frekanslı bir "intel" sözcüğüyle birlikte oluşması, eğitimde yüksek frekanslı bir sözcük "a" ile birlikte oluşmasından daha yararlıdır. Dahası, büyük miktarda (yüksek frekanslı) sözcüklerle eğitim yavaştır. Böylece, sözcük gömme modellerini eğitiyorken, yüksek frekanslı sözcükler *alt örneklenebilir* :cite:`Mikolov.Sutskever.Chen.ea.2013`. Özellikle, veri kümesindeki dizine alınmış her bir $w_i$ sözcüğü aşağıdaki olasılıkla atılacaktır.

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

burada $f(w_i)$, $w_i$ sözcüklerinin sayısının veri kümelerindeki toplam sözcük sayısına oranıdır ve $t$ sabit (deneyde $10^{-4}$) bir hiper parametredir. Sadece göreli frekans $f(w_i) > t$ (yüksek frekanslı) olursa $w_i$  sözcüğü atılabilir ve sözcüğün göreli frekansı ne kadar yüksek olursa, atılma olasılığı o kadar yüksektir.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Yüksek frekanslı sözcükleri alt örnekle."""
    # '<unk>' andıçlarını hariç tut
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Alt örnekleme sırasında `token` tutulursa True döndür
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Aşağıdaki kod parçacığı, alt örnekleme öncesi ve sonrasında cümle başına belirteç sayısının histogramını çizer. Beklendiği gibi, alt örnekleme, yüksek frekanslı sözcükleri düşürerek cümleleri önemli ölçüde kısaltır ve bu da eğitim hızlandırmasına yol açacaktır.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Bireysel belirteçler için, yüksek frekanslı "the" sözcüğünün örnekleme oranı 1/20'den azdır.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Buna karşılık, düşük frekanslı "join" sözcüğü tamamen tutulur.

```{.python .input}
#@tab all
compare_counts('join')
```

Alt örneklemeden sonra, belirteçleri külliyat için indekslerine eşliyoruz.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Merkezi Sözcükleri ve Bağlam Sözcüklerini Ayıklamak

Aşağıdaki `get_centers_and_contexts` işlevi, `corpus`'ten tüm merkez sözcükleri ve onların bağlam sözcüklerini ayıklar. Bağlam penceresi boyutu olarak rastgele olarak 1 ile `max_window_size` arasında bir tamsayıya eşit olarak örnekler. Herhangi bir merkez sözcük için, uzaklığı örneklenen bağlam penceresi boyutunu aşmayan sözcükler, bağlam sözcükleridir.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """skip-gramdaki merkez sözcüklerini ve bağlam sözcüklerini döndürür."""
    centers, contexts = [], []
    for line in corpus:
        # Bir "merkez sözcük-bağlam sözcüğü" çifti oluşturmak için 
        # her cümlede en az 2 kelime olması gerekir
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  #  `i` merkezli bağlam penceresi
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Merkez sözcüğü bağlam sözcüklerinden çıkar
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Ardından, sırasıyla 7 ve 3 sözcükten oluşan iki cümle içeren yapay bir veri kümesi oluşturuyoruz. Maksimum bağlam penceresi boyutunun 2 olmasını sağlayın ve tüm merkez sözcüklerini ve onların bağlam sözcüklerini yazdırın.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

PTB veri kümesi üzerinde eğitim yaparken, maksimum bağlam penceresi boyutunu 5'e ayarladık. Aşağıdaki, veri kümelerindeki tüm merkez sözcüklerini ve onların bağlam sözcüklerini ayıklar.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Negatif Örnekleme

Yaklaşıklama eğitim için negatif örnekleme kullanıyoruz. Gürültülü sözcükleri önceden tanımlanmış bir dağıtıma göre örneklemek için aşağıdaki `RandomGenerator` sınıfını tanımlıyoruz; burada (muhtemelen normalleştirilmemiş) örnekleme dağılımı `sampling_weights` argümanı üzerinden geçirilir.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Örnekleme ağırlıklarına göre {1, ..., n} arasından rastgele çek."""
    def __init__(self, sampling_weights):
        # Hariç tut 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # `k` rastgele örnekleme sonuçlarını önbelleğe al
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Örneğin, $P(X=1)=2/9, P(X=2)=3/9$ ve $P(X=3)=4/9$ örnekleme olasılıkları ile 1, 2 ve 3 indeksleri arasından 10 adet $X$ rasgele değişkenini çekebiliriz.

```{.python .input}
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Bir çift merkez sözcük ve bağlam sözcüğü için, rastgele `K` (deneyde 5) gürültü sözcüğü örnekleriz. Word2vec makalesindeki önerilere göre, $w$ gürültü sözcüğünün $P(w)$ örnekleme olasılığı, 0.75'in :cite:`Mikolov.Sutskever.Chen.ea.2013` gücüne yükseltilmiş olarak sözlükteki göreli frekansına ayarlanır.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Negatif örneklemede gürültü sözcüklerini döndür."""
    # Sözlükte 1, 2, ... (dizin 0 hariç tutulan bilinmeyen belirteçtir) 
    # olan kelimeler için örnekleme ağırlıkları
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Gürültü sözcükleri bağlam sözcükleri olamaz
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Minigruplarda Eğitim Örneklerini Yükleme
:label:`subsec_word2vec-minibatch-loading`

Bağlam sözcükleri ve örneklenmiş gürültü sözcükleri ile birlikte tüm merkezi sözcükler çıkarıldıktan sonra, eğitim sırasında yinelemeli olarak yüklenebilecek örneklerin minigruplarına dönüştürülecektir. 

Bir minigrupta, $i.$ örnek bir merkez sözcük ve onun $n_i$ bağlam sözcükleri ve $m_i$ gürültü sözcükleri içerir. Değişen bağlam penceresi boyutları nedeniyle $n_i+m_i$ farklı $i$'ler için değişiklik gösterir. Böylece, her örnek için bağlam sözcüklerini ve gürültü sözcüklerini `contexts_negatives` değişkeninde bitiştiririz ve bitiştirme uzunluğu $\max_i n_i+m_i$'a (`max_len`) ulaşana kadar sıfırlarla dolgularız. Kaybın hesaplanmasında dolguları hariç tutmak için bir maske değişkeni, `masks`, tanımlıyoruz. `masks`'taki elemanlar ve `contexts_negatives`'teki elemanlar arasında bire bir karşılık vardır, burada `masks`'daki sıfırlar (aksi takdirde birler) `contexts_negatives`'teki dolgulara karşılık gelir. 

Pozitif ve negatif örnekleri ayırt etmek için `contexts_negatives` içindeki gürültü sözcüklerinden `labels` değişkeni aracılığıyla bağlam sözcüklerini ayırırız. `masks`'e benzer şekilde, `labels`'daki elemanlar ve `contexts_negatives`'teki elemanlar arasında bire bir karşılık vardır ve `labels`'daki birler (aksi takdirde sıfırlar) `contexts_negatives`'teki bağlam sözcüklerine (olumlu örneklere) karşılık gelir. 

Yukarıdaki fikir aşağıdaki `batchify` işlevinde uygulanmaktadır. Girdi `data` toplu boyutuna eşit uzunlukta bir listedir; her öğesi `center` merkez sözcüklerinden, `context` bağlam sözcüklerinden ve `negative`  gürültü sözcüklerinden oluşan bir örnektir. Bu işlev, maske değişkeni de dahil olmak üzere eğitim sırasında hesaplamalar için yüklenebilecek bir minigrup döndürür.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Negatif örnekleme ile skip-gram için bir minigrup örnek döndür."""
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

Bu işlevi iki örnekten oluşan bir minigrup kullanarak test edelim.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Her Şeyi Bir Araya Getirmek

Son olarak, PTB veri kümesini okuyan ve veri yineleyicisini ve sözcük dağarcığını döndüren `load_data_ptb` işlevini tanımlıyoruz.

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """PTB veri kümesini indirin ve ardından belleğe yükleyin."""
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

Veri yineleyicisinin ilk minigrubunu yazdıralım.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Özet

* Yüksek frekanslı sözcükler eğitimde o kadar yararlı olmayabilir. Eğitimde hızlanmak için onları alt örnekleyebiliriz.
* Hesaplama verimliliği için örnekleri minigruplar halinde yükleriz. Dolguları dolgu olmayanlardan ve olumlu örnekleri olumsuz olanlardan ayırt etmek için başka değişkenler tanımlayabiliriz.

## Alıştırmalar

1. Alt örnekleme kullanmıyorsa, bu bölümdeki kodun çalışma süresi nasıl değişir?
1. `RandomGenerator` sınıfı `k` rasgele örnekleme sonuçlarını önbelleğe alır. `k`'yi diğer değerlere ayarlayın ve veri yükleme hızını nasıl etkilediğini görün.
1. Bu bölümün kodundaki hangi diğer hiper parametreler veri yükleme hızını etkileyebilir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1330)
:end_tab:
