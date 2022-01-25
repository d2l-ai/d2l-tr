# Kelime Benzerliği ve Benzerliği
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining`'te, küçük bir veri kümesi üzerinde bir word2vec modelini eğittik ve bir girdi sözcüğü için anlamsal olarak benzer kelimeleri bulmak için uyguladık. Uygulamada, büyük corpora üzerinde önceden eğitilmiş kelime vektörleri, daha sonra :numref:`chap_nlp_app`'te ele alınacak olan doğal dil işleme görevlerine uygulanabilir. Büyük corpora'dan önceden eğitilmiş kelime vektörlerinin semantiğini basit bir şekilde göstermek için, bunları benzerlik ve benzetme görevlerine uygulayalım.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Önceden Eğitimli Word Vektörlerini Yükleme

Aşağıda, [GloVe website](https://nlp.stanford.edu/projects/glove/)'ten indirilebilen 50, 100 ve 300 boyutlarında önceden eğitilmiş Eldiven gömme listeler. Önceden eğitilmiş fastText gömme parçaları birden çok dilde mevcuttur. Burada bir İngilizce sürümünü düşünün (300-boyutlu “wiki.en”) indirilebilir [fastText website](https://fasttext.cc/).

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Bu önceden eğitilmiş Eldiven ve fastText gömme yüklemeleri yüklemek için aşağıdaki `TokenEmbedding` sınıfını tanımlıyoruz.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Aşağıda 50 boyutlu Eldiven gömme yüklüyoruz (Vikipedi alt kümesi üzerinde önceden eğitilmiş). `TokenEmbedding` örneğini oluştururken, belirtilen gömme dosyasının henüz değilse indirilmesi gerekir.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Kelime dağarcığı boyutunu çıktılayın. Kelime bilgisi 400000 kelime (jeton) ve özel bir bilinmeyen belirteç içerir.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Kelime hazinesinde bir kelimenin indeksini alabiliriz ve tersi de geçerlidir.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Önceden Eğitimli Word Vektörlerini Uygulama

Yüklenen Eldiven vektörlerini kullanarak, anlamlarını aşağıdaki kelime benzerliği ve benzetme görevlerine uygulayarak göstereceğiz. 

### Kelime Benzerliği

:numref:`subsec_apply-word-embed`'e benzer şekilde, kelime vektörleri arasındaki kosinüs benzerliklerine dayanan bir giriş kelimesi için anlamsal olarak benzer kelimeleri bulmak için aşağıdaki `knn` ($k$ en yakın komşular) işlevini uyguluyoruz.

```{.python .input}
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Daha sonra, `TokenEmbedding` örneğinden önceden eğitilmiş sözcük vektörlerini kullanarak benzer kelimeleri ararız.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d`'teki önceden eğitilmiş kelime vektörlerinin kelime dağarcığı 400000 kelime ve özel bir bilinmeyen belirteç içerir. Giriş kelimesi ve bilinmeyen belirteci hariç, bu kelime arasında “çip” kelimesine en anlamsal olarak benzer üç kelimeyi bulalım.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Aşağıda “bebek” ve “güzel” benzer kelimeler çıktılar.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Kelime Analojisi

Benzer kelimeleri bulmanın yanı sıra, kelime vektörlerini kelime benzetme görevlerine de uygulayabiliriz. Örneğin, “erkek”: “kadın”። “oğul”: “kızı” bir kelime benzetme şeklidir: “erkek”, “oğul” olarak “kadın” demektir “kızı”. Özellikle, kelime benzetme tamamlama görevi şu şekilde tanımlanabilir: $a : b :: c : d$, $a$, $b$ ve $c$, $d$ bulmak için ilk üç kelime verilen bir kelime benzetme $a : b :: c : d$ için. $w$ kelimesinin vektörünü $\text{vec}(w)$ ile belirtin. Benzetmeyi tamamlamak için, vektörü $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$ sonucuna en çok benzeyen kelimeyi bulacağız.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Yüklü kelime vektörlerini kullanarak “erkek-kadın” benzetmeyi doğrulayalım.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Aşağıda bir “başkent-ülke” benzetme tamamlar: “Pekin”: “Çin”። “tokyo”: “japan”. Bu, önceden eğitilmiş kelime vektörlerinde semantik gösterir.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

“Kötü”: “en kötü”። “büyük”: “en büyük” gibi “sıfat-superlative sıfat” benzetme için, önceden eğitilmiş kelime vektörlerinin sözdizimsel bilgileri yakalayabileceğini görebiliriz.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Önceden eğitilmiş kelime vektörlerinde geçmiş zaman kavramını göstermek için, sözdizimini “şimdiki zaman geçmiş zaman” benzetmesini kullanarak test edebiliriz: “do”: “did”። “go”: “go”: “went”.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Özet

* Uygulamada, büyük corpora üzerinde önceden eğitilmiş kelime vektörleri doğal dil işleme görevlerine uygulanabilir.
* Önceden eğitilmiş kelime vektörleri kelime benzerlik ve benzetme görevlerine uygulanabilir.

## Egzersizler

1. `TokenEmbedding('wiki.en')` kullanarak fastText sonuçlarını test edin.
1. Kelime dağarcığı son derece büyük olduğunda, benzer kelimeleri nasıl bulabiliriz veya bir kelime benzerini daha hızlı tamamlayabiliriz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
