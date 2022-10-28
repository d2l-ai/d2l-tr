# Sözcük Benzerliği ve Benzeşim
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining` içinde, küçük bir veri kümesi üzerinde bir word2vec modelini eğittik ve bunu bir girdi sözcüğü için anlamsal olarak benzer kelimeleri bulmak için uyguladık. Uygulamada, büyük külliyatlar üzerinde önceden eğitilmiş kelime vektörleri, daha sonra :numref:`chap_nlp_app` içinde ele alınacak olan doğal dil işleme görevlerine uygulanabilir. Büyük külliyatlardan önceden eğitilmiş kelime vektörlerinin anlamını basit bir şekilde göstermek için, bunları sözcük benzerliğü ve benzeşimi görevlerine uygulayalım.

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

## Önceden Eğitilmiş Sözcük Vektörlerini Yükleme

Aşağıda, [GloVe web sitesinden](https://nlp.stanford.edu/projects/glove/) indirilebilen 50, 100 ve 300 boyutlarında önceden eğitilmiş GloVe gömme listeleri verilmektedir. Önceden eğitilmiş fastText gömmeleri birden çok dilde mevcuttur. Burada indirilebilen bir İngilizce sürümünü düşünelim (300-boyutlu "wiki.en") [fastText websitesi](https://fasttext.cc/).

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

Bu önceden eğitilmiş GloVe ve fastText gömmelerini yüklemek için aşağıdaki `TokenEmbedding` sınıfını tanımlıyoruz.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Andıç Gömme."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe websitesi: https://nlp.stanford.edu/projects/glove/
        # fastText websitesi: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # fastText'teki en üst satır gibi başlık bilgilerini atla
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

Aşağıda 50 boyutlu GloVe gömmelerini yüklüyoruz (Wikipedia alt kümesi üzerinde önceden eğitilmiş). `TokenEmbedding` örneğini oluştururken, belirtilen gömme dosyasının henüz yoksa indirilmesi gerekir.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Sözcük dağarcığı boyutunu çıktılayın. Sözcük bilgisi 400000 kelime (belirteç) ve özel bir bilinmeyen belirteci içerir.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Sözcük hazinesinde bir kelimenin indeksini alabiliriz ve tersi de geçerlidir.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Önceden Eğitilmiş Sözcük Vektörlerini Uygulama

Yüklenen GloVe vektörlerini kullanarak, anlamlarını aşağıdak sözcük benzerliği ve benzeşim görevlerine uygulayarak göstereceğiz. 

### Sözcük Benzerliği

:numref:`subsec_apply-word-embed` içindekine benzer şekilde, sözcük vektörleri arasındaki kosinüs benzerliklerine dayanan bir girdi sözcüğü için anlamsal olarak benzer kelimeleri bulmak için aşağıdaki `knn` ($k$ en yakın komşu) işlevini uygularız.

```{.python .input}
def knn(W, x, k):
    # Sayısal kararlılık için 1e-9 ekleyin
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Sayısal kararlılık için 1e-9 ekleyin
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
    for i, c in zip(topk[1:], cos[1:]):  # Girdi sözcüğünü hariç tut
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d`'teki önceden eğitilmiş kelime vektörlerinin kelime dağarcığı 400000 kelime ve özel bir bilinmeyen belirteç içerir. Girdi kelimesi ve bilinmeyen belirteci hariç, bu kelime arasında "chip (çip))" kelimesine en anlamsal olarak benzer üç kelimeyi bulalım.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Aşağıda "baby (bebek)" ve " beautiful (güzel)" kelimelerinin benzerleri çıktılanır.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Kelime Benzeşimi

Benzer kelimeleri bulmanın yanı sıra, kelime vektörlerini kelime benzetme görevlerine de uygulayabiliriz. Örneğin, "erkek":"kadın"::"oğul":"kız" bir kelime benzetme şeklidir: "erkek" ile "kadın", "kız" ile "oğul" benzerdir. Daha özel olarak, kelime benzeşim tamamlama görevi şu şekilde tanımlanabilir: Bir kelime benzeşimi $a : b :: c : d$ için, ilk üç kelime olan $a$, $b$ ve $c$ verildiğinde, $d$ bulunsun. $w$ kelimesinin vektörünü $\text{vec}(w)$ ile belirtilsin. Benzetmeyi tamamlamak için, vektörü $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$ sonucuna en çok benzeyen kelimeyi bulacağız.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Bilinmeyen kelimeleri çıkar
```

Yüklü kelime vektörlerini kullanarak "erkek-kadın" benzeşimini doğrulayalım.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Aşağıdaki bir "başkent-ülke" benzeşimini tamamlar: "Pekin":"Çin"::"Tokyo":"Japonya". Bu, önceden eğitilmiş kelime vektörlerindeki anlamı gösterir.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

"Kötü":"en kötü"::"büyük":"en büyük" gibi "sıfat-üstün sıfat" benzetme için, önceden eğitilmiş kelime vektörlerinin sözdizimsel bilgileri yakalayabileceğini görebiliriz.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Önceden eğitilmiş kelime vektörlerinde geçmiş zaman kavramını göstermek için, sözdizimini “şimdiki zaman - geçmiş zaman” benzetmesini kullanarak test edebiliriz: "yap (do)":“yaptım (did)"::"go (git)":"went (gittim)".

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Özet

* Uygulamada, büyük külliyatlar üzerinde önceden eğitilmiş kelime vektörleri doğal dil işleme görevlerine uygulanabilir.
* Önceden eğitilmiş kelime vektörleri kelime benzerliği ve benzeşimi görevlerine uygulanabilir.

## Alıştırmalar

1. `TokenEmbedding('wiki.en')` kullanarak fastText sonuçlarını test edin.
1. Kelime dağarcığı son derece büyük olduğunda, benzer kelimeleri nasıl bulabiliriz veya bir kelime benzeşimini daha hızlı tamamlayabiliriz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1336)
:end_tab:
