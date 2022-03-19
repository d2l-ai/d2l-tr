# Pretraining word2vec
:label:`sec_word2vec_pretraining`

:numref:`sec_word2vec`'te tanımlanan atlama gram modelini uygulamaya devam ediyoruz. Ardından, PTB veri kümesindeki negatif örnekleme kullanarak word2vec ön eğiteceğiz. Her şeyden önce, :numref:`sec_word2vec_data`'te açıklanan `d2l.load_data_ptb` işlevini arayarak veri yineleyicisini ve bu veri kümesi için kelime dağarcığını elde edelim

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Skip-Gram Modeli

Katmanları ve toplu matris çarpımlarını kullanarak atlama gram modelini uyguluyoruz. İlk olarak, gömme katmanların nasıl çalıştığını inceleyelim. 

### Katman Gömme

:numref:`sec_seq2seq`'te açıklandığı gibi, bir katman, bir simge dizini özellik vektörüyle eşler. Bu katmanın ağırlığı, satır sayısı sözlük boyutuna (`input_dim`) ve sütun sayısı her belirteç için vektör boyutuna eşit olan bir matristir (`output_dim`). Bir kelime gömme modeli eğitildikten sonra, bu ağırlık ihtiyacımız olan şeydir.

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Gömülü katmanın girişi, bir belirteç (sözcük) dizinidir. Herhangi bir belirteç dizini $i$ için vektör gösterimi, katıştırma katmanındaki ağırlık matrisinin $i.$ satırından elde edilebilir. Vektör boyutu (`output_dim`) 4 olarak ayarlandığından, gömme katmanı şekle sahip (2, 3) bir minik toplu işlem için şekilli (2, 3, 4) vektörleri döndürür (2, 3).

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### İleri yayılmayı tanımlama

İleri yayılımda, atlama grafiği modelinin girişi, `center` şeklin orta kelime indekslerini (toplu iş boyutu, 1) ve `max_len`'nın :numref:`subsec_word2vec-minibatch-loading`'te tanımlandığı `max_len` şeklin (toplu iş boyutu, `max_len`) birleştirilmiş bağlam ve gürültü kelime indekslerini içerir. Bu iki değişken önce belirteç dizinlerinden katman aracılığıyla vektörlere dönüştürülür, daha sonra toplu matris çarpımı (:numref:`subsec_batch_dot`'te açıklanmıştır) bir şekil çıktısı döndürür (toplu iş boyutu, 1, `max_len`). Çıktıdaki her öğe, bir merkez sözcük vektörünün nokta çarpımıdır ve bir bağlam veya parazit sözcük vektörüdür.

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Bazı örnek girişler için bu `skip_gram` işlevinin çıkış şeklini yazdıralım.

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Eğitim

Skip-gram modelini negatif örnekleme ile eğitmeden önce, öncelikle kayıp işlevini tanımlayalım. 

### İkili Çapraz Entropi Kaybı

:numref:`subsec_negative-sampling`'te negatif örnekleme için kayıp fonksiyonunun tanımına göre ikili çapraz entropi kaybını kullanacağız.

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

:numref:`subsec_word2vec-minibatch-loading`'teki maske değişkeninin ve etiket değişkeninin açıklamalarımızı hatırlayın. Aşağıdaki, verilen değişkenler için ikili çapraz entropi kaybını hesaplar.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Aşağıda, ikili çapraz entropi kaybında sigmoid aktivasyon fonksiyonu kullanılarak yukarıdaki sonuçların nasıl hesaplandığını (daha az verimli bir şekilde) göstermektedir. İki çıktıları maskelenmemiş tahminler üzerinden ortalama iki normalleştirilmiş kayıp olarak düşünebiliriz.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Model Parametreleri Başlatılıyor

Sırasıyla orta sözcükler ve bağlam sözcükleri olarak kullanıldıklarında kelime dağarcığındaki tüm kelimeler için iki gömme katman tanımlarız. `embed_size` sözcüğü vektör boyutu 100 olarak ayarlanır.

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Eğitim döngüsünün tanımlanması

Eğitim döngüsü aşağıda tanımlanmıştır. Dolgunun varlığı nedeniyle, kayıp fonksiyonunun hesaplanması önceki eğitim fonksiyonlarına kıyasla biraz farklıdır.

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Artık negatif örnekleme kullanarak bir skip-gram modelini eğitebiliriz.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Sözcük Gömlemeleri Uygulama
:label:`subsec_apply-word-embed`

Word2vec modelini eğittikten sonra, eğitimli modeldeki kelime vektörlerinin kosinüs benzerliğini kullanarak sözlükten bir giriş kelimesine en çok benzeyen kelimeleri bulabiliriz.

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Özet

* Gömülü katmanlar ve ikili çapraz entropi kaybını kullanarak negatif örnekleme ile bir skip-gram modelini eğitebiliriz.
* Kelime gömme uygulamaları, kelime vektörlerinin kosinüs benzerliğine dayanan belirli bir kelime için anlamsal olarak benzer kelimeleri bulmayı içerir.

## Egzersizler

1. Eğitimli modeli kullanarak, diğer girdi kelimeleri için anlamsal olarak benzer kelimeleri bulun. Hiperparametreleri ayarlayarak sonuçları iyileştirebilir misiniz?
1. Bir eğitim corpus çok büyük olduğunda, model parametrelerini güncellerken* mevcut minibatch içindeki orta sözcükler için bağlam sözcükleri ve gürültü kelimelerini sık sık örnekleriz. Başka bir deyişle, aynı merkez sözcük farklı eğitim çemlerinde farklı bağlam sözcükleri veya parazit sözcüklere sahip olabilir. Bu yöntemin faydaları nelerdir? Bu eğitim yöntemini uygulamaya çalışın.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
