# word2vec Ön Eğitimi
:label:`sec_word2vec_pretraining`

:numref:`sec_word2vec` içinde tanımlanan skip-gram modelini uygulamaya devam ediyoruz. Ardından, PTB veri kümesindeki negatif örnekleme kullanarak word2vec ön eğiteceğiz. Her şeyden önce, :numref:`sec_word2vec_data` içinde açıklanan `d2l.load_data_ptb` işlevini çağırarak veri yineleyicisini ve bu veri kümesi için sözcük dağarcığını elde edelim.

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

Katmanları ve toplu matris çarpımlarını kullanarak skip-gram modelini uyguluyoruz. İlk olarak, gömme katmanların nasıl çalıştığını inceleyelim. 

### Gömme Katmanı

:numref:`sec_seq2seq` içinde açıklandığı gibi, bir katman, bir belirteç dizinini öznitelik vektörüyle eşler. Bu katmanın ağırlığı, satır sayısı sözlük boyutuna (`input_dim`) ve sütun sayısı her belirteç için vektör boyutuna (`output_dim`) eşit olan bir matristir. Bir sözcük gömme modeli eğitildikten sonra, bu ağırlık ihtiyacımız olan şeydir.

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

Gömülü katmanın girdisi, bir belirteç (sözcük) dizinidir. Herhangi bir belirteç dizini $i$ için vektör gösterimi, gömme katmanındaki ağırlık matrisinin $i.$ satırından elde edilebilir. Vektör boyutu (`output_dim`) 4 olarak ayarlandığından, gömme katmanı (2, 3) şekle sahip bir minigrup işlemi için (2, 3, 4) şekilli vektörler döndürür.

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### İleri Yaymayı Tanımlama

İleri yayılımda, skip-gram modelinin girdisi, (toplu iş boyutu, 1) şekilli `center` merkez sözcük indekslerini ve `max_len`'in :numref:`subsec_word2vec-minibatch-loading` içinde tanımlandığı (toplu iş boyutu, `max_len`) şeklindeki `contexts_and_negatives` bitiştirilmiş bağlam ve gürültü sözcük indekslerini içerir. Bu iki değişken önce belirteç dizinlerinden gömme katmanı aracılığıyla vektörlere dönüştürülür, daha sonra toplu matris çarpımı (:numref:`subsec_batch_dot` içinde açıklanmıştır) (toplu iş boyutu, 1, `max_len`) şekilli bir çıktı döndürür. Çıktıdaki her eleman, bir merkez sözcük vektörü ile bir bağlam veya gürültü sözcük vektörünün nokta çarpımıdır .

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

Bazı örnek girdiler için bu `skip_gram` işlevinin çıktı şeklini yazdıralım.

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

:numref:`subsec_negative-sampling` içinde negatif örnekleme için kayıp fonksiyonunun tanımına göre ikili çapraz entropi kaybını kullanacağız.

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Maskeleme ile ikili çapraz entropi kaybı
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

:numref:`subsec_word2vec-minibatch-loading` içindeki maske değişkeninin ve etiket değişkeninin tanımlarını hatırlayın. Aşağıdaki ifade, verilen değişkenler için ikili çapraz entropi kaybını hesaplar.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Aşağısı, ikili çapraz entropi kaybında sigmoid etkinleştirme fonksiyonu kullanılarak yukarıdaki sonuçların nasıl hesaplandığını (daha az verimli bir şekilde) göstermektedir. İki çıktıyı, maskelenmemiş tahminlere göre ortalaması alınan iki normalleştirilmiş kayıp olarak düşünebiliriz.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Model Parametrelerini İlkleme

Sırasıyla merkez sözcükler ve bağlam sözcükleri olarak kullanıldıklarından sözcük dağarcığındaki tüm sözcükler için iki gömme katmanı tanımlarız. `embed_size` sözcük vektör boyutu 100 olarak ayarlanır.

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

### Eğitim Döngüsünün Tanımlanması

Eğitim döngüsü aşağıda tanımlanmıştır. Dolgunun varlığı nedeniyle, kayıp fonksiyonunun hesaplanması önceki eğitim fonksiyonlarına kıyasla biraz farklıdır.

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Normalleştirilmiş kayıpların toplamı, normalleştirilmiş kayıpların sayısı
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
    # Normalleştirilmiş kayıpların toplamı, normalleştirilmiş kayıpların sayısı
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

## Sözcük Gömmelerini Uygulama
:label:`subsec_apply-word-embed`

Word2vec modelini eğittikten sonra, eğitimli modeldeki sözcük vektörlerinin kosinüs benzerliğini kullanarak sözlükten bir girdi sözcüğüne en çok benzeyen sözcükleri bulabiliriz.

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Kosinüs benzerliğini hesaplayın. Sayısal kararlılık için 1e-9 ekleyin
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
    # Kosinüs benzerliğini hesaplayın. Sayısal kararlılık için 1e-9 ekleyin
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Özet

* Gömülü katmanlar ve ikili çapraz entropi kaybını kullanarak negatif örnekleme ile bir skip-gram modelini eğitebiliriz.
* Sözcük gömme uygulamaları, sözcük vektörlerinin kosinüs benzerliğine dayanan belirli bir sözcük için anlamsal olarak benzer sözcükleri bulmayı içerir.

## Alıştırmalar

1. Eğitilmiş modeli kullanarak, diğer girdi sözcükleri için anlamsal olarak benzer sözcükleri bulun. Hiper parametreleri ayarlayarak sonuçları iyileştirebilir misiniz?
1. Bir eğitim külliyatı çok büyük olduğunda, *model parametrelerini güncellerken* mevcut minigrup içindeki merkez sözcükler için bağlam sözcükleri ve gürültü sözcüklerini sık sık örnekleriz. Başka bir deyişle, aynı merkez sözcük farklı eğitim dönemlerinde farklı bağlam sözcüklerine veya gürültü sözcüklerine sahip olabilir. Bu yöntemin faydaları nelerdir? Bu eğitim yöntemini uygulamaya çalışın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1335)
:end_tab:
