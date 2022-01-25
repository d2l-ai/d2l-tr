# Duygu Analizi: Tekrarlayan Sinir Ağlarının Kullanımı
:label:`sec_sentiment_rnn`

Kelime benzerliği ve benzetme görevleri gibi, biz de duyarlılık analizine önceden eğitilmiş kelime vektörleri de uygulayabiliriz. :numref:`sec_sentiment`'teki IMDb inceleme veri kümesi çok büyük olmadığından, büyük ölçekli corpora üzerinde önceden eğitilmiş metin temsillerinin kullanılması modelin aşırı uyumunu azaltabilir. :numref:`fig_nlp-map-sa-rnn`'da gösterilen belirli bir örnek olarak, önceden eğitilmiş Eldiven modelini kullanarak her belirteci temsil edeceğiz ve bu belirteç temsillerini çok katmanlı çift yönlü bir RNN'ye besleyeceğiz ve metin sırası temsilini elde etmek için bu belirteç temsillerini :cite:`Maas.Daly.Pham.ea.2011`'e dönüştürülecek. Aynı aşağı akım uygulaması için daha sonra farklı bir mimari seçim düşüneceğiz. 

![This section feeds pretrained GloVe to an RNN-based architecture for sentiment analysis.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## RNN'lerle Tek Metni Temsil Etme

Duyarlılık analizi gibi metin sınıflandırmalarında, değişen uzunluktaki bir metin dizisi sabit uzunlukta kategorilere dönüştürülür. Aşağıdaki `BiRNN` sınıfında, bir metin dizisinin her belirteci, gömme katman (`self.embedding`) aracılığıyla bireysel önceden eğitilmiş Eldiven temsilini alırken, tüm dizi çift yönlü RNN (`self.encoder`) ile kodlanır. Daha somut olarak, hem başlangıç hem de son zaman adımlarında iki yönlü LSTM'nin gizli durumları (son katmanda) metin sırasının temsili olarak birleştirilir. Bu tek metin temsili daha sonra iki çıkışlı (“pozitif” ve “negatif”) tam bağlı bir katman (`self.decoder`) ile çıktı kategorilerine dönüştürülür.

```{.python .input}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        outs = self.decoder(encoding)
        return outs
```

Duygu analizi için tek bir metni temsil etmek üzere iki gizli katman içeren iki yönlü bir RNN oluşturalım.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);
```

## Önceden Eğitimli Word Vektörlerini Yükleme

Aşağıda önceden eğitilmiş 100 boyutlu yüklüyoruz (`embed_size` ile tutarlı olması gerekir) Kelime dağarcığındaki belirteçler için Eldiven ve gömme.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

Kelime dağarcığındaki tüm belirteçler için vektörlerin şeklini yazdırın.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

Bu önceden eğitilmiş kelime vektörlerini incelemelerde belirteçleri temsil etmek için kullanıyoruz ve eğitim sırasında bu vektörleri güncellemeyeceğiz.

```{.python .input}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## Modelin Eğitimi ve Değerlendirilmesi

Şimdi iki yönlü RNN'leri duygu analizi için eğitebiliriz.

```{.python .input}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Eğitimli model `net` kullanarak bir metin dizisinin duyarlılığını tahmin etmek için aşağıdaki işlevi tanımlıyoruz.

```{.python .input}
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Son olarak, iki basit cümlenin duygularını tahmin etmek için eğitimli modeli kullanalım.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Özet

* Önceden eğitilmiş sözcük vektörleri, bir metin dizisinde tek tek belirteçleri temsil edebilir.
* Çift yönlü RNN'ler, ilk ve son zaman adımlarında gizli durumlarının birleştirilmesi yoluyla gibi bir metin sırasını temsil edebilir. Bu tek metin temsili, tamamen bağlı bir katman kullanılarak kategorilere dönüştürülebilir.

## Egzersizler

1. Çeyin sayısını artırın. Eğitim ve test doğruluklarını geliştirebilir misiniz? Diğer hiperparametreleri ayarlamaya ne dersin?
1. 300 boyutlu Eldiven gömme gibi daha büyük önceden eğitilmiş sözcük vektörlerini kullanın. Sınıflandırma doğruluğunu arttırıyor mu?
1. SpaCy tokenization kullanarak sınıflandırma doğruluğunu artırabilir miyiz? SpaCy (`pip install spacy`) yüklemeniz ve İngilizce paketini (`python -m spacy download en`) yüklemeniz gerekir. Kodda, önce spaCy (`import spacy`) içe aktarın. Ardından, spaCy İngilizce paketini yükleyin (`spacy_en = spacy.load('en')`). Son olarak, `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` işlevini tanımlayın ve orijinal `tokenizer` işlevini değiştirin. GloVe ve spaCy içinde ifade belirteçlerinin farklı biçimlerine dikkat edin. Örneğin, “new york” ifadesi Glove'deki “new-york” şeklini ve spaCy tokenization işleminden sonra “new york” şeklini alır.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab:
