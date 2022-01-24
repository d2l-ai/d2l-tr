# Bahdanau'nun Dikkatine

:label:`sec_seq2seq_attention` 

:numref:`sec_seq2seq`'te makine çevirisi problemini inceledik, burada sıralı öğrenme için iki RNN temelinde bir kodlayıcı-dekoder mimarisi tasarladık. Özellikle, RNN kodlayıcısı bir değişken uzunluktaki diziyi sabit şekil bağlam değişkenine dönüştürür, daha sonra RNN kod çözücüsü, oluşturulan belirteçlere ve bağlam değişkenine göre belirteç tarafından çıktı (hedef) sırası belirteci oluşturur. Ancak, tüm giriş (kaynak) belirteçleri belirli bir belirteci kodlamak için yararlı olmasa da, giriş sırasının tamamını kodlayan *same* bağlam değişkeni hala her kod çözme adımında kullanılır. 

Graves, belirli bir metin dizisi için el yazısı oluşturmanın ayrı fakat ilgili bir zorluğunda, metin karakterlerini çok daha uzun kalem iziyle hizalamak için farklılaştırılabilir bir dikkat modeli tasarladı. Bu model, hizalamanın yalnızca tek bir yönde hareket ettiği :cite:`Graves.2013`. Hizalamayı öğrenme fikrinden esinlenen Bahdanau vd. :cite:`Bahdanau.Cho.Bengio.2014` ciddi tek yönlü hizalama sınırlaması olmaksızın farklılaştırılabilir bir dikkat modeli önerdi. Bir belirteci tahmin ederken, tüm giriş belirteçleri uygun değilse, model yalnızca giriş sırasının geçerli tahminle ilgili bölümlerine hizalar (veya katılır). Bu, bağlam değişkeninin dikkat havuzunun bir çıktısı olarak ele alınarak elde edilir. 

## Model

Aşağıdaki RNN kodlayıcı-kod çözücüsü için Bahdanau'nun dikkatini açıklarken, :numref:`sec_seq2seq`'te aynı notasyonu takip edeceğiz. Yeni dikkat tabanlı model :numref:`sec_seq2seq`'teki :numref:`sec_seq2seq`'teki $\mathbf{c}$ bağlam değişkeni :eqref:`eq_seq2seq_s_t`'te $\mathbf{c}_{t'}$ herhangi bir kod çözme saati adımında $t'$ ile değiştirildiği dışında aynıdır. Giriş dizisinde $T$ belirteçleri olduğunu varsayalım, kod çözme zamanı adımındaki bağlam değişkeni $t'$ dikkat havuzunun çıktısıdır: 

$$\mathbf{c}_{t'} = \sum_{t=1}^T \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_t) \mathbf{h}_t,$$

Burada dekoder gizli durumu $\mathbf{s}_{t' - 1}$ zaman adım $t' - 1$ sorgu ve kodlayıcı gizli durumları $\mathbf{h}_t$ hem anahtarlar hem de değerlerdir ve $\alpha$ dikkat ağırlığı :eqref:`eq_attn-scoring-alpha` ile tanımlanan katkı dikkat puanlama işlevini kullanarak :eqref:`eq_attn-scoring-alpha`'de olduğu gibi hesaplanır. 

:numref:`fig_seq2seq_details`'teki vanilya RNN kodlayıcı-dekoder mimarisinden biraz farklı olan Bahdanau'nun dikkatiyle aynı mimari :numref:`fig_s2s_attention_details`'te tasvir edilmiştir. 

![Layers in an RNN encoder-decoder model with Bahdanau attention.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Çözücüyü Dikkatle Tanımlama

RNN kodlayıcı-kod çözücüyü Bahdanau'nun dikkatiyle uygulamak için, sadece kod çözücüyü yeniden tanımlamamız gerekiyor. Öğrenilen dikkat ağırlıklarını daha rahat görselleştirmek için, aşağıdaki `AttentionDecoder` sınıfı [**dikkat mekanizmalarına sahip kod çözücüler için temel arabirim**] tanımlar.

```{.python .input}
#@tab all
#@save
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
```

Now let us [**implement
the RNN decoder with Bahdanau attention**]
in the following `Seq2SeqAttentionDecoder` class.
Kod çözücünün durumu ile başlatılır (i) kodlayıcı son katman gizli durumları her zaman adımlarında (tuşlar ve dikkat değerleri olarak); (ii) son zaman adımında kodlayıcı tüm katmanlı gizli durumu (kod çözücünün gizli durumunu başlatmak için); ve (iii) kodlayıcının geçerli uzunluğu ( dikkat havuzunda dolgu belirteçleri). Her kod çözme zaman adımında, kod çözücü son katman gizli durumu önceki zaman adımında dikkat sorgusu olarak kullanılır. Sonuç olarak, hem dikkat çıkışı hem de giriş gömme RNN kod çözücünün girişi olarak birleştirilir.

```{.python .input}
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`,
        # `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`num_steps`, `batch_size`, `vocab_size`)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
#@tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]),
                                      return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of `outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state, enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        # Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X) # Input `X` has shape: (`batch_size`, `num_steps`)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Shape of `context`: (`batch_size, 1, `num_hiddens`)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Concatenate on the feature dimension
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully-connected layer transformation, shape of `outputs`:
        # (`batch_size`, `num_steps`, `vocab_size`)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

In the following, we [**test the implemented
decoder**] with Bahdanau attention
using a minibatch of 4 sequence inputs
of 7 time steps.

```{.python .input}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.initialize()
X = d2l.zeros((4, 7))  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab pytorch
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)  # (`batch_size`, `num_steps`)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

```{.python .input}
#@tab tensorflow
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
X = tf.zeros((4, 7))
state = decoder.init_state(encoder(X, training=False), None)
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## [**Eğitim**]

:numref:`sec_seq2seq_training`'e benzer şekilde, burada hiperparemetreleri belirtiyoruz, bir kodlayıcı ve Bahdanau'nun dikkatini çeken bir kod çözücü oluşturuyor ve bu modeli makine çevirisi için eğitiyoruz. Yeni eklenen dikkat mekanizması nedeniyle, bu eğitim :numref:`sec_seq2seq_training`'te dikkat mekanizmaları olmadan çok daha yavaştır.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

Model eğitildikten sonra, [**birkaç İngilizce cümle**] Fransızca'ya çevirmek ve BLEU puanlarını hesaplamak için kullanıyoruz.

```{.python .input}
#@tab mxnet, pytorch
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab all
attention_weights = d2l.reshape(
    d2l.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
```

Son İngilizce cümleyi çevirirken [****görselleştirme**] ile, her sorgunun anahtar değer çiftleri üzerinde tekdüze olmayan ağırlıklar atadığını görebiliriz. Her kod çözme adımında, girdi dizilerinin farklı bölümlerinin dikkat havuzunda seçici olarak toplandığını gösterir.

```{.python .input}
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
#@tab tensorflow
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key posistions', ylabel='Query posistions')
```

## Özet

* Bir belirteci tahmin ederken, tüm giriş belirteçleri uygun değilse, Bahdanau dikkatine sahip RNN kodlayıcı-kod çözücü seçici olarak giriş dizisinin farklı bölümlerini toplar. Bu, bağlam değişkeninin katkı maddesi dikkat havuzunun bir çıktısı olarak ele alınarak elde edilir.
* RNN kodlayıcı-kod çözücüsü Bahdanau dikkat önceki zaman adımında kod çözücü gizli durumunu sorgu olarak ele alır ve kodlayıcı gizli durumları her zaman hem anahtar hem de değerler olarak adımlar.

## Egzersizler

1. Deneyde GRU LSTM ile değiştirin.
1. Katkı dikkat puanlama işlevini ölçekli nokta ürünüyle değiştirmek için deneyi değiştirin. Eğitim verimliliğini nasıl etkiler?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:
