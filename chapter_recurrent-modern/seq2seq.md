# Diziden Diziye Öğrenme
:label:`sec_seq2seq`

:numref:`sec_machine_translation`'te gördüğümüz gibi, makine çevirisinde hem girdi hem de çıktı değişken uzunlukta dizilerdir. Bu tür problemleri çözmek için :numref:`sec_encoder-decoder`'te genel bir kodlayıcı-kodçözücü mimarisi tasarladık. Bu bölümde, bu mimarinin kodlayıcısını ve kodçözücüsünü tasarlamak için iki RNN kullanacağız ve makine çevirisi :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014` için *diziden diziye* öğrenmeyi uygulayacağız.

Kodlayıcı-kodçözücü mimarisinin tasarım ilkesini takiben, RNN kodlayıcı girdi olarak değişken uzunlukta bir diziyi alabilir ve bir sabit şekilli gizli duruma dönüştürebilir. Başka bir deyişle, girdi dizisinin bilgileri RNN kodlayıcısının gizli durumunda *kodlanmış* olur. Çıktı dizisi andıç andıç oluşturmak için, ayrı bir RNN kodçözücü, girdi dizisinin kodlanmış bilgileriyle birlikte, hangi andıçların görüldüğünü (dil modellemesinde olduğu gibi) veya oluşturulduğuna bağlı olarak bir sonraki andıcı tahmin edebilir. :numref:`fig_seq2seq`, makine çevirisinde diziden diziye öğrenme için iki RNN'nin nasıl kullanılacağını gösterir.

![Bir RNN kodlayıcı ve bir RNN kodçözücü ile diziden diziye öğrenme.](../img/seq2seq.svg)
:label:`fig_seq2seq`

:numref:`fig_seq2seq`'te, özel "<eos>" andıcı dizinin sonunu işaretler. Model, bu andıç oluşturulduktan sonra tahminlerde bulunmayı durdurabilir. RNN kodçözücüsünün ilk zaman adımında, iki özel tasarım kararı vardır. İlk olarak, özel dizi başlangıç andıcı, "<bos>", bir girdidir. İkincisi, RNN kodlayıcısının son gizli durumu, kodçözücünün gizli durumunu ilklemek için kullanılır. :cite:`Sutskever.Vinyals.Le.2014`'teki gibi tasarımlarda, kodlanmış girdi dizisi bilgilerinin çıktı dizisini oluşturmak için kodçözücüsüne beslenmesi tam olarak da budur. :cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014` gibi diğer bazı tasarımlarda, kodlayıcının son gizli durumu, :numref:`fig_seq2seq`'te gösterildiği gibi her adımda girdilerin bir parçası olarak kodçözücüye beslenir. :numref:`sec_language_model`'deki dil modellerinin eğitimine benzer şekilde, etiketlerin bir andıç ile kaydırılmış orijinal çıktı dizisi olmasına izin verebiliriz: "<bos>“, “Ils”, “regardent”, “.” $\rightarrow$ “Ils”, “regardent”,”.“,"<eos>”.

Aşağıda, :numref:`fig_seq2seq`'ün tasarımını daha ayrıntılı olarak açıklayacağız. Bu modeli :numref:`sec_machine_translation`'te tanıtılan İngilizce-Fransız veri kümesinde makine çevirisi için eğiteceğiz.

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
```

## Kodlayıcı

Teknik olarak konuşursak, kodlayıcı değişken uzunluktaki bir girdi dizisini sabit şekilli *bağlam değişkeni* $\mathbf{c}$'ye dönüştürür ve bu bağlam değişkende girdi dizisinin bilgilerini kodlar. :numref:`fig_seq2seq`'te gösterildiği gibi, kodlayıcıyı tasarlamak için bir RNN kullanabiliriz.

Bir dizi örneği düşünelim (toplu küme boyutu: 1). Girdi dizimizin $x_1, \ldots, x_T$ olduğunu varsayalım, öyle ki $x_t$ girdi metin dizisindeki $t.$ andıç olsun. $t$ zaman adımında, RNN $x_t$ için girdi öznitelik vektörü $\mathbf{x}_t$'yi ve önceki zaman adımından gizli durum $\mathbf{h}_{t-1}$'yi şu anki gizli durum $\mathbf{h}_t$'ye dönüştürür. RNN'nin yinelemeli tabakasının dönüşümünü ifade etmek için $f$ işlevini kullanabiliriz:

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

Genel olarak, kodlayıcı, gizli durumları her zaman adamında özelleştirilmiş bir $q$ işlevi aracılığıyla bağlam değişkenine dönüştürür:

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

Örneğin, :numref:`fig_seq2seq`'te olduğu gibi $q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$'yi seçerken, bağlam değişkeni yalnızca son zaman adımındaki girdi dizisinin gizli durumu $\mathbf{h}_T$'dir.

Şimdiye kadar kodlayıcıyı tasarlamak için tek yönlü bir RNN kullandık, burada gizli bir durum yalnızca gizli durumun önceki ve o anki zaman adımındaki girdi altdizisine bağlıdır. Ayrıca çift yönlü RNN'leri kullanarak kodlayıcılar da oluşturabiliriz. Bu durumda, tüm dizinin bilgilerini kodlayan gizli durum, zaman adımından önceki ve sonraki altdiziye (geçerli zaman adımındaki girdi dahil) bağlıdır.

Şimdi RNN kodlayıcısını uygulamaya başlayalım. Girdi dizisindeki her andıç için öznitelik vektörünü elde ederken bir *gömme katmanı* kullandığımıza dikkat edin. Bir gömme katmanın ağırlığı, satır sayısı girdi kelime dağarcığının boyutuna (`vocab_size`) ve sütun sayısı öznitelik vektörünün boyutuna eşit olan bir matristir (`embed_size`). Herhangi bir girdi andıcı dizini $i$ için gömme katmanı, öznitelik vektörünü döndürmek üzere ağırlık matrisinin $i.$ satırını (0'dan başlayarak) getirir. Ayrıca, burada kodlayıcıyı uygulamak için çok katmanlı bir GRU seçiyoruz.

```{.python .input}
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        output, state = self.rnn(X, state)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

Yinelemeli katmanların döndürülen değişkenleri :numref:`sec_rnn-concise`'te açıklanmıştı. Yukarıdaki kodlayıcı uygulamasını göstermek için somut bir örnek kullanalım. Aşağıda, gizli birimlerin sayısı 16 olan iki katmanlı bir GRU kodlayıcısı oluşturuyoruz. `X` dizi girdilerinin bir minigrubu göz önüne alındığında (grup boyutu: 4, zaman adımı sayısı: 7), son katmanın gizli durumları (kodlayıcının yinelemeli katmanları tarafından döndürülen `output`) şekli (zaman adımlarının sayısı, grup boyutu, gizli birimlerin sayısı) olan tensörlerdir.

```{.python .input}
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = d2l.zeros((4, 7))
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab pytorch
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
```

Burada bir GRU kullanıldığından, son zaman adımındaki çok katmanlı gizli durumlar (gizli katmanların sayısı, grup boyutu, gizli birim sayısı) şeklindedir. Bir LSTM kullanılıyorsa, bellek hücresi bilgileri de `state`'te yer alır.

```{.python .input}
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state.shape
```

## Kodçözücü
:label:`sec_seq2seq_decoder`

Az önce de belirttiğimiz gibi, kodlayıcının çıkışının $\mathbf{c}$ bağlam değişkeni $x_1, \ldots, x_T$ tüm giriş sırasını kodlar. Eğitim veri setinden $y_1, y_2, \ldots, y_{T'}$ çıkış sırası göz önüne alındığında, her zaman adım $t'$ için (sembol, giriş dizilerinin veya kodlayıcıların $t$ zaman adımından farklıdır), kodçözücü çıkışının olasılığı $y_{t'}$ önceki çıkış alt sırası $y_1, \ldots, y_{t'-1}$ ve bağlam değişkeni üzerinde koşulludur $\mathbf{c}$, yani, $P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$.

Bu koşullu olasılığı diziler üzerinde modellemek için, kodçözücü olarak başka bir RNN kullanabiliriz. Herhangi bir zamanda adım $t^\prime$ çıktı sırası, RNN $y_{t^\prime-1}$ önceki zaman adımından ve $\mathbf{c}$ bağlam değişkeni giriş olarak alır, sonra onları ve önceki gizli durum $\mathbf{s}_{t^\prime-1}$ gizli duruma dönüştürür $\mathbf{s}_{t^\prime}$ geçerli zaman adımında. Sonuç olarak, kodçözücünün gizli katmanının dönüşümünü ifade etmek için $g$ işlevi kullanabiliriz:

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$

kodçözücünün gizli durumunu elde ettikten sonra, $t^\prime$ adımındaki çıkış için koşullu olasılık dağılımını $P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$'i hesaplamak için bir çıkış katmanı ve softmax işlemini kullanabiliriz.

:numref:`fig_seq2seq`'ü takiben, kodçözücüyü aşağıdaki gibi uygularken, kodçözücünün gizli durumunu başlatmak için kodlayıcının son zaman adımındaki gizli durumu doğrudan kullanırız. Bu, RNN kodlayıcı ve RNN kodçözücüsünün aynı sayıda katman ve gizli birimlere sahip olmasını gerektirir. Kodlanmış giriş sırası bilgilerini daha da dahil etmek için, bağlam değişkeni kodçözücü girişiyle her zaman adımda birleştirilir. Çıktı belirtecinin olasılık dağılımını tahmin etmek için, RNN kodçözücünün son katmanındaki gizli durumu dönüştürmek için tam bağlı bir katman kullanılır.

```{.python .input}
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        # `context` shape: (`batch_size`, `num_hiddens`)
        context = state[0][-1]
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = np.broadcast_to(context, (
            X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

Uygulanan kodçözücüyü göstermek için, aşağıda belirtilen kodlayıcıdan aynı hiperparametrelerle başlatıyoruz. Gördüğümüz gibi, kodçözücünün çıkış şekli olur (parti boyutu, zaman adımlarının sayısı, kelime dağarcığı boyutu), burada tensörün son boyutu tahmin edilen belirteç dağılımını depolar.

```{.python .input}
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

Özetlemek gerekirse, yukarıdaki RNN kodlayıcı-kodçözücü modelindeki katmanlar :numref:`fig_seq2seq_details`'te gösterilmektedir.

![Layers in an RNN encoder-decoder model.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

## Kayıp Fonksiyonu

Her adımda, kodçözücü çıktı belirteçleri için bir olasılık dağılımı öngörür. Dil modellemesine benzer şekilde, dağıtımı elde etmek ve optimizasyon için çapraz entropi kaybını hesaplamak için softmax uygulayabiliriz. Özel dolgu belirteçlerinin dizilerin sonuna eklendiğini hatırlayın :numref:`sec_machine_translation`, böylece değişen uzunluklardaki dizilerin aynı şeklin minibatch'lerine verimli bir şekilde yüklenebilmesini sağlayın. Bununla birlikte, dolgu belirteçlerinin tahmini kayıp hesaplamalarından hariç tutulmalıdır.

Bu amaçla, sıfır değerleriyle alakasız girişleri maskelemek için aşağıdaki `sequence_mask` işlevini kullanabiliriz, böylece daha sonra sıfır eşittir ile alakasız tahminlerin çarpımı sıfıra eşittir. Örneğin, dolgu belirteçleri hariç iki dizinin geçerli uzunluğu sırasıyla bir ve iki ise, ilk ve ilk iki girişten sonra kalan girişler sıfırlara temizlenir.

```{.python .input}
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

```{.python .input}
#@tab pytorch
#@save
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
```

Son birkaç eksendeki tüm girişleri de maskeleyebiliriz. İsterseniz, bu tür girişleri sıfır olmayan bir değerle değiştirmeyi bile belirtebilirsiniz.

```{.python .input}
X = d2l.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

```{.python .input}
#@tab pytorch
X = d2l.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
```

Artık alakasız tahminlerin maskelenmesine izin vermek için softmax çapraz entropi kaybını genişletebiliriz. Başlangıçta, tahmin edilen tüm belirteçler için maskeler bir olarak ayarlanır. Geçerli uzunluk verildikten sonra, herhangi bir dolgu belirtecine karşılık gelen maske sıfır olarak temizlenir. Sonunda, tüm belirteçlerin kaybı, kayıptaki dolgu belirteçlerinin ilgisiz tahminlerini filtrelemek için maske ile çarpılacaktır.

```{.python .input}
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        # `weights` shape: (`batch_size`, `num_steps`, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

```{.python .input}
#@tab pytorch
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

Akıl sağlığı kontrolü için üç özdeş sekansları oluşturabiliriz. Ardından, bu dizilerin geçerli uzunluklarının sırasıyla 4, 2 ve 0 olduğunu belirtebiliriz. Sonuç olarak, birinci dizinin kaybı ikinci dizinin iki katı kadar büyük olmalı, üçüncü dizinin sıfır kaybına sahip olmalıdır.

```{.python .input}
loss = MaskedSoftmaxCELoss()
loss(d2l.ones((3, 4, 10)), d2l.ones((3, 4)), np.array([4, 2, 0]))
```

```{.python .input}
#@tab pytorch
loss = MaskedSoftmaxCELoss()
loss(d2l.ones(3, 4, 10), d2l.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
```

## Eğitim
:label:`sec_seq2seq_training`

Aşağıdaki eğitim döngüsünde, :numref:`fig_seq2seq`'te gösterildiği gibi, kodçözücüye giriş olarak son belirteci hariç özel başlangıç dizisini ve orijinal çıkış sırasını birleştiririz. Buna *öğretmen zorlama* denir çünkü orijinal çıktı dizisi (belirteç etiketleri) kodçözücüye beslenir. Alternatif olarak, önceki zaman adımından*öngörülen* belirteci kodçözücüye geçerli giriş olarak da besleyebiliriz.

```{.python .input}
#@save
def train_s2s_ch9(model, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence (defined in Chapter 9)."""
    model.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [
                x.as_in_ctx(device) for x in batch]
            bos = np.array(
                [tgt_vocab['<bos>']] * Y.shape[0], ctx=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with autograd.record():
                Y_hat, _ = model(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_s2s_ch9(model, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence (defined in Chapter 9)."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    torch.nn.init.xavier_uniform_(m._parameters[param])
    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    model.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

Artık makine çevirisi veri kümesinde diziye öğrenme için bir RNN kodlayıcı-kodçözücü modeli oluşturabilir ve eğitebiliriz.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.EncoderDecoder(encoder, decoder)
train_s2s_ch9(model, train_iter, lr, num_epochs, tgt_vocab, device)
```

## Tahmin

Çıkış sırası belirtecini belirteç ile tahmin etmek için, her kodçözücü zaman adımında önceki zaman adımından tahmin edilen belirteç kodçözücüye girdi olarak beslenir. Eğitime benzer şekilde, başlangıç adımında dizinin başlangıcı (” <bos> “) belirteci kodçözücüye beslenir. Bu tahmin süreci :numref:`fig_seq2seq_predict`'te gösterilmektedir. Sıra sonu (” <eos> “) belirteci tahmin edildiğinde, çıktı sırasının tahmini tamamlanır.

![Predicting the output sequence token by token using an RNN encoder-decoder.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

:numref:`sec_beam-search`'te dizi üretimi için farklı stratejiler sunacağız.

```{.python .input}
#@save
def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device):
    """Predict sequences (defined in Chapter 9)."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # Once the end-of-sequence token is predicted, the generation of
        # the output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))
```

```{.python .input}
#@tab pytorch
#@save
def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device):
    """Predict sequences (defined in Chapter 9)."""
    # Set model to eval mode for inference
    model.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Once the end-of-sequence token is predicted, the generation of
        # the output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq))
```

## Tahmin edilen Dizilerin Değerlendirilmesi

Tahmin edilen bir diziyi etiket dizisi (zemin gerçeği) ile karşılaştırarak değerlendirebiliriz. BLEU (İki Dilli Değerlendirme Understudy), başlangıçta makine çevirisi sonuçlarını değerlendirmek için önerilen olsa da, farklı uygulamalar için çıktı dizilerinin kalitesini ölçmede yaygın olarak kullanılmaktadır. Prensip olarak, tahmin edilen dizideki herhangi bir $n$ gram için, BLEU bu $n$-gramın etiket dizisinde görüp görünmediğini değerlendirir.

$p_n$ ile $n$-gram hassasiyetini belirtin; bu, öngörülen ve etiket dizilerindeki eşleşen $n$-gramlık sayısının tahmin edilen sıradaki $n$-gram sayısına oranıdır. $A$, $B$, $C$, $D$, $D$, $E$, $F$ ve öngörülen bir dizi $A$, $B$, $B$, $C$, $D$. Elimizde $p_1 = 4/5$, $p_2 = 3/4$, $p_3 = 1/3$ ve $p_4 = 0$ var. Ayrıca, $\mathrm{len}_{\text{label}}$ ve $\mathrm{len}_{\text{pred}}$'ün sırasıyla etiket dizisindeki belirteçlerin sayıları ve tahmin edilen dizide olmasına izin verin. Daha sonra, BLEU olarak tanımlanır

$$ \exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

burada $k$ eşleşmesi için en uzun $n$-gramdır.

:eqref:`eq_bleu`'teki BLEU tanımına dayanarak, tahmin edilen sıra etiket dizisi ile aynı olduğunda, BLEU 1'dir. Dahası, daha uzun $n$-gram eşleştirme daha zor olduğundan, BLEU daha uzun $n$-gram hassasiyetine daha büyük bir ağırlık atar. Özellikle $p_n$ sabit olduğunda $p_n^{1/2^n}$ büyüdükçe $p_n^{1/2^n}$ artar. Ayrıca, daha kısa dizileri tahmin etmek daha yüksek bir $p_n$ değeri elde etme eğiliminde olduğundan, :eqref:`eq_bleu`'teki çarpım teriminden önceki katsayı daha kısa öngörülen dizileri cezalandırır. Örneğin, $k=2$, $A$, $B$, $C$, $D$, $E$, $F$ ve öngörülen dizi $A$, $B$, ancak $p_1 = p_2 = 1$, ceza faktörü $\exp(1-6/2) \approx 0.14$ BLEU'yu düşürür.

BLEU ölçüsünü aşağıdaki gibi uyguluyoruz.

```{.python .input}
#@tab all
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

Sonunda, birkaç İngilizce cümleyi Fransızca'ya çevirmek ve sonuçların BLEU'sını hesaplamak için eğitilmiş RNN kodlayıcı-kodçözücüsünü kullanıyoruz.

```{.python .input}
#@tab all
#@save
def translate(engs, fras, model, src_vocab, tgt_vocab, num_steps, device):
    """Translate text sequences."""
    for eng, fra in zip(engs, fras):
        translation = predict_s2s_ch9(
            model, eng, src_vocab, tgt_vocab, num_steps, device)
        print(
            f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

engs = ['go .', "i lost .", 'i\'m home .', 'he\'s calm .']
fras = ['va !', 'j\'ai perdu .', 'je suis chez moi .', 'il est calme .']
translate(engs, fras, model, src_vocab, tgt_vocab, num_steps, device)
```

## Özet

* Kodlayıcı-kodçözücü mimarisinin tasarımını takiben, dizi-dizi öğrenimi için bir model tasarlamak için iki RNN kullanabiliriz.
* Kodlayıcıyı ve kodçözücüyü uygularken, çok katmanlı RNN'leri kullanabiliriz.
* Kayıp hesaplanırken olduğu gibi alakasız hesaplamaları filtrelemek için maskeler kullanabiliriz.
* Kodlayıcı-kodçözücü eğitiminde, öğretmen zorlama yaklaşımı orijinal çıktı dizilerini (tahminlerin aksine) kodçözücüye besler.
* BLEU, tahmin edilen dizi ve etiket dizisi arasında $n$-gram eşleştirerek çıktı dizilerini değerlendirmek için popüler bir ölçüdür.

## Alıştırmalar

1. Çeviri sonuçlarını iyileştirmek için hiperparametreleri ayarlayabilir misiniz?
1. Kayıp hesaplamasında maskeler kullanmadan deneyi yeniden çalıştırın. Hangi sonuçları gözlemliyorsunuz? Neden?
1. Kodlayıcı ve kodçözücü katman sayısı veya gizli birimlerin sayısı bakımından farklıysa, kodçözücünün gizli durumunu nasıl başlatabiliriz?
1. Eğitimde, önceki zamanda öngörü besleyerek zorlayan öğretmeni kodçözücüye adım atın. Bu performansı nasıl etkiler?
1. GRU'yu LSTM ile değiştirerek deneyi yeniden çalıştırın.
1. kodçözücünün çıkış katmanını tasarlamanın başka yolları var mı?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1062)
:end_tab:
