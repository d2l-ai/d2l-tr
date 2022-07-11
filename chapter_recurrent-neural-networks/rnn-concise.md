# Yinelemeli Sinir Ağlarının Kısa Uygulaması
:label:`sec_rnn-concise`

:numref:`sec_rnn_scratch` RNN'lerin nasıl uygulandığını görmek için öğretici olsa da, bu uygun veya hızlı değildir. Bu bölümde, derin öğrenme çerçevesinin üst düzey API'leri tarafından sağlanan işlevleri kullanarak aynı dil modelinin nasıl daha verimli bir şekilde uygulanacağı gösterilecektir. Daha önce olduğu gibi zaman makinesi veri kümesini okuyarak başlıyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## [**Modelin Tanımlanması**]

Üst seviye API'ler, yinelemeli sinir ağlarının uygulamalarını sağlar. Yinelemeli sinir ağı tabakası `rnn_layer`'i tek bir gizli katman ve 256 gizli birimle inşa ediyoruz. Aslında, birden fazla katmana sahip olmanın ne anlama geldiğini henüz tartışmadık; onu :numref:`sec_deep_rnn` içinde göreceğiz. Şimdilik, birden çok katmanın, bir sonraki RNN katmanı için girdi olarak kullanılan bir RNN katmanının çıktısı anlamına geldiğini söylemek yeterlidir.

```{.python .input}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

```{.python .input}
#@tab tensorflow
num_hiddens = 256
rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,
    kernel_initializer='glorot_uniform')
rnn_layer = tf.keras.layers.RNN(rnn_cell, time_major=True,
    return_sequences=True, return_state=True)
```

:begin_tab:`mxnet`
Gizli durumu ilklemek basittir. Üye fonksiyonu `begin_state`'i çağırıyoruz . Bu, minigruptaki her örnek için (gizli katman sayısı, grup boyutu, gizli birim sayısı) şekilli bir ilk gizli durumu içeren liste (`state`) döndürür. Daha sonra tanıtılacak bazı modellerde (örneğin, uzun ömürlü kısa-dönem belleği), bu liste başka bilgiler de içerir.
:end_tab:

:begin_tab:`pytorch`
Şekli (gizli katman sayısı, grup boyutu, gizli birim sayısı) olan (**bir tensörü gizli durumu ilklemek için kullanırız**).
:end_tab:

```{.python .input}
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

```{.python .input}
#@tab tensorflow
state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
state.shape
```

[**Gizli bir durum ve bir girdi ile, çıktıyı güncellenmiş gizli durumla hesaplayabiliriz.**] `rnn_layer`'in “çıktı”'nın (`Y`) çıktı katmanlarının hesaplanmasını *içermediği* vurgulanmalıdır: *Her bir* zaman adımındaki gizli durumu ifade eder ve sonraki çıktı katmanına girdi olarak kullanılabilir.

:begin_tab:`mxnet`
Ayrıca, `rnn_layer` tarafından döndürülen güncelleştirilmiş gizli durum (`state_new`) minigrubun *son* zaman adımındaki gizli durumunu ifade eder. Sıralı bölümlemede bir dönem içinde sonraki minigrubun gizli durumunu ilklemede kullanılabilir. Birden çok gizli katman için, her katmanın gizli durumu bu değişkende saklanır (`state_new`). Daha sonra tanıtılacak bazı modellerde (örneğin, uzun ömürlü kısa-dönem belleği), bu değişken başka bilgiler de içerir.
:end_tab:

```{.python .input}
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

:numref:`sec_rnn_scratch` içindekine benzer şekilde, [**tam bir RNN modeli için bir `RNNModel` sınıfı tanımlarız.**] `rnn_layer`'in yalnızca gizli yinelemeli katmanları içerdiğini unutmayın, ayrı bir çıktı katmanı oluşturmamız gerekir.

```{.python .input}
#@save
class RNNModel(nn.Block):
    """RNN modeli."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # Tam bağlı katman önce `Y`'nin şeklini 
        # (`num_steps` * `batch_size`, `num_hiddens`) olarak değiştirir. 
        # Çıktı (`num_steps` * `batch_size`, `vocab_size`) şeklindedir.
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

```{.python .input}
#@tab pytorch
#@save
class RNNModel(nn.Module):
    """RNN modeli."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # RNN çift yönlü ise (daha sonra tanıtılacaktır), 
        # `num_directions` 2 olmalı, yoksa 1 olmalıdır.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # Tam bağlı katman önce `Y`'nin şeklini 
        # (`num_steps` * `batch_size`, `num_hiddens`) olarak değiştirir. 
        # Çıktı (`num_steps` * `batch_size`, `vocab_size`) şeklindedir.
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` gizli durum olarak bir tensör alır
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), 
                                device=device)
        else:
            # `nn.LSTM` bir gizli durum çokuzu alır
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # Daha sonra `tf.keras.layers.LSTMCell` gibi RNN ikiden fazla değer döndürür
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
```

## Eğitim ve Tahmin

Modeli eğitmeden önce, [**rastgele ağırlıklara sahip bir modelle bir tahmin yapalım.**]

```{.python .input}
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab pytorch
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab tensorflow
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = RNNModel(rnn_layer, vocab_size=len(vocab))

d2l.predict_ch8('time traveller', 10, net, vocab)
```

Oldukça açık, bu model hiç çalışmıyor. Ardından, :numref:`sec_rnn_scratch` içinde tanımlanan aynı hiper parametrelerle `train_ch8`'i çağırdık ve [**modelimizi üst düzey API'lerle eğitiyoruz.**]

```{.python .input}
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

Son bölümle karşılaştırıldığında, bu model, kodun derin öğrenme çerçevesinin üst düzey API'leriyle daha iyi hale getirilmesinden dolayı daha kısa bir süre içinde olsa da, benzer bir şaşkınlığa ulaşmaktadır.

## Özet

* Derin öğrenme çerçevesinin üst düzey API'leri RNN katmanının bir uygulanmasını sağlar.
* Üst düzey API'lerin RNN katmanı, çıktının çıktı katmanı hesaplamasını içermediği bir çıktı ve güncellenmiş bir gizli durum döndürür.
* Üst düzey API'lerin kullanılması, sıfırdan uygulama yaratmaktan daha hızlı RNN eğitimine yol açar.

## Alıştırmalar

1. RNN modelini üst düzey API'leri kullanarak aşırı eğitebilir misiniz?
1. RNN modelinde gizli katman sayısını artırırsanız ne olur? Modelin çalışmasını sağlayabilecek misiniz?
1. Bir RNN kullanarak :numref:`sec_sequence` içindeki özbağlanımlı modelini uygulayın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/2211)
:end_tab:
