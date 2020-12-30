# Derin Yinelemeli Sinir Ağları
:label:`sec_deep_rnn`

Şimdiye kadar, tek tek yönlü gizli katmanlı RNN'leri tartıştık. İçinde gizli değişkenlerin ve gözlemlerin nasıl etkileşime girdiğinin özgül fonksiyonel formu oldukça keyfidir. Farklı etkileşim türlerini modellemek için yeterli esnekliğe sahip olduğumuz sürece bu büyük bir sorun değildir. Bununla birlikte, tek bir katman ile, bu oldukça zor olabilir. Doğrusal modellerde, daha fazla katman ekleyerek bu sorunu düzelttik. RNN'ler içinde bu biraz daha zor, çünkü öncelikle ek doğrusal olmayanlığın nasıl ve nerede ekleneceğine karar vermemiz gerekiyor.

Aslında, birden fazla RNN katmanını üst üste yığabiliriz. Bu, birkaç basit katmanın kombinasyonu sayesinde esnek bir mekanizma ile sonuçlanır. Özellikle, veriler yığının farklı düzeylerinde alakalı olabilir. Örneğin, finansal piyasa koşulları (hisse alım ve satım piyasası) ile ilgili üst düzey verileri elimizde mevcut tutmak isteyebiliriz, ancak daha düşük bir seviyede yalnızca kısa vadeli zamansal dinamikleri kaydederiz.

Yukarıdaki soyut tartışmaların ötesinde, :numref:`fig_deep_rnn`'ü inceleyerek ilgilendiğimiz model ailesini anlamak muhtemelen en kolay yoldur. $L$ adet gizli katmanlı derin bir RNN'yı göstermektedir. Her gizli durum, hem geçerli katmanın bir sonraki zaman adımına hem de bir sonraki katmanın şu anki zaman adımına sürekli olarak iletilir.

![Derin RNN mimarisi.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

## Fonksiyonel Bağlılıklar

:numref:`fig_deep_rnn`'te tasvir edilen $L$ gizli katmanın derin mimarisi içindeki işlevsel bağımlıkları resmileştirebiliriz. Aşağıdaki tartışmamız öncelikle sıradan RNN modeline odaklanmaktadır, ancak diğer dizi modelleri için de geçerlidir.

Bir $t$ zaman adımında $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ minigrup girdisine sahip olduğumuzu varsayalım (örnek sayısı: $n$, her örnekte girdi sayısı: $d$). Aynı zamanda adımda, $l^\mathrm{th}$ gizli katmanın ($l=1,\ldots,L$), $\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$ gizli durumunun (gizli birimlerin sayısı: $h$) ve $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ çıktı katmanı değişkeninin (çıktı sayısı: $q$) olduğunu varsayalım. $\mathbf{H}_t^{(0)} = \mathbf{X}_t$, $\phi_l$ etkinleştirme işlevini kullanan $l^\mathrm{th}$ gizli katmanın gizli durumu aşağıdaki gibi ifade edilir:

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

burada $\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$ ve $\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$ ağırlıkları $\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$ ekgirdisi ile birlikte $l^\mathrm{th}$ gizli katmanın model parametreleridir.

Sonunda, çıktı katmanının hesaplanması yalnızca son $L$. gizli katmanın gizli durumuna dayanır:

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

burada ağırlık $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ ve ek girdi $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$, çıktı katmanının model parametreleridir.

MLP'lerde olduğu gibi, $L$ gizli katmanların sayısı ve $h$ gizli birimlerin sayısı hiperparametrelerdir. Başka bir deyişle, bunlar bizim tarafımızdan ayarlanabilir veya belirtilebilir. Buna ek olarak, :eqref:`eq_deep_rnn_H`'teki gizli durum hesaplamalarını GRU veya LSTM ile değiştirerek bir derin geçitli RNN elde edebiliriz.

## Kısa Uygulama

Neyse ki, bir RNN'nin birden fazla katmanını uygulamak için gerekli olan lojistik detayların çoğu, üst düzey API'lerde kolayca mevcuttur. İşleri basit tutmak için sadece bu tür yerleşik işlevleri kullanarak uygulamayı gösteriyoruz. Örnek olarak bir LSTM modelini ele alalım. Kod, daha önce :numref:`sec_lstm`'te kullandığımıza çok benzer. Aslında, tek fark, tek bir katmanlı varsayılanı seçmek yerine açıkça katman sayısını belirtmemizdir. Her zamanki gibi, veri kümesini yükleyerek başlıyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

Hiperparametre seçimi gibi mimari kararlar :numref:`sec_lstm`'tekine çok benzer. Farklı andıçlara sahip olduğumuz için aynı sayıda girdi ve çıktı seçiyoruz, yani `vocab_size`. Gizli birimlerin sayısı hala 256'dır. Tek fark, şimdi apaçık olmayan gizli katmanların sayısını `num_layers` değerini belirterek seçmemizdir.

```{.python .input}
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
device = d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

## Eğitim ve Tahmin

Şu andan itibaren LSTM modeli ile iki katman oluşturuyoruz, bu oldukça karmaşık mimari eğitimi önemli ölçüde yavaşlatıyor.

```{.python .input}
#@tab all
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Özet

* Derin RNN'lerde, gizli durum bilgileri şu anki katmanın sonraki zaman adımına ve sonraki katmanın şu anki zaman adımına geçirilir.
* LSTM'ler, GRU'lar veya sıradan RNN gibi derin RNN'lerin birçok farklı seçenekleri vardır. Bu modellerin tümü, derin öğrenme çerçevesinin üst düzey API'lerinin bir parçası olarak mevcuttur.
* Modellerin ilklenmesi dikkat gerektirir. Genel olarak, derin RNN'ler doğru yakınsamayı sağlamak için önemli miktarda iş gerektirir (öğrenme hızı ve kırpma gibi).

## Alıştırmalar

1. :numref:`sec_rnn_scratch`'te tartıştığımız tek katmanlı uygulamayı kullanarak sıfırdan iki katmanlı bir RNN uygulamaya çalışın.
2. LSTM'yi GRU ile değiştirin ve doğruluk ve eğitim hızını karşılaştırın.
3. Eğitim verilerini birden fazla kitap içerecek şekilde artırın. Şaşkınlık ölçeğinde ne kadar düşüğe inebilirsin?
4. Metni modellerken farklı yazarların kaynaklarını birleştirmek ister misiniz? Bu neden iyi bir fikir? Ne ters gidebilir ki?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1058)
:end_tab:
