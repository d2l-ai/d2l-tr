# Uzun Ömürlü Kısa-Dönem Belleği (LSTM)
:label:`sec_lstm`

Gizli değişkenli modellerde uzun vadeli bilgi koruma ve kısa vadeli girdi atlama sorunu uzun zamandır var olmuştur. Bunu ele almak için öncü yaklaşımlardan biri uzun ömürlü kısa-dönem belleği (LSTM) :cite:`Hochreiter.Schmidhuber.1997` oldu. GRU'nun birçok özelliklerini paylaşıyor. İlginçtir ki, LSTM'ler GRU'lardan biraz daha karmaşık bir tasarıma sahiptir, ancak GRU'lardan neredeyse yirmi yıl önce ortaya konmuştur.

## Geçitli Bellek Hücresi

Muhtemelen LSTM'nin tasarımı bir bilgisayarın mantık kapılarından esinlenilmiştir. LSTM, gizli durumla aynı şekle sahip bir *bellek hücresi* (veya kısaca *hücre*) tanıtır (bazı çalışmalar, bellek hücresini gizli durumun özel bir türü olarak görür), ki ek bilgileri kaydetmek için tasarlanmıştır. Hafıza hücresini kontrol etmek için birkaç geçide ihtiyacımız vardır. Hücreden girdileri okumak için bir geçit gerekiyor. Biz buna *çıktı geçidi* olarak atıfta bulunacağız. Hücreye veri ne zaman okunacağına karar vermek için ikinci bir geçit gereklidir. Bunu *girdi geçidi* olarak adlandırıyoruz. Son olarak, *unutma geçidi* tarafından yönetilen hücrenin içeriğini sıfırlamak için bir mekanizmaya ihtiyacımız var. Böyle bir tasarımın motivasyonu GRU'larla aynıdır, yani özel bir mekanizma aracılığıyla gizli durumdaki girdileri ne zaman hatırlayacağınıza ve ne zaman gözardı edeceğinize karar verebilmek. Bunun pratikte nasıl çalıştığını görelim.

### Girdi Geçidi, Unutma Geçidi ve Çıktı Geçidi

Tıpkı GRU'larda olduğu gibi, LSTM kapılarına beslenen veriler, :numref:`lstm_0`'te gösterildiği gibi, geçerli zaman adımındaki giriş ve önceki zaman adımının gizli durumudur. Giriş, unutma. ve çıkış kapıları değerlerini hesaplamak için sigmoid aktivasyon fonksiyonuna sahip üç tam bağlı katman tarafından işlenir. Sonuç olarak, üç kapının değerleri $(0, 1)$ aralığındadır.

![Computing the input gate, the forget gate, and the output gate in an LSTM model.](../img/lstm-0.svg)
:label:`lstm_0`

Matematiksel olarak, $h$ gizli birimler olduğunu varsayalım, toplu iş boyutu $n$ ve giriş sayısı $d$. Böylece, giriş $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ ve önceki zaman adımının gizli durumu $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$'dir. Buna göre, $t$ zaman adımındaki kapılar şu şekilde tanımlanır: giriş kapısı $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, unut kapısı $\mathbf{F}_t \in \mathbb{R}^{n \times h}$ ve çıkış kapısı $\mathbf{O}_t \in \mathbb{R}^{n \times h}$'dır. Bunlar aşağıdaki gibi hesaplanır:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

burada $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ ve $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$ ağırlık parametreleridir ve $\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$ önyargı parametreleridir.

### Aday Bellek Hücresi

Sonra hafıza hücresini tasarlıyoruz. Çeşitli kapıların eylemini henüz belirtmediğimizden, öncelikle *candidate* bellek hücresini $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$'i tanıtıyoruz. Hesaplaması, yukarıda açıklanan üç kapıdakine benzer, ancak aktivasyon fonksiyonu olarak $(-1, 1)$ için bir değer aralığına sahip bir $\tanh$ işlevini kullanarak. Bu, $t$ zaman adımında aşağıdaki denklem yol açar:

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$

burada $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$ ve $\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$ ağırlık parametreleridir ve $\mathbf{b}_c \in \mathbb{R}^{1 \times h}$ bir önyargı parametresidir.

Aday bellek hücresinin hızlı bir gösterimi :numref:`lstm_1`'te gösterilmiştir.

![Computing the candidate memory cell in an LSTM model.](../img/lstm-1.svg)
:label:`lstm_1`

### Bellek Hücresi

GRU'larda, girişi ve unutmayı (veya atlamayı) yönetecek bir mekanizmamız vardır. Benzer şekilde, LSTM'lerde bu tür amaçlar için iki özel kapımız var: $\mathbf{I}_t$ giriş kapısı $\tilde{\mathbf{C}}_t$ aracılığıyla yeni verileri ne kadar hesaba kattığımızı ve $\mathbf{F}_t$, eski bellek hücresi içeriğinin $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$'nin ne kadarını tuttuğumuzu kontrol eder. Daha önce olduğu gibi aynı noktasal çarpım hilesini kullanarak, aşağıdaki güncelleme denklemine ulaşırız:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

Unut kapısı her zaman yaklaşık 1 ise ve giriş kapısı her zaman yaklaşık 0 ise, geçmiş bellek hücreleri $\mathbf{C}_{t-1}$ zamanla kaydedilir ve geçerli zaman adımına geçirilir. Bu tasarım, kaybolan degrade sorununu hafifletmek ve diziler içindeki uzun menzilli bağımlılıkları daha iyi yakalamak için tanıtıldı.

Böylece :numref:`lstm_2`'teki akış şemasına ulaşırız.

![Computing the memory cell in an LSTM model.](../img/lstm-2.svg)

:label:`lstm_2`

### Gizli Durum

Son olarak, gizli durumu nasıl hesaplayacağımızı tanımlamamız gerekiyor $\mathbf{H}_t \in \mathbb{R}^{n \times h}$. Çıkış kapısının devreye girdiği yer burası. LSTM'de, bellek hücresinin $\tanh$'in sadece bir kapılı versiyonudur. Bu, $\mathbf{H}_t$ değerlerinin her zaman $(-1, 1)$ aralığında olmasını sağlar.

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

Çıkış kapısı 1'e yaklaştığında, tüm bellek bilgilerini etkin bir şekilde öngörüye aktarırız, oysa 0'a yakın çıkış kapısı için tüm bilgileri yalnızca bellek hücresinde saklarız ve daha fazla işlem yapmayız.

:numref:`lstm_3`, veri akışının grafiksel bir resme sahiptir.

![Computing the hidden state in an LSTM model.](../img/lstm-3.svg)
:label:`lstm_3`

## Sıfırdan Uygulama

Şimdi sıfırdan bir LSTM uygulayalım. :numref:`sec_rnn_scratch`'teki deneylerle aynı, önce zaman makinesi veri kümesini yükleriz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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

### Model Parametrelerini İlkleme

Daha sonra model parametrelerini tanımlamamız ve başlatmamız gerekiyor. Daha önce olduğu gibi, hiperparametre `num_hiddens` gizli birimlerin sayısını tanımlar. 0.01 standart sapma ile Gauss dağılımını takiben ağırlıkları başlatırız ve önyargıları 0'a ayarlarız.

```{.python .input}
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### Modelin Tanımlanması

Başlatma işlevinde, LSTM'nin gizli durumunun 0 değeri ve şekli (toplu iş boyutu, gizli birimlerin sayısı) olan bir *ekleme* bellek hücresi döndürmesi gerekir. Bu nedenle aşağıdaki devlet başlatma olsun.

```{.python .input}
def init_lstm_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device),
            np.zeros((batch_size, num_hiddens), ctx=device))
```

```{.python .input}
#@tab pytorch
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

Gerçek model, daha önce tartıştığımız gibi tanımlanmıştır: üç kapı ve bir yardımcı bellek hücresi sağlama. Çıktı katmanına yalnızca gizli durumun iletildiğini unutmayın. Bellek hücresi $\mathbf{C}_t$ doğrudan çıktı hesaplama katılmaz.

```{.python .input}
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * np.tanh(C)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)
```

```{.python .input}
#@tab pytorch
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

### Eğitim ve Tahmin

:numref:`sec_rnn_scratch`'te tanıtılan `RNNModelScratch` sınıfını başlatarak :numref:`sec_gru`'te yaptığımız gibi bir LSTM'yi eğitmemize izin verin.

```{.python .input}
#@tab all
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Kısa Uygulama

Üst düzey API'leri kullanarak doğrudan bir `LSTM` modeli oluşturabiliriz. Bu, yukarıda açıkça yaptığımız tüm yapılandırma ayrıntılarını kapsüller. Daha önce ayrıntılı olarak yazdığımız birçok ayrıntı için Python yerine derlenmiş operatörleri kullandığı için kod önemli ölçüde daha hızlıdır.

```{.python .input}
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

LSTM'ler, önemsiz olmayan durum kontrolü ile prototipik latent değişken otoregresif modeldir. Birçok türevleri yıllar içinde önerilmiştir, örn. birden fazla katman, artık bağlantılar, farklı düzenlilik türleri. Bununla birlikte, eğitim LSTM'leri ve diğer dizi modelleri (GRU'lar gibi), dizinin uzun menzilli bağımlılığı nedeniyle oldukça maliyetlidir. Daha sonra bazı durumlarda kullanılabilen Transformers gibi alternatif modellerle karşılaşacağız.

## Özet

* LSTM'lerin üç tip kapıları vardır: giriş kapıları, unutma kapıları ve bilgi akışını kontrol eden çıkış kapıları.
* Gizli katman çıktısı LSTM gizli durumu ve bellek hücresini içerir. Çıktı katmanına yalnızca gizli durum iletilir. Hafıza hücresi tamamen içsel.
* LSTM'ler kaybolan ve patlayan degradeleri hafifletebilir.

## Alıştırmalar

1. Hiperparametreleri ayarlayın ve çalışma süresi, şaşkınlık ve çıktı dizisi üzerindeki etkilerini analiz edin.
1. Karakter dizileri aksine doğru kelimeleri üretmek için modeli nasıl değiştirmeniz gerekir?
1. Belirli bir gizli boyut için GRU'lar, LSTM'ler ve normal RNN'ler için hesaplama maliyetini karşılaştırın. Eğitim ve çıkarım maliyetine özel dikkat gösterin.
1. Aday bellek hücresi $-1$ ve $1$ $\tanh$ işlevini kullanarak değer aralığının $-1$ ve $1$ arasında olduğundan emin olmak için $\tanh$ işlevini yeniden kullanmak neden gizli durum gerekiyor?
1. Karakter sırası tahmini yerine zaman serisi tahmini için bir LSTM modeli uygulayın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1057)
:end_tab:
