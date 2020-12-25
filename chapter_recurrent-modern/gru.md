# Geçitli Yinelemeli Birimler (GRU)
:label:`sec_gru`

:numref:`sec_bptt`'te, gradyanların RNN'lerde nasıl hesaplandığını tartıştık. Özellikle matrislerin uzun çarpımlarının kaybolan veya patlayan gradyanlara yol açabileceğini bulduk. Bu tür gradyan sıradışılıklarının pratikte ne anlama geldiğini hakkında kısaca düşünelim:

* Gelecekteki tüm gözlemleri tahmin etmek için erken bir gözlemin son derece önemli olduğu bir durumla karşılaşabiliriz. Biraz karmaşık bir durumu düşünün; ilk gözlem bir sağlama toplamı (checksum) içerir ve hedef de sağlama toplamının dizinin sonunda doğru olup olmadığını fark etmektir. Bu durumda, ilk andıcın etkisi hayati önem taşır. Hayati önem taşıyan erken bilgileri bir *bellek hücresinde* depolamak için bazı mekanizmalara sahip olmak isteriz. Böyle bir mekanizma olmadan, bu gözlemlere çok büyük bir gradyan atamak zorunda kalacağız, çünkü sonraki tüm gözlemleri etkilerler.
* Bazı andıçların uygun gözlem taşımadığı durumlarla karşılaşabiliriz. Örneğin, bir web sayfasını ayrıştırırken, sayfada iletilen duygunun değerlendirilmesi amacıyla alakasız olan yardımcı HTML kodu olabilir. Gizli durum temsilinde bu tür andıçları *atlamak* için birtakım mekanizmaya sahip olmak isteriz.
* Bir dizinin parçaları arasında mantıksal bir kırılma olduğu durumlarla karşılaşabiliriz. Örneğin, bir kitaptaki bölümler arasında bir geçiş veya menkul kıymetler piyasasında hisse değerleri arasında bir geçiş olabilir. Bu durumda iç durum temsilimizi *sıfırlamak* için bir araca sahip olmak güzel olurdu.

Bunu ele almak için bir dizi yöntem önerilmiştir. İlk öncülerden biri :numref:`sec_lstm`'de tartışacağımız uzun ömürlü kısa-dönem belleğidir :cite:`Hochreiter.Schmidhuber.1997`. Geçitli yinelemeli birim (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`, genellikle benzer performans sunan ve :cite:`Chung.Gulcehre.Cho.ea.2014` hesaplanmanın önemli ölçüde daha hızlı olduğu biraz daha elverişli bir türdür. Sadeliğinden dolayı, GRU ile başlayalım.

## Geçitli Gizli Durum

Sıradan RNN ve GRU'lar arasındaki anahtar ayrım, ikincisinin gizli durumu geçitlemeyi desteklemesidir. Bu, gizli bir durumun *güncellenmesi* gerektiği zamanlara ve ayrıca *sıfırlanması* gerektiği zamanlara yönelik özel mekanizmalarımız olduğu anlamına gelir. Bu mekanizmalar öğrenilir ve yukarıda listelenen kaygıları ele alır. Örneğin, ilk andıç büyük önem taşıyorsa, ilk gözlemden sonra gizli durumu güncellememeyi öğreneceğiz. Aynı şekilde, ilgisiz geçici gözlemleri atlamayı öğreneceğiz. Son olarak, gerektiğinde gizli durumu sıfırlamayı öğreneceğiz. Bunları aşağıda ayrıntılı olarak tartışıyoruz.

### Sıfırlama Geçidi ve Güncelleme Geçidi

Tanışmamız gereken ilk kavramlar, *sıfırlama geçidi* ve *güncelleme geçidi*dir. Onları $(0, 1)$'te girdileri olan vektörler olacak şekilde tasarlıyoruz, böylece dışbükey bileşimleri gerçekleştirebiliriz. Örneğin, bir sıfırlama geçidi, önceki durumun ne kadarını hala hatırlamak isteyebileceğimizi kontrol etmemizi sağlar. Aynı şekilde, bir güncelleme geçidi yeni durumun ne kadarının eski durumun bir kopyası olacağını kontrol etmemizi sağlayacaktır.

Bu kapıları mühendisleştirerek başlıyoruz. :numref:`fig_gru_1`, mevcut zaman adımının girişi ve önceki zaman adımının gizli durumu göz önüne alındığında, bir GRU'daki hem sıfırlama hem de güncelleme kapıları için girişleri göstermektedir. İki kapının çıkışları, sigmoid aktivasyon işlevine sahip iki tam bağlı katman tarafından verilir.

![Computing the reset gate and the update gate in a GRU model.](../img/gru-1.svg)
:label:`fig_gru_1`

Matematiksel olarak, belirli bir zaman adımı $t$ için, girişin bir minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (örnek sayısı: $n$, giriş sayısı: $d$) olduğunu varsayalım ve önceki zaman adımının gizli durumu $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (gizli birimlerin sayısı: $h$). Daha sonra, sıfırlama kapısı $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ ve güncelleştirme kapısı $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ aşağıdaki gibi hesaplanır:

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

burada $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ ve $\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ ağırlık parametreleridir ve $\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$ önyargılardır. Yayının (bkz. :numref:`subsec_broadcasting`) toplamı sırasında tetiklendiğini unutmayın. Giriş değerlerini $(0, 1)$ aralığına dönüştürmek için sigmoid işlevleri (:numref:`sec_mlp`'te tanıtıldığı gibi) kullanıyoruz.

### Aday Gizli Devlet

Ardından, $\mathbf{R}_t$'i sıfırlama kapısını :eqref:`rnn_h_with_state`'teki normal gizli durum güncelleme mekanizmasıyla entegre edelim. Aşağıdakilere yol açar
*aday gizli durum*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ zaman adımında $t$:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
:eqlabel:`gru_tilde_H`

burada $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ ve $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ ağırlık parametreleridir, $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ önyargı ve $\odot$ sembolü Hadamard (elementwise) ürün operatörüdür. Burada, aday gizli durumdaki değerlerin $(-1, 1)$ aralığında kalmasını sağlamak için tanh şeklinde bir doğrusal olmayan bir özellik kullanıyoruz.

Sonuç, güncelleme kapısının eylemini dahil etmemiz gerektiğinden bir *aday* oldu. :eqref:`rnn_h_with_state` ile karşılaştırıldığında, şimdi önceki durumların etkisi $\mathbf{R}_t$ ve $\mathbf{H}_{t-1}$'nin :eqref:`gru_tilde_H`'te elementsel çarpımı ile azaltılabilir. Sıfırlama kapısı $\mathbf{R}_t$ girişleri 1'e yakın olduğunda, :eqref:`rnn_h_with_state`'te olduğu gibi bir vanilya RNN kurtarırız. Sıfırlama kapısı $\mathbf{R}_t$'nın 0'a yakın olan tüm girişleri için, aday gizli durum, giriş olarak $\mathbf{X}_t$ olan bir MLP'nin sonucudur. Önceden var olan herhangi bir gizli durum böylece*reset* varsayılanlara ayarlanır.

:numref:`fig_gru_2`, sıfırlama kapısını uyguladıktan sonra hesaplama akışını gösterir.

![Computing the candidate hidden state in a GRU model.](../img/gru-2.svg)
:label:`fig_gru_2`

### Gizli Devlet

Son olarak, güncelleme kapısının etkisini dahil etmemiz gerekiyor $\mathbf{Z}_t$. Bu, yeni gizli devlet $\mathbf{H}_t \in \mathbb{R}^{n \times h}$'in sadece eski devlet $\mathbf{H}_{t-1}$ ve yeni aday devletin $\tilde{\mathbf{H}}_t$'ün ne kadar kullanıldığını belirler. $\mathbf{Z}_t$ güncelleme kapısı bu amaçla kullanılabilir, sadece hem $\mathbf{H}_{t-1}$ hem de $\tilde{\mathbf{H}}_t$ arasındaki elementsel dışbükey kombinasyonları alarak kullanılabilir. Bu, GRU için son güncelleştirme denklemine yol açar:

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

Güncelleme kapısı $\mathbf{Z}_t$ 1'e yakın olduğunda, sadece eski durumu koruruz. Bu durumda $\mathbf{X}_t$'den gelen bilgiler esas olarak göz ardı edilir, bağımlılık zincirinde $t$ zaman adımını etkin bir şekilde atlanır. Buna karşılık, $\mathbf{Z}_t$ 0'a yakın olduğunda, yeni gizli durum $\mathbf{H}_t$ aday gizli devlet $\tilde{\mathbf{H}}_t$'ye yaklaşır. Bu tasarımlar, RNN'lerdeki kaybolan degrade problemiyle başa çıkmamıza ve büyük zaman adım mesafeleri olan diziler için daha iyi yakalama bağımlılıklarıyla başa çıkmamıza yardımcı olabilir. Örneğin, güncelleme kapısı, tüm bir alt dizinin tüm zaman adımları için 1'e yakınsa, başlangıç zamanındaki eski gizli durum, alt sıranın uzunluğuna bakılmaksızın kolayca korunur ve sonuna kadar geçirilir.

:numref:`fig_gru_3`, güncelleme kapısı harekete geçtikten sonra hesaplama akışını gösterir.

![Computing the hidden state in a GRU model.](../img/gru-3.svg)
:label:`fig_gru_3`

Özetle, GRU'lar aşağıdaki iki ayırt edici özelliğe sahiptir:

* Sıfırlama kapıları dizilerdeki kısa vadeli bağımlılıkları yakalamaya yardımcı olur.
* Güncelleme kapıları dizilerdeki uzun vadeli bağımlılıkları yakalamaya yardımcı olur.

## Sıfırdan Uygulama

GRU modelini daha iyi anlamak için, sıfırdan uygulamamıza izin verin. :numref:`sec_rnn_scratch`'te kullandığımız zaman makinesi veri kümesini okuyarak başlıyoruz. Veri kümesini okuma kodu aşağıda verilmiştir.

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

### Model Parametrelerini Başlatma

Bir sonraki adım model parametrelerini başlatmaktır. Ağırlıkları standart sapma ile bir Gauss dağılımından 0.01 olarak çiziyoruz ve önyargı 0'a ayarlıyoruz. Hiperparametre `num_hiddens`, gizli birimlerin sayısını tanımlar. Güncelleme kapısı, sıfırlama kapısı, aday gizli durumu ve çıkış katmanı ile ilgili tüm ağırlıkları ve önyargıları başlatacağız.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### Modelin Tanımlanması

Şimdi gizli durum başlatma işlevini tanımlayacağız `init_gru_state`. :numref:`sec_rnn_scratch`'te tanımlanan `init_rnn_state` işlevi gibi, bu işlev, değerleri sıfırlar olan bir şekle (toplu boyut, gizli birim sayısı) sahip bir tensör döndürür.

```{.python .input}
def init_gru_state(batch_size, num_hiddens, device):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

Şimdi GRU modelini tanımlamaya hazırız. Yapısı, güncelleme denklemlerinin daha karmaşık olması dışında, temel RNN hücresinin yapısı ile aynıdır.

```{.python .input}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

### Eğitim ve Tahmin

Eğitim ve öngörü, :numref:`sec_rnn_scratch`'teki gibi tam olarak aynı şekilde çalışır. Eğitimden sonra, sırasıyla sağlanan “zaman yolcusu” ve “yolcu” ön eklerini takip eden eğitim setindeki şaşkınlığı ve tahmin edilen diziyi yazdırırız.

```{.python .input}
#@tab all
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Özlü Uygulama

Üst düzey API'lerde, doğrudan bir GPU modelini oluşturabiliriz. Bu, yukarıda açıkça yaptığımız tüm yapılandırma ayrıntılarını kapsüller. Kod, daha önce yazdığımız birçok ayrıntı için Python yerine derlenmiş operatörleri kullandığı için önemli ölçüde daha hızlıdır.

```{.python .input}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Özet

* Geçmeli RNN'ler, büyük zaman adım mesafeleri olan diziler için bağımlılıkları daha iyi yakalayabilir.
* Sıfırlama kapıları dizilerdeki kısa vadeli bağımlılıkları yakalamaya yardımcı olur.
* Güncelleme kapıları dizilerdeki uzun vadeli bağımlılıkları yakalamaya yardımcı olur.
* GRU'lar, sıfırlama kapısı açıldığında aşırı durum olarak temel RNN'leri içerir. Ayrıca güncelleme kapısını açarak sonradan atlayabilirler.

## Alıştırmalar

1. Zaman adımında $t'$ zaman adım $t > t'$ çıktısını tahmin etmek için girdiyi kullanmak istediğimizi varsayalım. Her zaman adım için sıfırlama ve güncelleme kapıları için en iyi değerler nelerdir?
1. Hiperparametreleri ayarlayın ve çalışma süresi, şaşkınlık ve çıktı dizisi üzerindeki etkilerini analiz edin.
1. `rnn.RNN` ve `rnn.GRU` uygulamaları için çalışma zamanı, şaşkınlık ve çıkış dizelerini birbirleriyle karşılaştırın.
1. Yalnızca bir GRU'nun parçalarını, örneğin yalnızca bir sıfırlama kapısı veya yalnızca bir güncelleme kapısı ile uygularsanız ne olur?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1056)
:end_tab:
