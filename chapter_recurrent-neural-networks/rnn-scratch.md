# Yinelemeli Sinir Ağlarının Sıfırdan Uygulanması
:label:`sec_rnn_scratch`

Bu bölümde, :numref:`sec_rnn`'deki açıklamalarımıza göre, karakter düzeyinde bir dil modeli için sıfırdan bir RNN uygulayacağız. Bu bir model H. G. Wells'in *Zaman Makinesi* ile eğitilecektir. Daha önce olduğu gibi, önce :numref:`sec_language_model`'te tanıtılan veri kümesini okuyarak başlıyoruz.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## Bire Bir Kodlama

Her andıcın `train_iter`'te sayısal bir indeks olarak temsil edildiğini hatırlayın. Bu indeksleri doğrudan sinir ağına beslemek öğrenmeyi zorlaştırabilir. Genellikle her andıcı daha açıklayıcı bir öznitelik vektörü olarak temsil ediyoruz. En kolay gösterim, :numref:`subsec_classification-problem`'te tanıtılan *bire bir kodlama*dır (one-hot coding).

Özetle, her bir indeksi farklı bir birim vektörüne eşleriz: Kelime dağarcığındaki farklı andıçların sayısının $N$ (`len(vocab)`) olduğunu ve andıç indekslerinin 0 ile $N-1$ arasında değiştiğini varsayalım. Bir andıç indeksi $i$ tamsayısı ise, o zaman $N$ uzunluğunda tümü 0 olan bir vektör oluştururuz ve $i$ konumundaki elemanı 1'e ayarlarız. Bu vektör, orijinal andıcin bire bir kodlama vektörüdür. İndeksleri 0 ve 2 olan bire bir kodlama vektörleri aşağıda gösterilmiştir.

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

Her seferinde örnek aldığımız minigrubun şekli (grup boyutu, zaman adımlarının sayısı)'dır. `one_hot` işlevi, böyle bir minigrubu, son boyutun kelime dağarcığı uzunluğuna eşit olduğu üç boyutlu bir tensöre dönüştürür (`len(vocab)`). Girdiyi sıklıkla deviriyoruz (transpose), böylece (zaman adımlarının sayısı, grup boyutu, kelime dağarcığı uzunluğu) şeklinde bir çıktı elde edeceğiz. Bu, bir minigrubun gizli durumlarını zaman adımlarıyla güncellerken en dıştaki boyutta daha rahat döngü yapmamızı sağlayacaktır.

```{.python .input}
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

## Model Parametrelerini İlkleme

Ardından, RNN modeli için model parametrelerini ilkliyoruz. Gizli birimlerin sayısı `num_hiddens` ayarlanabilir bir hiperparametredir. Dil modellerini eğitirken, girdiler ve çıktılar aynı kelime dağarcığındandır. Bu nedenle, kelime dağarcığı uzunluğuna eşit olan aynı boyuta sahiptirler.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    # Hidden layer parameters
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

## RNN Modeli

Bir RNN modeli tanımlamak için, ilk olarak ilkleme esnasında gizli durumu döndürmek için bir `init_rnn_state` işlevi gerekir. Şekli (grup boyutu, gizli birimlerin sayısı) olan 0 ile doldurulmuş bir tensör döndürür. Çokuzlu (tuple) kullanmak, daha sonraki bölümlerde karşılaşacağımız gibi gizli durumun birden çok değişken içerdiği halleri işlemeyi kolaylaştırır.

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

Aşağıdaki `rnn` işlevi, gizli durumu ve çıktıyı bir zaman adımında nasıl hesaplayacağınızı tanımlar. RNN modelinin `inputs` değişkeninin en dış boyutunda döngü yaptığını, böylece minigrubun gizli durumları `H`'nin her zaman adımında güncellendiğini unutmayın. Ayrıca, burada etkinleştirme fonksiyonu olarak $\tanh$ işlevi kullanılır. :numref:`sec_mlp`'te açıklandığı gibi, $\tanh$ işlevinin ortalama değeri, elemanlar gerçel sayılar üzerinde eşit olarak dağıtıldığında 0'dır.

```{.python .input}
def rnn(inputs, state, params):
    # Shape of `inputs`: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

Gerekli tüm işlevler tanımlandıktan sonra, bu işlevleri sarmalamak ve sıfırdan uygulanan bir RNN modelinin parametreleri depolamak için bir sınıf oluşturuyoruz.

```{.python .input}
class RNNModelScratch:  #@save
    """An RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input}
#@tab tensorflow
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state, params):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)
```

Çıktıların doğru şekillere sahip olup olmadığını kontrol edelim, örn. gizli durumun boyutunun değişmeden kalmasını sağlamak için.

```{.python .input}
#@tab mxnet
num_hiddens = 512
model = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                        init_rnn_state, rnn)
state = model.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = model(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
model = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                        init_rnn_state, rnn)
state = model.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = model(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab tensorflow
# defining tensorflow training strategy
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    model = RNNModelScratch(len(vocab), num_hiddens, 
                            init_rnn_state, rnn)
state = model.begin_state(X.shape[0])
params = get_params(len(vocab), num_hiddens)
Y, new_state = model(X, state, params)
Y.shape, len(new_state), new_state[0].shape
```

Çıktı şekli (zaman sayısı $\times$ grup boyutu, kelime dağarcığı uzunluğu) iken, gizli durum şeklinin aynı kaldığını, yani (grup boyutu, gizli birimlerin sayısı) olduğunu görebiliriz.

## Tahmin

Önce kullanıcı tarafından sağlanan `prefix`'i (önek), ki birkaç karakter içeren bir dizgidir, takip eden yeni karakterler oluşturmak için tahmin işlevi tanımlayalım. `prefix`'teki bu başlangıç karakterleri arasında döngü yaparken, herhangi bir çıktı oluşturmadan gizli durumu bir sonraki adımına geçirmeye devam ediyoruz. Buna, modelin kendisini güncellediği (örn. gizli durumu güncellediği) ancak tahminlerde bulunmadığı *ısınma* dönemi denir. Isınma döneminden sonra, gizli durum genellikle başlangıçtaki ilkleme değerinden daha iyidir. Artık tahmin edilen karakterleri oluşturup yayarız.

```{.python .input}
def predict_ch8(prefix, num_preds, model, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = model.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_preds, model, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab tensorflow
def predict_ch8(prefix, num_preds, model, vocab, params):  #@save
    """Generate new characters following the `prefix`."""
    state = model.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), (1, 1)).numpy()
    for y in prefix[1:]:  # Warm-up period
        _, state = model(get_input(), state, params)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = model(get_input(), state, params)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Şimdi `predict_ch8` işlevini test edebiliriz. Öneki `time traveller ` olarak belirtiyoruz ve 10 ek karakter üretiyoruz. Ağı eğitmediğimiz göz önüne alındığında, saçma tahminler üretecektir.

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, model, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, model, vocab, params)
```

## Gradyan Kırpma

$T$ uzunluğunda bir dizi için, bir yinelemede bu $T$ zaman adımlarının üzerindeki gradyanları hesaplarız, bu da geri yayma sırasında $\mathcal{O}(T)$ uzunluğunda bir matris çarpımları zincirine neden olur. :numref:`sec_numerical_stability`'te belirtildiği gibi, bu da sayısal kararsızlığa neden olabilir, örneğin $T$ büyük olduğunda gradyanlar patlayabilir veya kaybolabilir. Bu nedenle, RNN modelleri genellikle eğitimi kararlı tutmak için ekstra yardıma ihtiyaç duyar.

Genel olarak, bir eniyileme problemini çözerken, model parametresi için güncelleme adımları atıyoruz, mesela $\mathbf{x}$ vektör formunda, bir minigrup üzerinden negatif gradyan $\mathbf{g}$ yönünde. Örneğin, öğrenme oranı $\eta > 0$ ise, bir yinelemede $\mathbf{x}$'i $\mathbf{x} - \eta \mathbf{g}$ olarak güncelleriz. $f$ amaç fonksiyonunun iyi davrandığını varsayalım, diyelim ki, $L$ sabiti ile *Lipschitz sürekli* olsun. Yani, herhangi bir $\mathbf{x}$ ve $\mathbf{y}$ için:

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

Bu durumda, parametre vektörünü $\eta \mathbf{g}$ ile güncelledğimizi varsayarsak, o zaman

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

demektir ki $L \eta \|\mathbf{g}\|$'dan daha büyük bir değişiklik gözlemlemeyeceğiz. Bu hem bir lanet hem de bir nimettir. Lanet tarafında, ilerleme kaydetme hızını sınırlar; oysa nimet tarafında, yanlış yönde hareket edersek işlerin ne ölçüde ters gidebileceğini sınırlar.

Bazen gradyanlar oldukça büyük olabilir ve eniyileme algoritması yakınsamada başarısız olabilir. Öğrenme hızı $\eta$'yı azaltarak bunu ele alabiliriz. Ama ya sadece nadiren büyük gradyanlar alırsak? Bu durumda böyle bir yaklaşım tamamen yersiz görünebilir. Popüler bir alternatif, $\mathbf{g}$ gradyanını belirli bir yarıçaptaki bir küreye geri yansıtmaktır, diyelim ki $\theta$ için:

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

Bunu yaparak, gradyanın normunun $\theta$'yı asla geçmediğini ve güncellenen gradyanın $\mathbf{g}$'nin orijinal yönüyle tamamen hizalandığını biliyoruz. Ayrıca, herhangi bir minigrubun (ve içindeki herhangi bir örneklemin) parametre vektörü üzerinde uygulayabileceği etkiyi sınırlaması gibi arzu edilen yan etkiye sahiptir. Bu, modele belirli bir derecede gürbüzlük kazandırır. Gradyan kırpma gradyan patlaması için hızlı bir düzeltme sağlar. Sorunu tamamen çözmese de, onu hafifletmede kullanılan birçok teknikten biridir.

Aşağıda, sıfırdan uygulanan veya üst düzey API'ler tarafından oluşturulan bir modelin gradyanlarını kırpmak için bir işlev tanımlıyoruz. Ayrıca, tüm model parametrelerinin üzerinden gradyan normunu hesapladığımızı unutmayın.

```{.python .input}
def grad_clipping(model, theta):  #@save
    """Clip the gradient."""
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(model, theta):  #@save
    """Clip the gradient."""
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab tensorflow
def grad_clipping(grads, theta): #@save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in grads))
    norm = tf.cast(norm, tf.float32)
    new_grad = []
    if tf.greater(norm, theta):
        for grad in grads:
            new_grad.append(grad * theta / norm)
    else:
        for grad in grads:
            new_grad.append(grad)
    return new_grad
```

## Eğitim

Modeli eğitmeden önce, modeli bir dönemde eğitmek için bir işlev tanımlayalım. :numref:`sec_softmax_scratch`'teki model eğitimimizden üç yerde farklılık gösterir:

1. Dizili veriler için farklı örnekleme yöntemleri (rastgele örnekleme ve sıralı bölümleme), gizli durumların ilklenmesinde farklılıklara neden olacaktır.
1. Model parametrelerini güncellemeden önce gradyanları kırpıyoruz. Bu, eğitim süreci sırasında bir noktada gradyanlar patladığında bile modelin ıraksamamasını sağlar.
1. Modeli değerlendirmek için şaşkınlığı kullanıyoruz. :numref:`subsec_perplexity`'te tartışıldığı gibi, bu farklı uzunluktaki dizilerin karşılaştırılabilir olmasını sağlar.

Özellikle, sıralı bölümleme kullanıldığında, gizli durumu yalnızca her dönemin başında ilkleriz. Sonraki minigruptaki $i$. altdizi örneği şimdiki $i$. altdizi örneği bitişik olduğundan, şimdiki minigrup sonundaki gizli durumu sonraki minigrup başında gizli durumu ilklemek için kullanılır. Bu şekilde, gizli durumda saklanan dizinin tarihsel bilgileri bir dönem içinde bitişik altdizilere aktarılabilir. Ancak, herhangi bir noktada gizli durumun hesaplanması, aynı dönemdeki tüm önceki minigruplara bağlıdır, ki bu da gradyan hesaplamasını zorlaştırır. Hesaplama maliyetini azaltmak için, herhangi bir minigrup işleminden önce gradyanı ayırırız, böylece gizli durumun gradyan hesaplaması her zaman minigrup içindeki zaman adımlarıyla sınırlı kalır.

Rastgele örneklemeyi kullanırken, her örnek rastgele bir konumla örneklendiğinden, her yineleme için gizli durumu yeniden ilklememiz gerekir. :numref:`sec_softmax_scratch`'teki `train_epoch_ch3` işlevi gibi, `updater` da model parametrelerini güncellemek için genel bir işlevdir. Sıfırdan uygulanan `d2l.sgd` işlevi veya derin bir öğrenme çerçevesindeki yerleşik eniyileme işlevi olabilir.

```{.python .input}
def train_epoch_ch8(model, train_iter, loss, updater, device,  #@save
                    use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = model.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = model(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab pytorch
def train_epoch_ch8(model, train_iter, loss, updater, device,  #@save
                    use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = model.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(model, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation 
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = model(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(model, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(model, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch8(model, train_iter, loss, updater,   #@save
                    params, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = model.begin_state(batch_size=X.shape[0])
        with tf.GradientTape(persistent=True) as g:
            g.watch(params)
            y_hat, state= model(X, state, params)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))
        
        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(d2l.size(y)) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

Eğitim fonksiyonu sıfırdan veya üst düzey API'ler kullanılarak uygulanan bir RNN modelini destekler.

```{.python .input}
def train_ch8(model, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(model, gluon.Block):
        model.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab pytorch
#@save
def train_ch8(model, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(model, nn.Module):
        updater = torch.optim.SGD(model.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch8(model, train_iter, vocab, num_hiddens, lr, num_epochs, strategy,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    with strategy.scope():
        params = get_params(len(vocab), num_hiddens)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, params)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
             model, train_iter, loss, updater, params, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

Şimdi RNN modelini eğitebiliriz. Veri kümesinde yalnızca 10000 andıç kullandığımızdan, modelin daha iyi yakınsaması için daha fazla döneme ihtiyacı var.

```{.python .input}
#@tab mxnet,pytorch
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, num_hiddens, lr, num_epochs, strategy)
```

Son olarak, rastgele örnekleme yöntemini kullanmanın sonuçlarını kontrol edelim.

```{.python .input}
#@tab mxnet,pytorch
train_ch8(model, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
params = get_params(len(vocab_random_iter), num_hiddens)
train_ch8(model, train_random_iter, vocab_random_iter, num_hiddens, lr,
          num_epochs, strategy, use_random_iter=True)
```

Yukarıdaki RNN modelini sıfırdan uygulamak öğretici olsa da, pek de uygun değildir. Bir sonraki bölümde, RNN modelinin nasıl geliştirileceğini göreceğiz, örneğin nasıl daha kolay uygulanmasını ve daha hızlı çalışmasını sağlarız.

## Özet

* Kullanıcı tarafından sağlanan metin önekini takip eden metin üretmek için RNN tabanlı karakter düzeyinde bir dil modeli eğitebiliriz.
* Basit bir RNN dil modeli girdi kodlama, RNN modelleme ve çıktı üretme içerir.
* RNN modellerinin eğitim için durum ilklenmesi gerekir, ancak rasgele örnekleme ve sıralı bölümleme farklı yollar kullanır.
* Sıralı bölümleme kullanırken, hesaplama maliyetini azaltmak için gradyanı ayırmamız gerekir.
* Isınma süresi, herhangi bir tahmin yapmadan önce bir modelin kendisini güncellemesine (örneğin, ilkleme değerinden daha iyi bir gizli durum elde etmesine) olanak tanır.
* Gradyan kırpma gradyan patlamasını önler, ancak kaybolan gradyanları düzeltemez.

## Alıştırmalar

1. Bire bir kodlamanın, her nesne için farklı bir gömme seçmeye eşdeğer olduğunu gösterin.
1. Şaşkınlığı iyileştirmek için hiperparametreleri (örn. dönemlerin sayısı, gizli birimlerin sayısı, bir minigruptaki zaman adımlarının sayısı ve öğrenme oranı) ayarlayın.
    * Ne kadar alçağa düşebilirsiniz?
    * Bire bir kodlamayı öğrenilebilir gömmelerle değiştirin. Bu daha iyi bir performansa yol açar mı?
    * H. G. Wells'in, örn. [*Dünyalar Savaşı*](http://www.gutenberg.org/ebooks/36) gibi, diğer kitaplarıyla ne kadar iyi çalışacaktır?
1. En olası sonraki karakteri seçmek yerine örnekleme kullanarak tahmin işlevini değiştirin.
    * Ne olur?
    * Modeli, örneğin $\alpha > 1$ için $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$'ten örnekleme yaparak daha olası çıktılara doğru yanlı yapın.
1. Gradyan kırpmadan bu bölümdeki kodu çalıştırın. Ne olur?
1. Sıralı bölümlemeyi gizli durumları hesaplama çizgesinden ayırmayacak şekilde değiştirin. Çalışma süresi değişiyor mu? Şaşkınlığa ne olur?
1. Bu bölümde kullanılan etkinleştirme işlevini ReLU ile değiştirin ve bu bölümdeki deneyleri tekrarlayın. Hala gradyan kırpmaya ihtiyacımız var mı? Neden?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/1052)
:end_tab:
