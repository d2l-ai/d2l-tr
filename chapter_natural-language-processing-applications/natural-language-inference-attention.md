# Doğal Dil Çıkarımı: Dikkat Kullanma
:label:`sec_natural-language-inference-attention`

:numref:`sec_natural-language-inference-and-dataset`'da doğal dil çıkarım görevini ve SNLI veri kümesini tanıttık. Karmaşık ve derin mimarilere dayanan birçok model göz önüne alındığında, Parikh ve ark. Doğal dil çıkarımına dikkat mekanizmaları ile hitap etmeyi önerdi ve bunu “ayrıştırılabilir dikkat modeli” olarak nitelendirdi :cite:`Parikh.Tackstrom.Das.ea.2016`. Bu, tekrarlayan veya kıvrımlı katmanları olmayan bir modelle sonuçlanır ve SNLI veri kümelerinde o anda çok daha az parametre ile en iyi sonucu elde eder. Bu bölümde, :numref:`fig_nlp-map-nli-attention`'te tasvir edildiği gibi doğal dil çıkarımı için bu dikkat tabanlı yöntemi (MLP'lerle) açıklayacağız ve uygulayacağız. 

![This section feeds pretrained GloVe to an architecture based on attention and MLPs for natural language inference.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`

## Model

Tesislerde ve hipotezlerdeki belirteçlerin sırasını korumaktan daha basit, sadece bir metin dizisindeki belirteçleri diğerindeki her belirteçle hizalayabiliriz ve tam tersi de geçerlidir, daha sonra tesisler ve hipotezler arasındaki mantıksal ilişkileri tahmin etmek için bu bilgileri karşılaştırabilir ve toplayabiliriz. Makine çevirisinde kaynak ve hedef cümleler arasında belirteçlerin hizalanmasına benzer şekilde, tesis ve hipotezler arasındaki belirteçlerin hizalanması dikkat mekanizmaları ile düzgün bir şekilde gerçekleştirilebilir. 

![Natural language inference using attention mechanisms.](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention` dikkat mekanizmalarını kullanarak doğal dil çıkarım yöntemini tasvir eder. Yüksek düzeyde, ortaklaşa eğitilmiş üç adımdan oluşur: katılmak, karşılaştırmak ve birleştirmek. Onları aşağıda adım adım göstereceğiz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### Katılmak

İlk adım, bir metin dizisindeki belirteçleri diğer dizideki her belirteçle hizalamaktır. Varsayalım ki öncülün “uykuya ihtiyacım var” ve hipotezin “yorgunum” olduğunu varsayalım. Semantik benzerlik nedeniyle, “i” hipotezinde “i” ile öncül içinde “i” ile hizalamak ve “yorgun” hipotezinde öncül içinde “uyku” ile hizalamak isteyebiliriz. Aynı şekilde, hipotezdeki “i” öncülünde “i” ile hizalamak ve “ihtiyaç” ve “uyku” ile hipotezde “yorgun” ile hizalamak isteyebiliriz. Bu hizalamanın, ideal olarak büyük ağırlıkların hizalanacak belirteçlerle ilişkilendirildiği ağırlıklı ortalama kullanarak*yumuşak* olduğunu unutmayın. Gösterim kolaylığı için, :numref:`fig_nli_attention` böyle bir hizalamayı *sert* bir şekilde gösterir. 

Şimdi dikkat mekanizmalarını kullanarak yumuşak hizalamayı daha ayrıntılı olarak tanımlıyoruz. $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$ ve $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ tarafından belirtin, sırasıyla $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) $d$ boyutlu bir sözcük vektörü olan simge sayısı $m$ ve $n$ olan öncül ve hipotezi belirtin. Yumuşak hizalama için dikkat ağırlıklarını $e_{ij} \in \mathbb{R}$ olarak hesaplıyoruz 

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

burada $f$ işlevi aşağıdaki `mlp` işlevinde tanımlanan bir MLP olduğu. $f$ çıkış boyutu `mlp` `num_hiddens` bağımsız değişkeni tarafından belirtilir.

```{.python .input}
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

Bu :eqref:`eq_nli_e` $f$ girişleri alır $\mathbf{a}_i$ ve $\mathbf{b}_j$ ayrı ayrı yerine bir çift giriş olarak bir çift alır, vurgulanmalıdır. Bu *ayrıştırma* hilesi $mn$ uygulamaları (kuadratik karmaşıklık) yerine sadece $f$ $f$ uygulamalarına (doğrusal karmaşıklık) yol açar. 

:eqref:`eq_nli_e`'te dikkat ağırlıklarını normalleştirerek, varsayımdaki tüm belirteç vektörlerinin ağırlıklı ortalamasını hesaplıyoruz ve bu hipotezin temsilini elde etmek için $i$ tarafından endeksli belirteç ile yumuşak bir şekilde hizalanan hipotezin temsilini elde ederiz: 

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Benzer şekilde, hipotezde $j$ tarafından endekslenen her belirteç için öncül belirteçlerinin yumuşak hizalamasını hesaplarız: 

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Aşağıda `Attend` giriş tesislerinde `A` ve giriş hipotezleri `B` ile giriş hipotezlerinin yumuşak hizalanmasını (`beta`) ve giriş hipotezleri ile (`alpha`) yumuşak hizalamasını hesaplamak için `Attend` sınıfını tanımlıyoruz.

```{.python .input}
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### Karşılaştırıyor

Bir sonraki adımda, bir dizideki bir simgeyi, bu belirteçle yumuşak bir şekilde hizalanan diğer diziyle karşılaştırırız. Yumuşak hizalamada, bir dizideki tüm belirteçlerin, muhtemelen farklı dikkat ağırlıklarına sahip olsa da, diğer sırayla bir belirteçle karşılaştırılacağını unutmayın. Kolay gösterim için, :numref:`fig_nli_attention`, jetonları hizalanmış belirteçlerle *sert* bir şekilde çiftleştirir. Örneğin, katılan adımın öncüldeki “ihtiyaç” ve “uyku” hipotezinde “yorgun” ile hizalandığını, çifti “yorul-uyku ihtiyacı” karşılaştırılacağını belirlediğini varsayalım. 

Karşılaştırma adımında, bir diziden belirteçlerin birleştirmesini (operatör $[\cdot, \cdot]$) ve diğer diziden hizalanmış belirteçleri $g$ (bir MLP) işlevine besleriz: 

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab` 

:eqref:`eq_nli_v_ab`'te, $\mathbf{v}_{A,i}$, öncüldeki $i$ belirteci ve $i$ belirteci ile yumuşak bir şekilde hizalanan tüm hipotez belirteçleri arasındaki karşılaştırmadır; $\mathbf{v}_{B,j}$ ise, hipotezdeki jeton $j$ ile yumuşak bir şekilde hizalanan simge $j$ arasındaki karşılaştırma ve jeton $j$ ile yumuşak bir şekilde hizalanan tüm öncül belirteçleri. Aşağıdaki `Compare` sınıfı, adım karşılaştırılması gibi tanımlar.

```{.python .input}
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### Toplama

İki karşılaştırma vektörü seti $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) ve $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$) elinizde, son adımda mantıksal ilişkiyi ortaya çıkaracak şekilde bu bilgileri toplayacağız. Her iki seti de özetleyerek başlıyoruz: 

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Daha sonra mantıksal ilişkinin sınıflandırma sonucunu elde etmek için $h$ (bir MLP) işlevine her iki özetleme sonuçlarının birleştirilmesini besleriz: 

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

Toplama adımı aşağıdaki `Aggregate` sınıfında tanımlanır.

```{.python .input}
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### Her Şeyleri Bir Araya Getirmek

Katılan, karşılaştırarak ve adımları bir araya getirerek, bu üç adımı ortaklaşa eğitmek için ayrıştırılabilir dikkat modelini tanımlıyoruz.

```{.python .input}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## Modelin Eğitimi ve Değerlendirilmesi

Şimdi SNLI veri kümesinde tanımlanmış ayrıştırılabilir dikkat modelini eğiteceğiz ve değerlendireceğiz. Veri kümesini okuyarak başlıyoruz. 

### Veri kümesini okuma

SNLI veri kümesini :numref:`sec_natural-language-inference-and-dataset`'te tanımlanan işlevi kullanarak indirip okuyoruz. Parti boyutu ve sıra uzunluğu sırasıyla $256$ ve $50$ olarak ayarlanır.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### Modeli Oluşturma

Giriş belirteçlerini temsil etmek için önceden eğitilmiş 100 boyutlu Eldiven gömme kullanıyoruz. Böylece, :eqref:`eq_nli_e` olarak :eqref:`eq_nli_e` ve $\mathbf{b}_j$ vektörlerin boyutunu önceden tanımlıyoruz. :eqref:`eq_nli_e` ve :eqref:`eq_nli_v_ab`'te :eqref:`eq_nli_v_ab`'te $g$ fonksiyonlarının çıkış boyutu 200 olarak ayarlanır. Ardından bir model örneği oluştururuz, parametrelerini başlatırız ve giriş belirteçlerinin vektörlerini başlatmak için GloVe gömülü yükleriz.

```{.python .input}
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### Modelin Eğitimi ve Değerlendirilmesi

:numref:`sec_multi_gpu`'teki `split_batch` işlevinin aksine, metin dizileri (veya görüntüler) gibi tek girişleri alan `split_batch` işlevinin aksine, minibatch'lerde binalar ve hipotezler gibi birden fazla girişi almak için `split_batch_multi_inputs` işlevi tanımlıyoruz.

```{.python .input}
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

Şimdi modeli SNLI veri kümesinde eğitebilir ve değerlendirebiliriz.

```{.python .input}
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### Modeli Kullanma

Son olarak, bir çift öncül ve hipotez arasındaki mantıksal ilişkiyi ortaya çıkaracak tahmin işlevini tanımlayın.

```{.python .input}
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

Eğitimli modeli, örnek bir cümle çifti için doğal dil çıkarım sonucunu elde etmek için kullanabiliriz.

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## Özet

* Çürüyebilir dikkat modeli, binalar ve hipotezler arasındaki mantıksal ilişkileri tahmin etmek için üç adımdan oluşur: katılmak, karşılaştırmak ve birleştirmek.
* Dikkat mekanizmalarıyla, bir metin dizisinde belirteçleri diğerindeki her belirteçle hizalayabiliriz ve tam tersi de geçerlidir. Bu hizalama, ideal olarak büyük ağırlıkların hizalanacak belirteçlerle ilişkilendirildiği ağırlıklı ortalama kullanılarak yumuşaktır.
* Ayrışma hilesi, dikkat ağırlıklarını hesaplarken kuadratik karmaşıklıktan daha arzu edilen doğrusal bir karmaşıklığa yol açar.
* Doğal dil çıkarımı gibi doğal dil işleme görevi için giriş temsili olarak önceden eğitilmiş sözcük vektörlerini kullanabiliriz.

## Egzersizler

1. Modeli diğer hiperparametre kombinasyonları ile eğitin. Test setinde daha iyi doğruluk elde edebilir misiniz?
1. Doğal dil çıkarımı için ayrıştırılabilir dikkat modelinin en büyük dezavantajları nelerdir?
1. Herhangi bir cümle çifti için anlamsal benzerlik düzeyini (örneğin, 0 ile 1 arasında sürekli bir değer) elde etmek istediğimizi varsayalım. Veri kümesini nasıl toplayıp etiketleyeceğiz? Dikkat mekanizmaları ile bir model tasarlayabilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:
