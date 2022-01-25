# Duygu Analizi: Konvolsiyonel Sinir Ağlarının Kullanımı 
:label:`sec_sentiment_cnn`

:numref:`chap_cnn`'te, bitişik pikseller gibi yerel özelliklere uygulanan iki boyutlu CNN'lerle iki boyutlu görüntü verilerini işlemek için mekanizmaları inceledik. Başlangıçta bilgisayar görüşü için tasarlanmış olsa da, CNN'ler doğal dil işleme için de yaygın olarak kullanılmaktadır. Basitçe söylemek gerekirse, herhangi bir metin dizisini tek boyutlu bir görüntü olarak düşünün. Bu şekilde, tek boyutlu CNN'ler metin olarak $n$ gram gibi yerel özellikleri işleyebilir. 

Bu bölümde, tek metni temsil etmek için bir CNN mimarisi tasarlamak için nasıl göstermek için*textCNN* modelini kullanacağız :cite:`Kim.2014`. Duygu analizi için GloVe ön eğitimi ile RNN mimarisi kullanan :numref:`fig_nlp-map-sa-rnn` ile karşılaştırıldığında, :numref:`fig_nlp-map-sa-cnn`'teki tek fark mimarinin seçiminde yatmaktadır. 

![This section feeds pretrained GloVe to a CNN-based architecture for sentiment analysis.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
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

## Tek Boyutlu Konvolümanlar

Modeli tanıtmadan önce, tek boyutlu bir evrimin nasıl çalıştığını görelim. Çapraz korelasyon operasyonuna dayanan iki boyutlu bir evrimin sadece özel bir durumu olduğunu unutmayın. 

![One-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

:numref:`fig_conv1d`'te gösterildiği gibi, tek boyutlu durumda, konvolüsyon penceresi giriş tensör boyunca soldan sağa doğru kayar. Kayma sırasında, belirli bir konumdaki evrişim penceresinde bulunan giriş subtensör (örn. :numref:`fig_conv1d` yılında $0$ ve $1$) ve çekirdek tensör (örneğin, :numref:`fig_conv1d`'te $1$ ve $2$) elementwise çarpılır. Bu çarpımların toplamı, çıkış tensörünün karşılık gelen pozisyonunda tek skaler değeri (örneğin, :numref:`fig_conv1d`'te $0\times1+1\times2=2$) verir. 

Aşağıdaki `corr1d` işlevinde tek boyutlu çapraz korelasyon uyguluyoruz. Bir giriş tensör `X` ve bir çekirdek tensör `K` göz önüne alındığında, çıkış tensörünü `Y` döndürür.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

Yukarıdaki tek boyutlu çapraz korelasyon uygulamasının çıkışını doğrulamak için `X`'dan `X` giriş tensörünü ve `K`'ten `K`'i inşa edebiliriz.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

Birden çok kanallı tek boyutlu giriş için, evrişim çekirdeğinin aynı sayıda giriş kanalına sahip olması gerekir. Daha sonra her kanal için, girişin bir boyutlu tensör ve konvolüsyon çekirdeğinin tek boyutlu tensör üzerinde bir çapraz korelasyon işlemi gerçekleştirin, tek boyutlu çıkış tensör üretmek için sonuçları tüm kanallar üzerinden topladı. :numref:`fig_conv1d_channel` tek boyutlu çapraz korelasyon işlemini gösterir 3 giriş kanalı ile. 

![One-dimensional cross-correlation operation with 3 input channels. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`

Birden fazla giriş kanalı için tek boyutlu çapraz korelasyon işlemini uygulayabilir ve sonuçları :numref:`fig_conv1d_channel`'te doğrulayabiliriz.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

Çok girişli kanal tek boyutlu çapraz korelasyon tek girişli kanal iki boyutlu çapraz korelasyonlar eşdeğer olduğunu unutmayın. Göstermek için, :numref:`fig_conv1d_channel`'deki çok girişli kanal tek boyutlu çapraz korelasyonun eşdeğer bir formu, :numref:`fig_conv1d_2d`'teki tek girişli kanal iki boyutlu çapraz korelasyondur; burada konvolüsyon çekirdeğinin yüksekliği giriş tensörünküyle aynı olmalıdır. 

![Two-dimensional cross-correlation operation with a single input channel. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

:numref:`fig_conv1d` ve :numref:`fig_conv1d_channel`'teki her iki çıkışta da yalnızca bir kanal vardır. :numref:`subsec_multi-output-channels`'da açıklanan çoklu çıkış kanallarına sahip iki boyutlu kıvrımlarla aynı şekilde, tek boyutlu kıvrımlar için birden fazla çıkış kanalı da belirtebiliriz. 

## En Fazla Zamanlı Havuz

Benzer şekilde, zaman adımlarında en önemli özellik olarak sıra gösterimlerinden en yüksek değeri ayıklamak için havuzlama özelliğini kullanabiliriz. TextCNN'de kullanılan *max-over-time havuzlama*, tek boyutlu küresel maksimum havuzlama :cite:`Collobert.Weston.Bottou.ea.2011` gibi çalışır. Her kanalın değerleri farklı zaman adımlarında depoladığı çok kanallı giriş için, her kanaldaki çıktı, bu kanalın maksimum değeridir. En fazla zaman havuzunun farklı kanallarda farklı sayıda zaman adımına izin verdiğini unutmayın. 

## TextCNN Modeli

Tek boyutlu evrişim ve max-over-time havuzu kullanarak, textCNN modeli giriş olarak bireysel önceden eğitilmiş belirteç temsillerini alır, sonra alır ve aşağı akış uygulama için sıra temsillerini dönüştürür. 

$d$ boyutlu vektörlerle temsil edilen $n$ belirteçleri olan tek bir metin dizisi için giriş tensörünün kanal genişliği, yüksekliği ve sayısı sırasıyla $n$, $1$ ve $d$'tür. textCNN modeli, girdiyi çıktıya aşağıdaki gibi dönüştürür: 

1. Birden çok tek boyutlu evrişim çekirdeğini tanımlar ve girişler üzerinde ayrı olarak konvolüsyon işlemlerini gerçekleştirir. Farklı genişliklere sahip konvolution çekirdekleri, bitişik belirteçlerin farklı sayıları arasında yerel özellikleri yakalayabilir.
1. Tüm çıkış kanallarında en fazla zaman havuzu gerçekleştirin ve ardından tüm skaler havuzlama çıktılarını bir vektör olarak birleştirin.
1. Tamamen bağlı katmanı kullanarak birleştirilmiş vektörü çıktı kategorilerine dönüştürün. Bırakma, aşırı uyumu azaltmak için kullanılabilir.

![The model architecture of textCNN.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn`, textCNN'in model mimarisini somut bir örnekle göstermektedir. Giriş, her belirteç 6 boyutlu vektörlerle temsil edildiği 11 belirteçli bir cümledir. Bu yüzden genişliği 11 olan 6 kanallı bir girişe sahibiz. Sırasıyla 4 ve 5 çıkış kanalı ile 2 ve 4 genişliklerindeki iki tek boyutlu evrişim çekirdeğini tanımlayın. Genişlikli $11-2+1=10$ ve $11-4+1=8$ genişliğine sahip 5 çıkış kanalı üretirler. Bu 9 kanalın farklı genişliklerine rağmen, maksimum zaman havuzlama, birleştirilmiş 9 boyutlu bir vektör verir ve bu da nihayet ikili duyarlılık tahminleri için 2 boyutlu bir çıkış vektörüne dönüştürülür. 

### Modeli Tanımlama

TextCNN modelini aşağıdaki sınıfta uyguluyoruz. :numref:`sec_sentiment_rnn`'teki çift yönlü RNN modeliyle karşılaştırıldığında, tekrarlayan katmanları evrimsel katmanlarla değiştirmenin yanı sıra, iki gömme katmanı da kullanıyoruz: biri eğitilebilir ağırlıklara ve diğeri sabit ağırlıklara sahip.

```{.python .input}
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

Bir textCNN örneği oluşturalım. 3, 4 ve 5 çekirdek genişliklerine sahip 3 kıvrımlı katmana sahiptir ve hepsi 100 çıkış kanalı bulunur.

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights);
```

### Önceden Eğitimli Word Vektörlerini Yükleme

:numref:`sec_sentiment_rnn` ile aynı şekilde, önceden eğitilmiş 100 boyutlu Eldiven gömme yerleştirmeleri, başlatılan belirteç temsilleri olarak yükleriz. Bu belirteç temsilleri (gömme ağırlıkları) `embedding`'da eğitilecek ve `constant_embedding`'te sabitlenecektir.

```{.python .input}
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### Modelin Eğitimi ve Değerlendirilmesi

Şimdi textCNN modelini duygu analizi için eğitebiliriz.

```{.python .input}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Aşağıda iki basit cümle için duyguları tahmin etmek için eğitimli modeli kullanıyoruz.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## Özet

* Tek boyutlu CNN'ler metin olarak $n$-gram gibi yerel özellikleri işleyebilir.
* Çok girişli kanal tek boyutlu çapraz korelasyonlar, tek girişli kanal iki boyutlu çapraz korelasyonlar eşdeğerdir.
* Max-over-time havuzu, farklı kanallarda farklı sayıda zaman adımına olanak tanır.
* textCNN modeli, tek boyutlu evrimsel katmanlar ve maksimum zaman havuzlama katmanları kullanarak tek tek belirteç temsillerini aşağı akış uygulama çıkışlarına dönüştürür.

## Egzersizler

1. Hiperparametreleri ayarlayın ve duyarlılık analizi için :numref:`sec_sentiment_rnn` ve bu bölümde sınıflandırma doğruluğu ve hesaplama verimliliği gibi iki mimariyi karşılaştırın.
1. :numref:`sec_sentiment_rnn`'ün egzersizlerinde tanıtılan yöntemleri kullanarak modelin sınıflandırma doğruluğunu daha da geliştirebilir misiniz?
1. Giriş temsillerine konumsal kodlama ekleyin. Sınıflandırma doğruluğunu arttırıyor mu?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1425)
:end_tab:
