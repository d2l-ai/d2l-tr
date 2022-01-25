# Transformatörlerden Çift Yönlü Enkoder Temsilleri (BERT)
:label:`sec_bert`

Doğal dil anlayışı için çeşitli kelime gömme modelleri tanıttık. Ön eğitim sonrasında çıktı, her satırın önceden tanımlanmış bir sözcük dağarcığını temsil eden bir vektör olduğu bir matris olarak düşünülebilir. Aslında, bu kelime gömme modelleri tüm*bağlam-bağımsız*. Bu mülkü göstererek başlayalım. 

## Bağlamdan Bağımsızlıktan Bağlam Duyarlısına

:numref:`sec_word2vec_pretraining` ve :numref:`sec_synonyms`'teki deneyleri hatırlayın. Örneğin, word2vec ve GloVe her ikisi de kelimenin bağlamından bağımsız olarak (varsa) aynı önceden eğitilmiş vektörü aynı sözcüğe atar. Resmi olarak, herhangi bir belirteç $x$ bağlamdan bağımsız gösterimi, yalnızca $x$ girdisi olarak alan $f(x)$ işlevidir. Doğal dillerde çoklukla ve karmaşık anlambilimin bolluğu göz önüne alındığında, bağlamdan bağımsız temsillerin belirgin sınırlamaları vardır. Örneğin, “vinç uçuyor” ve “vinç sürücüsü geldi” bağlamlarındaki “vinç” kelimesinin tamamen farklı anlamları vardır; bu nedenle, aynı kelimeye bağlamlara bağlı olarak farklı temsiller atanabilir. 

Bu, kelimelerin temsillerinin bağlamlarına bağlı olduğu *bağlam duyarlı* kelime temsillerinin gelişimini motive eder. Bu nedenle, $x$ belirteci bağlam duyarlı gösterimi $x$ hem $x$ hem de bağlamı $c(x)$ bağlı olarak $f(x, c(x))$ bir işlevdir. Popüler bağlam duyarlı temsiller arasında TagLM (dil-model-artırılmış sıra etiketleyici) :cite:`Peters.Ammar.Bhagavatula.ea.2017`, CoVe (Bağlam Vektörleri) :cite:`McCann.Bradbury.Xiong.ea.2017` ve ELMo (Dil Modellerinden Gömülmeler) :cite:`Peters.Neumann.Iyyer.ea.2018` bulunur. 

Örneğin, tüm diziyi girdi olarak alarak, ELMo, giriş dizisinden her sözcüğe bir temsil atan bir işlevdir. Özellikle, ELMo, önceden eğitilmiş çift yönlü LSTM'den tüm ara katman temsillerini çıktı temsili olarak birleştirir. Daha sonra ELMo temsili, mevcut modelde ELMo temsilini ve simgelerin orijinal temsilini (örneğin, Eldiven) birleştirmek gibi ek özellikler olarak bir aşağı akış görevinin mevcut denetlenen modeline eklenecektir. Bir yandan, önceden eğitilmiş çift yönlü LSTM modelindeki tüm ağırlıklar ELMo temsilleri eklendikten sonra dondurulur. Öte yandan, mevcut denetimli model belirli bir görev için özel olarak özelleştirilmiştir. O dönemde farklı görevler için farklı en iyi modellerden yararlanarak ELMo, altı doğal dil işleme görevinde sanat durumunu iyileştirdi: duygu analizi, doğal dil çıkarımı, anlamsal rol etiketleme, çekirdek çözümü, adlandırılmış varlık tanıma ve soru yanıtlama. 

## Göreve Özgü Görevden Agnostik

ELMo, çeşitli doğal dil işleme görevleri setine yönelik çözümleri önemli ölçüde geliştirmiş olsa da, her çözüm hala*görev özelli* mimarisine dayanıyor. Bununla birlikte, her doğal dil işleme görevi için belirli bir mimari oluşturmak pratik olarak önemsizdir. GPT (Generative Pre-Training) modeli, bağlama duyarlı gösterimler :cite:`Radford.Narasimhan.Salimans.ea.2018` için genel bir *görev-agnostic* modeli tasarlama çabasını temsil eder. Bir transformatör kod çözücü üzerine inşa edilen GPT, metin dizilerini temsil etmek için kullanılacak bir dil modelini ön planlar. Bir aşağı akış göreve GPT uygularken, dil modelinin çıktısı görevin etiketini tahmin etmek için ek bir doğrusal çıktı katmanına beslenir. Önceden eğitilmiş modelin parametrelerini donduran ELMo ile keskin bir kontrast olarak, GPT aşağı akış görevinin denetimli öğrenimi sırasında önceden eğitilmiş transformatör dekoderindeki parametreleri *tüm* ince ayarlar. GPT, doğal dil çıkarımı, soru cevaplama, cümle benzerliği ve sınıflandırma gibi on iki görev üzerinde değerlendirildi ve model mimarisinde minimum değişikliklerle dokuzunda sanat durumunu iyileştirdi. 

Bununla birlikte, dil modellerinin otoregresif doğası nedeniyle, GPT sadece ileriye bakar (soldan sağa). “Banka” solundaki bağlamda duyarlı olduğu için “bankaya yatırmak için bankaya gittim” ve “oturmak için bankaya gittim” bağlamında, GPT farklı anlamlara sahip olsa da, “banka” için aynı temsili geri dönecektir. 

## BERT: Her İki Dünyanın En İyisini Birleştirme

Gördüğümüz gibi, ELMo bağlamı iki yönlü olarak kodlar, ancak göreve özgü mimarileri kullanır; GPT ise göreve özgüdür ancak bağlamı soldan sağa kodlar. Her iki dünyanın en iyilerini bir araya getiren BERT (Transformers Çift Yönlü Encoder Temsilleri) bağlamı çift yönlü olarak kodlar ve :cite:`Devlin.Chang.Lee.ea.2018` çeşitli doğal dil işleme görevleri için minimum mimari değişiklikleri gerektirir. Önceden eğitilmiş bir transformatör kodlayıcısı kullanarak BERT, çift yönlü bağlamına dayalı herhangi bir belirteci temsil edebilir. Aşağı akış görevlerinin denetimli öğrenimi sırasında BERT iki açıdan GPT'ye benzer. İlk olarak, BERT temsilleri, her belirteç için tahmin etmek gibi görevlerin niteliğine bağlı olarak model mimarisinde minimum değişikliklerle birlikte eklenen bir çıktı katmanına beslenecektir. İkinci olarak, önceden eğitilmiş transformatör kodlayıcının tüm parametreleri ince ayarlanırken, ek çıkış katmanı sıfırdan eğitilecektir. :numref:`fig_elmo-gpt-bert`, ELMo, GPT ve BERT arasındaki farkları tasvir eder. 

![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

BERT, (i) tek metin sınıflandırması (örn. Duygu analizi), (ii) metin çifti sınıflandırması (örneğin, doğal dil çıkarımı), (iii) soru yanıtlama, (iv) metin etiketleme (örneğin, adlandırılmış varlık tanıma) geniş kategorilerinde on bir doğal dil işleme görevinde sanat durumunu iyileştirdi. . 2018 yılında, bağlam duyarlı ELMo'dan görev-agnostik GPT ve BERT'e kadar, doğal diller için derin temsillerin kavramsal olarak basit ama ampirik olarak güçlü ön eğitimi, çeşitli doğal dil işleme görevlerine çözümler devrim yaratmıştır. 

Bu bölümün geri kalanında BERT ön eğitimine dalacağız. Doğal dil işleme uygulamaları :numref:`chap_nlp_app`'te açıklandığında, aşağı akım uygulamaları için BERT ince ayarını göstereceğiz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## Girdi Temsili
:label:`subsec_bert_input_rep`

Doğal dil işlemede, bazı görevler (örn. Duygu analizi) giriş olarak tek bir metin alırken, diğer bazı görevlerde (örneğin, doğal dil çıkarımı) girdi bir çift metin dizisi oluşturur. BERT giriş sırası, hem tek metin hem de metin çiftlerini açıkça temsil eder. Birincisinde, BERT giriş sırası özel sınıflandırma belirteci “<cls>”, bir metin dizisinin belirteçleri ve “<sep>” özel ayırma belirteci birleştirilmedir. İkincisinde, BERT giriş sırası “<cls>”, ilk metin dizisinin belirteçleri, “<sep>”, ikinci metin dizisinin belirteçleri ve “<sep>” birleştirilmedir. “BERT giriş dizisi” terminolojisini diğer “diziler” türlerinden sürekli olarak ayırt edeceğiz. Örneğin, bir *BERT giriş sırası*, bir *text dizisi* veya iki *text dizisi* içerebilir. 

Metin çiftlerini ayırt etmek için, öğrenilen segment gömme $\mathbf{e}_A$ ve $\mathbf{e}_B$ sırasıyla birinci dizinin belirteç gömme ve ikinci diziye eklenir. Tek metin girişleri için sadece $\mathbf{e}_A$ kullanılır. 

Aşağıdaki `get_tokens_and_segments` giriş olarak bir cümle veya iki cümle alır, daha sonra BERT giriş sırasının belirteçlerini ve bunların karşılık gelen segment kimliklerini döndürür.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT, transformatör kodlayıcısını çift yönlü mimarisi olarak seçer. Trafo kodlayıcısında yaygın olarak, BERT giriş dizisinin her pozisyonuna konumsal gömme eklenir. Ancak, orijinal transformatör kodlayıcısından farklı olan BERT, *learnable* konumsal gömme kullanır. Özetlemek gerekirse, :numref:`fig_bert-input` BERT giriş sırasının gömme token gömme, segment gömme ve konum gömme toplamı olduğunu gösterir. 

! [BERT giriş sırasının gömülmeleri belirteç gömme, segment gömme ve konumsal gömme toplamıdır.](.. /img/bert-input.svg) :label:`fig_bert-input` 

Aşağıdaki `BERTEncoder` sınıfı :numref:`sec_transformer`'te uygulandığı gibi `TransformerEncoder` sınıfına benzer. `TransformerEncoder`'dan farklı olan `BERTEncoder`, segment gömme ve öğrenilebilir konumsal gömme parçaları kullanır.

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

Kelime dağarcığının 10000 olduğunu varsayalım. `BERTEncoder`'ün ileri çıkarımını göstermek için, bunun bir örneğini oluşturalım ve parametrelerini başlatalim.

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

`tokens`'yı, her belirteç kelime dağarcığının bir dizisi olduğu 2 BERT giriş dizisi olarak tanımlıyoruz. `BERTEncoder` girişiyle `BERTEncoder` ileri çıkarımı, her belirteçin uzunluğu hiperparametre `num_hiddens` tarafından önceden tanımlanmış bir vektör tarafından temsil edildiği kodlanmış sonucu döndürür. Bu hiperparametre genellikle transformatör kodlayıcın*gizli boyut* (gizli birim sayısı) olarak adlandırılır.

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## Ön Eğitim Görevleri
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder`'ün ileri çıkarımı, giriş metninin her belirteci ve eklenen özel belirteçlerin “” ve “<cls>” işaretlerinin BERT temsilini verir<seq>. Ardından, BERT öncesi eğitim için kayıp fonksiyonunu hesaplamak için bu temsilleri kullanacağız. Ön eğitim aşağıdaki iki görevden oluşur: maskeli dil modelleme ve sonraki cümle tahmini. 

### Maskeli Dil Modelleme
:label:`subsec_mlm`

:numref:`sec_language_model`'te gösterildiği gibi, bir dil modeli, sol tarafındaki bağlamı kullanarak bir belirteci öngörür. Her belirteci temsil etmek için bağlamı iki yönlü olarak kodlamak için BERT, belirteçleri rastgele maskeler ve maskelenmiş belirteçleri kendi kendine denetimli bir şekilde tahmin etmek için çift yönlü bağlamdaki belirteçleri kullanır. Bu göre*maskeli dil modeli* olarak adlandırılır. 

Bu ön eğitim görevinde, belirteçlerin% 15'i tahmin için maskelenmiş belirteçler olarak rastgele seçilecektir. Etiketini kullanarak hile yapmadan maskelenmiş bir belirteci tahmin etmek için, basit bir yaklaşım, her zaman <mask>BERT giriş dizisinde özel bir “” belirteci ile değiştirmektir. Bununla birlikte, yapay özel belirteç “<mask>” asla ince ayarda görünmeyecektir. Ön eğitim ve ince ayar arasında böyle bir uyuşmazlığı önlemek için, eğer bir belirteç tahmin için maskelenmişse (örneğin, “büyük” maskelenecek ve “bu film harika” olarak tahmin edilmek üzere seçilir), girişte şu şekilde değiştirilir: 

* <mask>zamanın%80'i için özel bir “” simgesi (örneğin, “bu film harika” olur “bu film <mask>“);
* zamanın%10'u için rastgele bir simge (örneğin, “bu film harika” “bu film içiyor” olur);
* zamanın%10'unda değişmeyen etiket simgesi (örneğin, “bu film harika” “bu film harika” olur).

%15 zamanının% 10'unda rastgele bir belirteç eklendiğini unutmayın. Bu ara sıra gürültü, BERT'i, çift yönlü bağlam kodlamasında maskelenmiş belirteç (özellikle etiket belirteci değişmeden kaldığında) karşı daha az önyargılı olmasını teşvik eder. 

BERT ön eğitiminin maskelenmiş dil modeli görevinde maskeli belirteçleri tahmin etmek için aşağıdaki `MaskLM` sınıfını uyguluyoruz. Tahmin, tek gizli katmanlı bir MLP kullanır (`self.mlp`). İleri çıkarımda, iki giriş gerekir: `BERTEncoder`'nın kodlanmış sonucu ve tahmin için belirteç konumları. Çıktı, bu pozisyonlardaki tahmin sonuçlarıdır.

```{.python .input}
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

`MaskLM`'ün ileri çıkarımını göstermek için, `mlm` örneğini oluşturup başlatırız. `BERTEncoder`'in ileri çıkarımından `encoded_X`'nin 2 BERT girdi dizisini temsil ettiğini hatırlayın. `mlm_positions`'i `encoded_X` BERT giriş dizisinde tahmin etmek için 3 endeksleri olarak tanımlıyoruz. `mlm` ileri çıkarımı `encoded_X` `encoded_X` tüm maskeli pozisyonlarda `mlm_positions` tahmini sonuçları `mlm_Y_hat` döndürür. Her tahmin için, sonucun boyutu kelime dağarcığına eşittir.

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

Tahmin edilen belirteçlerin `mlm_Y` `mlm_Y` maskeleri altında zemin doğruluk etiketleri ile, BERT ön eğitimindeki maskeli dil modeli görevinin çapraz entropi kaybını hesaplayabiliriz.

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### Sonraki Cümle Tahmini
:label:`subsec_nsp`

Maskelenmiş dil modellemesi sözcükleri temsil etmek için çift yönlü bağlam kodlayabilse de, metin çiftleri arasındaki mantıksal ilişkiyi açıkça modellemez. BERT, iki metin dizisi arasındaki ilişkiyi anlamaya yardımcı olmak için, ön eğitimde ikili sınıflandırma görevi, *sonraki cümle tahmini* olarak değerlendirir. Ön eğitim için cümle çiftleri oluştururken, çoğu zaman “True” etiketi ile ardışık cümlelerdir; zamanın diğer yarısı için ikinci cümle rastgele “Yanlış” etiketi ile corpustan örneklenir. 

Aşağıdaki `NextSentencePred` sınıfı, ikinci cümlenin BERT giriş sırasındaki ilk cümlenin bir sonraki cümle olup olmadığını tahmin etmek için tek gizli katmanlı bir MLP kullanır. Transformatör kodlayıcısındaki öz-dikkat nedeniyle, “<cls>” özel belirtecinin BERT temsili, her iki cümleyi de girdiden kodlar. Bu nedenle, MLP sınıflandırıcısının çıkış katmanı (`self.output`) giriş olarak `X` alır, burada `X` girişi kodlanmış “<cls>” belirteci olan MLP gizli katmanın çıkışıdır.

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

Bir `NextSentencePred` örneğinin ileri çıkarımının her BERT giriş sırası için ikili tahminleri döndürdüğünü görebiliriz.

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

2 ikili sınıflandırmanın çapraz entropi kaybı da hesaplanabilir.

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

Yukarıda bahsedilen ön eğitim görevlerindeki tüm etiketlerin manuel etiketleme çabası olmaksızın ön eğitim korpusundan önemsiz olarak elde edilebileceği dikkat çekicidir. Orijinal BERT, BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015` ve İngilizce Vikipedi'nin birleştirilmesi üzerine önceden eğitilmiştir. Bu iki metin corpora çok büyüktür: sırasıyla 800 milyon kelime ve 2,5 milyar kelime var. 

## Her Şeyleri Bir Araya Getirmek

BERT öncesi eğitim yaparken, nihai kayıp fonksiyonu hem maskeli dil modelleme hem de sonraki cümle tahmini için kayıp fonksiyonlarının doğrusal bir kombinasyonudur. Artık `BERTModel` sınıfını `BERTEncoder`, `MaskLM` ve `NextSentencePred`'in üç sınıfını kurarak tanımlayabiliriz. İleri çıkarım kodlanmış BERT temsilleri `encoded_X`, maskeli dil modelleme `mlm_Y_hat` tahminleri ve sonraki cümle tahminleri `nsp_Y_hat` döndürür.

```{.python .input}
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## Özet

* Word2vec ve GloVe gibi Word katıştırma modelleri bağlamdan bağımsızdır. Aynı önceden eğitilmiş vektörü, kelimenin bağlamından bağımsız olarak (varsa) aynı sözcüğe atar. Doğal dillerde polimsi veya karmaşık semantiği iyi ele almaları zordur.
* ELMo ve GPT gibi bağlam duyarlı sözcük temsilleri için, sözcüklerin temsilleri bağlamlarına bağlıdır.
* ELMo bağlamı iki yönlü olarak kodlar ancak göreve özgü mimarileri kullanır (ancak, her doğal dil işleme görevi için belirli bir mimariyi oluşturmak pratik olarak önemsizdir); GPT görev özgüdür ancak bağlamı soldan sağa kodlar.
* BERT her iki dünyanın en iyilerini birleştirir: bağlamı çift yönlü olarak kodlar ve çok çeşitli doğal dil işleme görevleri için minimal mimari değişiklikleri gerektirir.
* BERT giriş sırasının gömülmeleri belirteç gömme, segment gömme ve konumsal gömme toplamıdır.
* BERT öncesi eğitim iki görevden oluşur: maskeli dil modelleme ve sonraki cümle tahmini. Birincisi, sözcükleri temsil etmek için çift yönlü bağlamı kodlayabilirken, ikincisi ise metin çiftleri arasındaki mantıksal ilişkiyi açıkça modelliyor.

## Egzersizler

1. BERT neden başarılı oluyor?
1. Diğer tüm şeylerin eşit olması, maskeli bir dil modeli, soldan sağa dil modelinden daha fazla veya daha az ön eğitim adımı gerektirir mi? Neden?
1. BERT'in orijinal uygulamasında `BERTEncoder` (`d2l.EncoderBlock` aracılığıyla) konumsal ilerleme ağı ve `MaskLM`'teki tam bağlı katman, Gauss hata doğrusal birimi (GELU) :cite:`Hendrycks.Gimpel.2016`'ü aktivasyon işlevi olarak kullanır. GELU ve ReLU arasındaki farkı araştırın.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:
