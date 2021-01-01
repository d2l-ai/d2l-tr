# Çift Yönlü Yinelemeli Sinir Ağları
:label:`sec_bi_rnn`

Dizi öğrenmede, şu ana kadar amacımızın, şimdiye kadar gördüklerimizle bir sonraki çıktıyı modellemek olduğunu varsaydık; örneğin, bir zaman serisi veya bir dil modeli bağlamında. Bu tipik bir senaryo olsa da, karşılaşabileceğimiz tek senaryo değil. Sorunu göstermek için, bir metin dizisinde boş doldurmada aşağıdaki üç görevi gözünüzde canlandırın:

* Ben `___`.
* `___` açım.
* `___` açım ve yarım kuzu yiyebilirim.

Mevcut bilgi miktarına bağlı olarak boşlukları “mutluyum”, “yarı” ve “çok” gibi çok farklı kelimelerle doldurabiliriz. Açıkça ifadenin sonu (varsa) hangi kelimenin alınacağı hakkında önemli bilgiler taşır. Bundan yararlanamayan bir dizi modeli, ilgili görevlerde kötü performans gösterecektir. Örneğin, adlandırılmış varlık tanımada iyi iş yapmak için (örneğin, “Yeşil”'in “Bay Yeşil”'i ya da rengi ifade edip etmediğini tanımak) daha uzun menzilli bağlam aynı derecede hayati önem taşır. Problemi ele alırken biraz ilham almak için olasılıksal çizge modellerde biraz dolanalım.

## Gizli Markov Modellerinde Dinamik Programlama

Bu alt bölüm, dinamik programlama problemini göstermeyi amaçlar. Belirli teknik detaylar derin öğrenme modellerini anlamak için önemli değildir, ancak neden derin öğrenmeyi kullanabileceğimizi ve neden belirli mimarileri seçebileceğimizi anlamaya yardımcı olurlar.

Problemi olasılıksal çizge modelleri kullanarak çözmek istiyorsak, mesela aşağıdaki gibi gizli bir değişken modeli tasarlayabiliriz. Herhangi bir $t$ zamanda adımında, $P(x_t \mid h_t)$ olasılığında $x_t$ aracılığıyla gözlenen salımını yöneten bazı gizli $h_t$ değişkenimiz olduğunu varsayalım. Dahası, herhangi bir $h_t \to h_{t+1}$ geçişi, bir durum geçiş olasılığı $P(h_{t+1} \mid h_{t})$ ile verilir. Bu olasılıksal çizge modeli :numref:`fig_hmm`'te olduğu gibi bir *saklı Markov modeli*dir.

![Saklı Markov Modeli.](../img/hmm.svg)
:label:`fig_hmm`

Böylece, $T$ gözlemlerinin bir dizisi için gözlemlenen ve gizli durumlar üzerinde aşağıdaki ortak olasılık dağılımına sahibiz:

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

Şimdi $x_i$ bazı $x_j$ hariç tüm gözlemlediğimizi varsayalım ve $P(x_j \mid x_{-j})$, burada $x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$ hesaplamak hedefimizdir. $P(x_j \mid x_{-j})$'te gizli bir değişken olmadığından, $h_1, \ldots, h_T$ için olası tüm seçenek kombinasyonlarını toplamayı düşünüyoruz. Herhangi bir $h_i$'un $k$ farklı değerlere (sonlu sayıda durum) sahip olması durumunda, bu, $k^T$ terimi toplamamız gerektiği anlamına gelir; genellikle görev imkansız! Neyse ki bunun için zarif bir çözüm var: *dinamik programlama*.

Nasıl çalıştığını görmek için, sırayla $h_1, \ldots, h_T$ gizli değişkenler üzerinde toplamayı düşünün. :eqref:`eq_hmm_jointP`'e göre, bu verim:

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

Genel olarak*ileri özyineleme* olarak

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

Yineleme $\pi_1(h_1) = P(h_1)$ olarak başlatılır. Soyut terimlerle bu $\pi_{t+1} = f(\pi_t, x_t)$ olarak yazılabilir, burada $f$ bazı öğrenilebilir işlevdir. Bu, RNN bağlamında şimdiye kadar tartıştığımız gizli değişken modellerindeki güncelleme denklemine çok benziyor!

İleri özyinelemeye tamamen benzer şekilde, geriye dönük bir özyineleme ile aynı gizli değişken kümesini de özetleyebiliriz. Bu verim:

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

Böylece *geriye dönük özyineleme* olarak yazabiliriz

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

başlatma ile $\rho_T(h_T) = 1$. Hem ileri hem de geri özyinelemeler $T$ latent değişkenleri $\mathcal{O}(kT)$ (doğrusal) zaman içinde $(h_1, \ldots, h_T)$ yerine üstel zaman içinde toplamamızı sağlar. Bu, grafiksel modellerle olasılık çıkarımının en büyük avantajlarından biridir. Aynı zamanda algoritma :cite:`Aji.McEliece.2000` geçen genel bir mesaj çok özel bir örneğidir. Hem ileri hem de geri özyinelemeleri birleştirerek,

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

Soyut terimlerle geriye dönük özyinelemenin $\rho_{t-1} = g(\rho_t, x_t)$ olarak yazılabileceğini unutmayın; burada $g$ öğrenilebilir bir işlevdir. Yine, bu çok bir güncelleme denklemi gibi görünüyor, Sadece biz RNN şimdiye kadar gördük aksine geriye doğru çalışan. Gerçekten de, gizli Markov modelleri, mevcut olduğunda gelecekteki verileri bilmekten fayda sağlar. Sinyal işleme bilim adamları, interpolasyon v.s. ekstrapolasyon olarak gelecekteki gözlemleri bilerek ve bilmemenin iki vakası arasında ayrım. Daha fazla ayrıntı için sıralı Monte Carlo algoritmaları üzerine kitabın tanıtım bölümüne bakın :cite:`Doucet.De-Freitas.Gordon.2001`.

## Çift Yönlü Model

Eğer gizli Markov modellerinde olduğu gibi benzer bir bakış yeteneği sunan RNN'lerde bir mekanizmaya sahip olmak istiyorsak, şimdiye kadar gördüğümüz RNN tasarımını değiştirmeliyiz. Neyse ki, bu kavramsal olarak kolaydır. Bir RNN'yi yalnızca ilk belirtecinden başlayarak ileri modda çalıştırmak yerine, arkadan öne çalışan son belirteciden başka bir tane başlatırız.
*Çift yönlü RNN*, bu tür bilgileri daha esnek bir şekilde işlemek için bilgileri geriye doğru ileten gizli bir katman ekler. :numref:`fig_birnn`, tek bir gizli katmanla çift yönlü RNN mimarisini göstermektedir.

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

Aslında, bu, gizli Markov modellerinin dinamik programlanmasında ileri ve geri tekrarlamalara çok benzemez. Ana ayrım, önceki durumda bu denklemlerin belirli bir istatistiksel anlamı olmasıdır. Artık bu kadar kolay erişilebilir yorumlardan yoksunlar ve biz onlara genel ve öğrenilebilir işlevler olarak davranabiliriz. Bu geçiş, modern derin ağların tasarımına rehberlik eden ilkelerin çoğunu özetliyor: önce, klasik istatistiksel modellerin fonksiyonel bağımlılıklarının türünü kullanın ve daha sonra bunları genel bir biçimde parameterize edin.

### Tanımı

Çift yönlü RNN'ler :cite:`Schuster.Paliwal.1997` tarafından tanıtıldı. Çeşitli mimarilerin ayrıntılı bir tartışma için de kağıt bkz :cite:`Graves.Schmidhuber.2005`. Böyle bir ağın özelliklerine bakalım.

Herhangi bir zaman için adım $t$, bir minibatch girişi verilen $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (örnek sayısı: $n$, her örnekte giriş sayısı: $d$) ve gizli katman etkinleştirme işlevinin $\phi$ olmasına izin verin. Çift yönlü mimaride, bu zaman adımı için ileri ve geriye doğru gizli durumların sırasıyla $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ ve $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ olduğunu varsayıyoruz, burada $h$ gizli birimlerin sayısı. İleri ve geriye doğru gizli durum güncelleştirmeleri aşağıdaki gibidir:

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

burada $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$ ağırlıkları ve $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$ önyargıları tüm model parametreleridir.

Daha sonra, $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ gizli durumunu çıkış katmanına beslemek için $\overrightarrow{\mathbf{H}}_t$ ve $\overleftarrow{\mathbf{H}}_t$, ileri ve geri gizli durumlarını birleştiririz. Birden fazla gizli katman içeren derin çift yönlü RNN'lerde, bu tür bilgiler sonraki çift yönlü katmana *giriş* olarak aktarılır. Son olarak, çıkış katmanı $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ çıkışını hesaplar (çıkış sayısı: $q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Burada, $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ ağırlık matrisi ve $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ önyargı, çıkış katmanının model parametreleridir. Aslında, iki yön farklı sayıda gizli üniteye sahip olabilir.

### Hesaplamalı Maliyet ve Uygulamaları

Çift yönlü RNN'nin temel özelliklerinden biri, dizinin her iki ucundan gelen bilgilerin çıktıyı tahmin etmek için kullanıldığıdır. Yani, mevcut olanı tahmin etmek için hem gelecekteki hem de geçmiş gözlemlerden gelen bilgileri kullanırız. Bir sonraki belirteç tahmininde bu istediğimiz şey değil. Sonuçta, bir sonraki simgeyi tahmin ederken bir sonraki simgeyi bilme lüksüne sahip değiliz. Dolayısıyla, çift yönlü bir RNN kullansaydık, çok iyi bir doğruluk elde edemezdik: Eğitim sırasında şimdiki zamanı tahmin etmek için geçmiş ve gelecekteki verilere sahibiz. Test süresi boyunca sadece geçmiş veri ve dolayısıyla zayıf doğruluk var. Bunu aşağıda bir deneyde göstereceğiz.

Yaralanmaya hakaret eklemek için, çift yönlü RNN'ler de son derece yavaştır. Bunun başlıca nedenleri ileri yayılma iki yönlü katmanlarda hem ileri hem de geri yinelemeler gerektirir ve geri yayılma ileri yayılım sonuçlarına bağlı olmasıdır. Bu nedenle, degradeler çok uzun bir bağımlılık zincirine sahip olacaktır.

Pratikte çift yönlü katmanlar çok idareli ve yalnızca eksik kelimeleri doldurma, belirteçleri açıklama ekleme (örneğin, adlandırılmış varlık tanıma için) ve bir dizi işleme boru hattında bir adım olarak toptan dizileri kodlama gibi dar bir uygulama kümesi için kullanılır (örneğin, makine çevirisi için). :numref:`sec_bert` ve :numref:`sec_sentiment_rnn`'te, metin dizilerini kodlamak için çift yönlü RNN'lerin nasıl kullanılacağını tanıtacağız.

## Yanlış bir uygulama için çift yönlü RNN eğitimi

İki yönlü RNN'lerin geçmiş ve gelecekteki verileri kullanmaları ve sadece dil modellerine uyguladıkları gerçeğine ilişkin tüm tavsiyeleri görmezden gelirsek, kabul edilebilir bir şaşkınlıkla ilgili tahminler alırız. Bununla birlikte, modelin gelecekteki belirteçlerini tahmin etme yeteneği, aşağıdaki deney gösterildiği gibi ciddi bir şekilde tehlikeye atılmıştır. Makul şaşkınlığa rağmen, birçok yinelemeden sonra bile anlamsız üretir. Aşağıdaki kodu, yanlış bağlamda kullanılmasına karşı uyarıcı bir örnek olarak ekliyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

Çıktı, yukarıda açıklanan nedenlerden dolayı açıkça tatmin edici değildir. İki yönlü RNN'lerin daha etkili kullanımları hakkında bir tartışma için lütfen :numref:`sec_sentiment_rnn`'teki duyarlılık analizi uygulamasına bakın.

## Özet

* İki yönlü RNN'lerde, her zaman adımı için gizli durum, geçerli zaman adımından önce ve sonra veriler tarafından eş zamanlı olarak belirlenir.
* Çift yönlü RNN'ler olasılıksal grafik modellerde ileri geri algoritma ile çarpıcı bir benzerlik taşır.
* Çift yönlü RNN'ler çoğunlukla sekans kodlaması ve iki yönlü bağlam verilen gözlemlerin tahmini için yararlıdır.
* Çift yönlü RNN'ler, uzun degrade zincirleri nedeniyle antrenman yapmak için çok maliyetlidir.

## Alıştırmalar

1. Farklı yönler farklı sayıda gizli birim kullanıyorsa, $\mathbf{H}_t$'ün şekli nasıl değişecek?
1. Birden fazla gizli katmanla çift yönlü bir RNN tasarlayın.
1. Polysemy doğal dillerde yaygındır. Örneğin, “banka” kelimesinin “nakit yatırmak için bankaya gittim” ve “oturmak için bankaya gittim” bağlamlarında farklı anlamları vardır. Bir bağlam dizisi ve bir kelime verilen bir sinir ağı modelini nasıl tasarlayabiliriz, bağlamda kelimenin vektör temsili döndürülür? Polemiyi işlemek için hangi tür sinir mimarileri tercih edilir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1059)
:end_tab:
