# Çift Yönlü Yinelemeli Sinir Ağları
:label:`sec_bi_rnn`

Dizi öğrenmede, şu ana kadar amacımızın, şimdiye kadar gördüklerimizle bir sonraki çıktıyı modellemek olduğunu varsaydık; örneğin, bir zaman serisi veya bir dil modeli bağlamında. Bu tipik bir senaryo olsa da, karşılaşabileceğimiz tek senaryo değil. Sorunu göstermek için, bir metin dizisinde boşluk doldurmada aşağıdaki üç görevi gözünüzde canlandırın:

* Ben `___`.
* `___` açım.
* `___` açım ve yarım kuzu yiyebilirim.

Mevcut bilgi miktarına bağlı olarak boşlukları “mutluyum”, “yarı” ve “çok” gibi çok farklı kelimelerle doldurabiliriz. Açıkça ifadenin sonu (varsa) hangi kelimenin alınacağı hakkında önemli bilgiler taşır. Bundan yararlanamayan bir dizi modeli, ilgili görevlerde kötü performans gösterecektir. Örneğin, adlandırılmış varlık tanımada iyi iş yapmak için (örneğin, “Yeşil”'in “Bay Yeşil”'i ya da rengi ifade edip etmediğini tanımak) daha uzun menzilli bağlam aynı derecede hayati önem taşır. Problemi ele alırken biraz ilham almak için olasılıksal çizge modellerde biraz dolanalım.

## Gizli Markov Modellerinde Dinamik Programlama

Bu alt bölüm, dinamik programlama problemini göstermeyi amaçlar. Belirli teknik detaylar derin öğrenme modellerini anlamak için önemli değildir, ancak neden derin öğrenmeyi kullanabileceğimizi ve neden belirli mimarileri seçebileceğimizi anlamaya yardımcı olurlar.

Problemi olasılıksal çizge modelleri kullanarak çözmek istiyorsak, mesela aşağıdaki gibi gizli bir değişken modeli tasarlayabiliriz. Herhangi bir $t$ zaman adımında, $P(x_t \mid h_t)$ olasılığında $x_t$ aracılığıyla gözlenen salınımı yöneten bazı gizli $h_t$ değişkenimiz olduğunu varsayalım. Dahası, herhangi bir $h_t \to h_{t+1}$ geçişi, bir durum geçiş olasılığı $P(h_{t+1} \mid h_{t})$ ile verilir. Bu olasılıksal çizge modeli :numref:`fig_hmm` şeklinde olduğu gibi bir *saklı Markov modeli*dir.

![Saklı Markov Modeli.](../img/hmm.svg)
:label:`fig_hmm`

Böylece, $T$ gözlemlerinin bir dizisi için gözlemlenen ve gizli durumlar üzerinde aşağıdaki bileşik olasılık dağılımına sahibiz:

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ öyleki } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

Şimdi bazı $x_j$'ler hariç tüm $x_i$'leri gözlemlediğimizi varsayalım ve amacımız $P(x_j \mid x_{-j})$'yı hesaplamaktır ve burada $x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$'dir. $P(x_j \mid x_{-j})$'te saklı bir değişken olmadığından, $h_1, \ldots, h_T$ için olası tüm seçenek kombinasyonlarını toplamayı düşünürüz. Herhangi bir $h_i$'nin $k$ farklı değerlere (sonlu sayıda durum) sahip olması durumunda, bu, $k^T$ terimi toplamamız gerektiği anlamına gelir; bu da genellikle imkansız bir işlemdir! Neyse ki bunun için şık bir çözüm var: *Dinamik programlama*.

Nasıl çalıştığını görmek için, sırayla $h_1, \ldots, h_T$ saklı değişkenleri üzerinde toplamayı düşünün. :eqref:`eq_hmm_jointP` denklemine göre, şu ifadeye varırız:

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

Genel bir *ileriye özyineleme*ye sahip oluruz:

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

Özyineleme $\pi_1(h_1) = P(h_1)$ olarak ilklenir. Soyut terimlerle bu $\pi_{t+1} = f(\pi_t, x_t)$ olarak yazılabilir, burada $f$ bir öğrenilebilir işlevdir. Bu, RNN bağlamında şimdiye kadar tartıştığımız saklı değişken modellerindeki güncelleme denklemine çok benziyor!

İleriye özyinelemeye tamamen benzer şekilde, geriye özyineleme ile aynı saklı değişken kümesi üzerinden toplayabiliriz. Böylece şu ifadeye varırız:

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

Böylece *geriye özyineleme* olarak yazabiliriz:

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

$\rho_T(h_T) = 1$ olarak ilkleriz. Hem ileriye hem de geriye özyinelemeler $T$ saklı değişkenlerini üstel zaman yerine bütün $(h_1, \ldots, h_T)$ değerler için $\mathcal{O}(kT)$ (doğrusal) zaman içinde toplamamızı sağlar. Bu, çizgesel modellerle olasılık çıkarımının en büyük avantajlarından biridir. Aynı zamanda genel mesaj geçişi algoritmasının çok özel bir örneğidir :cite:`Aji.McEliece.2000`. Hem ileriye hem de geri özyinelemeleri birleştirerek, şöyle bir hesaplamaya ulaşabiliriz:

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

Soyut terimlerle geriye özyinelemenin $\rho_{t-1} = g(\rho_t, x_t)$ olarak yazılabileceğini unutmayın; burada $g$ öğrenilebilir bir işlevdir. Yine, bu aşırı şekilde bir güncelleme denklemi gibi görünüyor, sadece bizim RNN'de şimdiye kadar gördüğümüzün aksine geriye doğru çalışıyor. Gerçekten de, gizli Markov modelleri, mevcut olduğunda gelecekteki verileri bilmekten fayda sağlar. Sinyal işleme bilim insanları, gelecekteki gözlemleri bilme ve bilmemenin arasında aradeğerleme (interpolation) ve dışdeğerleme (extrapolation) diye ayrım yapar. Daha fazla ayrıntı için dizili Monte Carlo algoritmaları kitabının tanıtım bölümüne bakın :cite:`Doucet.De-Freitas.Gordon.2001`.

## Çift Yönlü Model

Eğer RNN'lerde gizli Markov modellerinde olana benzer bir ileri-bakış yeteneği sunan bir mekanizmaya sahip olmak istiyorsak, şimdiye kadar gördüğümüz RNN tasarımını değiştirmeliyiz. Neyse ki, bu kavramsal olarak kolaydır. Bir RNN'yi sadece ilk andıçtan başlayarak ileri modda çalıştırmak yerine, son andıçtan arkadan öne çalışan başka bir tane daha başlatırız. *Çift yönlü RNN*, bu tür bilgileri daha esnek bir şekilde işlemek için bilgileri geriye doğru ileten gizli bir katman ekler. :numref:`fig_birnn`, tek bir gizli katmanlı çift yönlü RNN mimarisini göstermektedir.

![Çift yönlü RNN mimarisi.](../img/birnn.svg)
:label:`fig_birnn`

Aslında, bu, gizli Markov modellerinin dinamik programlanmasındaki ileriye ve geriye özyinelemelerden çok farklı değildir. Ana ayrım, önceki durumda bu denklemlerin belirli bir istatistiksel anlamı olmasıdır. Artık bu kadar kolay erişilebilir yorumlardan yoksunlar ve biz onlara genel ve öğrenilebilir işlevler olarak davranabiliriz. Bu geçiş, modern derin ağların tasarımına rehberlik eden ilkelerin çoğunu özetliyor: Önce, klasik istatistiksel modellerin fonksiyonel bağımlılıklarının türünü kullanın ve daha sonra bunları genel bir biçimde parameterize edin.

### Tanım

Çift yönlü RNN'ler :cite:`Schuster.Paliwal.1997` çalışmasında tanıtıldı. Çeşitli mimarilerin ayrıntılı bir tartışması için de :cite:`Graves.Schmidhuber.2005` çalışmasında bakınız. Böyle bir ağın özelliklerine bakalım.

Herhangi bir $t$ zaman adımı için, bir minigrup girdisi $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (örnek sayısı: $n$, her örnekteki girdi sayısı: $d$) verildiğinde gizli katman etkinleştirme işlevinin $\phi$ olduğunu varsayalım. Çift yönlü mimaride, bu zaman adımı için ileriye ve geriye doğru gizli durumların sırasıyla $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ ve $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ olduğunu varsayıyoruz, burada $h$ gizli birimlerin sayısıdır. İleriye ve geriye doğru gizli durum güncelleştirmeleri aşağıdaki gibidir:

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

burada $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$ ağırlıkları ve $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$ ek girdileri tüm model parametreleridir.

Daha sonra, $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ gizli durumunu çıktı katmanına beslemek için $\overrightarrow{\mathbf{H}}_t$ ve $\overleftarrow{\mathbf{H}}_t$, ileriye ve geriye gizli durumlarını birleştiririz. Birden fazla gizli katman içeren derin çift yönlü RNN'lerde, bu tür bilgiler sonraki çift yönlü katmana *girdi* olarak aktarılır. Son olarak, çıktı katmanı $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ çıktısını hesaplar (çıktı sayısı: $q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Burada, $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ ağırlık matrisi ve $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ ek girdisi, çıktı katmanının model parametreleridir. Aslında, iki yön farklı sayıda gizli birime sahip olabilir.

### Hesaplama Maliyeti ve Uygulamalar

Çift yönlü RNN'nin temel özelliklerinden biri, dizinin her iki ucundan gelen bilgilerin çıktıyı tahmin etmek için kullanıldığıdır. Yani, mevcut olanı tahmin etmek için hem gelecekteki hem de geçmişteki gözlemlerden gelen bilgileri kullanırız. Bir sonraki andıç tahmininde bu istediğimiz şey değildir. Sonuçta, sonraki andıcı tahmin ederken bir sonraki andıcı bilme lüksüne sahip değiliz. Dolayısıyla, çift yönlü bir RNN kullansaydık, çok iyi bir doğruluk elde edemezdik: Eğitim sırasında şimdiki zamanı tahmin etmek için geçmişteki ve gelecekteki verilere sahibiz. Test süresi boyunca ise elimizde sadece geçmişteki veri var ve dolayısıyla düşük doğruluğumuz olur. Bunu aşağıda bir deneyde göstereceğiz.

Yaraya tuz ekler gibi üstelik çift yönlü RNN'ler de son derece yavaştır. Bunun başlıca nedeni ileri yaymanın iki yönlü katmanlarda hem ileri hem de geri özyinelemeler gerektirmesi ve geri yaymanın ileri yayma sonuçlarına bağlı olmasıdır. Bu nedenle, gradyanlar çok uzun bir bağlılık zincirine sahip olacaktır.

Pratikte çift yönlü katmanlar çok az kullanılır ve yalnızca eksik kelimeleri doldurma, andıçlara açıklama ekleme (örneğin, adlandırılmış nesne tanıma için) ve dizileri toptan kodlayarak diziyi veri işleme hattında bir adım işlemek gibi (örneğin, makine çevirisi için) dar bir uygulama kümesi için kullanılır. :numref:`sec_bert` ve :numref:`sec_sentiment_rnn` içinde, metin dizilerini kodlamak için çift yönlü RNN'lerin nasıl kullanılacağını tanıtacağız.

## (**Yanlış Bir Uygulama İçin Çift Yönlü RNN Eğitmek**)

İki yönlü RNN'lerin geçmişteki ve gelecekteki verileri kullanmalarına gerçeğine ilişkin tüm ikazları görmezden gelirsek ve basitçe dil modellerine uygularsak, kabul edilebilir bir şaşkınlık değeriyle tahminler elde edebiliriz. Bununla birlikte, modelin gelecekteki andıçlarını tahmin etme yeteneği, aşağıdaki deneyin gösterdiği gibi ciddi bir şekilde zarar sokulmuştur. Makul şaşkınlığa rağmen, birçok yinelemeden sonra bile anlamsız ifadeler üretir. Aşağıdaki kodu, yanlış bağlamda kullanmaya karşı uyarıcı bir örnek olarak ekliyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Veriyi yükle
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Çift yönlü LSTM modelini `bidirectional=True` olarak ayarlayarak tanımlayın.
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Modeli eğitin
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Veriyi yükle
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Çift yönlü LSTM modelini `bidirectional=True` olarak ayarlayarak tanımlayın.
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Modeli eğitin
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

Çıktı, yukarıda açıklanan nedenlerden dolayı bariz şekilde tatmin edici değildir. İki yönlü RNN'lerin daha etkili kullanımları hakkında bir tartışma için lütfen :numref:`sec_sentiment_rnn` içindeki duygusallık analizi uygulamasına bakın.

## Özet

* İki yönlü RNN'lerde, her zaman adımı için gizli durum, şu anki zaman adımından önceki ve sonraki veriler tarafından eş zamanlı olarak belirlenir.
* Çift yönlü RNN'ler olasılıksal çizge modellerdeki ileri-geri algoritması ile çarpıcı bir benzerlik taşır.
* Çift yönlü RNN'ler çoğunlukla dizi kodlaması ve çift yönlü bağlam verilen gözlemlerin tahmini için yararlıdır.
* Çift yönlü RNN'lerin uzun gradyan zincirleri nedeniyle eğitilmesi maliyetlidir.

## Alıştırmalar

1. Farklı yönler farklı sayıda gizli birim kullanıyorsa, $\mathbf{H}_t$'nin şekli nasıl değişecektir?
1. Birden fazla gizli katmanlı çift yönlü bir RNN tasarlayın.
1. Çok anlamlılık doğal dillerde yaygındır. Örneğin, “banka” kelimesinin “nakit yatırmak için bankaya gittim” ve “oturmak için banka doğru gittim” bağlamlarında farklı anlamları vardır. Bir bağlam dizisi ve bir kelime, bağlamda kelimenin vektör temsilini döndüren bir sinir ağı modelini nasıl tasarlayabiliriz? Çok anlamlılığın üstesinden gelmek için hangi tür sinir mimarileri tercih edilir?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1059)
:end_tab:
