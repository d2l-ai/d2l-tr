# AutoRec: Otomatik Kodlayıcılar ile Değerlendirme Tahmini

Matrisi çarpanlara ayırma modeli, derecelendirme tahmini görevinde iyi bir performans elde etmesine rağmen, aslında doğrusal bir modeldir. Bu nedenle, bu tür modeller, kullanıcıların tercihlerini öngörebilecek karmaşık doğrusal olmayan ve dallı budaklı ilişkileri yakalama yeteneğine sahip değildir. Bu bölümde, doğrusal olmayan bir sinir ağı işbirlikçi filtreleme modeli olan AutoRec'i :cite:`Sedhain.Menon.Sanner.ea.2015` tanıtıyoruz. Otomatik kodlayıcı mimarisiyle işbirlikçi filtrelemeyi (İF) tanımlar ve doğrusal olmayan dönüşümleri açık geri bildirim temelinde İF'ye entegre etmeyi amaçlar. Sinir ağlarının herhangi bir sürekli fonksiyona yaklaşabildiği kanıtlanmıştır, bu da onu matris çarpanlarına ayırmanın sınırlamasını ele almaya ve matris çarpanlarına ayırmanın ifadesini zenginleştirmeye uygun hale getirir. 

Bir yandan AutoRec, girdi katmanı, gizli bir katman ve geri çatma (çıktı) katmanından oluşan bir otomatik kodlayıcı ile aynı yapıya sahiptir. Otomatik kodlayıcı, girdileri gizli (ve genellikle düşük boyutlu) temsillere kodlamak için girdisini çıktısına kopyalamayı öğrenen bir sinir ağıdır. AutoRec'de, kullanıcıları/öğeleri açıkça düşük boyutlu alana gömmek yerine, girdi olarak etkileşim matrisinin sütununu/satırını kullanır ve ardından çıktı katmanındaki etkileşim matrisini geri çatar. 

Öte yandan, AutoRec geleneksel bir otomatik kodlayıcıdan farklıdır: Gizli temsilleri öğrenmek yerine, AutoRec çıktı katmanını öğrenmeye/geri çatmaya odaklanır. Girdi olarak kısmen gözlemlenen bir etkileşim matrisi kullanır ve tamamlanmış bir derecelendirme matrisini geri çatmayı amaçlar. Bu esnada, girdinin eksik girdileri, tavsiye amacıyla çıktı katmanında geri çatma yoluyla doldurulur.  

AutoRec iki çeşidi vardır: Kullanıcı tabanlı ve öğe tabanlı. Kısaca, burada sadece öğe tabanlı AutoRec'i tanıtıyoruz. Kullanıcı tabanlı AutoRec buna göre türetilebilir. 

## Model

$\mathbf{R}_{*i}$, bilinmeyen derecelendirmelerin varsayılan olarak sıfıra ayarlandığı derecelendirme matrisinin $i.$ sütununu göstersin. Sinir mimarisi şu şekilde tanımlanır: 

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

burada $f(\cdot)$ ve $g(\cdot)$ etkinleştirme fonksiyonlarını temsil eder, $\mathbf{W}$ ve $\mathbf{V}$ ağırlık matrisleri, $\mu$ ve $b$ ek girdilerdir. $h( \cdot )$ AutoRec tüm ağını göstersin. $h(\mathbf{R}_{*i})$ çıktısı, derecelendirme matrisinin $i.$ sütununun geri çatmasıdır. 

Aşağıdaki amaç işlevi geri çatma hatasının en aza indirilmesini amaçlamaktadır: 

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

burada $\| \cdot \|_{\mathcal{O}}$, yalnızca gözlenen derecelendirmelerin katkısı dikkate alındığı anlamına gelir, yani yalnızca gözlenen girdilerle ilişkili ağırlıklar geri yayma sırasında güncellenir.

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Modelin Uygulanması

Tipik bir otomatik kodlayıcı bir kodlayıcı ve bir kodçözücüden oluşur. Kodlayıcı, girdiyi gizli temsillere iz düşürür ve kodçözücü gizli katmanı geri çatma katmanıyla eşler. Bu yöntemi takip ediyoruz ve kodlayıcıyı ve kodçözücüyü yoğun katmanlarla oluşturuyoruz. Kodlayıcının etkinleştirilmesi varsayılan olarak `sigmoid` diye ayarlanır ve kodçözücü için hiçbir etkinleştirme uygulanmaz. Aşırı öğrenmeyi azaltmak için kodlama dönüşümünden sonra hattan düşürme eklenir. Gözlenmeyen girdilerin gradyanları, yalnızca gözlenen derecelendirmelerin model öğrenme sürecine katkıda bulunmasını sağlamak için maskelenir.

```{.python .input  n=2}
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Eğitim sırasında gradyanı maskeleyin
            return pred * np.sign(input)
        else:
            return pred
```

## Değerlendiriciyi Yeniden Uygulamak

Girdi ve çıktı değiştiğinden, değerlendirme işlevini yeniden uygulamalıyız, ancak doğruluk ölçütü olarak RMSE'yi kullanıyoruz.

```{.python .input  n=3}
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # test RMSE'sini hesapla
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Model Eğitimi ve Değerlendirilmesi

Şimdi, MovieLens veri kümesinde AutoRec'i eğitip değerlendirelim. Testin RMSE'sinin matrisi çarpanlarına ayırma modelinden daha düşük olduğunu ve sinir ağlarının derecelendirme tahmini görevindeki etkinliğini doğruladığını açıkça görebiliriz.

```{.python .input  n=4}
devices = d2l.try_all_gpus()
# MovieLens 100K veri kümesini yükle
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model ilkleme, eğitim ve değerlendirme
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Özet

* Doğrusal olmayan katmanları ve hattan düşürme düzenlileştirmesini entegre ederken, matrisi çarpanlara ayırma algoritmasını otomatik kodlayıcılarla çerçeveleyebiliriz.
* MovieLens 100K veri kümesindeki deneyler, AutoRec'in matris çarpanlarına ayırmaya kıyasla üstün performans sağladığını gösteriyor.

## Alıştırmalar

* Model performansı üzerindeki etkisini görmek için AutoRec'in gizli boyutunu değiştirin.
* Daha fazla gizli katman eklemeyi deneyin. Model performansını iyileştirmede yararlı mıdır?
* Kodçözücü ve kodlayıcı etkinleştirme işlevlerinin daha iyi bir kombinasyonunu bulabilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/401)
:end_tab:
