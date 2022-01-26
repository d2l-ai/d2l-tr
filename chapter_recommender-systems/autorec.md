# AutoRec: Otomatik kodlayıcılar ile Değerlendirme Tahmini

Matris çarpanlara çıkarma modeli, derecelendirme tahmini görevinde iyi bir performans elde etse de, aslında doğrusal bir modeldir. Bu nedenle, bu tür modeller, kullanıcıların tercihlerini tahmin edebilecek karmaşık doğrusal olmayan ve karmaşık ilişkileri yakalama yeteneğine sahip değildir. Bu bölümde, doğrusal olmayan bir sinir ağı işbirlikçi filtreleme modeli olan AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015` tanıtıyoruz. Otomatik kodlayıcı mimarisiyle işbirliğine dayalı filtrelemeyi (CF) tanımlar ve doğrusal olmayan dönüşümleri açık geri bildirimler temelinde CF'ye entegre etmeyi amaçlar. Sinir ağlarının herhangi bir sürekli fonksiyona yaklaşabildiği kanıtlanmıştır, bu da matris çarpanlarına ilişkin sınırlandırmanın sınırlandırılmasını ve matris çarpanlaştırmasının ifade gücünü zenginleştirmeye uygun hale getirilmiştir. 

Bir yandan AutoRec, giriş katmanı, gizli bir katman ve yeniden yapılanma (çıktı) katmanından oluşan bir otomatik kodlayıcı ile aynı yapıya sahiptir. Otomatik kodlayıcı, girdileri gizli (ve genellikle düşük boyutlu) temsillere kodlamak için girişini çıktısına kopyalamayı öğrenen bir sinir ağıdır. AutoReg'de, kullanıcıları/öğeleri açıkça düşük boyutlu alana gömmek yerine, girdi olarak etkileşim matrisinin sütununu/satırını kullanır ve ardından çıktı katmanındaki etkileşim matrisini yeniden oluşturur. 

Öte yandan, AutoRec geleneksel bir otomatik kodlayıcıdan farklıdır: gizli temsilleri öğrenmek yerine, AutoRec çıktı katmanını öğrenme/yeniden yapılandırmaya odaklanır. Tamamlanmış bir derecelendirme matrisini yeniden oluşturmayı amaçlayan giriş olarak kısmen gözlenen bir etkileşim matrisini kullanır. Bu arada, girdinin eksik girişleri, öneri amacıyla yeniden yapılanma yoluyla çıkış katmanına doldurulur.  

AutoRec iki çeşidi vardır: kullanıcı tabanlı ve öğe tabanlı. Kısalık için, burada sadece öğe tabanlı AutoRec tanıtıyoruz. Kullanıcı tabanlı AutoRec buna göre türetilebilir. 

## Model

$\mathbf{R}_{*i}$'in, bilinmeyen derecelendirmelerin varsayılan olarak sıfıra ayarlandığı derecelendirme matrisinin $i^\mathrm{th}$ sütununu göstermesine izin verin. Sinir mimarisi şu şekilde tanımlanır: 

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

burada $f(\cdot)$ ve $g(\cdot)$ aktivasyon fonksiyonlarını temsil eder, $\mathbf{W}$ ve $\mathbf{V}$ ağırlık matrisleri, $\mu$ ve $b$ önyargılardır. $h( \cdot )$ AutoRec tüm ağını göstersin. $h(\mathbf{R}_{*i})$ çıkışı, derecelendirme matrisinin $i^\mathrm{th}$ sütununun yeniden yapılandırılmasıdır. 

Aşağıdaki objektif işlevi yeniden yapılanma hatasının en aza indirilmesini amaçlamaktadır: 

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

burada $\| \cdot \|_{\mathcal{O}}$, yalnızca gözlenen derecelendirmelerin katkısı dikkate alındığı anlamına gelir, yani yalnızca gözlenen girişlerle ilişkili ağırlıklar geri yayılım sırasında güncellenir.

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Modelin Uygulanması

Tipik bir otomatik kodlayıcı bir kodlayıcı ve bir kod çözücüden oluşur. Kodlayıcı, girdiyi gizli temsillere yansıtır ve kod çözücü gizli katmanı yeniden yapılanma katmanıyla eşler. Bu uygulamayı takip ediyoruz ve kodlayıcıyı ve kod çözücüyü yoğun katmanlarla oluşturuyoruz. Kodlayıcının etkinleştirilmesi varsayılan olarak `sigmoid` olarak ayarlanır ve kod çözücü için hiçbir etkinleştirme uygulanmaz. Aşırı uyumu azaltmak için kodlama dönüşümünden sonra bırakma dahildir. Gözlenmeyen girdilerin degradeleri, yalnızca gözlenen derecelendirmelerin model öğrenme sürecine katkıda bulunmasını sağlamak için maskelenir.

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
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## Değerlendiriciyi Yeniden Uygulanıyor

Giriş ve çıkış değiştiğinden, değerlendirme işlevini yeniden uygulamalıyız, ancak doğruluk ölçüsü olarak RMSE'yi kullanıyoruz.

```{.python .input  n=3}
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## Modelin Eğitimi ve Değerlendirilmesi

Şimdi, MovieLens veri kümesinde AutoReg'i eğitip değerlendirelim. RMSE testinin matris çarpanlarına modelinden daha düşük olduğunu ve nöral ağların derecelendirme tahmini görevindeki etkinliğini doğruladığını açıkça görebiliriz.

```{.python .input  n=4}
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
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
# Model initialization, training, and evaluation
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

* Matris çarpanlara ayırma algoritmasını otomatik kodlayıcılar ile çerçeveleyebiliriz, doğrusal olmayan katmanları ve bırakma düzenini entegre edebiliriz. 
* MovieLens 100K veri kümesindeki deneyler, AutoReg'in matris çarpanlarına kıyasla üstün performans sağladığını gösteriyor.

## Egzersizler

* Model performansı üzerindeki etkisini görmek için AutoReg'in gizli boyutunu değiştir.
* Daha fazla gizli katman eklemeyi deneyin. Model performansını iyileştirmek yararlı mıdır?
* Kod çözücü ve kodlayıcı aktivasyon işlevlerinin daha iyi bir kombinasyonunu bulabilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:
