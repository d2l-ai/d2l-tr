# Derin Çarpanlara Ayırma Makinaları

Etkili özellik kombinasyonlarını öğrenmek, tıklama oranı tahmini görevinin başarısı açısından kritik öneme sahiptir. Çarpanlara aykırma makineleri lineer paradigmada özellik etkileşimleri modellemesi (örn. Bilineer etkileşimler). Bu genellikle doğal özellik geçiş yapılarının genellikle çok karmaşık ve doğrusal olmayan olduğu gerçek dünya verileri için yetersizdir. Daha da kötüsü, ikinci dereceden özellik etkileşimleri genellikle pratikte çarpanlara aykırma makinelerinde kullanılır. Çarpma makineleriyle yüksek derecede özellik kombinasyonlarının modellenmesi teorik olarak mümkündür, ancak sayısal istikrarsızlık ve yüksek hesaplama karmaşıklığı nedeniyle genellikle kabul edilmez. 

Etkili bir çözüm derin sinir ağları kullanmaktır. Derin sinir ağları, özellik temsili öğreniminde güçlüdür ve sofistike özellik etkileşimlerini öğrenme potansiyeline sahiptir. Bu nedenle, derin sinir ağlarını çarpanlara ayırma makinelerine entegre etmek doğaldır. Çarpanlara doğrusal olmayan dönüşüm katmanları eklenmesi, hem düşük mertebeden özellik kombinasyonlarını hem de yüksek mertebeden özellik kombinasyonlarını modelleme olanağı sağlar. Dahası, girişlerden doğrusal olmayan doğal yapılar da derin sinir ağları ile yakalanabilir. Bu bölümde FM ve derin sinir ağlarını birleştiren derin çarpanlara geçirme makineleri (DeepFM) :cite:`Guo.Tang.Ye.ea.2017` adlı temsili bir model sunacağız. 

## Model Mimarileri

DeepFM, bir FM bileşeninden ve paralel bir yapıya entegre edilmiş derin bir bileşenden oluşur. FM bileşeni, düşük mertebeden özellik etkileşimlerini modellemek için kullanılan 2 yönlü çarpanlara sahip makinelerle aynıdır. Derin bileşen, yüksek mertebeden özellik etkileşimleri ve doğrusal olmayan değerleri yakalamak için kullanılan bir MLP'dir. Bu iki bileşen aynı giriş/gömme paylaşır ve çıkışları nihai tahmin olarak özetlenir. DeepFM ruhunun hem ezberlemeyi hem de genellemeyi yakalayabilen Wide\ & Deep mimarisine benzediğini belirtmek gerekir. DeepFM'in Wide\ & Deep modeline göre avantajları, özellik kombinasyonlarını otomatik olarak belirleyerek el yapımı özellik mühendisliğinin çabasını azaltmaktır. 

Kısalık için FM bileşeninin açıklamasını atlıyoruz ve çıkışı $\hat{y}^{(FM)}$ olarak belirtiyoruz. Okuyucular daha fazla ayrıntı için son bölüme sevk edilir. $\mathbf{e}_i \in \mathbb{R}^{k}$, $i^\mathrm{th}$ alanının gizli özellik vektörünü göstersin. Derin bileşenin girişi, seyrek kategorik özellik girişi ile bakılan tüm alanların yoğun gömülmelerinin birleştirilmesiyle, şu şekilde gösterilir: 

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

burada $f$ alan sayısıdır. Daha sonra aşağıdaki sinir ağına beslenir: 

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

burada $\alpha$ aktivasyon fonksiyonudur. $\mathbf{W}_{l}$ ve $\mathbf{b}_{l}$, $l^\mathrm{th}$ katmanındaki ağırlık ve önyargıdır. $y_{DNN}$'ün tahminin çıktısını göstermesine izin verin. DeepFM'in nihai tahmini hem FM hem de DNN çıkışlarının toplamıdır. Yani elimizde: 

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

burada $\sigma$ sigmoid fonksiyonudur. DeepFM mimarisi aşağıda gösterilmiştir. [Illustration of the DeepFM model](../img/rec-deepfm.svg) 

Derin sinir ağlarını FM ile birleştirmenin tek yolu DeepFM'in olmadığını belirtmek gerekir. Ayrıca özellik etkileşimleri :cite:`He.Chua.2017` üzerine doğrusal olmayan katmanlar ekleyebiliriz.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## DeepFM Uygulaması DeepFM'in uygulanması FM ile benzerdir. FM parçasını değişmeden tutuyoruz ve aktivasyon fonksiyonu olarak `relu` ile bir MLP bloğu kullanıyoruz. Bırakma modeli düzenli hale getirmek için de kullanılır. MLP nöronların sayısı `mlp_dims` hiperparametre ile ayarlanabilir.

```{.python .input  n=2}
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## Modelin Eğitimi ve Değerlendirilmesi Veri yükleme işlemi FM ile aynıdır. DeepFM'nin MLP bileşenini bir piramit yapısına sahip üç katmanlı yoğun bir ağa ayarladık (30-20-10). Diğer tüm hiperparametreler FM ile aynı kalır.

```{.python .input  n=4}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

FM ile karşılaştırıldığında DeepFM daha hızlı yakınlaşır ve daha iyi performans sağlar. 

## Özet

* Sinir ağlarının FM'ye entegre edilmesi, karmaşık ve yüksek mertebeden etkileşimleri modellemesini sağlar.
* DeepFM, reklam veri kümelerinde orijinal FM'den daha iyi performans gösterir.

## Egzersizler

* Model performansı üzerindeki etkisini kontrol etmek için MLP yapısını değiştir.
* Veri kümesini Criteo olarak değiştirin ve orijinal FM modeliyle karşılaştırın.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:
