# Çarpanlara Ayırma Makinaları

2010 yılında Steffen Rendle tarafından önerilen çarpanlara ayırma makineleri (FM) :cite:`Rendle.2010`, sınıflandırma, regresyon ve sıralama görevleri için kullanılabilen denetimli bir algoritmadır. Hızla fark edildi ve tahminler ve öneriler yapmak için popüler ve etkili bir yöntem haline geldi. Özellikle, doğrusal regresyon modelinin ve matris çarpanlara çıkarma modelinin genelleştirilmesidir. Dahası, polinom çekirdeği olan destek vektör makinelerini andırır. Çarpanlara ayırma makinelerinin doğrusal regresyon ve matris çarpanlaştırması üzerindeki güçlü yönleri şunlardır: (1) $\chi$ yönlü değişken etkileşimleri modelleyebilir, burada $\chi$ polinom düzeninin sayısıdır ve genellikle ikiye ayarlanır. (2) Çarpanlara ayırma makineleri ile ilişkili hızlı optimizasyon algoritması azaltabilir polinom hesaplama süresi doğrusal karmaşıklığa, özellikle yüksek boyutlu seyrek girişler için son derece verimli hale getirir. Bu nedenlerden dolayı çarpanlara ayırma makineleri modern reklam ve ürün önerilerinde yaygın olarak kullanılmaktadır. Teknik detaylar ve uygulamalar aşağıda açıklanmıştır. 

## 2-Yönlü Çarpanlara Ayak Makinaları

Resmi olarak, $x \in \mathbb{R}^d$ bir numunenin özellik vektörlerini göstersin ve $y$ gerçek değerli etiket veya ikili sınıf “tıklatma/tıklama-tıklama” gibi sınıf etiketi olabilen karşılık gelen etiketi gösterir. İkinci derece bir çarpanlara ayırma makinesi modeli şu şekilde tanımlanır: 

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

burada $\mathbf{w}_0 \in \mathbb{R}$ küresel önyargıdır; $\mathbf{w} \in \mathbb{R}^d$ i-th değişkeninin ağırlıklarını gösterir; $\mathbf{V} \in \mathbb{R}^{d\times k}$ özellik gömme özelliğini temsil eder; $\mathbf{v}_i$ $\mathbf{V}$'ün $i^\mathrm{th}$ satırını temsil eder; $k$ gizli faktörlerin boyutsallığıdır; $\langle\cdot, \cdot \rangle$ iki vektörün nokta ürünüdür. $\langle\cdot, \cdot \rangle$, iki vektörün nokta ürünüdür. 293617 modeli etkileşim $i^\mathrm{th}$ ve $j^\mathrm{th}$ özelliği arasında. Bazı özellik etkileşimleri kolayca anlaşılabilir, böylece uzmanlar tarafından tasarlanabilirler. Bununla birlikte, diğer özellik etkileşimlerinin çoğu verilerde gizlidir ve tanımlanması zordur. Böylece modelleme özelliği etkileşimleri otomatik olarak özellik mühendisliği çabalarını büyük ölçüde azaltabilir. İlk iki terimin doğrusal regresyon modeline karşılık geldiği ve son terimin matris çarpanlara ayırma modelinin bir uzantısı olduğu açıktır. $i$ özelliği bir öğeyi temsil eder ve $j$ özelliği bir kullanıcıyı temsil ediyorsa, üçüncü terim tam olarak kullanıcı ve öğe gömme arasındaki nokta ürünüdür. FM'nin daha yüksek siparişlere genelleme yapabileceğini de belirtmek gerekir (derece > 2). Bununla birlikte, sayısal kararlılık genelleme zayıflatabilir. 

## Verimli Bir Optimizasyon Ölçütü

Çarpma makinelerinin düz ileri bir yöntemle optimize edilmesi, tüm çift yönlü etkileşimlerin hesaplanması gerektiği için $\mathcal{O}(kd^2)$ karmaşıklığına yol açar. Bu verimsizlik sorununu çözmek için, FM'nin üçüncü terimini yeniden düzenleyebiliriz, bu da hesaplama maliyetini büyük ölçüde düşürebilir ve doğrusal bir zaman karmaşıklığına yol açar ($\mathcal{O}(kd)$). İkili etkileşim teriminin yeniden formülasyonu aşağıdaki gibidir: 

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Bu reformülasyon ile model karmaşıklığı büyük ölçüde azaltılır. Dahası, seyrek özellikler için, yalnızca sıfır olmayan elementlerin hesaplanması gerekir, böylece genel karmaşıklık sıfır olmayan özelliklerin sayısına doğrusal olur. 

FM modelini öğrenmek için, regresyon görevi için MSE kaybını, sınıflandırma görevleri için çapraz entropi kaybı ve sıralama görevi için BPR kaybını kullanabiliriz. Stokastik degrade iniş ve Adam gibi standart iyileştiriciler optimizasyon için uygundur.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Model Uygulaması Aşağıdaki kod çarpanlara ayırma makinelerini uygular. FM'nin doğrusal bir regresyon bloğu ve verimli bir özellik etkileşim bloğu içerdiğini görmek açıktır. CTR tahminini sınıflandırma görevi olarak ele aldığımızdan, final puanı üzerinde sigmoid fonksiyon uyguluyoruz.

```{.python .input  n=2}
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## Reklam Veri Kümesini Yükle Online reklam veri kümesini yüklemek için son bölümdeki CTR veri sarmalayıcısını kullanırız.

```{.python .input  n=3}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Modeli Sonrasında eğitiyoruz, modeli eğitiyoruz. Öğrenme oranı 0,02 olarak ayarlanır ve gömme boyutu varsayılan olarak 20'ye ayarlanır. `Adam` iyileştirici ve `SigmoidBinaryCrossEntropyLoss` kayıp model eğitimi için kullanılır.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Özet

* FM, regresyon, sınıflandırma ve sıralama gibi çeşitli görevlere uygulanabilen genel bir çerçevedir.
* Özellik etkileşimi/geçiş, tahmin görevleri için önemlidir ve 2 yönlü etkileşim FM ile etkili bir şekilde modellenebilir.

## Egzersizler

* FM'yi Avazu, MovieLens ve Criteo veri kümeleri gibi diğer veri kümelerinde test edebilir misiniz?
* Performans üzerindeki etkisini kontrol etmek için gömme boyutu farklı, matris çarpanlarına benzer bir desen gözlemleyebilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:
