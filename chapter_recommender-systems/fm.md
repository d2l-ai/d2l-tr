# Çarpanlara Ayırma Makineleri

2010 yılında Steffen Rendle tarafından önerilen çarpanlara ayırma makineleri (FM) :cite:`Rendle.2010`, sınıflandırma, bağlanım ve sıralama görevleri için kullanılabilen bir gözetimli algoritmadır. Hızla fark edildi ve tahminler ve öneriler yapmak için popüler ve etkili bir yöntem haline geldi. Özellikle, doğrusal bağlanım modelinin ve matris çarpanlara ayırma modelinin genelleştirilmesidir. Dahası, polinom çekirdeği olan destek vektör makinelerini andırır. Çarpanlara ayırma makinelerinin doğrusal bağlanım ve matris çarpanlara ayırma üzerindeki güçlü yönleri şunlardır: (1) $\chi$ yönlü değişken etkileşimleri modelleyebilir, burada $\chi$ polinom kuvvetinin sayısıdır ve genellikle ikiye ayarlanır. (2) Çarpanlara ayırma makineleriyle ilişkili hızlı bir eniyileme algoritması, polinom hesaplama süresini doğrusal karmaşıklığa indirerek, özellikle yüksek boyutlu seyrek girdiler için son derece verimli hale getirebilir. Bu nedenlerden dolayı çarpanlara ayırma makineleri modern reklam ve ürün tavsiyelerinde yaygın olarak kullanılmaktadır. Teknik detaylar ve uygulamalar aşağıda açıklanmıştır. 

## 2-Yönlü Çarpanlara Ayırma Makineleri

Biçimsel olarak, $x \in \mathbb{R}^d$ bir örneklemin öznitelik vektörlerini göstersin ve $y$ gerçek değerli etiket veya ikili sınıf “tıklanma/tıklanmama” gibi sınıf etiketi olabilen karşılık gelen etiketi gösterir. İkinci derece bir çarpanlara ayırma makinesi modeli şu şekilde tanımlanır: 

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

burada $\mathbf{w}_0 \in \mathbb{R}$ küresel ek girdidir; $\mathbf{w} \in \mathbb{R}^d$ i. değişkeninin ağırlıklarını gösterir; $\mathbf{V} \in \mathbb{R}^{d\times k}$ öznitelik gömmelerini temsil eder; $\mathbf{v}_i$ $\mathbf{V}$'nin $i.$ satırını temsil eder; $k$ gizli çarpanların boyutsallığıdır; $\langle\cdot, \cdot \rangle$ iki vektörün nokta çarpımıdır. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$, $i.$ ve $j.$ özniteliği arasındaki etkileşimi modeller. Bazı öznitelik etkileşimleri kolayca anlaşılabilir, böylece uzmanlar tarafından tasarlanabilirler. Bununla birlikte, diğer öznitelik etkileşimlerinin çoğu verilerde gizlidir ve tanımlanması zordur. Böylece, öznitelik etkileşimlerini otomatik olarak modellemek, öznitelik mühendisliğindeki çabaları büyük ölçüde azaltabilir. İlk iki terimin doğrusal bağlanım modeline karşılık geldiği ve son terimin matris çarpanlara ayırma modelinin bir uzantısı olduğu açıktır. $i$ özniteliği bir öğeyi temsil eder ve $j$ özniteliği bir kullanıcıyı temsil ediyorsa, üçüncü terim tam olarak kullanıcı ve öğe gömmeleri arasındaki nokta çarpımıdır. FM'nin daha yüksek kuvvetlere genelleme yapabileceğini de belirtmek gerekir (kuvvet > 2). Bununla birlikte, sayısal kararlılık genellemeyi zayıflatabilir. 

## Verimli Bir Optimizasyon Ölçütü

Çarpanlara ayırma makinelerinin düz ileri bir yöntemle optimize edilmesi, tüm ikili etkileşimlerin hesaplanması gerektiği için $\mathcal{O}(kd^2)$ karmaşıklığına yol açar. Bu verimsizlik sorununu çözmek için, FM'nin üçüncü terimini yeniden düzenleyebiliriz, bu da hesaplama maliyetini büyük ölçüde düşürebilir ve doğrusal bir zaman karmaşıklığına yol açar ($\mathcal{O}(kd)$). İkili etkileşim teriminin yeniden formülleştirilmesi aşağıdaki gibidir: 

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Bu yeniden formülleştirme ile model karmaşıklığı büyük ölçüde azaltılır. Dahası, seyrek öznitelikler için, yalnızca sıfır olmayan elemanların hesaplanması gerekir, böylece genel karmaşıklık sıfır olmayan özniteliklerin sayısına doğrusal oranda olur. 

FM modelini öğrenmek için, bağlanım görevi için MSE kaybını, sınıflandırma görevleri için çapraz entropi kaybını ve sıralama görevi için BPR kaybını kullanabiliriz. Rasgele gradyan inişi ve Adam gibi standart eniyileştiriciler eniyileme için uygundur.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Model Uygulaması 
Aşağıdaki kod çarpanlara ayırma makinelerini uygular. FM'nin doğrusal bir bağlanım bloğu ve verimli bir öznitelik etkileşim bloğu içerdiğini görmek barizdir. CTR tahminini sınıflandırma görevi olarak ele aldığımızdan, nihai puan üzerinde sigmoid işlevi uyguluyoruz.

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

## Reklam Veri Kümesini Yükleme 
Çevrimiçi reklam veri kümesini yüklemek için son bölümdeki CTR veri sarmalayıcısını kullanırız.

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

## Modeli Eğitme
Sonrasında, modeli eğitiyoruz. Öğrenme oranı 0.02 olarak ayarlanır ve gömme boyutu varsayılan olarak 20'ye ayarlanır. `Adam` eniyileştiricisi ve `SigmoidBinaryCrossEntropyLoss` kaybı model eğitimi için kullanılır.

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

* FM, bağlanım, sınıflandırma ve sıralama gibi çeşitli görevlere uygulanabilen genel bir çerçevedir.
* Öznitelik etkileşimi/kesiti, tahmin görevleri için önemlidir ve 2 yönlü etkileşim FM ile verimli bir şekilde modellenebilir.

## Alıştırmalar

* FM'yi Avazu, MovieLens ve Criteo veri kümeleri gibi diğer veri kümelerinde test edebilir misiniz?
* Performans üzerindeki etkisini kontrol etmek için gömme boyutunu değiştirin, matris çarpanlarına ayırma ile benzer bir örüntü gözlemleyebilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/406)
:end_tab:
