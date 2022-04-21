# Zengin Öznitelikli Tavsiye Sistemleri

Etkileşim verileri, kullanıcıların tercih ve ilgi alanlarının en temel göstergesidir. Daha önce tanıtılan modellerde kritik bir rol oynar. Ancak, etkileşim verileri genellikle son derece seyrektir ve bazen gürültülü olabilir. Bu sorunu çözmek için öğelerin öznitelikleri, kullanıcı profilleri ve hatta etkileşimin hangi bağlamda gerçekleştiği gibi yan bilgileri tavsiye modeline entegre edebiliriz. Bu özniteliklerin kullanılması, özellikle etkileşim verilerinin eksik olduğu durumlarda, bu özniteliklerin kullanıcıların ilgi alanlarının etkili bir tahmincisi olabileceği için önerilerde bulunmaya yardımcı olur. Bu nedenle, tavsiye modellerinin bu özniteliklerle başa çıkma ve modele bir miktar içerik/bağlam farkındalığı verme kabiliyetine sahip olması esastır. Bu tür öneri modellerini göstermek için, çevrimiçi reklam önerilerinde tıklama oranı (CTR) hakkında başka bir görev tanıtıyoruz :cite:`McMahan.Holt.Sculley.ea.2013` ve anonim bir reklam verisi sunuyoruz. Hedefli reklam hizmetleri yaygın ilgi gördüler ve genellikle tavsiye motorları olarak çerçevelenirler. Kullanıcıların kişisel zevkine ve ilgi alanlarına uyan reklamların önerilmesi, tıklama oranının iyileştirilmesi açısından önemlidir. 

Dijital pazarlamacılar, müşterilere reklam göstermek için çevrimiçi reklamcılık kullanır. Tıklama oranı, reklamverenlerin gösterim sayısı başına reklamlarında aldıkları tıklama sayısını ölçen bir metriktir ve aşağıdaki formülle hesaplanan bir yüzde değer olarak ifade edilir:  

$$ \text{CTR} = \frac{\#\text{Tıklanmalar}} {\#\text{Gösterimler}} \times 100 \% .$$

Tıklama oranı, tahmin algoritmalarının etkinliğini gösteren önemli bir sinyaldir. Tıklama oranı tahmini, bir web sitesindeki bir şeyin tıklanma olabilirliğini tahmin etme görevidir. CTR tahmini modelleri sadece hedefli reklam sistemlerinde değil, aynı zamanda genel öğe (örn. filmler, haberler, ürünler) öneri sistemlerinde, e-posta kampanyalarında ve hatta arama motorlarında da kullanılabilir. Ayrıca kullanıcı memnuniyeti, dönüşüm oranı ile de yakından ilişkilidir ve reklamverenlerin gerçekçi beklentileri belirlemelerine yardımcı olabileceğinden kampanya hedeflerini belirlemede yardımcı olabilir.

```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Çevrimiçi Reklam Veri Kümesi

İnternet ve mobil teknolojinin önemli gelişmeleriyle, çevrimiçi reklamcılık önemli bir gelir kaynağı haline geldi ve İnternet endüstrisinde gelirin büyük çoğunluğunu oluşturmaktadır. Gündelik ziyaretçilerin ödeme yapan müşterilere dönüştürülmesi için kullanıcıların ilgi alanlarını çeken reklamları veya  ilgili reklamları görüntülemek önemlidir. Tanıtılan veri kümesi bir çevrimiçi reklam veri kümesidir. Bir reklamın tıklanıp (1) tıklanmadığını (0) belirten hedef değişkeni temsil eden ilk sütunla birlikte 34 alandan oluşur. Diğer tüm sütunlar kategorik özniteliklerdir. Sütunlar reklam kimliğini, site veya uygulama kimliğini, cihaz kimliğini, zamanı, kullanıcı profillerini vb. temsil edebilir. Özniteliklerin gerçek anlamı anonimleştirme ve gizlilik endişesi nedeniyle açıklanmamıştır. 

Aşağıdaki kod veri kümesini sunucumuzdan indirir ve yerel veri klasörüne kaydeder.

```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Sırasıyla 15000 ve 3000 örnekten/satırdan oluşan bir eğitim kümesi ve bir test kümesi bulunmaktadır.

## Veri Kümesi Sarmalayıcı

Veri yükleme kolaylığı için, CSV dosyasından reklam veri kümesini yükleyen ve `DataLoader` tarafından kullanılabilen bir `CTRDataset`'i uyguluyoruz.

```{.python .input  n=13}
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

Aşağıdaki örnek, eğitim verilerini yükler ve ilk kaydı yazdırır.

```{.python .input  n=16}
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Görüldüğü gibi, tüm 34 alan kategorik özniteliklerdir. Her değer, karşılık gelen girdinin birebir kodlama dizinini temsil eder. $0$ etiketi, tıklanmadığı anlamına gelir. Bu `CTRDataset`, Criteo ekran reklam yarışması [veri kümesi](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) ve Avazu tıklama oranı tahmini [veri kümesi](https://www.kaggle.com/c/avazu-ctr-prediction) gibi diğer veri kümelerini yüklemek için de kullanılabilir.   

## Özet
* Tıklama oranı reklam sistemlerinin ve tavsiye sistemlerinin etkinliğini ölçmek için kullanılan önemli bir metriktir.
* Tıklama oranı tahmini genellikle ikili sınıflandırma problemine dönüştürülür. Hedef, verilen özniteliklere göre bir reklamın/öğenin tıklanıp tıklanmayacağını tahmin etmektir. 

## Alıştırmalar

* Sağlanan `CTRDataset` ile Criteo ve Avazu veri kümelerini yükleyebilir misiniz? Gerçek değerli özniteliklerden oluşan Criteo veri kümesinin kodunu biraz gözden geçirmeniz gerekebilir.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/405)
:end_tab:
