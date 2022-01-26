# Zengin Özellikli Tavsiye Sistemleri

Etkileşim verileri, kullanıcıların tercih ve ilgi alanlarının en temel göstergesidir. Eski tanıtılan modellerde kritik bir rol oynar. Ancak, etkileşim verileri genellikle son derece seyrektir ve bazen gürültülü olabilir. Bu sorunu gidermek için, öğelerin özellikleri, kullanıcı profilleri ve hatta etkileşimin hangi bağlamda öneri modeline gerçekleştiği gibi yan bilgileri entegre edebiliriz. Bu özelliklerin kullanılması, özellikle etkileşim verileri eksik olduğunda, bu özelliklerin kullanıcıların ilgi alanlarının etkili bir belirleyicisi olabileceği konusunda önerilerde bulunmada faydalıdır. Bu nedenle, öneri modelleri de bu özelliklerle başa çıkmak ve modele bazı içerik/bağlam farkındalık kazandırmak için yeteneğine sahip olması önemlidir. Bu tür öneri modellerini göstermek için çevrimiçi reklam önerileri :cite:`McMahan.Holt.Sculley.ea.2013` için tıklama oranı (TO) hakkında başka bir görev sunuyoruz ve anonim bir reklam verisi sunuyoruz. Hedeflenen reklam hizmetleri yaygın ilgi gördü ve genellikle tavsiye motorları olarak çerçevelenir. Kullanıcıların kişisel zevkine ve ilgi alanlarına uyan reklamların önerilmesi, tıklama oranının iyileştirilmesi açısından önemlidir. 

Dijital pazarlamacılar, müşterilere reklam göstermek için çevrimiçi reklamcılık kullanır. Tıklama oranı, reklamverenlerin gösterim sayısı başına reklamlarında aldıkları tıklama sayısını ölçen bir metriktir ve aşağıdaki formülle hesaplanan yüzde olarak ifade edilir:  

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

Tıklama oranı, tahmin algoritmalarının etkinliğini gösteren önemli bir sinyaldir. Tıklama oranı tahmini, bir web sitesindeki bir şeyin tıklanma olasılığını tahmin etme görevidir. CTR tahmini modelleri sadece hedeflenen reklam sistemlerinde değil, aynı zamanda genel öğede (örn. filmler, haberler, ürünler) öneri sistemleri, e-posta kampanyaları ve hatta arama motorlarında da kullanılabilir. Ayrıca kullanıcı memnuniyeti, dönüşüm oranı ile de yakından ilişkilidir ve reklamverenlerin gerçekçi beklentileri belirlemelerine yardımcı olabileceğinden kampanya hedefleri belirlemede yardımcı olabilir.

```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Çevrimiçi Reklam VeriKümesi

İnternet ve mobil teknolojinin önemli gelişmeleriyle, çevrimiçi reklamcılık önemli bir gelir kaynağı haline geldi ve İnternet endüstrisinde gelirin büyük çoğunluğunu oluşturmaktadır. Gündelik ziyaretçilerin ödeme yapan müşterilere dönüştürülmesi için kullanıcıların ilgi alanlarını çeken ilgili reklamları veya reklamları görüntülemek önemlidir. Tanıtılan veri kümesi bir online reklam veri kümesidir. Bu, hedef değişkeni temsil eden ilk sütun, bir reklamın (1) tıklanıp tıklanmadığını (0) belirten 34 alandan oluşur. Diğer tüm sütunlar kategorik özelliklerdir. Sütunlar reklam kimliğini, site veya uygulama kimliğini, cihaz kimliğini, saati, kullanıcı profillerini vb. temsil edebilir. Özelliklerin gerçek semantiği anonimleştirme ve gizlilik endişesi nedeniyle açıklanmamıştır. 

Aşağıdaki kod veri kümesini sunucumuzdan indirir ve yerel veri klasörüne kaydeder.

```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Bir antrenman seti ve sırasıyla 15000 ve 3000 örnek/hattan oluşan bir test seti bulunmaktadır. 

## Veri kümesi Sarıcı

Veri yükleme kolaylığı için, CSV dosyasından reklam veri kümesini yükleyen ve `DataLoader` tarafından kullanılabilen bir `CTRDataset`'ü uyguluyoruz.

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

Görüldüğü gibi, tüm 34 alan kategorik özelliklerdir. Her değer, karşılık gelen girdinin tek sıcak dizinini temsil eder. $0$ etiketi, tıklanmadığı anlamına gelir. Bu `CTRDataset`, Criteo ekran reklam mücadelesi [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) ve Avazu tıklama oranı tahmini [Dataset](https://www.kaggle.com/c/avazu-ctr-prediction) gibi diğer veri kümelerini yüklemek için de kullanılabilir.   

## Özet* Tıklama oranı reklam sistemlerinin ve tavsiye sistemlerinin etkinliğini ölçmek için kullanılan önemli bir metriktir.* Tıklama oranı tahmini genellikle ikili sınıflandırma problemine dönüştürülür. Hedef, verilen özelliklere göre bir reklam/öğenin tıklanıp tıklanmayacağını tahmin etmektir. 

## Egzersizler

* Sağlanan `CTRDataset` ile Criteo ve Avazu veri kümesini yükleyebilir misiniz. Gerçek değerli özelliklerden oluşan Criteo veri kümesinin kodunu biraz gözden geçirmeniz gerekebilir.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:
