# Veri Ön İşleme
:label:`sec_pandas`

Şimdiye kadar, zaten tensörlerde saklanan verilerde oynama yapmak için çeşitli teknikleri tanıştırdık.
Gerçek dünyadaki problemleri çözmede derin öğrenme uygularken, genellikle tensör biçiminde güzel hazırlanmış veriler kullanmak yerine ham verileri ön işleyerek başlarız.
Python'daki popüler veri analitik araçları arasında `pandas` paketi yaygın olarak kullanılmaktadır.
Python'un geniş ekosistemindeki diğer birçok uzatma paketi gibi, `pandas` da tensörler ile birlikte çalışabilir.
Bu nedenle, ham verileri `pandas` ile ön işleme ve bunları tensör formatına dönüştürme adımlarını kısaca ele alacağız.
Daha sonraki bölümlerde daha çok veri ön işleme tekniğini ele alacağız.

## Veri Kümesini Okuma

Örnek olarak, (**`../data/house_tiny.csv` isimli csv (virgülle ayrılmış değerler) dosyasında saklanan yapay bir veri kümesi yaratarak**) başlıyoruz. Diğer formatlarda saklanan veriler de benzer şekilde işlenebilir.

Aşağıda veri kümesini satır satır bir csv dosyasına yazıyoruz.

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
veri_dosyasi = os.path.join('..', 'data', 'house_tiny.csv')
with open(veri_dosyasi, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Sütun adları
    f.write('NA,Pave,127500\n')  # Her satır bir örnek temsil eder
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

[**Oluşturulan csv dosyasından ham veri kümesini yüklemek için**] `pandas` paketini içe aktarıyoruz ve `read_csv` işlevini çağırıyoruz.
Bu veri kümesinde dört satır ve üç sütun bulunur; burada her satırda oda sayısı ("NumRooms"), sokak tipi ("Alley") ve evin fiyatı ("Price") açıklanır.

```{.python .input}
#@tab all
# Eğer pandas kurulu ise alt satırdaki yorumu kaldır
# !pip install pandas
import pandas as pd

veri = pd.read_csv(veri_dosyasi)
print(veri)
```

## Eksik Verileri İşleme

"NaN" girdilerinin eksik değerler olduğuna dikkat edin.
Eksik verilerle başa çıkmak için, tipik yöntemler arasında *itham etme* ve *silme* yer alır; burada itham etme eksik değerleri ikame edenlerle değiştirir, silme ise eksik değerleri yok sayar. Burada ithamı ele alacağız.

Tamsayı-konuma dayalı indeksleme (`iloc`) ile, `veri`yi `girdiler`e ve `ciktilar`a (çıktılar) böldük, burada girdi ilk iki sütuna, çıktı ise son sütuna denk gelir.
`Girdiler`deki eksik sayısal değerler için, [**"NaN" girdilerini aynı sütunun ortalama değeri ile değiştiriyoruz**].

```{.python .input}
#@tab all
girdiler, ciktilar = veri.iloc[:, 0:2], veri.iloc[:, 2]
girdiler = girdiler.fillna(girdiler.mean())
print(girdiler)
```

[**`Girdiler`deki kategorik veya ayrık değerler için `NaN`'i bir kategori olarak kabul ediyoruz.**]
"Alley" sütunu yalnızca "Pave" ve "NaN" olmak üzere iki tür kategorik değer aldığından, `pandas` bu sütunu otomatik olarak "Alley_Pave" ve "Alley_nan" sütunlarına dönüştürebilir.
Sokak tipi "Pave" olan bir satır "Alley_Pave" ve "Alley_nan" değerlerini 1 ve 0 olarak tayin eder.
Sokak türü eksik olan bir satır, değerlerini 0 ve 1 olarak tayin eder.

```{.python .input}
#@tab all
girdiler = pd.get_dummies(girdiler, dummy_na=True)
print(girdiler)
```

## Tensör Formatına Dönüştürme

Artık [**`girdiler`deki ve `ciktilar`daki tüm girdi değerleri sayısal olduğundan, bunlar tensör formatına dönüştürülebilir.**]
Veriler bu formatta olduğunda, daha önce tanıttığımız tensör işlevleri ile daha fazla oynama yapılabilir :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(girdiler.values), np.array(ciktilar.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(girdiler.values), torch.tensor(ciktilar.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(girdiler.values), tf.constant(ciktilar.values)
X, y
```

## Özet

* Python'un geniş ekosistemindeki diğer birçok eklenti paketi gibi, `pandas` da tensörler ile birlikte çalışabilir.
* İtham etme ve silme, eksik verilerle baş etmek için kullanılabilir.

## Alıştırmalar

Daha fazla satır ve sütun içeren bir ham veri kümesi oluşturun.

1. En fazla eksik değerlere sahip sütunu silin.
2. Ön işlenmiş veri kümesini tensör formatına dönüştürün.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/195)
:end_tab:
