# Veri Önişleme
:label:`sec_pandas`

Şimdiye kadar, zaten tensörlerde saklanan verilerde oynama yapmak için çeşitli teknikler tanıştırdık.
Gerçek dünyadaki problemleri çözmeye derin öğrenme uygularken, genellikle tensör formatında güzel hazırlanmış veriler yerine ham verileri önişleme ile başlarız.
Python'daki popüler veri analitik araçları arasında `pandas` paketi yaygın olarak kullanılmaktadır.
Python'un geniş ekosistemindeki diğer birçok uzatma paketi gibi, `pandas` da tensörler ile birlikte çalışabilir.
Bu nedenle, ham verileri `pandas` ile önişleme ve bunları tensör formatına dönüştürme adımlarını kısaca ele alacağız.
Daha sonraki bölümlerde daha çok veri önişleme tekniğini ele alacağız.

## Veri Kümesini Okuma

Örnek olarak, csv (virgülle ayrılmış değerler) dosyasında `../data/house_tiny.csv` saklanan yapay bir veri kümesi yaratarak başlıyoruz. Diğer formatlarda saklanan veriler de benzer şekilde işlenebilir.
Aşağıdaki `mkdir_if_not_exist` işlevi `../data` dizininin var olmasını garanti eder.

```{.python .input}
#@tab all
import os

def mkdir_if_not_exist(path):  #@save
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)
```

Aşağıda veri kümesini bir csv dosyasına satır satır yazıyoruz.

```{.python .input}
#@tab all
data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row is a data instance
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

Oluşturulan csv dosyasından ham veri kümesini yüklemek için `pandas` paketini içe aktarıyoruz ve `read_csv` işlevini çağırıyoruz.
Bu veri kümesinde dört satır ve üç sütun bulunur; burada her satırda oda sayısı ("NumRooms"), sokak tipi ("Alley") ve evin fiyatı ("Price") açıklanır.

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Eksik Verileri İşleme

"NaN" girdilerinin eksik değerler olduğuna dikkat edin.
Eksik verilerle başa çıkmak için, tipik yöntemler arasında *yükleme* ve *silme* yer alır; burada yükleme eksik değerleri ikame edenlerle değiştirir, silme ise eksik değerleri yok sayar. Burada yüklemeyi ele alacağız.

Tamsayı-konuma dayalı indeksleme (`iloc`) ile, `veri`yi `girdi`lere ve `çıktı`lara böldük, burada girdi ilk iki sütuna, çıktı ise son sütuna denk gelir.
`Girdiler`deki eksik sayısal değerler için, "NaN" girdilerini aynı sütunun ortalama değeri ile değiştiriyoruz.

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

`Girdiler`deki kategorik veya ayrık değerler için `NaN`i bir kategori olarak kabul ediyoruz.
"Alley" sütunu yalnızca "Pave" ve "NaN" olmak üzere iki tür kategorik değer aldığından, `pandas` bu sütunu otomatik olarak "Alley_Pave" ve "Alley_nan" sütunlarına dönüştürebilir.
Sokak tipi "Pave" olan bir satır "Alley_Pave" ve "Alley_nan" değerlerini 1 ve 0 olarak tayin eder.
Sokak türü eksik olan bir satır, değerlerini 0 ve 1 olarak tayin eder.

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## Tensör Formatına Dönüştürme

Artık `girdiler`deki ve `çıktılar`daki tüm giriş değerleri sayısal olduğundan, bunlar tensör formatına dönüştürülebilir.
Veriler bu formatta olduğunda, daha önce tanıttığımız tensör işlevleri ile daha fazla oynama yapılabilir :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## Özet

* Python'un geniş ekosistemindeki diğer birçok uzantı paketi gibi, `pandas` da tensörler ile birlikte çalışabilir.
* Yükleme ve silme, eksik verilerle baş etmek için kullanılabilir.

## Alıştırmalar

Daha fazla satır ve sütun içeren bir ham veri kümesi oluşturun.

1. En fazla eksik değerlere sahip sütunu silin.
2. Önişlenmiş veri kümesini tensör formatına dönüştürün.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/195)
:end_tab:
