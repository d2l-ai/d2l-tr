# Doğal Dil Çıkarımı ve Veri Kümesi
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment` içinde, duygu analizi sorununu tartıştık. Bu görev, tek bir metin dizisini, duygu kutupları kümesi gibi önceden tanımlanmış kategorilere sınıflandırmayı amaçlamaktadır. Bununla birlikte, bir cümlenin diğerinden çıkarılıp çıkarılamayacağına karar vermek veya anlamsal olarak eşdeğer cümleleri tanımlayarak fazlalıkları ortadan kaldırmak gerektiğinde, bir metin dizisini nasıl sınıflandıracağını bilmek yetersizdir. Bunun yerine, metin dizileri çiftleri üzerinde akıl yürütebilmemiz gerekir. 

## Doğal Dil Çıkarımı

*Doğal dil çıkarımı*, her ikisinin de bir metin dizisi olduğu bir *öncülden* bir *hipotezin* çıkarılıp çıkarılamayacağını inceler. Başka bir deyişle, doğal dil çıkarımı, bir çift metin dizisi arasındaki mantıksal ilişkiyi belirler. Bu tür ilişkiler genellikle üç tipe ayrılır: 

* *Gerekçe*: Hipotez, öncülden çıkarılabilir.
* *Çelişki*: Hipotezin olumsuzlanması öncülden çıkarılabilir.
* *Tarafsızlık*: Diğer tüm durumlar.

Doğal dil çıkarımı aynı zamanda metinsel gerekçe görevi olarak da bilinir. Örneğin, aşağıdaki çift, *gerekçe* olarak etiketlenecektir, çünkü hipotezdeki “sevgi göstermek” öncülünden “birbirine sarılmak” çıkarılabilir. 

> Öncül: İki kadın birbirlerine sarılıyor. 

> Hipotez: İki kadın sevgi gösteriyor. 

"Kodlama örneğini çalıştırmak", "uyku" yerine "uyumamayı" gösterdiğinden, aşağıda bir *çelişki* örneği verilmiştir.

> Öncül: Bir adam Derin Öğrenmeye Dalış'tan kodlama örneğini çalıştırıyor. 

> Hipotez: Adam uyuyor. 

Üçüncü örnek bir *tarafsızlık* ilişkisini gösterir, çünkü ne "ünlü" ne de "ünlü değil" "bizim için performans gösteriyor" gerçeğinden çıkarılamaz.  

> Öncül: Müzisyenler bizim için performans gösteriyorlar. 

> Hipotez: Müzisyenler ünlüdür. 

Doğal dil çıkarımı doğal dili anlamak için merkezi bir konu olmuştur. Bilgi getiriminden açık alan sorularını yanıtlamaya kadar geniş uygulamalara sahiptir. Bu sorunu incelemek için popüler bir doğal dil çıkarım kıyaslama veri kümesini araştırarak başlayacağız. 

## Stanford Doğal Dil Çıkarımı (SNLI) Veri Kümesi

Stanford Doğal Dil Çıkarımı (Stanford Natural Language Inference - SNLI) Külliyatı 500000'in üzerinde etiketli İngilizce cümle çiftleri içeren bir koleksiyondur :cite:`Bowman.Angeli.Potts.ea.2015`. Ayıklanan SNLI veri kümesini `../data/snli_1.0` yoluna indirip saklıyoruz.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### Veri Kümesini Okuma

Orijinal SNLI veri kümesi, deneylerimizde gerçekten ihtiyacımız olandan çok daha zengin bilgi içeriyor. Bu nedenle, veri kümesinin yalnızca bir kısmını çıkarmak için bir `read_snli` işlevi tanımlarız, ardından öncüllerin, hipotezlerin ve bunların etiketlerinin listesini döndürürüz.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """SNLI veri kümesini öncüller, hipotezler ve etiketler halinde okuyun."""
    def extract_text(s):
        # Bizim tarafımızdan kullanılmayacak bilgileri kaldırın
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Ardışık iki veya daha fazla boşluğu boşlukla değiştirin
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Şimdi ilk 3 çift öncül ve hipotezin yanı sıra onların etiketlerini yazdıralım ("0", "1" ve "2" sırasıyla "gerekçe", "çelişki" ve "tarafsızlık"'a karşılık gelir).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

Eğitim kümesinin yaklaşık 550000 çifti vardır ve test kümesi yaklaşık 10000 çifte sahiptir. Aşağıdakiler, hem eğitim kümesindeki hem de test kümesindeki üç etiketin, "gerekçe", "çelişki" ve "tarafsızlık", dengelendiğini göstermektedir.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### Veri kümesini Yüklemek İçin Bir Sınıf Tanımlama

Aşağıda, Gluon'daki `Dataset` sınıfından türetilmiş SNLI veri kümesini yüklemek için bir sınıf tanımlıyoruz. Sınıf kurucusundaki `num_steps` bağımsız değişkeni, bir metin dizisinin uzunluğunu belirtir, böylece dizilerin her minigrup işlemi aynı şekle sahip olur. Başka bir deyişle, daha uzun dizideki ilk `num_steps` olanlardan sonraki belirteçler kırpılırken, "&lt;pad&gt;" özel belirteçleri uzunlukları `num_steps` olana kadar daha kısa dizilere eklenecektir. `__getitem__` işlevini uygulayarak, `idx` endeksi ile öncüle, hipoteze ve etikete keyfi olarak erişebiliriz.

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """SNLI veri kümesini yüklemek için özelleştirilmiş bir veri kümesi."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """SNLI veri kümesini yüklemek için özelleştirilmiş bir veri kümesi."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### Her Şeyi Bir Araya Koymak

Artık `read_snli` işlevini ve `SNLIDataset` sınıfını SNLI veri kümesini indirmek ve eğitim kümesinin kelime dağarcığıyla birlikte hem eğitim hem de test kümeleri için `DataLoader` örneklerini döndürmek için çalıştırabiliriz. Eğitim kümesinden inşa edilen kelime dağarcığını test kümesininki gibi kullanmamız dikkat çekicidir. Sonuç olarak, test kümesindeki herhangi bir yeni belirteç, eğitim kümesinde eğitilen modelce bilinmeyecektir.

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """SNLI veri kümesini indirin ve veri yineleyicilerini ve kelime dağarcığını döndürün."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """SNLI veri kümesini indirin ve veri yineleyicilerini ve kelime dağarcığını döndürün."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Burada toplu iş boyutunu 128'e ve dizi uzunluğunu 50'ye ayarlıyoruz ve veri yineleyicilerini ve kelime dağarcığını elde etmek için `load_data_snli` işlevini çağırıyoruz. Sonra kelime dağarcığı boyutunu yazdırıyoruz.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Şimdi ilk minigrubun şeklini yazdırıyoruz. Duygu analizinin aksine, iki girdi, `X[0]` ve `X[1]`, öncül ve hipotez çiftlerini temsil ediyor.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Özet

* Doğal dil çıkarımı, her ikisinin de bir metin dizisi olduğu, bir öncülden bir hipotezin çıkarılıp çıkarılamayacağını inceler.
* Doğal dil çıkarımlarında, öncüller ve hipotezler arasındaki ilişkiler, gerekçe, çelişki ve tarafsızlık olarak sayılabilir.
* Stanford Doğal Dil Çıkarımı (Stanford Natural Language Inference - SNLI) Külliyatı, doğal dil çıkarımında popüler bir kıyaslama veri kümesidir.

## Alıştırmalar

1. Makine çevirisi uzun zamandır bir çıktı çevirisi ile bir gerçek referans değer çeviri arasındaki yüzeysel $n$-gramlar eşleşmesine dayalı olarak değerlendirilmektedir. Doğal dil çıkarımını kullanarak makine çevirisi sonuçlarını değerlendirmek için bir ölçü tasarlayabilir misiniz?
1. Kelime dağarcığı boyutunu azaltmak için hiper parametreleri nasıl değiştirebiliriz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1388)
:end_tab:
