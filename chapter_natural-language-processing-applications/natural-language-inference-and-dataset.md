# Doğal Dil Çıkarımı ve Veri Kümesi
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment`'te, duygu analizi sorununu tartıştık. Bu görev, tek bir metin dizisini, duyarlılık kutupları kümesi gibi önceden tanımlanmış kategorilere sınıflandırmayı amaçlamaktadır. Ancak, bir cümlenin başka bir biçimde çıkarılıp çıkarılamayacağına karar verme ihtiyacı olduğunda veya anlamsal olarak eşdeğer cümleleri tanımlayarak fazlalığı ortadan kaldırmaya ihtiyaç duyulduğunda, bir metin dizisinin nasıl sınıflandırılacağını bilmek yetersizdir. Bunun yerine, metin dizileri çiftleri üzerinde akıl edebilmek gerekir. 

## Doğal Dil Çıkarımı

*Doğal dil çıkarımı* bir *hipotez* olup olmadığını inceler
, her ikisinin de metin dizisi olduğu bir *öncül* den çıkarılabilir. Başka bir deyişle, doğal dil çıkarımı, bir çift metin dizisi arasındaki mantıksal ilişkiyi belirler. Bu tür ilişkiler genellikle üç tipe ayrılır: 

* *Katkı*: Hipotez, öncülden çıkarılabilir.
* *Çelişme*: Hipotezin olumsuzluğu öncülden çıkarılabilir.
* *Tarafsiz*: Diğer tüm davalar.

Doğal dil çıkarımı aynı zamanda metinsel bağlılık görevi olarak da bilinir. Örneğin, aşağıdaki çift *entailment* olarak etiketlenecektir, çünkü hipotezdeki “sevgi göstermek” öncülünde “birbirine sarılmaktan” çıkarılabilir. 

> Premise: İki kadın birbirlerine sarılıyor. 

> Hipotez: İki kadın sevgi gösteriyor. 

Aşağıdaki*contradiction* örneği “kodlama örneğini çalıştırmak” yerine “uyku” yerine “uyku değil” anlamına geldiği için bir örnektir. 

> Premise: Bir adam Derin Öğrenmeye Dalış bölümünden kodlama örneğini çalıştırıyor. 

> Hipotez: Adam uyuyor. 

Üçüncü örnek bir *tarafsızlık* ilişkisini gösterir, çünkü ne “ünlü” ne de “ünlü değil” “bizim için performans gösteriyor” gerçeğinden çıkarılamaz.  

> Premise: Müzisyenler bizim için performans gösteriyorlar. 

> Hipotez: Müzisyenler ünlüdür. 

Doğal dil çıkarımı doğal dili anlamak için merkezi bir konu olmuştur. Bilgi alımdan açık alan soru yanıtlamaya kadar geniş uygulamalara sahiptir. Bu sorunu incelemek için popüler bir doğal dil çıkarım kıyaslama veri kümesini araştırarak başlayacağız. 

## Stanford Doğal Dil Çıkarımı (SNLI) Veri Kümesi

Stanford Natural Language Inference (SNLI) Corpus üzerinde 500000 etiketli İngilizce cümle çiftleri :cite:`Bowman.Angeli.Potts.ea.2015` bir koleksiyondur. Çıkarılan SNLI veri kümesini `../data/snli_1.0` yolunda indirip saklıyoruz.

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

Orijinal SNLI veri seti, deneylerimizde gerçekten ihtiyacımız olandan çok daha zengin bilgiler içeriyor. Böylece, bir işlev tanımlamak `read_snli` yalnızca veri kümesinin bir kısmını ayıklamak, sonra tesislerinde, hipotezlerin ve etiketlerinin listelerini döndürmek için.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
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

Şimdi ilk 3 çift öncül ve hipotezin yanı sıra etiketlerini yazdıralım (“0", “1" ve “2" sırasıyla “gerekçe”, “çelişki” ve “nötr” e karşılık gelir).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

Eğitim setinin yaklaşık 550000 çiftleri vardır ve test seti yaklaşık 10000 çifte sahiptir. Aşağıdakiler, hem eğitim setinde hem de test setinde “gerekçe”, “çelişki” ve “nötr” üç etiketinin dengelendiğini göstermektedir.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### Veri kümesini yüklemek için bir sınıf tanımlama

Aşağıda, Gluon'daki `Dataset` sınıfından miras alarak SNLI veri kümesini yüklemek için bir sınıf tanımlıyoruz. Sınıf yapıcısındaki `num_steps` bağımsız değişkeni, bir metin dizisinin uzunluğunu belirtir, böylece dizilerin her mini toplu işlemi aynı şekle sahip olur. Başka bir deyişle, daha uzun sırayla ilk `num_steps`'ten sonra belirteçler kesilirken, özel belirteçleri “<pad>” uzunluğu `num_steps` olana kadar daha kısa dizilere eklenecektir. `__getitem__` işlevini uygulayarak, `idx` endeksi ile öncül, hipotez ve etikete keyfi olarak erişebiliriz.

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
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
    """A customized dataset to load the SNLI dataset."""
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

### Her Şeyleri Bir Araya Getirmek

Artık `read_snli` işlevini ve `SNLIDataset` sınıfını SNLI veri kümesini indirmek ve eğitim setinin kelime dağarcığıyla birlikte hem eğitim hem de test setleri için `DataLoader` örneklerini iade edebiliriz. Eğitim setinden inşa edilen kelime dağarcığını test setinin olduğu gibi kullanmamız dikkat çekicidir. Sonuç olarak, test setindeki herhangi bir yeni belirteç, eğitim setinde eğitilen modele bilinmeyecektir.

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
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
    """Download the SNLI dataset and return data iterators and vocabulary."""
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

Burada toplu iş boyutunu 128 ve sıra uzunluğunu 50'ye ayarladık ve veri yineleyicilerini ve kelime dağarcığını almak için `load_data_snli` işlevini çağırıyoruz. Sonra kelime dağarcığı boyutunu yazdırıyoruz.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Şimdi ilk mini batch şeklini yazdırıyoruz. Duygu analizinin aksine, iki giriş `X[0]` ve `X[1]` bina ve hipotez çiftlerini temsil ediyor.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Özet

* Doğal dil çıkarımı, bir hipotezin her ikisinin de metin dizisi olduğu bir öncülden çıkarılıp çıkarılamayacağını inceler.
* Doğal dil çıkarımlarında, bina ve hipotezler arasındaki ilişkiler, bağlılık, çelişki ve tarafsız sayılabilir.
* Stanford Natural Language Inference (SNLI) Corpus, doğal dil çıkarımının popüler bir kıyaslama veri kümasıdır.

## Egzersizler

1. Makine çevirisi uzun bir çıktı çevirisi ve bir zemin doğruluk çevirisi arasında yüzeysel $n$ gram eşleştirme dayalı değerlendirilmiştir. Doğal dil çıkarımı kullanarak makine çevirisi sonuçlarını değerlendirmek için bir ölçü tasarlayabilir misiniz?
1. Kelime boyutunu azaltmak için hiperparametreleri nasıl değiştirebiliriz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:
