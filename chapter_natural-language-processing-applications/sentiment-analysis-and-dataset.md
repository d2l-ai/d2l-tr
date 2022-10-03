# Duygu Analizi ve Veri Kümesi
:label:`sec_sentiment`

Çevrimiçi sosyal medya ve inceleme platformlarının çoğalmasıyla birlikte, karar verme süreçlerini desteklemek için büyük bir potansiyel taşıyan bir çok görüş içeren veri kaydedildi.
*Duygu analizi* ürün incelemeleri, blog yorumları ve forum tartışmaları gibi insanların ürettikleri metinlerdeki duyguları inceler. Siyaset (örn. politikalara yönelik kamu duygularının analizi), finans (örneğin, pazarın duygularının analizi) ve pazarlama (örneğin, ürün araştırması ve marka yönetimi) kadar çeşitli alanlarda geniş uygulamalara sahiptir. 

Duygular ayrık kutuplar veya ölçekler (örneğin, pozitif ve negatif) olarak sınıflandırılabileceğinden, duygu analizini, değişen uzunluktaki bir metin dizisini sabit uzunlukta bir metin kategorisine dönüştüren bir metin sınıflandırma görevi olarak düşünebiliriz. Bu bölümde, duygu analizi için Stanford'un [film incelemesi büyük veri kümesini](https://ai.stanford.edu/~amaas/data/sentiment/) kullanacağız. IMDb'den indirilen 25000 film incelemesini içeren bir eğitim kümesi ve bir test kümesinden oluşur. Her iki veri kümesinde de, farklı duygu kutuplarını gösteren eşit sayıda “pozitif” ve “negatif” etiketleri bulunur.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

##  Veri Kümesini Okuma

İlk olarak, bu IMDb inceleme veri kümesini `../data/aclImdb` yoluna indirin ve genişletin.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Ardından, eğitim ve test veri kümelerini okuyun. Her örnek bir inceleme ve onun etiketidir: “Pozitif” için 1 ve “negatif” için 0.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """IMDb inceleme veri kümesi metin dizilerini ve etiketlerini okuyun."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

## Veri Kümesini Ön İşlemesi

Her kelimeye bir belirteç gibi davranarak ve 5 kereden az görünen sözcükleri filtreleyerek, eğitim veri kümesinden bir kelime dağarcığı oluşturuyoruz.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Belirteçlendirme işleminden sonra, inceleme uzunluklarının belirteçler cinsinden histogramını çizelim.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('inceleme başına belirteç sayısı')
d2l.plt.ylabel('adet')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Beklediğimiz gibi, incelemeler değişen uzunluklara sahiptir. Bu tür incelemelerin minigrup işlemlerini her seferinde işlemek için, :numref:`sec_machine_translation` içindeki makine çevirisi veri kümesi için ön işleme adımına benzer olan budama ve dolgulama ile her incelemenin uzunluğunu 500 olarak ayarladık.

```{.python .input}
#@tab all
num_steps = 500  # dizi uzunluğu
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Veri Yineleyiciler Oluşturma

Şimdi veri yineleyiciler oluşturabiliriz. Her yinelemede, bir örnek minigrubu döndürülür.

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Her Şeyleri Bir Araya Koymak

Son olarak, yukarıdaki adımları `load_data_imdb` işlevinde sarıyoruz. İşlev eğitim ve test veri yineleyicileri ve IMDb inceleme veri kümesinin kelime dağarcığını döndürür.

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Veri yineleyicilerini ve IMDb inceleme veri kümesinin kelime dağarcığını döndür."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Veri yineleyicilerini ve IMDb inceleme veri kümesinin kelime dağarcığını döndür."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Özet

* Duygu analizi, değişen uzunluktaki bir metin dizisini sabit uzunlukta bir metin kategorisine dönüştüren bir metin sınıflandırma problemi olarak kabul edilen, insanların ürettikleri metindeki duygularını inceler.
* Ön işlemeden sonra Stanford'un film inceleme büyük veri kümesini (IMDb inceleme veri kümesini) kelime dağarcıklı veri yineleyicilerine yükleyebiliriz.

## Alıştırmalar

1. Duygu analizi modellerinin eğitimi hızlandırmak için bu bölümdeki hangi hiper parametreleri değiştirebiliriz?
1. [Amazon incelemeleri](https://snap.stanford.edu/data/web-Amazon.html) veri kümesini duygu analizi için veri yineleyicilerine ve etiketlere yükleme için bir işlev uygulayabilir misiniz?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1387)
:end_tab:
