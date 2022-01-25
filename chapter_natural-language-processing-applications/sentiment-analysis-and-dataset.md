# Duygu Analizi ve Veri Kümesi
:label:`sec_sentiment`

Çevrimiçi sosyal medya ve inceleme platformlarının çoğalmasıyla birlikte, karar verme süreçlerini desteklemek için büyük bir potansiyel taşıyan bir çok inatçı veri kaydedildi.
*Duygu analizi*
ürün incelemeleri, blog yorumları ve forum tartışmaları gibi insanların ürettikleri metinlerdeki duyguları inceler. Siyaset (örn. politikalara yönelik kamu duygularının analizi), finans (örneğin, pazarın duygularının analizi) ve pazarlama (örneğin, ürün araştırması ve marka yönetimi) kadar çeşitli alanlara geniş uygulamalara sahiptir. 

Duygular ayrık kutuplar veya ölçekler (örneğin, pozitif ve negatif) olarak sınıflandırılabileceğinden, duyarlılık analizini, değişen uzunluktaki bir metin dizisini sabit uzunlukta bir metin kategorisine dönüştüren bir metin sınıflandırma görevi olarak düşünebiliriz. Bu bölümde, duygu analizi için Stanford'un [büyük film incelemesi veri kümesini](https://ai.stanford.edu/~amaas/data/sentiment/) kullanacağız. IMDb'den indirilen 25000 film incelemesini içeren bir eğitim seti ve bir test setinden oluşur. Her iki veri kümesinde de, farklı duyarlılık kutuplarını gösteren eşit sayıda “pozitif” ve “negatif” etiket bulunur.

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

İlk olarak, `../data/aclImdb` yolunda bu IMDb inceleme veri kümesini indirin ve ayıklayın.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Ardından, eğitimi okuyun ve veri kümelerini test edin. Her örnek bir inceleme ve etiketidir: “pozitif” için 1 ve “negatif” için 0.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
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

## Veri Kümesinin Önişlenmesi

Her kelimeyi bir belirteç olarak tedavi ederek ve 5 kereden az görünen sözcükleri filtreleyerek, eğitim veri kümesinden bir kelime dağarcığı oluşturuyoruz.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Belirteçlendirme işleminden sonra, belirteçlerdeki inceleme uzunluklarının histogramını çizelim.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Beklediğimiz gibi, incelemeler değişen uzunluklara sahiptir. Bu tür incelemelerin mini toplu işlemlerini her seferinde işlemek için, :numref:`sec_machine_translation`'teki makine çevirisi veri kümesi için ön işleme adımına benzer olan kesme ve dolgu ile her incelemenin uzunluğunu 500 olarak ayarladık.

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Veri Yineleyiciler Oluşturma

Şimdi veri yineleyiciler oluşturabiliriz. Her yinelemede, bir mini toplu örnekler döndürülür.

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

## Her Şeyleri Bir Araya Getirmek

Son olarak, yukarıdaki adımları `load_data_imdb` işlevine sarıyoruz. Eğitim ve test veri yineleyicileri ve IMDb inceleme veri kümesinin kelime dağarcığını döndürür.

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
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
    """Return data iterators and the vocabulary of the IMDb review dataset."""
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

* Duyarlılık analizi, değişen uzunluktaki metin dizisini dönüştüren bir metin sınıflandırma problemi olarak kabul edilen, üretilen metinlerdeki insanların duygularını inceler
sabit uzunlukta bir metin kategorisine girin.
* Ön işlemden sonra Stanford'un büyük film incelemesi veri kümesini (IMDb inceleme veri kümesini) bir kelime dağarcığıyla veri yineleyicilerine yükleyebiliriz.

## Egzersizler

1. Eğitim duygu analizi modellerini hızlandırmak için bu bölümdeki hangi hiperparametreleri değiştirebiliriz?
1. [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html) veri kümesini veri yineleyicilerine ve duyarlılık analizi için etiketlere yüklemek için bir işlev uygulayabilir misiniz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:
