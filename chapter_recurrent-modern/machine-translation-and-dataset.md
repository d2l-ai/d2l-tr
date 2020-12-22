# Makine Çevirisi ve Dataset
:label:`sec_machine_translation`

Doğal dil işlemenin anahtarı olan dil modellerini tasarlamak için RNN kullandık. Diğer bir amiral gemisi kıyaslaması, giriş dizilerini çıktı dizilerine dönüştüren *sıra dönüşüm* modelleri için merkezi bir sorun alanı olan *makine çevirisi*. Çeşitli modern yapay zeka uygulamalarında önemli bir rol oynayan dizi iletim modelleri, bu bölümün geri kalanının ve :numref:`chap_attention`'ün odağını oluşturacaktır. Bu amaçla, bu bölüm makine çevirisi sorununu ve daha sonra kullanılacak veri kümesini tanıtır.

*Makine çevirisi*
bir dizinin bir dilden diğerine otomatik çevirisi. Aslında, bu alan, özellikle II. Dünya Savaşı'nda dil kodlarını kırmak için bilgisayarların kullanılması göz önüne alınarak, dijital bilgisayarlar icat edildikten kısa bir süre sonra 1940'lara kadar uzanabilir. Onlarca yıldır, bu alanda istatistiksel yaklaşımlar :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`, sinir ağlarını kullanarak uçtan uca öğrenmenin yükselmesinden önce baskın olmuştur. İkincisi genellikle denir
*Sinir makinesi çevirisi*
kendini ayırt etmek
*İstatistiksel makine çevirisi*
çeviri modeli ve dil modeli gibi bileşenlerde istatistiksel analiz içerir.

Uçtan uca öğrenmeyi vurgulayan bu kitap, sinirsel makine çeviri yöntemlerine odaklanacaktır. Dersleri tek bir dilde olan :numref:`sec_language_model`'teki dil modeli problemimizden farklı olarak, makine çevirisi veri kümeleri sırasıyla kaynak dilde ve hedef dilde bulunan metin dizileri çiftlerinden oluşmaktadır. Bu nedenle, dil modelleme için önişleme rutinini yeniden kullanmak yerine, makine çevirisi veri kümelerini önişlemek için farklı bir yol gerekir. Aşağıda, önceden işlenmiş verilerin eğitim için minibatch'lere nasıl yükleneceğini gösteriyoruz.

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
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## Veri Kümesini İndirme ve Önişleme

Başlangıç olarak, [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/)'ten oluşan bir İngiliz-Fransız veri kümesini indiriyoruz. Veri kümedeki her satır, İngilizce metin dizisinin sekmeyle ayrılmış bir çifti ve çevrilmiş Fransızca metin dizisidir. Her metin dizisinin sadece bir cümle veya birden çok cümleden oluşan bir paragraf olabileceğini unutmayın. İngilizce'nin Fransızca'ya çevrildiği bu makine çevirisi probleminde, İngilizce*kaynak dil*, Fransızca ise *hedef dil*.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

Veri kümesini indirdikten sonra, ham metin verileri için birkaç önişleme adımına devam ediyoruz. Örneğin, kırılmayan alanı boşlukla değiştirir, büyük harfleri küçük harflere dönüştürür ve sözcüklerle noktalama işaretleri arasına boşluk ekleriz.

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## Tokenization

:numref:`sec_language_model`'teki karakter düzeyinde tokenizasyondan farklı olarak, makine çevirisi için burada kelime düzeyinde tokenizasyonu tercih ediyoruz (son teknoloji modelleri daha gelişmiş tokenizasyon teknikleri kullanabilir). Aşağıdaki `tokenize_nmt` işlevi, her belirteç bir sözcük veya noktalama işareti olduğu ilk `num_examples` metin sırası çiftlerini belirteçler. Bu işlev, belirteç listelerinin iki listesini döndürür: `source` ve `target`. Özellikle, `source[i]` kaynak dilde (İngilizce burada) $i^\mathrm{th}$ metin dizisinden belirteçlerin bir listesidir ve `target[i]` hedef dilde (Fransızca burada).

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

Metin dizisi başına belirteç sayısının histogramını çizelim. Bu basit İngilizce-Fransız veri kümesinde, metin dizilerinin çoğunun 20'den az belirteci vardır.

```{.python .input}
#@tab all
d2l.set_figsize()
_, _, patches = d2l.plt.hist(
    [[len(l) for l in source], [len(l) for l in target]],
    label=['source', 'target'])
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right');
```

## Kelime hazinesi

Makine çeviri veri kümesi dil çiftlerinden oluştuğundan, hem kaynak dil hem de hedef dil için ayrı ayrı iki kelime hazinesi oluşturabiliriz. Kelime düzeyinde tokenization ile, kelime dağarcığı boyutu, karakter düzeyinde belirteç kullanarak bundan önemli ölçüde daha büyük olacaktır. Bunu hafifletmek için, burada aynı bilinmeyen (” <unk> “) belirteci ile 2 defadan az görünen seyrek belirteçleri tedavi ediyoruz. Bunun yanı sıra, <pad> mini batchlerde aynı uzunlukta dolgu (” “) dizileri ve dizilerin başlangıcını (” <bos> “) veya sonunu (” <eos> “) işaretlemek için gibi ek özel belirteçleri belirtiyoruz. Bu tür özel belirteçler, doğal dil işleme görevlerinde yaygın olarak kullanılır.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## Veri kümesini yükleme
:label:`subsec_mt_data_loading`

Dil modellemesinde, her dizi örneğinin, bir cümlenin bir kesimi veya birden fazla cümle üzerindeki bir yayılma, sabit bir uzunluğa sahip olduğunu hatırlayın. Bu `num_steps` (zaman adımları veya belirteçleri sayısı) bağımsız değişken :numref:`sec_language_model` tarafından belirtilmiştir. Makine çevirisinde, her örnek, her metin dizisinin farklı uzunluklara sahip olabileceği bir kaynak ve hedef metin dizisi çiftidir.

Hesaplamalı verimlilik için, yine de bir mini toplu metin dizilerini *kestirme* ve *padding* ile işleyebiliriz. Aynı minibatch'deki her dizinin aynı uzunlukta olması gerektiğini varsayalım `num_steps`. Bir metin dizisi `num_steps` jetonundan daha azsa, <pad> uzunluğu `num_steps`'e ulaşana kadar özel "" belirteci sonuna eklemeye devam edeceğiz. Aksi takdirde, metin sırasını yalnızca ilk `num_steps` jetonlarını alıp geri kalanını atarak keseceğiz. Bu şekilde, her metin dizisi aynı şekle sahip mini batches olarak yüklenecek aynı uzunluğa sahip olacaktır.

Aşağıdaki `truncate_pad` işlevi metin dizilerini daha önce açıklandığı gibi keser veya pedler.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

Şimdi, metin dizilerini eğitim için mini batchlere dönüştürmek için bir işlev tanımlıyoruz. Dizinin sonunu <eos> belirtmek için her dizinin sonuna özel “” belirteci ekliyoruz. Bir model belirteç sonra bir dizi belirteci oluşturarak tahmin edildiğinde, “<eos>” belirteci oluşturma çıktı sırası tamamlandığını önerebilir. Ayrıca, dolgu belirteçleri hariç her metin dizisinin uzunluğunu da kaydediyoruz. Bu bilgi, daha sonra ele alacağımız bazı modeller tarafından gerekli olacaktır.

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## Her şeyini bir araya koymak

Son olarak, veri yineleyiciyi hem kaynak dil hem de hedef dil için kelime hazineleri ile birlikte döndürmek için `load_data_nmt` işlevini tanımlıyoruz.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

İngilizce-Fransız veri kümesinden ilk minibatch'i okuyalım.

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## Özet

* Makine çevirisi, bir dizinin bir dilden diğerine otomatik çevirisini ifade eder.
* Kelime düzeyinde tokenization kullanarak, kelime dağarcığı boyutu, karakter düzeyinde belirteç kullanarak bundan önemli ölçüde daha büyük olacaktır. Bunu hafifletmek için, seyrek belirteçleri aynı bilinmeyen belirteç olarak ele alabiliriz.
* Metin dizilerini kesebilir ve doldırabiliriz, böylece hepsi mini batchlerde yüklenecek aynı uzunluğa sahip olur.

## Egzersizler

1. `load_data_nmt` işlevindeki `num_examples` bağımsız değişkeni farklı değerlerini deneyin. Bu, kaynak dilin ve hedef dilin kelime dağarcığı boyutlarını nasıl etkiler?
1. Çince ve Japonca gibi bazı dillerde metin, kelime sınır göstergelerine (örn., boşluk) sahip değildir. Sözcük düzeyinde tokenizasyon bu gibi durumlar için hala iyi bir fikir mi? Neden ya da neden olmasın?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
