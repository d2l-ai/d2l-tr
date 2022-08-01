# Makine Çevirisi ve Veri Kümesi
:label:`sec_machine_translation`

Doğal dil işlemenin anahtarı olan dil modellerini tasarlamak için RNN kullandık. Diğer bir amiral gemisi kıyaslaması, girdi dizilerini çıktı dizilerine dönüştüren *dizi dönüştürme* modelleri için merkezi bir problem düzlemi olan *makine çevirisi*dir. Çeşitli modern yapay zeka uygulamalarında önemli bir rol oynayan dizi dönüştürme modelleri, bu bölümün geri kalanının ve sonraki bölümün, :numref:`chap_attention`, odağını oluşturacaktır. Bu amaçla, bu bölüm makine çevirisi sorununu ve daha sonra kullanılacak veri kümesini anlatır.

*Makine çevirisi* bir dizinin bir dilden diğerine otomatik çevirisidir. Aslında, bu alan, özellikle II. Dünya Savaşı'nda dil kodlarını kırmak için bilgisayarların kullanılması göz önüne alınarak, sayısal bilgisayarların icat edilmesinin kısa bir süre sonrasından 1940'lara kadar uzanabilir. Onlarca yıldır, bu alanda, istatistiksel yaklaşımlar, :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`, sinir ağlarını kullanarak uçtan uca öğrenmenin yükselmesinin öncesine kadar baskın olmuştur. İkincisi, çeviri modeli ve dil modeli gibi bileşenlerde istatistiksel analiz içeren *istatistiksel makine çevirisinden* ayırt edilmesi için genellikle *sinirsel makine çevirisi* olarak adlandırılır.

Uçtan uca öğrenmeyi vurgulayan bu kitap, sinirsel makine çevirisi yöntemlerine odaklanacaktır. Külliyatı tek bir dil olan :numref:`sec_language_model` içindeki dil modeli problemimizden farklı olarak, makine çevirisi veri kümeleri sırasıyla kaynak dilde ve hedef dilde bulunan metin dizileri çiftlerinden oluşmaktadır. Bu nedenle, dil modelleme için ön işleme rutinini yeniden kullanmak yerine, makine çevirisi veri kümelerini ön işlemek için farklı bir yol gerekir. Aşağıda, önceden işlenmiş verilerin eğitim için minigruplara nasıl yükleneceğini gösteriyoruz.

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

## [**Veri Kümesini İndirme ve Ön işleme**]

Başlangıç olarak, [Tatoeba Projesi'nden iki dilli cümle çiftleri](http://www.manythings.org/anki/)'nden oluşan bir İngiliz-Fransız veri kümesini indiriyoruz. Veri kümedeki her satır, bir sekmeyle ayrılmış İngilizce metin dizisi ve çevrilmiş Fransızca metin dizisi çiftidir. Her metin dizisinin sadece bir cümle veya birden çok cümleden oluşan bir paragraf olabileceğini unutmayın. İngilizce'nin Fransızca'ya çevrildiği bu makine çevirisi probleminde, İngilizce *kaynak dil*, Fransızca ise *hedef dil*dir.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """İnglizce-Fransızca veri kümesini yükle."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

Veri kümesini indirdikten sonra, ham metin verileri için [**birkaç ön işleme adımı ile devam ediyoruz**]. Örneğin, aralıksız boşluğu boşlukla değiştirir, büyük harfleri küçük harflere dönüştürür ve sözcüklerle noktalama işaretleri arasına boşluk ekleriz.

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """İngilizce-Fransızca veri kümesini ön işle."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Bölünmez boşluğu boşlukla değiştirin ve büyük harfleri küçük 
    # harflere dönüştürün
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Sözcükler ve noktalama işaretleri arasına boşluk ekleyin
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## [**Andıçlama**]

:numref:`sec_language_model` içindeki karakter düzeyinde andıçlara ayırmaktan farklı olarak, makine çevirisi için burada kelime düzeyinde andıçlamayı tercih ediyoruz (son teknoloji modeller daha gelişmiş andıçlama teknikleri kullanabilir). Aşağıdaki `tokenize_nmt` işlevi, her andıç bir sözcük veya noktalama işareti olduğu ilk `num_examples` tane metin dizisi çiftini andıçlar. Bu işlev, andıç listelerinden oluşan iki liste döndürür: `source` (kaynak) ve `target` (hedef). Özellikle, `source[i]` kaynak dilde (İngilizce burada) $i$. metin dizisinden andıçların bir listesidir ve `target[i]` hedef dildekileri (Fransızca burada) içerir.

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """İngilizce-Fransızca veri kümesini andıçla."""
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

[**Metin dizisi başına andıç sayısının histogramını çizelim.**] Bu basit İngilizce-Fransız veri kümesinde, metin dizilerinin çoğunun 20'den az andıcı vardır.

```{.python .input}
#@tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Liste uzunluğu çiftleri için histogramı çizin."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
```

## [**Kelime Dağarcığı**]

Makine çeviri veri kümesi dil çiftlerinden oluştuğundan, hem kaynak dil hem de hedef dil için ayrı ayrı iki kelime hazinesi oluşturabiliriz. Kelime düzeyinde andıçlamada, kelime dağarcığı boyutu, karakter düzeyinde andıç kullanandan önemli ölçüde daha büyük olacaktır. Bunu hafifletmek için, burada 2 defadan az görünen seyrek andıçları aynı bilinmeyen ("&lt;unk&gt;") andıcı ile ifade ediyoruz. Bunun yanı sıra, minigruplarda dizileri aynı uzunlukta dolgulamak için ("&lt;pad&gt;")  ve dizilerin başlangıcını işaretlemek için ("&lt;bos&gt;") veya sonunu işaretlemek için ("&lt;eos&gt;") gibi ek özel andıçlar belirtiyoruz. Bu tür özel andıçlar, doğal dil işleme görevlerinde yaygın olarak kullanılır.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## Veri Kümesini Okuma
:label:`subsec_mt_data_loading`

Dil modellemesinde, [**her dizi örneğinin**], bir cümlenin bir kesimine veya birden fazla cümle üzerine bir yayılan [**sabit bir uzunluğa**] sahip olduğunu hatırlayın. Bu :numref:`sec_language_model` içindeki `num_steps` (zaman adımları veya andıç sayısı) bağımsız değişkeni tarafından belirtilmiştir. Makine çevirisinde, her örnek, her metin dizisinin farklı uzunluklara sahip olabileceği bir kaynak ve hedef metin dizisi çiftidir.

Hesaplamada verimlilik için, yine de bir minigrup metin dizisini *kırkma (truncation)* ve *dolgu* ile işleyebiliriz. Aynı minigruptaki her dizinin aynı `num_steps` uzunluğunda olması gerektiğini varsayalım. Bir metin dizisi `num_steps` andıçtan daha azsa, uzunluğu `num_steps`'e ulaşana kadar özel "&lt;pad&gt;" andıcını sonuna eklemeye devam edeceğiz. Aksi takdirde, metin sırasını yalnızca ilk `num_steps` andıcını alıp geri kalanını atarak keseceğiz. Bu şekilde, her metin dizisi aynı şekle sahip minigruplar olarak yüklenebileceği aynı uzunluğa sahip olacaktır.

Aşağıdaki `truncate_pad` işlevi metin dizilerini daha önce açıklandığı gibi (**keser veya dolgular**).

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Dizileri dolgula veya kırk"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

Şimdi, [**metin dizilerini eğitimde minigruplara dönüştürmek**] için bir işlev tanımlıyoruz. Dizinin sonunu belirtmek için her dizinin sonuna özel “&lt;eos&gt;” andıcını ekliyoruz. Bir model bir diziyi her andıç sonrası bir andıç oluşturarak tahmin ettiğinde, modelin “&lt;eos&gt;” andıcını oluşturması çıktı dizisini tamamlandığını ifade edebilir. Ayrıca, dolgu andıçlarını hariç tutarak her metin dizisinin uzunluğunu da kaydediyoruz. Bu bilgi, daha sonra ele alacağımız bazı modellerde gerekli olacaktır.

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Makine çevirisinin metin dizilerini minigruplara dönüştürün."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

##[**Her Şeyi Bir Araya Koyma**]

Son olarak, veri yineleyiciyi hem kaynak dil hem de hedef dil için kelime dağarcıkları ile birlikte döndüren `load_data_nmt` işlevini tanımlıyoruz.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Çeviri veri kümesinin yineleyicisini ve sözcük dağarcıklarını döndür."""
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

[**İngilizce-Fransız veri kümesinden ilk minigrubu okuyalım.**]

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
* Kelime düzeyinde andıçlama kullanırsak, kelime dağarcığının boyutu, karakter düzeyinde andıçlama kullanmaya göre önemli ölçüde daha büyük olacaktır. Bunu hafifletmek için, seyrek kullanılan andıçları aynı bilinmeyen andıç olarak ifade alabiliriz.
* Metin dizilerini kırkabilir ve dolgulayabiliriz, böylece hepsi minigruplarda yüklenirken aynı uzunluğa sahip olurlar.

## Alıştırmalar

1. `load_data_nmt` işlevindeki `num_examples` değişkeninin farklı değerlerini deneyin. Bu, kaynak ve hedef dillerin kelime dağarcığı boyutlarını nasıl etkiler?
1. Çince ve Japonca gibi bazı dillerde metin, kelime sınır göstergelerine (örn., boşluk) sahip değildir. Sözcük düzeyinde andıçlama bu gibi durumlar için hala iyi bir fikir midir? Neden?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1060)
:end_tab:
