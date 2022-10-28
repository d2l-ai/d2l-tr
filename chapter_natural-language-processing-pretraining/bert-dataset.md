# BERT Ön Eğitimi için Veri Kümesi
:label:`sec_bert-dataset`

BERT modelini :numref:`sec_bert` içinde uygulandığı şekilde ön eğitirken, iki ön eğitim görevini kolaylaştırmak için veri kümesini ideal formatta oluşturmamız gerekir: Maskeli dil modelleme ve sonraki cümle tahmini. Bir yandan, orijinal BERT modeli, bu kitabın en okuyucuları için koşmayı zorlaştıran iki büyük külliyatın,  BookCorpus ve İngilizce Wikipedia'nın (bkz. :numref:`subsec_bert_pretraining_tasks`), bitiştirilmesinde ön eğitilmiştir. Öte yandan, kullanıma hazır önceden eğitilmiş BERT modeli tıp gibi belirli alanlardan gelen uygulamalar için uygun olmayabilir. Bu nedenden, BERT'i özelleştirilmiş bir veri kümesi üzerinde ön eğitmek popüler hale gelmektedir. BERT ön eğitiminin gösterilmesini kolaylaştırmak için daha küçük bir külliyat, WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016` kullanıyoruz. 

:numref:`sec_word2vec_data` içinde word2vec ön eğitimi için kullanılan PTB veri kümesiyle karşılaştırıldığında, WikiText-2 (i) orijinal noktalama işaretlerini korur ve bir sonraki cümle tahmini için uygun hale getirir; (ii) orijinal harf büyüklüğünü (büyük-küçük) ve sayıları korur; (iii) iki kat daha büyüktür.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

WikiText-2 veri kümesinde, her satır herhangi bir noktalama işareti ile önceki belirteci arasına boşluk eklenen bir paragrafı temsil eder. En az iki cümle içeren paragraflar korunur. Cümleleri bölerken, noktayı sadece basitlik için sınırlayıcı olarak kullanıyoruz. Daha karmaşık cümle bölme teknikleriyle ilgili tartışmaları bu bölümün sonundaki alıştırmalara bırakıyoruz.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Büyük harfler küçük harflere dönüştürülür.
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Ön Eğitim Görevleri için Yardımcı İşlevleri Tanımlama

Aşağıda, iki BERT ön eğitim görevi için yardımcı fonksiyonları uygulayarak başlıyoruz: Sonraki cümle tahmini ve maskeli dil modelleme. Bu yardımcı işlevler, daha sonra, bu ham metin külliyatı, BERT ön eğitimi için ideal biçimli veri kümesine dönüştürülürken çağrılacaktır.

### Sonraki Cümle Tahmini Görevi Oluşturma

:numref:`subsec_nsp` açıklamalarına göre, `_get_next_sentence` işlevi ikili sınıflandırma görevi için bir eğitim örneği oluşturur.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` bir listelerin listelerinin listesidir
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

Aşağıdaki işlev `_get_next_sentence` işlevini çağırarak `paragraph` girdisinden sonraki cümle tahmini için eğitim örnekleri oluşturur. Burada `paragraph`, her cümlenin bir belirteç listesi olduğu bir cümleler listesidir. `max_len` bağımsız değişkeni, ön eğitim sırasında BERT girdi dizisinin maksimum uzunluğunu belirtir.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 1 tane '<cls>' belirteci ve 2 tane '<sep>' belirteci düşünün
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### Maskeli Dil Modelleme Görevi Oluşturma
:label:`subsec_prepare_mlm_data`

BERT girdi dizisinden maskelenmiş dil modelleme görevi için eğitim örnekleri oluşturmak için aşağıdaki `_replace_mlm_tokens` işlevini tanımlıyoruz. Girdilerinde, `tokens`, BERT girdi dizisini temsil eden belirteçlerin bir listesidir, `candidate_pred_positions`, özel belirteçler hariç BERT girdi dizisinin belirteç indekslerinin bir listesidir (maskeli dil modelleme görevinde özel belirteçler tahmin edilmez) ve `num_mlm_preds` tahminlerin sayısını gösterir (tahmin etmek için %15 rastgele belirteci geri çağırın). :numref:`subsec_mlm` içindeki maskelenmiş dil modelleme görevinin tanımını takiben, her tahmin konumunda, girdi özel bir “&lt;mask&gt;” belirteci veya rastgele bir belirteç ile değiştirilebilir veya değişmeden kalabilir. Sonunda, işlev olası değiştirmeden sonra girdi belirteçlerini, tahminlerin gerçekleştiği belirteç endekslerini ve bu tahminler için etiketleri döndürür.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # Girdinin değiştirilmiş '<mask>' veya rastgele belirteçler içerebileceği 
    # maskelenmiş bir dil modelinin girdisi için belirteçlerin 
    # yeni bir kopyasını oluşturun
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Maskeli dil modelleme görevinde tahmin için %15 rastgele 
    # belirteç almak için karıştır
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # Zamanın %80'inde sözcüğü '<mask>' simgesiyle değiştirin
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # Zamanın %10'unda kelimeyi değiştirmeden bırakın
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # Zamanın %10'unda kelimeyi rastgele bir kelimeyle değiştirin
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

Yukarıda bahsedilen `_replace_mlm_tokens` işlevini çağırarak, aşağıdaki işlev bir BERT girdi dizisini (`tokens`) girdi olarak alır ve girdi belirteçlerinin dizinlerini (:numref:`subsec_mlm` içinde açıklandığı gibi olası belirteç değişiminden sonra), belirteç tahminlerin gerçekleştiği indeksleri ve bu tahminler için etiket indekslerini döndürür.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` bir dizgiler listesidir
    for i, token in enumerate(tokens):
        # Maskeli dil modelleme görevinde özel belirteçler tahmin edilmez
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # Rastgele belirteçlerin %15'i maskelenmiş dil modelleme görevinde tahmin edilir
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Metni Ön Eğitim Veri Kümesine Dönüştürme

Şimdi BERT ön eğitimi için bir `Dataset` sınıfını özelleştirmeye neredeyse hazırız. Bundan önce, girdilere özel “&lt;mask&gt;” belirteçlerini eklemek için `_pad_bert_inputs` bir yardımcı işlevi tanımlamamız gerekiyor. Onun argümanı `examples`, iki ön eğitim görevi için yardımcı işlevlerden `_get_nsp_data_from_paragraph` ve `_get_mlm_data_from_tokens` çıktılarını içerir.

```{.python .input}
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens`, '<pad>' belirteçlerinin sayısını hariç tutar
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Dolgulu belirteçlerin tahminleri, 0 ağırlıklar çarpımı 
        # yoluyla kayıpta filtrelenecektir.
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens`, '<pad>' belirteçlerinin sayısını hariç tutar
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Dolgulu belirteçlerin tahminleri, 0 ağırlıklar çarpımı 
        # yoluyla kayıpta filtrelenecektir.
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

İki ön eğitim görevinin eğitim örneklerini üretmek için yardımcı fonksiyonları ve girdi dolgulama için yardımcı işlevini bir araya getirerek, BERT ön eğitimi için WikiText-2 veri kümesi olarak aşağıdaki `_WikiTextDataset` sınıfını özelleştiriyoruz. `__getitem__` işlevini uygulayarak, WikiText-2 külliyatından bir çift cümleden oluşturulan ön eğitim (maskeli dil modelleme ve sonraki cümle tahmini) örneklerine keyfi olarak erişebiliriz.

Orijinal BERT modeli kelime boyutu 30000 :cite:`Wu.Schuster.Chen.ea.2016` olan WordPiece gömmeleri kullanır. WordPiece'nin belirteçlere ayırma yöntemi, :numref:`subsec_Byte_Pair_Encoding` içinde orijinal sekizli çifti kodlama algoritmasının hafif bir değişiğidir. Basitlik için, belirteçlere ayırmak için `d2l.tokenize` işlevini kullanıyoruz. Beş kereden az görünen seyrek belirteçler filtrelenir.

```{.python .input}
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Girdi `paragraphs[i]`, bir paragrafı temsil eden cümle dizilerinin 
        # bir listesidir; çıktı `paragraphs[i]` bir paragrafı temsil eden 
        # cümlelerin bir listesidir, burada her cümle bir belirteç listesidir
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Bir sonraki cümle tahmini görevi için veri alın
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Maskeli dil modeli görevi için veri alın
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Girdiyi dolgula
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Girdi `paragraphs[i]`, bir paragrafı temsil eden cümle dizilerinin 
        # bir listesidir; çıktı `paragraphs[i]` bir paragrafı temsil eden 
        # cümlelerin bir listesidir, burada her cümle bir belirteç listesidir
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Bir sonraki cümle tahmini görevi için veri alın
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Maskeli dil modeli görevi için veri alın
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Girdiyi dolgula
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

`_read_wiki` işlevini ve `_WikiTextDataset` sınıfını kullanarak, WikiText-2 veri kümesini indirmek ve ondan ön eğitim örnekleri oluşturmak için aşağıdaki `load_data_wiki`'yı tanımlıyoruz.

```{.python .input}
#@save
def load_data_wiki(batch_size, max_len):
    """WikiText-2 veri kümesini yükleyin."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """WikiText-2 veri kümesini yükleyin."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

Toplu iş boyutunu 512 olarak ve BERT girdi dizisinin maksimum uzunluğunu 64 olarak ayarlayarak, BERT ön eğitim örneklerinden oluşan bir minigrubun şekillerini yazdırıyoruz. Her BERT girdi dizisinde, maskelenmiş dil modelleme görevi için $10$ ($64 \times 0.15$) konumun tahmin edildiğini unutmayın.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

Sonunda, kelime dağarcığına bir göz atalım. Sık görülen belirteçleri filtreledikten sonra bile, PTB veri kümesinden iki kat daha büyüktür.

```{.python .input}
#@tab all
len(vocab)
```

## Özet

* PTB veri kümesiyle karşılaştırıldığında, WikiText-2 veri kümesi orijinal noktalama işaretlerini, büyük/küçük harf ve sayıları korur ve iki kat daha büyüktür.
* WikiText-2 külliyatındaki bir çift cümleden oluşturulan ön eğitim (maskeli dil modellemesi ve sonraki cümle tahmini) örneklerine keyfi olarak erişebiliriz.

## Alıştırmalar

1. Basitlik açısından, nokta cümleleri bölmede tek sınırlayıcı olarak kullanılır. SpaCy ve NLTK gibi diğer cümle ayırma tekniklerini deneyin. Örnek olarak NLTK'yi ele alalım. Önce NLTK'yi kurmanız gerekiyor: `pip install nltk`. Kodda, ilk `import nltk` çağırın. Daha sonra Punkt cümle belirteci ayıklayıcıyı indirin: `nltk.download('punkt')`.  `sentences = 'This is great ! Why not ?'` gibi cümleleri ayırmak için `nltk.tokenize.sent_tokenize(sentences)` çağırmak iki cümlelik bir liste döndürür: `['This is great !', 'Why not ?']`
1. Seyrek belirteçleri filtrelemezsek kelime dağarcığı boyutu ne olur?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1496)
:end_tab:
