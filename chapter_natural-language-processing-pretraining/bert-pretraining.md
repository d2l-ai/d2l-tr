# BERT Ön Eğitimi
:label:`sec_bert-pretraining`

:numref:`sec_bert` içinde uygulanan BERT modeli ve :numref:`sec_bert-dataset` içindeki WikiText-2 veri kümesinden oluşturulan ön eğitim örnekleriyle bu bölümde BERT'in WikiText-2 veri kümesi üzerinde ön eğitimini yapacağız.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

Başlamak için WikiText-2 veri kümesini maskelenmiş dil modellemesi ve sonraki cümle tahmini için ön eğitim örneklerinin minigrubu olarak yükleriz. Toplu iş boyutu 512 ve BERT girdi dizisinin maksimum uzunluğu 64'tür. Orijinal BERT modelinde maksimum uzunluğun 512 olduğunu unutmayın.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## BERT Ön Eğitimi

Orijinal BERT'in farklı model boyutlarında :cite:`Devlin.Chang.Lee.ea.2018` iki sürümü vardır. Temel model ($\text{BERT}_{\text{BASE}}$) 768 gizli birim (gizli boyut) ve 12 öz-dikkat kafası olan 12 katman (dönüştürücü kodlayıcı blokları) kullanır. Büyük model ($\text{BERT}_{\text{LARGE}}$), 1024 gizli birimli ve 16 öz-dikkat kafalı 24 katman kullanır. Dikkate değer şekilde, birincisinin 110 milyon parametresi varken, ikincisi 340 milyon parametreye sahiptir. Kolayca gösterim için, 2 katman, 128 gizli birim ve 2 öz-dikkat kafası kullanarak küçük bir BERT tanımlıyoruz.

```{.python .input}
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_layers=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

Eğitim döngüsünü tanımlamadan önce, `_get_batch_loss_bert` yardımcı işlevini tanımlıyoruz. Eğitim örneklerinin parçası göz önüne alındığında, bu işlev hem maskelenmiş dil modellemesi hem de sonraki cümle tahmini görevlerinin kaybını hesaplar. BERT ön eğitiminin son kaybı sadece hem maskeli dil modelleme kaybının hem de bir sonraki cümle tahmini kaybının toplamıdır.

```{.python .input}
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # İleri ilet
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Hesaplanan maskeli dil modeli kaybı
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Sonraki cümle tahmin kaybını hesapla
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # İleri ilet
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Hesaplanan maskeli dil modeli kaybı
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Sonraki cümle tahmin kaybını hesapla
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Yukarıda belirtilen iki yardımcı işlevini çağıran aşağıdaki `train_bert` işlevi, WikiText-2 (`train_iter`) veri kümesindeki BERT (`net`) ön eğitim prosedürünü tanımlar. BERT eğitimi çok uzun sürebilir. `train_ch13` işlevinde olduğu gibi eğitim için dönemlerin sayısını belirtmek yerine (bkz. :numref:`sec_image_augmentation`), aşağıdaki işlevin `num_steps` girdisi, eğitim için yineleme adımlarının sayısını belirtir.

```{.python .input}
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Maskeli dil modelleme kayıplarının toplamı,
    # sonraki cümle tahmin kayıplarının toplamı, 
    # cümle çiftlerinin sayısı, adet
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Maskeli dil modelleme kayıplarının toplamı,
    # sonraki cümle tahmin kayıplarının toplamı, 
    # cümle çiftlerinin sayısı, adet
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

BERT ön eğitimi sırasında hem maskeli dil modelleme kaybını hem de sonraki cümle tahmini kaybını çizdirebiliriz.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## BERT ile Metni Temsil Etme

BERT ön eğitiminden sonra onu, tek metin, metin çiftleri veya bunlardaki herhangi bir belirteci temsil etmek için kullanabiliriz. Aşağıdaki işlev, `tokens_a` ve `tokens_b`'teki tüm belirteçler için BERT (`net`) temsillerini döndürür.

```{.python .input}
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

"A crane is flying" cümlesini düşünün. :numref:`subsec_bert_input_rep` içinde tartışıldığı gibi BERT girdi temsilini hatırlayın. Özel belirteçleri ekledikten sonra “&lt;cls&gt;” (sınıflandırma için kullanılır) ve “&lt;sep&gt;” (ayırma için kullanılır), BERT girdi dizisi altı uzunluğa sahiptir. Sıfır “&lt;cls&gt;” belirteci dizini olduğundan, `encoded_text[:, 0, :]` tüm girdi cümlesinin BERT temsilidir. Çokanlamlılık belirteci "crane"'yi değerlendirmek için, belirteçin BERT temsilinin ilk üç öğesini de yazdırıyoruz.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

Şimdi bir cümle çifti düşünün "a crane driver came" ("bir vinç sürücüsü geldi") ve "he just left" ("az önce gitti"). Benzer şekilde, `encoded_pair[:, 0, :]`, önceden eğitilmiş BERT'ten tüm cümle çiftinin kodlanmış sonucudur. Çokanlamlılık belirteci "crane"'nin (vinç veya turna) ilk üç öğesinin, bağlam farklı olduğu zamanlardan farklı olduğunu unutmayın. Bu BERT temsillerinin bağlam duyarlı olduğunu destekler.

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Andıçlar: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

:numref:`chap_nlp_app` içinde, aşağı akış doğal dil işleme uygulamaları için önceden eğitilmiş bir BERT modelinde ince ayar yapacağız. 

## Özet

* Orijinal BERT, temel modelin 110 milyon parametreye sahip olduğu ve büyük modelin 340 milyon parametreye sahip olduğu iki sürüme sahiptir.
* BERT ön eğitiminden sonra onu, tek metin, metin çiftleri veya bunlardaki herhangi bir belirteci temsil etmek için kullanabiliriz.
* Deneyde, aynı belirteç, bağlamları farklı olduğunda farklı BERT temsiline sahiptir. Bu BERT temsillerinin bağlam duyarlı olduğunu destekler.

## Alıştırmalar

1. Deneyde, maskeli dil modelleme kaybının bir sonraki cümle tahmini kaybından önemli ölçüde daha yüksek olduğunu görebiliriz. Neden?
2. BERT girdi dizisinin maksimum uzunluğunu 512 olarak ayarlayın (orijinal BERT modeliyle aynı). Orijinal BERT modelinin $\text{BERT}_{\text{LARGE}}$ gibi yapılandırmalarını kullanın. Bu bölümü çalıştırırken herhangi bir hatayla karşılaşıyor musunuz? Neden?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1497)
:end_tab:
