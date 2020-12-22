# Tekrarlayan Sinir Ağları
:label:`chap_rnn`

Şimdiye kadar iki tür veriyle karşılaştık: tablo verileri ve görüntü verileri. İkincisi için, içlerindeki düzenlilikten yararlanmak için özel katmanlar tasarladık. Başka bir deyişle, bir görüntüdeki piksellere izin vereceksek, analog TV zamanlarındaki bir test modelinin arka planına çok benzeyen bir şeyin içeriğinden bahsetmek çok daha zor olurdu.

En önemlisi, şimdiye kadar, verilerimizin hepsinin bir dağıtımdan çekildiğini ve tüm örneklerin bağımsız ve aynı şekilde dağıtıldığını (i.d.) varsaydık. Ne yazık ki, bu çoğu veri için doğru değildir. Örneğin, bu paragraftaki sözcükler sırayla yazılır ve rastgele geçirilirse anlamını deşifre etmek oldukça zor olur. Aynı şekilde, bir videoda görüntü kareleri, bir konuşmadaki ses sinyali ve bir web sitesinde gezinme davranışı sıralı sırayı takip eder. Bu nedenle, bu tür veriler için özel modellerin bunları tanımlamada daha iyi olacağını varsaymak mantıklıdır.

Başka bir sorun, yalnızca bir girdi olarak bir dizi almakla kalmayıp, diziye devam etmemiz beklenebileceği gerçeğinden kaynaklanmaktadır. Örneğin, görev $2, 4, 6, 8, 10, \ldots$ serisine devam etmek olabilir. Bu, zaman serisi analizinde, borsayı, hastanın ateş eğrisini veya yarış arabası için gereken ivmeyi tahmin etmek oldukça yaygındır. Yine bu tür verileri işleyebilecek modellere sahip olmak istiyoruz.

Kısacası, CNN'ler uzamsal bilgileri verimli bir şekilde işleyebilirken, *tekrarlayan sinir ağları* (RNN) sıralı bilgileri daha iyi işleyecek şekilde tasarlanmıştır. RNN'ler, mevcut çıktıları belirlemek için geçmiş bilgileri, akım girdileri ile birlikte depolamak için durum değişkenlerini tanıtır.

Tekrarlayan ağları kullanma örneklerinin çoğu metin verilerine dayanmaktadır. Bu nedenle, bu bölümde dil modellerini vurgulayacağız. Dizi verilerinin daha resmi bir incelemesinden sonra, metin verilerinin ön işlenmesi için pratik teknikler sunuyoruz. Daha sonra, bir dil modelinin temel kavramlarını tartışıyoruz ve bu tartışmayı RNN'lerin tasarımına ilham kaynağı olarak kullanıyoruz. Sonunda, bu tür ağları eğitirken karşılaşılabilecek sorunları keşfetmek için RNN'ler için degrade hesaplama yöntemini açıklıyoruz.

```toc
:maxdepth: 2

sequence
text-preprocessing
language-models-and-dataset
rnn
rnn-scratch
rnn-concise
bptt
```
