# Yinelemeli Sinir Ağları
:label:`chap_rnn`

Şimdiye kadar iki tür veriyle karşılaştık: Tablo verileri ve imge verileri. İkincisi için, içlerindeki düzenlilikten yararlanmak için özel katmanlar tasarladık. Başka bir deyişle, bir imgedeki piksellerin yerini değiştirirsek, analog TV zamanlarındaki gibi bir test deseninin arka planına çok benzeyen bir şeyin içeriğinden bahsetmek çok daha zor olurdu.

En önemlisi, şimdiye kadar, verilerimizin hepsinin bir dağılımdan çekildiğini ve tüm örneklerin bağımsız ve özdeşçe dağıtıldığını (i.i.d.) varsaydık. Ne yazık ki, bu çoğu veri için doğru değildir. Örneğin, bu paragraftaki sözcükler sırayla yazılmıştır ve rastgele yer değiştirilirlerse anlamları açığa çıkarmak oldukça zor olur. Aynı şekilde, bir videoda imge çerçeveleri, bir konuşmadaki ses sinyali ve bir web sitesinde gezinme davranışı dizili sırayı takip eder. Bu nedenle, bu tür veriler için özelleşmiş modellerin bunları tanımlamada daha iyi olacağını varsaymak mantıklıdır.

Başka bir sorun, yalnızca bir girdi olarak bir diziyi almakla kalmayıp, diziye devam etmemiz beklenebileceği gerçeğinden kaynaklanmaktadır. Örneğin, görevimiz $2, 4, 6, 8, 10, \ldots$ serisine devam etmek olabilir. Bu, zaman serisi analizinde, borsayı, hastanın ateş değeri eğrisini veya yarış arabası için gereken ivmeyi tahmin etme de oldukça yaygındır. Yine bu tür verileri işleyebilecek modellere sahip olmak istiyoruz.

Kısacası, CNN'ler uzamsal bilgileri verimli bir şekilde işleyebilirken, *yinelemeli sinir ağları* (RNN) dizili bilgileri daha iyi işleyecek şekilde tasarlanmıştır. RNN'ler, mevcut çıktıları belirlemek için geçmiş bilgileri, mevcut girdiler ile birlikte depolayan durum değişkenlerini kullanır.

Yinelemeli ağları kullanan örneklerin çoğu metin verilerine dayanmaktadır. Bu nedenle, bu bölümde dil modellerini vurgulayacağız. Dizi verilerinin daha resmi bir incelemesinden sonra, metin verilerinin ön işlenmesi için pratik teknikler sunacağız. Daha sonra, bir dil modelinin temel kavramlarını tartışıyoruz ve bu tartışmayı RNN'lerin tasarımına ilham kaynağı olarak kullanıyoruz. Sonunda, bu tür ağları eğitirken karşılaşılabilecek sorunları keşfetmek için RNN'ler için gradyan hesaplama yöntemini açıklıyoruz.

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
