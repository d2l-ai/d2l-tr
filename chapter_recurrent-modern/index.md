# Modern Tekrarlayan Sinir Ağları
:label:`chap_modern_rnn`

Biz RNN temellerini tanıttık, hangi daha iyi sekans verileri işleyebilir. Tanıtım için, metin verilerine RNN tabanlı dil modelleri uyguladık. Ancak, bu tür teknikler günümüzde çok çeşitli dizi öğrenme problemleriyle karşı karşıya kaldıklarında uygulayıcılar için yeterli olmayabilir.

Örneğin, pratikte önemli bir konu RNN'lerin sayısal dengesizliğidir. Degrade kırpma gibi uygulama püf noktalarını uygulamış olsak da, bu sorun daha sofistike dizisi modelleriyle daha da hafifletilebilir. Özellikle, kapı RNN'ler pratikte çok daha yaygındır. Bu tür yaygın olarak kullanılan ağlardan ikisini, yani *kapılı tekrarlayan birimler* (GRU'lar) ve *uzun kısa süreli bellek* (LSTM) sunarak başlayacağız. Ayrıca, şimdiye kadar tartışılan tek bir yönsüz gizli katman ile RNN mimarisini genişleteceğiz. Derin mimarileri birden fazla gizli katmanla tanımlayacağız ve iki yönlü tasarımı hem ileri hem de geri tekrarlayan hesaplamalarla tartışacağız. Bu tür genişlemeler, modern tekrarlayan ağlarda sıklıkla benimsenir. Bu RNN varyantlarını açıklarken, :numref:`chap_rnn`'te tanıtılan aynı dil modelleme problemini dikkate almaya devam ediyoruz.

Aslında, dil modellemesi, sıralı öğrenmenin yapabileceklerinin sadece küçük bir kısmını ortaya koymaktadır. Otomatik konuşma tanıma, metinden konuşmaya ve makine çevirisi gibi çeşitli dizi öğrenme problemlerinde hem girişler hem de çıktılar keyfi uzunlukta dizilerdir. Bu tür verilerin nasıl sığacağını açıklamak için makine çevirisini örnek olarak ele alacağız ve RNN'lere ve dizi üretimi için ışın aramasına dayalı kodlayıcı-kod çözücü mimarisini tanıtacağız.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```
