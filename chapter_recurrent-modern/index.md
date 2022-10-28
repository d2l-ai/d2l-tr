 # Modern Yinelemeli Sinir Ağları
:label:`chap_modern_rnn`

Dizi verileri daha iyi işleyebilen RNN'nin temellerini tanıttık. Tanışmak amacıyla, metin verilerine RNN tabanlı dil modelleri uyguladık. Ancak, bu tür teknikler günümüzdeki çok çeşitli dizi öğrenme problemleriyle karşı karşıya kaldıklarında uygulayıcılar için yeterli olmayabilir.

Örneğin, pratikte önemli bir konu RNN'lerin sayısal dengesizliğidir. Gradyan kırpma gibi uygulama püf noktalarını uygulamış olsak da, bu sorun daha gelişmiş dizi modelleriyle daha da hafifletilebilir. Özellikle, geçitli RNN'ler pratikte çok daha yaygındır. Bu tür yaygın olarak kullanılan ağlardan ikisini, yani *geçitli yinelemeli birimler (gated recurrent units)* (GRU'lar) ve *uzun ömürlü kısa-dönem belleği (long short-term memory)* (LSTM) sunarak başlayacağız. Ayrıca, şimdiye kadar tartışılan tek yönlü gizli katmanlı RNN mimarisini genişleteceğiz. Derin mimarileri birden fazla gizli katmanla tanımlayacağız ve iki yönlü tasarımı hem ileri hem de geri yinelemeli hesaplamalarla tartışacağız. Bu tür genişlemeler, modern yinelemeli ağlarda sıklıkla benimsenir. Bu RNN türlerini açıklarken, :numref:`chap_rnn` içinde tanıtılan aynı dil modelleme problemini dikkate almaya devam ediyoruz.

Aslında, dil modellemesi, dizi öğrenmenin yapabileceklerinin sadece küçük bir kısmını ortaya koymaktadır. Otomatik konuşma tanıma, metinden konuşmaya dönüştürme ve makine çevirisi gibi çeşitli dizi öğrenme problemlerinde hem girdiler hem de çıktılar keyfi uzunluktaki dizilerdir. Bu tür verilere nasıl uyarlanacağını açıklamak için makine çevirisini örnek olarak ele alacağız ve dizi üretimi için ışın aramasını ve kodlayıcı-kodçözücü mimarisine dayalı RNN'leri tanıtacağız.

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
