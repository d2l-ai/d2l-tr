# Çok Katmanlı Algılayıcılar
:label:`chap_perceptrons`

Bu bölümde, ilk gerçek *derin* ağınızı tanıtacağız. En basit derin ağlara çok katmanlı algılayıcılar denir ve bunlar, her biri aşağıdaki katmandakilere (girdi aldıkları) ve yukarıdakilere (sırayla etkiledikleri) tam bağlı olan birden fazla nöron (sinir hücresi, burada hesaplama ünitesi) katmanından oluşur. Yüksek kapasiteli modelleri eğittiğimizde, aşırı eğitme (overfitting) riskiyle karşı karşıyayız. Bu nedenle, aşırı eğitme, eksik eğitme (underfitting) ve model seçimi kavramlarıyla ilk titiz karşılaşmanızı sağlamamız gerekecek. Bu sorunlarla mücadele etmenize yardımcı olmak için, ağırlık sönümü (weight decay) ve hattan düşme (dropout) gibi düzenlileştirme tekniklerini tanıtacağız. Derin ağları başarılı bir şekilde eğitmenin anahtarı olan sayısal kararlılık ve parametre ilkleme gibi ilgili sorunları da tartışacağız. Baştan sona, size sadece kavramları değil, aynı zamanda derin ağları kullanma pratiğini de sağlam bir şekilde kavratmayı amaçlıyoruz. Bu bölümün sonunda, şimdiye kadar sunduğumuz şeyleri gerçek bir vakaya uyguluyoruz: Ev fiyatı tahmini. Modellerimizin hesaplama performansı, ölçeklenebilirliği ve verimliliği ile ilgili konuları sonraki bölümlerde değerlendiriyoruz.

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```

