# Eniyileme Algoritmaları
:label:`chap_optimization`

Kitabı bu noktaya kadar sırayla okurduysanız, şimdiden derin öğrenme modellerini eğitmek için bir takım eniyileme (optimizasyon) algoritmasını kullanmış oldunuz. Eğitim kümesinde değerlendirildiği gibi, model parametrelerini güncellemeye devam etmemize ve kayıp fonksiyonunun değerini en aza indirmemize izin veren araçlardı. Nitekim, basit bir ortamda amaç işlevleri en aza indirmek için bir kara kutu cihazı olarak eniyileme uygulayan herkes, böyle bir dizi sihirli yöntemin (“SGD” ve “Adam” gibi isimlerle) bulunduğunu bilerek kendini memnun edebilir. 

Bununla birlikte, daha iyisini yapmak için biraz daha derin bilgi gereklidir. Optimizasyon algoritmaları derin öğrenme için önemlidir. Bir yandan karmaşık bir derin öğrenme modelini eğitmek saatler, günler, hatta haftalar sürebilir. Optimizasyon algoritmasının başarımı, modelin eğitim verimliliğini doğrudan etkiler. Öte yandan, farklı optimizasyon algoritmalarının ilkelerini ve onların hiper parametrelerinin rolünü anlamak, derin öğrenme modellerinin başarımını artırmak için hiper parametreleri hedeflenen bir şekilde kurmamızı sağlayacaktır. 

Bu bölümde, yaygın derin öğrenme optimizasyon algoritmalarını derinlemesine araştırıyoruz. Derin öğrenmede ortaya çıkan neredeyse tüm optimizasyon problemleri *dışbükey (convex) olmayandır*. Bununla birlikte, *dışbükey* problemleri bağlamında algoritmaların tasarımının ve çözümlemesinin çok öğretici olduğu kanıtlanmıştır. Bu nedenle, bu bölüm dışbükey optimizasyonda bir özbilgi ve dışbükey bir amaç fonksiyon üzerinde çok basit bir rasgele gradyan iniş algoritması için kanıt içerir.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```
