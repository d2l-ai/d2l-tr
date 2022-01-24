# Optimizasyon Algoritmaları
:label:`chap_optimization`

Kitabı bu noktaya kadar sırayla okursanız, derin öğrenme modellerini eğitmek için bir dizi optimizasyon algoritması kullanmış oldunuz. Eğitim setinde değerlendirildiği gibi, model parametrelerini güncellemeye devam etmemize ve kayıp fonksiyonunun değerini en aza indirmemize izin veren araçlardı. Nitekim, basit bir ortamda objektif işlevleri en aza indirmek için bir kara kutu cihazı olarak optimizasyonu tedavi eden herkes, böyle bir prosedürün bir dizi büyü bulunduğunu bilerek kendini içerebilir (“SGD” ve “Adam” gibi isimlerle). 

Bununla birlikte, iyi yapmak için biraz daha derin bilgi gereklidir. Optimizasyon algoritmaları derin öğrenme için önemlidir. Bir yandan karmaşık bir derin öğrenme modelini eğitmek saatler, günler, hatta haftalar sürebilir. Optimizasyon algoritmasının performansı, modelin eğitim verimliliğini doğrudan etkiler. Öte yandan, farklı optimizasyon algoritmalarının ilkelerini ve hiperparametrelerinin rolünü anlamak, derin öğrenme modellerinin performansını artırmak için hiperparametreleri hedeflenen bir şekilde ayarlamamızı sağlayacaktır. 

Bu bölümde, yaygın derin öğrenme optimizasyon algoritmalarını derinlemesine araştırıyoruz. Derin öğrenmede ortaya çıkan neredeyse tüm optimizasyon problemleri*dışbükeyler* değildir. Bununla birlikte, *dışbükey problemleri bağlamında algoritmaların tasarımı ve analizinin çok öğretici olduğu kanıtlanmıştır. Bu nedenle, bu bölüm dışbükey optimizasyonda bir astar ve dışbükey bir objektif fonksiyon üzerinde çok basit bir stokastik degrade iniş algoritması için kanıt içerir.

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
