# Konvolüsyonel Sinir Ağları
:label:`chap_cnn`

Önceki bölümlerde, her örnek iki boyutlu bir piksel ızgarasından oluşan görüntü verilerine karşı çıktık. Siyah-beyaz veya renkli görüntüleri kullanıp kullanmadığımıza bağlı olarak, her piksel konumu sırasıyla bir veya birden çok sayısal değerle ilişkilendirilebilir. Şimdiye kadar, bu zengin yapıyla başa çıkma şeklimiz çok tatmin edici değildi. Her görüntünün mekansal yapısını tek boyutlu vektörlere düzleştirerek tamamen bağlı bir MLP ile besleyerek attık. Bu ağlar özelliklerin sırasına göre değişmez olduğundan, piksellerin mekansal yapısına karşılık gelen bir sırayı koruyup korumadığımıza veya MLP'nin parametrelerini takmadan önce tasarım matrisinin sütunlarına izin verdiğimize bakılmaksızın benzer sonuçlar elde edebiliriz. Tercihen, görüntü verilerinden öğrenmeye yönelik etkili modeller oluşturmak için yakındaki piksellerin tipik olarak birbiriyle ilişkili olduğu ön bilgimizden yararlanırız.

Bu bölümde, tam olarak bu amaç için tasarlanmış güçlü bir sinir ağları ailesi olan *evrimsel sinir ağları* (CNN) tanıtılmaktadır. CNN tabanlı mimariler artık bilgisayar görüşü alanında her yerde bulunmaktadırlar ve o kadar baskın hale gelmişlerdir ki, bugün hiç kimse ticari bir uygulama geliştiremeyecek veya görüntü tanıma, nesne algılama veya semantik segmentasyon ile ilgili bir yarışmaya katılacaktır.

Modern CNN'ler, halk dilinde tasarımlarını biyoloji, grup teorisi ve deneysel müdahalenin sağlıklı dozundan ilham almalarına borçludur. Doğru modellere ulaşmada örnek verimliliğine ek olarak, CNN'ler hem tam bağlı mimarilere göre daha az parametre gerektirdikleri hem de GPU çekirdeklerinde kıvrımların paralelleştirilmesi kolay olduğu için hesaplama açısından verimli olma eğilimindedir. Sonuç olarak, uygulayıcılar genellikle CNN'leri mümkün olduğunda uygularlar ve tekrarlayan sinir ağlarının geleneksel olarak kullanıldığı ses, metin ve zaman serisi analizi gibi tek boyutlu bir dizi yapısına sahip görevlerde bile giderek daha güvenilir rakipler olarak ortaya çıkmışlardır. CNN'lerin bazı zekice uyarlamaları onları grafik yapılandırılmış verilere ve tavsiye sistemlerine taşımaya getirdi.

İlk olarak, tüm evrimsel ağların omurgasını oluşturan temel işlemleri yürüyeceğiz. Bunlar, kıvrımsal katmanların kendileri, dolgu ve adım gibi küçük cesur detaylar, bitişik mekansal bölgeler boyunca bilgi toplamak için kullanılan havuzlama katmanları, her katmanda birden fazla kanal kullanımı ve modern mimarilerin yapısının dikkatli bir şekilde tartışılması içerir. Bu bölümü, modern derin öğrenmenin doğuşundan çok önce başarıyla konuşlandırılan ilk evrimsel ağ olan LeNet'in tam bir çalışma örneği ile sonuçlandıracağız. Bir sonraki bölümde, tasarımları modern uygulayıcılar tarafından yaygın olarak kullanılan tekniklerin çoğunu temsil eden bazı popüler ve nispeten yeni CNN mimarilerinin tam uygulamalarına dalacağız.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
