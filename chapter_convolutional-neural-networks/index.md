# Evrişimli Sinir Ağları
:label:`chap_cnn`

Önceki bölümlerde, her örnek iki boyutlu bir piksel ızgarasından oluşan imge verileriyle karşılaştık. Siyah-beyaz veya renkli imgeleri kullanıp kullanmadığımıza bağlı olarak, her piksel konumu sırasıyla bir veya birden çok sayısal değerle ilişkilendirilebilir. Şimdiye kadar, bu zengin yapıyla başa çıkma yolumuz çok da tatmin edici değildi. Her imgenin mekansal yapısını tek boyutlu vektörlere düzleştirerek onları tam bağlı bir MLP'ye besledik. Bu ağlar özniteliklerin sıralamasından etkilenmez olduğundan, piksellerin konumsal yapısına karşılık gelen bir sırayı koruyup korumadığımıza veya MLP'nin parametrelerini oturtmadan önce tasarım matrisimizin sütunlarını devşirmemize (permute) bakılmaksızın benzer sonuçlar elde edebiliriz. Tercihen, imge verilerinden öğrenmeye yönelik etkili modeller oluşturmak için yakındaki piksellerin tipik olarak birbiriyle ilişkili olduğu ön bilgimizden yararlanırız.

Bu bölümde, tam olarak bu amaç için tasarlanmış güçlü bir sinir ağları ailesi olan *evrişimli sinir ağları (convolutional neural networks)* (CNN) tanıtılmaktadır. CNN tabanlı mimariler artık bilgisayarla görme alanında her yerde bulunmaktadırlar ve o kadar baskın hale gelmişlerdir ki, bugün bu yaklaşım üzerinden inşa etmeden bir kimsenin ticari bir uygulama geliştirmesi veya imge tanıma, nesne algılama veya anlamsal bölünme (semantic segmentation) ile ilgili bir yarışmaya katılması zordur.

Bilinen adlarıyla modern CNN'ler, tasarımlarını biyolojiden, grup teorisiden ve aşırı olmayan miktarda deneysel müdahaleden alınan ilhamlara borçludur. Doğru modellere ulaşmada örneklem verimliliğine ek olarak, CNN'ler hem tam bağlı mimarilere göre daha az parametre gerektirdiklerinden, hem de GPU çekirdeklerinde evrişimlerin paralelleştirilmesi kolay olduğundan hesaplama açısından verimli olma eğilimindedirler. Sonuç olarak, uygulayıcılar genellikle CNN'leri ne zaman mümkün olursa uygularlar ve böylece yinelemeli sinir ağlarının geleneksel olarak kullanıldığı ses, metin ve zaman serisi analizi gibi tek boyutlu bir dizi yapısına sahip görevlerde bile giderek daha güvenilir rakipler olarak ortaya çıkmışlardır. CNN'lerin bazı zekice uyarlamaları onları çizge yapılı verileri ve tavsiye sistemleri de taşır hale getirdi.

İlk olarak, tüm evrişimli ağların omurgasını oluşturan temel işlemleri irdeleyeceğiz. Bunlar, evrişimli katmanlarının kendisini, dolgu (padding) ve uzun adım (stride) gibi esaslı ince detayları, komşu konumsal bölgeler boyunca bilgi toplamak için kullanılan ortaklama (pooling) katmanlarını, her katmanda birden fazla kanal kullanımını ve modern mimarilerin yapısının dikkatli bir şekilde tartışılmasını içerir. Bu bölümü, modern derin öğrenmenin doğuşundan çok önce başarıyla konuşlandırılan ilk evrişimli ağ olan LeNet'in tam bir çalışan örneği ile sonlandıracağız. Bir sonraki bölümde, tasarımları modern uygulayıcılar tarafından yaygın olarak kullanılan tekniklerin çoğunu temsil eden bazı popüler ve nispeten yeni CNN mimarilerinin tam uygulamalarına dalacağız.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
