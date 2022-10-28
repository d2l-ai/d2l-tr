# Derin Öğrenme Hesaplamaları
:label:`chap_computation`

Devasa veri kümeleri ve güçlü donanımın yanı sıra, harika yazılım araçları, 
derin öğrenmenin hızlı ilerlemesinde yadsınamaz bir rol oynamıştır. 2007'de 
yayınlanan çığır açan Theano kütüphanesinden başlayarak, esnek açık kaynaklı 
araçlar, araştırmacıların modelleri hızlı bir şekilde prototip haline 
getirmelerine, standart bileşenleri geri dönüştürürken tekrarlanan 
çalışmalardan kaçınmalarına ve aynı zamanda alt seviyede değişiklikler 
yapma yeteneğini sürdürmelerine olanak tanıdı. Zamanla, derin öğrenme 
kütüphaneleri giderek daha kaba soyutlamalar sunmak için evrimleşti. 
Yarı iletken tasarımcıların transistörlerin belirlenmesinden mantıksal 
devrelere kod yazmaya geçmesi gibi, sinir ağları araştırmacıları da 
tek tek yapay nöronların davranışları hakkında düşünmekten ağları 
tüm katmanlar açısından kavramaya geçtiler ve şimdi genellikle zihindeki 
çok daha kaba *bloklara* sahip mimariler tasarladılar.

Şimdiye kadar, bazı temel makine öğrenmesi kavramlarını tanıtmaktan 
tamamen işlevsel derin öğrenme modellerine yükseliyoruz. Son bölümde, 
bir MLP'nin her bileşenini sıfırdan uyguladık ve 
hatta aynı modelleri zahmetsizce kullanıma sunmak için yüksek düzey 
API'lerden nasıl yararlanılacağını gösterdik. Sizi bu kadar mesafeye bu kadar 
hızlı ulaştırmak için, kütüphaneleri *çağırdık*, ancak *nasıl çalıştıkları* 
ile ilgili daha gelişmiş ayrıntıları atladık. Bu bölümde, derin öğrenme 
hesaplamasının temel bileşenlerine daha derinlemesine, yani model oluşturma, 
parametre erişimi ve ilkleme, ısmarlama (özel kesim) katmanlar ve bloklar 
tasarlama, modelleri diskten okuma, diske yazma ve çarpıcı bir hızlandırma 
elde etmek için GPU'lardan yararlanmaya bakacağız. Bu 
içgörüler, sizi *son kullanıcıdan* *güçlü kullanıcıya* taşıyarak, olgun 
bir derin öğrenme kütüphanesinin faydalarından yararlanmak için gereken 
araçları sağlarken, kendi icat ettikleriniz de dahil olmak üzere daha karmaşık 
modelleri uygulama esnekliğini korur! Bu bölüm herhangi bir yeni model 
veya veri kümesi tanıtmasa da, takip eden gelişmiş modelleme bölümleri 
büyük ölçüde bu tekniklere dayanmaktadır.

```toc
:maxdepth: 2

model-construction
parameters
deferred-init
custom-layer
read-write
use-gpu
```

