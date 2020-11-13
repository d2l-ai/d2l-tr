# Modern Konvolüsyonel Sinir Ağları
:label:`chap_modern_cnn`

Artık CNN'leri birlikte kablolama temellerini anladığımıza göre, sizi modern CNN mimarilerinin bir turuna götüreceğiz. Bu bölümde, her bölüm bir noktada (veya şu anda) birçok araştırma projesinin ve konuşlandırılmış sistemlerin inşa edildiği temel model olan önemli bir CNN mimarisine karşılık gelmektedir. Bu ağların her biri kısaca baskın bir mimariydi ve çoğu, 2010 yılından bu yana bilgisayar görüşünde denetimli öğrenmede ilerleme barometresi olarak hizmet eden ImageNet yarışmasında kazananlar veya ikincilerdi.

Bu modeller, büyük ölçekli bir görme zorluğu üzerinde geleneksel bilgisayar görme yöntemlerini yenmek için konuşlandırılan ilk büyük ölçekli ağ olan AlexNet'i içerir; bir dizi tekrarlayan eleman bloklarını kullanan VGG ağı; tüm sinir ağlarını yamalı olarak birleştiren ağdaki ağ (NiN) girişleri; Paralel birleştirmelere sahip ağları kullanan GoogLeNet; bilgisayar görüşünde en popüler raf dışı mimari olmaya devam eden artık ağlar (ResNet); ve hesaplanması pahalı ancak bazı son kriterler belirlemiş yoğun bağlı ağlar (DenseNet).

*Derin sinir ağları fikri oldukça basit olsa da (bir grup katmanı bir araya getirin), performans mimariler ve hiperparametre seçenekleri arasında çılgınca farklılık gösterebilir. Bu bölümde açıklanan sinir ağları sezgi, birkaç matematiksel anlayış ve bir sürü deneme yanılma ürünüdür. Bu modelleri kronolojik sırayla sunuyoruz, kısmen tarih duygusunu iletmek için, böylece alanın nereye gittiği ile ilgili kendi sezgilerinizi oluşturabilir ve belki de kendi mimarilerinizi geliştirebilirsiniz. Örneğin, bu bölümde açıklanan toplu normalleştirme ve artık bağlantılar, derin modellerin eğitimi ve tasarlanması için iki popüler fikir sunmuştur.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
```
