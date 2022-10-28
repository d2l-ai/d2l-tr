# Modern Evrişimli Sinir Ağları
:label:`chap_modern_cnn`

Artık CNN'leri biraraya bağlama temellerini anladığımıza göre, sizlerle modern CNN mimarilerinde bir tur atacağız. Bu ünitede, her bölüm bir noktada (veya şu anda) birçok araştırma projesinde ve konuşlandırılmış sistemlerde inşanın temel modeli olan önemli bir CNN mimarisine karşılık gelmektedir. Bu ağların her biri kısaca baskın bir mimariydi ve çoğu, 2010 yılından bu yana bilgisayarla görmede gözetimli öğrenmenin ilerlemesinin göstergesi olarak hizmet eden ImageNet yarışmasında kazananlar veya ikincilerdi.

Bu modeller, büyük ölçekli bir görme yarışmasında geleneksel bilgisayarla görme yöntemlerini yenmek için konuşlandırılan ilk büyük ölçekli ağ olan AlexNet'i içerir; bir dizi tekrarlayan eleman bloklarını kullanan VGG ağı; girdileri tüm sinir ağlarını yamalı olarak evriştiren ağ içindeki ağ (NiN); paralel birleştirmelere sahip ağları kullanan GoogLeNet; bilgisayarla görmede en popüler kullanıma hazır mimari olmaya devam eden artık ağlar (residual networks - ResNet) ve hesaplanması külfetli ancak bazı alanlarda en yüksek başarı performanslarını ortaya koyan yoğun bağlı ağlar (DenseNet).

*Derin* sinir ağları fikri oldukça basit olsa da (bir grup katmanı bir araya getirin), performans mimarilere ve hiper parametre seçeneklerine  bağlı olarak aşırı farklılık gösterebilir. Bu bölümde açıklanan sinir ağları sezginin, birkaç matematiksel içgörünün ve bir sürü deneme yanılmanın ürünüdür. Bu modelleri tarihsel sırayla sunuyoruz, kısmen de tarih duygusunu aktarmak için, böylece alanın nereye gittiği ile ilgili kendi sezgilerinizi oluşturabilir ve belki de kendi mimarilerinizi geliştirebilirsiniz. Örneğin, bu bölümde açıklanan toptan normalleştirme ve artık bağlantılar, derin modellerin eğitimi ve tasarlanması için iki popüler fikir sunmuştur.

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
