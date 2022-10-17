# Ek: Derin Öğrenme için Matematik
:label:`chap_appendix_math`

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*) ve bu kitabın yazarları


Modern derin öğrenmenin harika yanlarından biri, altındaki matematiğin çoğunun tam olarak idrak edilmeden anlaşılabilir ve kullanılabilir olmasıdır. Bu, alanın olgunlaştığının bir işaretidir. Çoğu yazılım geliştiricisinin artık hesaplanabilir işlevler teorisi hakkında endişelenmesi gerekmediği gibi, derin öğrenme uygulayıcılarının da en büyük olabilirlik (maximum likelihood) öğrenmesinin teorik temelleri hakkında endişelenmesi gerekmemesidir.

Ancak henüz tam olarak sonda değiliz.

Uygulamada, bazen mimari seçimlerin gradyan akışını nasıl etkilediğini veya belirli bir kayıp fonksiyonu ile eğitim yaptığınızda saklı varsayımları anlamanız gerekecektir. Entropinin (düzensizlik) bu dünyada neyi ölçtüğünü ve modelinizde karakter başına bitlerin tam olarak ne anlama geldiğini anlamanıza nasıl yardımcı olabileceğini bilmeniz gerekebilir. Bunların hepsi daha derin matematiksel anlayış gerektirir.

Bu ek, modern derin öğrenmenin temel teorisini anlamak için ihtiyacınız olan matematiksel altyapıyı sağlamayı amaçlamaktadır, ancak tam kapsamlı değildir. Doğrusal cebiri daha derinlemesine incelemeye başlayacağız. Çeşitli dönüşümlerin verilerimiz üzerindeki etkilerini görselleştirmemizi sağlayacak tüm genel doğrusal cebirsel nesnelerin ve işlemlerin geometrik bir anlayışını geliştiriyoruz. Temel unsurlardan biri, öz ayrışımların (eigen-decomposition) temellerinin geliştirilmesidir.

Daha sonra, gradyanın neden en dik iniş yönü olduğunu ve neden geri yaymanın olduğu şekli aldığını tam olarak anlayabileceğimiz noktaya kadar türevsel hesap (diferansiyel kalkülüs) teorisini geliştireceğiz. Daha sonra integral hesabı, bir sonraki konumuz, olasılık teorisini desteklemek için gereken ölçüde tartışılacak.

Pratikte sıklıkla karşılaşılan sorunlar kesin değildir ve bu nedenle belirsiz şeyler hakkında konuşmak için bir dile ihtiyacımız vardır. Rastgele değişkenler teorisini ve en sık karşılaşılan dağılımları gözden geçiriyoruz, böylece modelleri olasılıksal olarak tartışabiliriz. Bu, olasılıksal bir sınıflandırma tekniği olan saf (naif) Bayes sınıflandırıcısının temelini sağlar.

Olasılık teorisi ile yakından ilgili olan şey, istatistik alanıdır. İstatistik, kısa bir bölümde hakkını vererek incelemek için çok büyük bir alan olsa da, özellikle tüm makine öğrenmesi uygulayıcılarının bilmesi gereken temel kavramları tanıtacağız: Tahmin edicileri değerlendirmek ve karşılaştırmak, hipotez testleri yapmak ve güven aralıkları oluşturmak.

Son olarak, bilgi depolama ve aktarımının matematiksel alanı olan bilgi teorisi konusuna dönüyoruz. Bu, bir modelin bir araştırma alanında ne kadar bilgi tuttuğunu nicel olarak tartışabileceğimiz temel dili sağlar.

Birlikte ele alındığında bunlar, derin öğrenmeyi derinlemesine anlamaya giden yola başlamak için gereken matematiksel kavramların özünü oluştururlar.


```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```

