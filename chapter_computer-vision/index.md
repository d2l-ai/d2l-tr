# Bilgisayarla Görme
:label:`chap_cv`

Tıbbi teşhis, kendi kendini yöneten araçlar, kameralı izleme veya akıllı filtreler olsun, bilgisayarla görme alanındaki birçok uygulama mevcut ve gelecekteki yaşamlarımızla yakından ilişkilidir. Son yıllarda derin öğrenme, bilgisayarla görme sistemlerinin performansını artırmak için dönüştürücü güç olmuştur. En gelişmiş bilgisayarla görme uygulamalarının derin öğrenmeden neredeyse ayrılmaz olduğu söylenebilir. Bu bölüm bilgisayarla görme alanı üzerinde duracak ve yakın zamanda akademide ve endüstride etkili olan yöntemleri ve uygulamaları inceleyecektir. 

:numref:`chap_cnn` ve :numref:`chap_modern_cnn` içinde, bilgisayarla görmede yaygın olarak kullanılan çeşitli evrişimli sinir ağlarını inceledik ve bunları basit imge sınıflandırma görevlerine uyguladık. Bu bölümün başında, model genelleştirmesini geliştirebilecek iki yöntemi, yani *imge artırımı* ve *ince ayarlama*, tanımlayacağız ve bunları imge sınıflandırmasına uygulayacağız. Derin sinir ağları imgeleri birden çok katmanda etkili bir şekilde temsil edebildiğinden, bu tür katmanlı gösterimler, *nesne tespiti*, *anlamsal bölümleme* ve *stil aktarımı* gibi çeşitli bilgisayarla görme görevlerinde başarıyla kullanılmıştır. Bilgisayarla görmede katmanlı temsillerden yararlanmanın temel fikrini takiben, nesne tespiti için önemli bileşenler ve teknikler ile başlayacağız. Ardından, imgelerin anlamsal bölümlemesi için *tamamen evrişimli ağlar*ı nasıl kullanılacağını göstereceğiz. Sonra bu kitabın kapağı gibi imgeler oluşturmak için stil aktarım tekniklerinin nasıl kullanılacağını açıklayacağız. Sonunda, bu bölümün materyallerini ve önceki birkaç bölümü iki bilindik bilgisayarla görme kıyaslama veri kümesine uygulayarak sonlandırıyoruz.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```
