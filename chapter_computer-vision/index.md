# Bilgisayar Görme
:label:`chap_cv`

Tıbbi teşhis, kendi kendini yöneten araçlar, kamera izleme veya akıllı filtreler olsun, bilgisayar görme alanındaki birçok uygulama mevcut ve gelecekteki yaşamlarımızla yakından ilişkilidir. Son yıllarda derin öğrenme, bilgisayar görme sistemlerinin performansını artırmak için dönüştürücü güç olmuştur. En gelişmiş bilgisayar görme uygulamalarının derin öğrenimden neredeyse ayrılmaz olduğu söylenebilir. Bu bölümde bilgisayar görme alanı üzerinde durulacak ve yakın zamanda akademik ve endüstride etkili olan yöntem ve uygulamaları inceleyecektir. 

:numref:`chap_cnn` ve :numref:`chap_modern_cnn`'te, bilgisayar görüşünde yaygın olarak kullanılan çeşitli evrimsel sinir ağlarını inceledik ve bunları basit görüntü sınıflandırma görevlerine uyguladık. Bu bölümün başında, model genelleştirmesini geliştirebilecek iki yöntem, yani *görüntü artırma* ve*ince ayarlama* ve bunları görüntü sınıflandırmasına uygulayacağız. Derin sinir ağları görüntüleri birden çok seviyedeki etkili bir şekilde temsil edebildiğinden, bu tür katmanlı gösterimler, *nesne algılaması*, *semantik segmentasyon* ve *stil aktarımı* gibi çeşitli bilgisayar görme görevlerinde başarıyla kullanılmıştır. Bilgisayar görüşünde katmanlı temsillerden yararlanmanın temel fikrini takiben, nesne algılama için önemli bileşenler ve teknikler ile başlayacağız. Ardından, görüntülerin semantik segmentasyonu için*tamamen evrimsel ağlar* nasıl kullanılacağını göstereceğiz. Sonra bu kitabın kapağı gibi görüntüler oluşturmak için stil aktarma tekniklerinin nasıl kullanılacağını açıklayacağız. Sonunda, bu bölümün materyallerini ve iki popüler bilgisayar vizyonu kıyaslama veri kümesinde önceki bölümleri uygulayarak bu bölümü sonuçlandırıyoruz.

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
