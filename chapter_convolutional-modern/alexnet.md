# Derin Konvolüsyonel Sinir Ağları (AlexNet)
:label:`sec_alexnet`

CNN'ler LeNet'in tanıtımını takiben bilgisayar görüşü ve makine öğrenimi topluluklarında iyi bilinse de, alana hemen hakim olmadılar. LeNet, erken küçük veri kümelerinde iyi sonuçlar elde etse de, CNN'leri daha büyük, daha gerçekçi veri kümeleri üzerinde eğitmenin performansı ve fizibilitesi henüz belirlenmemişti. Aslında, 1990'ların başları ile 2012 yılının dönüm noktası sonuçları arasındaki müdahale süresinin büyük bir bölümünde, sinir ağları genellikle destek vektör makineleri gibi diğer makine öğrenimi yöntemleri tarafından aşıldı.

Bilgisayar görüşü için, bu karşılaştırma belki de adil değildir. Bu, evrimsel ağlara girişler ham veya hafif işlenmiş (örneğin, merkezleme yoluyla) piksel değerlerinden oluşsa da, uygulayıcılar asla ham pikselleri geleneksel modellere beslemeyeceklerdir. Bunun yerine, tipik bilgisayar görüş boru hatları manuel mühendislik özelliği çıkarma boru hatlarından oluşuyordu. *Özellikleri öğrenmek* yerine, özellikler* hazırlanmış*. İlerlemenin çoğu, özellikler için daha akıllı fikirlere sahip olmandan geldi ve öğrenme algoritması genellikle sonradan düşünceye düşürüldü.

Bazı sinir ağı hızlandırıcıları 1990'larda mevcut olmasına rağmen, çok sayıda parametreyle derin çok kanallı, çok katmanlı CNN'ler yapmak için henüz yeterince güçlü değillerdi. Dahası, veri kümeleri hala nispeten küçüktü. Bu engellere ek olarak, parametre başlatma sezgisel yöntemleri, stokastik gradyan inişin akıllı varyantları, ezilmeyen aktivasyon fonksiyonları ve etkili düzenleme teknikleri de dahil olmak üzere sinir ağlarını eğitmek için anahtar hileler hala eksikti.

Böylece, *uçtan uç* (pikselden sınıflandırmaya) sistemleri eğitmek yerine, klasik boru hatları daha çok şöyle görünüyordu:

1. İlginç bir veri kümesi elde edin. İlk günlerde, bu veri setleri pahalı sensörlere ihtiyaç duyuyordu (o sırada 1 megapiksel görüntüler son teknolojiydi).
2. Veri kümesini, optik, geometri, diğer analitik araçlar ve bazen şanslı lisansüstü öğrencilerin rastlantısal keşiflerine dayanan elle hazırlanmış özelliklerle ön işleyin.
3. Verileri SIFT (ölçek-değişmez özellik dönüşümü) :cite:`Lowe.2004`, SURF (hızlandırılmış sağlam özellikler) :cite:`Bay.Tuytelaars.Van-Gool.2006` veya herhangi bir sayıda elle ayarlanmış diğer boru hattı gibi standart özellik çıkarıcıları kümesinde besleyin.
4. Bir sınıflandırıcı eğitmek için ortaya çıkan temsilleri en sevdiğiniz sınıflandırıcıya (muhtemelen doğrusal bir model veya çekirdek yöntemi) dökün.

Makine öğrenimi araştırmacılarıyla konuştuysanız, makine öğreniminin hem önemli hem de güzel olduğuna inanırlardı. Zarif teoriler çeşitli sınıflandırıcıların özelliklerini kanıtladı. Makine öğrenimi alanı gelişen, titiz ve son derece yararlıydı. Ancak, bir bilgisayar görüşü araştırmacısı ile konuşsanız, çok farklı bir hikaye duyarsınız. Görüntü tanımanın kirli gerçeği, size söylerlerdi, algoritmaları öğrenme değil, özelliklerin ilerlemeyi yönlendirdiğidir. Bilgisayar görüşü araştırmacıları haklı olarak, biraz daha büyük veya daha temiz bir veri kümesi veya biraz geliştirilmiş özellik çıkarma boru hattı herhangi bir öğrenme algoritmasından daha nihai doğruluk için çok daha önemli olduğuna inanıyordu.

## Öğrenme Temsilleri

İşleri devletinin bir başka yolu da, boru hattının en önemli kısmının temsil olmasıdır. Ve 2012 yılına kadar temsil mekanik olarak hesaplandı. Aslında, yeni bir özellik fonksiyonu seti mühendisliği, sonuçları iyileştirme ve yöntemi yazma belirgin bir kağıt türüydü. SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, HOG (odaklı gradyan histogramları) :cite:`Dalal.Triggs.2005`, [bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) ve benzeri özellik çıkarıcılar tüneğe hükmetti.

Yann LeCun, Geoff Hinton, Yoshua Bengio, Andrew Ng, Shun-ichi Amari ve Juergen Schmidhuber de dahil olmak üzere bir başka araştırmacı grubunun farklı planları vardı. Onlar özellikleri kendilerini öğrenilmesi gerektiğine inanıyordu. Dahası, makul derecede karmaşık olması için, özelliklerin hiyerarşik olarak, her biri öğrenilebilir parametrelere sahip birden çok ortak öğrenilen katmanla oluşması gerektiğine inanıyorlardı. Görüntü durumunda, en düşük katmanlar kenarları, renkleri ve dokuları algılayabilir. Gerçekten de, Alex Krizhevsky, Ilya Sutskever ve Geoff Hinton, bir CNN yeni bir varyantı önerdi
*AlexNet*,
2012 ImageNet meydan okumasında mükemmel performans elde etti. AlexNet, :cite:`Krizhevsky.Sutskever.Hinton.2012`'ün atılım yapan ImageNet sınıflandırma kağıdının ilk yazarı olan Alex Krizhevsky'nin adını aldı.

İlginçtir ki, ağın en düşük katmanlarında, model bazı geleneksel filtrelere benzeyen özellik çıkarıcıları öğrendi. :numref:`fig_filters`, AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012` kağıdından çoğaltılıyor ve alt düzey görüntü tanımlayıcılarını açıklıyor.

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Ağdaki daha yüksek katmanlar, gözler, burunlar, çimen bıçakları vb. gibi daha büyük yapıları temsil edecek şekilde bu temsillerin üzerine inşa edilebilir. Daha yüksek katmanlar bile insanlar, uçaklar, köpekler veya frizbi gibi tüm nesneleri temsil edebilir. Nihayetinde, son gizli durum, farklı kategorilere ait verilerin kolayca ayrılabilmesi için içeriğini özetleyen görüntünün kompakt bir temsilini öğrenir.

Çok katmanlı CNN'ler için nihai atılım 2012'de gelirken, çekirdek bir grup araştırmacı, görsel verilerin hiyerarşik temsillerini uzun yıllar öğrenmeye çalışarak kendilerini bu fikre adamıştı. 2012'deki nihai atılım iki temel faktöre atfedilebilir.

### Eksiksiz Bileşen: Veri

Birçok katmana sahip derin modeller, dışbükey optimizasyonlara (örneğin doğrusal ve çekirdek yöntemleri) dayalı geleneksel yöntemlerden önemli ölçüde daha iyi performans gösterdikleri rejime girmek için büyük miktarda veri gerektirir. Bununla birlikte, bilgisayarların sınırlı depolama kapasitesi, sensörlerin göreceli giderleri ve 1990'lardaki nispeten daha sıkı araştırma bütçeleri göz önüne alındığında, çoğu araştırma küçük veri kümelerine dayanıyordu. Çok sayıda makale, UCI veri kümelerinin koleksiyonuna değindi, bunların birçoğu düşük çözünürlükte doğal olmayan ortamlarda yakalanan sadece yüzlerce veya (birkaç) binlerce görüntü içeriyordu.

2009'da ImageNet veri kümesi yayımlandı ve araştırmacıları 1000 farklı nesne kategorisinden her biri 1000 farklı 1 milyon örnekten modelleri öğrenmeye zorladı. Bu veri kümesini tanıtan Fei-Fei Li liderliğindeki araştırmacılar, her kategori için büyük aday setlerini ön filtrelemek için Google Image Search kullandı ve ilgili kategoriye ait olup olmadığını her görüntü için onaylamak için Amazon Mechanical Turk crowdsourcing boru hattını kullandı. Bu ölçek eşi görülmemiş. ImageNet Challenge olarak adlandırılan ilişkili rekabet, bilgisayar görme ve makine öğrenimi araştırmalarını ileriye itti, araştırmacıları hangi modellerin daha önce akademisyenlerin düşündüklerinden daha büyük bir ölçekte en iyi performans gösterdiğini belirlemelerine zorladı.

### Eksiksiz Malzeme: Donanım

Derin öğrenme modelleri, bilgi işlem döngülerinin doymak bilmeyen tüketicileridir. Eğitim yüzlerce devir alabilir ve her yineleme, hesaplamalı olarak pahalı doğrusal cebir işlemlerinin birçok katmanından veri geçirmeyi gerektirir. Bu, 1990'ların ve 2000'lerin başında, daha verimli bir şekilde optimize edilmiş dışbükey hedeflere dayanan basit algoritmaların tercih edilmesinin başlıca nedenlerinden biridir.

*Grafik işlem birimleri* (GPU'lar) bir oyun değiştirici olduğu kanıtlandı
derin öğrenmeyi mümkün kılmada. Bu yongalar uzun bilgisayar oyunlarından yararlanmak için grafik işlemeyi hızlandırmak için geliştirilmiştir. Özellikle, birçok bilgisayar grafik görevi için gerekli olan yüksek verim $4 \times 4$ matris vektör ürünleri için optimize edilmişlerdir. Neyse ki, bu matematik, evrimsel katmanları hesaplamak için gerekli olana çarpıcı bir şekilde benzer. Bu süre zarfında NVIDIA ve ATI, GPU'ları genel bilgi işlem işlemleri için optimize etmeye ve bunları *genel amaçlı GPU'lar* (GPGPU) olarak pazarlamaya başlamıştı.

Bazı sezgiler sağlamak için, modern bir mikroişlemcinin (CPU) çekirdeklerini göz önünde bulundurun. Çekirdeklerin her biri, yüksek bir saat frekansında çalışan ve büyük önbellekleri (birkaç megabayta kadar L3) spor yapan oldukça güçlüdür. Her çekirdek, şube belirleyicileri, derin bir boru hattı ve çok çeşitli programları çalıştırmasını sağlayan diğer çan ve ıslıklarla birlikte çok çeşitli talimatları uygulamak için çok uygundur. Bununla birlikte, bu belirgin güç aynı zamanda Aşil topuğudur: genel amaçlı çekirdekler inşa etmek çok pahalıdır. Çok sayıda yonga alanı, sofistike bir destek yapısı (bellek arabirimleri, çekirdekler arasında önbelleğe alma mantığı, yüksek hızlı ara bağlantılar vb.) gerektirirler ve herhangi bir görevde nispeten kötüdürler. Modern dizüstü bilgisayarlar 4 adede kadar çekirdeğe sahiptir ve hatta üst düzey sunucular bile 64 çekirdeği nadiren aşmaktadır, çünkü maliyet etkin değildir.

Karşılaştırma olarak, GPU'lar $100 \sim 1000$ küçük işleme elemanlarından oluşur (ayrıntılar NVIDIA, ATI, ARM ve diğer yonga satıcıları arasında biraz farklılık gösterir), genellikle daha büyük gruplar halinde gruplandırılır (NVIDIA bunları çözgü olarak adlandırır). Her çekirdek nispeten zayıf olsa da, bazen 1GHz altı saat frekansında bile olsa, GPU'ların büyüklük sıralamasını CPU'lardan daha hızlı hale getiren bu tür çekirdeklerin toplam sayısıdır. Örneğin, NVIDIA'nın yeni Volta nesli, özel talimatlar için çip başına 120 TFlop (ve daha genel amaçlı olanlar için 24 TFlop'a kadar) sunarken, CPU'ların kayan nokta performansı bugüne kadar 1 TFlop'u aşmadı. Bunun mümkün olmasının nedeni aslında oldukça basittir: Birincisi, güç tüketimi saat frekansı ile*dört dereceden olarak* büyümeye eğilimlidir. Bu nedenle, 4 kat daha hızlı çalışan bir CPU çekirdeğinin güç bütçesi için (tipik bir sayı), $1/4$ hızında 16 GPU çekirdeğini kullanabilirsiniz, bu da performansın $16 \times 1/4 = 4$ katını verir. Ayrıca, GPU çekirdekleri çok daha basittir (aslında, uzun bir süre için genel amaçlı kod yürütebilir* bile değillerdi), bu da onları daha verimli hale getirir. Son olarak, derin öğrenmede birçok işlem yüksek bellek bant genişliği gerektirir. Yine, GPU'lar burada en az 10 kat daha geniş olan otobüslerle parlıyor.

2012'ye geri dön. Alex Krizhevsky ve Ilya Sutskever GPU donanımı üzerinde çalışabilecek derin bir CNN uyguladığında büyük bir atılım oldu. CNN'lerdeki hesaplamalı darboğazların, kıvrımların ve matris çarpımlarının, donanımda paralelleştirilebilecek tüm işlemler olduğunu fark ettiler. 3GB belleğe sahip iki NVIDIA GTX 580s kullanarak hızlı kıvrımlar uyguladılar. [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) kodu, birkaç yıldır endüstri standardı olduğu ve derin öğrenme patlamasının ilk birkaç yılını desteklediği için yeterince iyiydi.

## AlexNet

8 katmanlı CNN kullanan AlexNet, ImageNet Büyük Ölçekli Görsel Tanıma Yarışması'nı olağanüstü derecede büyük bir farkla kazandı. Bu ağ, ilk kez, öğrenme yoluyla elde edilen özelliklerin el ile tasarlanmış özellikleri aşabildiğini ve bilgisayar görüşünde önceki paradigmayı kırabileceğini gösterdi.

:numref:`fig_alexnet`'ün gösterdiği gibi AlexNet ve LeNet'in mimarileri çok benzer. Modelin iki küçük GPU'ya uyması için 2012'de gerekli olan tasarım tuhaflıklarından bazılarını kaldırarak AlexNet'in biraz aerodinamik bir versiyonunu sunduğumuzu unutmayın.

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet ve LeNet'in tasarım felsefeleri çok benzer, ancak önemli farklılıklar da vardır. İlk olarak, AlexNet nispeten küçük LeNet5'ten çok daha derin. AlexNet sekiz katmandan oluşur: beş kıvrımsal katman, iki tam bağlı gizli katman ve bir tam bağlı çıkış katmanı. İkincisi, AlexNet aktivasyon fonksiyonu olarak sigmoid yerine ReLU'yu kullandı. Bize aşağıdaki ayrıntıları inceleyelim.

### Mimarlık

AlexNet'in ilk katmanında, evrim penceresi şekli $11\times11$'dir. ImageNet'teki çoğu görüntü MNIST görüntülerinden on kat daha yüksek ve daha geniş olduğundan, ImageNet verilerindeki nesneler daha fazla piksel işgal etme eğilimindedir. Sonuç olarak, nesneyi yakalamak için daha büyük bir evrim penceresi gereklidir. İkinci kattaki evrişim pencere şekli $5\times5$'ya, ardından $3\times3$'e indirgenir. Buna ek olarak, birinci, ikinci ve beşinci kıvrımsal katmanlardan sonra, ağ $3\times3$ pencere şekli ve 2'lik bir adım ile maksimum havuzlama katmanları ekler. Ayrıca, AlexNet'in LeNet'ten on kat daha fazla evrişim kanalı vardır.

Son konvolüsyonel tabakadan sonra 4096 çıkışlı iki tam bağlı katman vardır. Bu iki büyük tam bağlı katman, yaklaşık 1 GB'lık model parametreleri üretir. Erken GPU'lardaki sınırlı bellek nedeniyle, orijinal AlexNet çift veri akışı tasarımı kullandı, böylece iki GPU'larının her biri modelin yalnızca yarısını depolamaktan ve hesaplamaktan sorumlu olabilir. Neyse ki, GPU belleği artık nispeten bol, bu yüzden nadiren GPU'lar arasında modelleri parçalamamız gerekiyor (AlexNet modelinin versiyonumuz bu açıdan orijinal kağıttan sapıyor).

### Etkinleştirme İşlevleri

Ayrıca, AlexNet sigmoid aktivasyon işlevini daha basit bir ReLU aktivasyon fonksiyonuna değiştirdi. Bir yandan, ReLU aktivasyon işlevinin hesaplanması daha kolaydır. Örneğin, sigmoid etkinleştirme işlevinde bulunan üsleme işlemi yoktur. Öte yandan, ReLU etkinleştirme işlevi, farklı parametre başlatma yöntemleri kullanıldığında model eğitimini kolaylaştırır. Bunun nedeni, sigmoid etkinleştirme işlevinin çıktısı 0 veya 1'e çok yakın olduğunda, bu bölgelerin degradesinin neredeyse 0 olmasıdır, böylece geri yayılım model parametrelerinin bazılarını güncelleştirmeye devam edemez. Buna karşılık, pozitif aralıktaki ReLU etkinleştirme işlevinin degrade her zaman 1'dir. Bu nedenle, model parametreleri düzgün başlatılmazsa, sigmoid işlevi pozitif aralıkta neredeyse 0 degrade elde edebilir, böylece model etkili bir şekilde eğitilemez.

### Kapasite Kontrolü ve Ön İşleme

AlexNet, tam bağlı katmanın model karmaşıklığını bırakarak (:numref:`sec_dropout`) kontrol ederken, LeNet sadece ağırlık çürümesini kullanır. Verileri daha da artırmak için AlexNet'in eğitim döngüsü, ters çevirme, kırpma ve renk değişiklikleri gibi çok fazla görüntü büyütme ekledi. Bu, modeli daha sağlam hale getirir ve daha büyük numune boyutu aşırı uyumu etkili bir şekilde azaltır. Veri büyütme işlemlerini :numref:`sec_image_augmentation`'te daha ayrıntılı olarak tartışacağız.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

Her katmanın çıkış şeklini gözlemlemek için hem yükseklik hem de 224 genişliğinde tek kanallı bir veri örneği oluşturuyoruz. :numref:`fig_alexnet`'teki AlexNet mimarisine uyuyor.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)
```

## Veri kümesini okuma

AlexNet gazetede ImageNet üzerinde eğitilmiş olsa da, bir ImageNet modelini modern bir GPU'da bile saatler veya günler sürebileceğinden, burada Moda-MNIST kullanıyoruz. AlexNet'in doğrudan Moda-MNIST üzerine uygulanmasıyla ilgili sorunlardan biri, görüntülerinin ImageNet görüntülerinden daha düşük çözünürlüğe ($28 \times 28$ piksel) sahip olmasıdır. İşleri yürütebilmek için onları $224 \times 224$'ya yükseltiyoruz (genellikle akıllı bir uygulama değil, ama burada AlexNet mimarisine sadık olmak için yapıyoruz). Bu yeniden boyutlandırma `resize` bağımsız değişkeni ile `d2l.load_data_fashion_mnist` işlevinde gerçekleştiriyoruz.

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## Eğitim

Şimdi AlexNet'i eğitmeye başlayabiliriz. :numref:`sec_lenet`'teki LeNet ile karşılaştırıldığında, buradaki ana değişiklik, daha derin ve daha geniş ağ, daha yüksek görüntü çözünürlüğü ve daha maliyetli kıvrımlar nedeniyle daha küçük bir öğrenme hızı ve çok daha yavaş eğitim kullanılmasıdır.

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

## Özet

* AlexNet, LeNet'e benzer bir yapıya sahiptir, ancak büyük ölçekli ImageNet veri kümesine uyacak şekilde daha fazla kıvrımsal katman ve daha büyük bir parametre alanı kullanır.
* Bugün AlexNet çok daha etkili mimariler tarafından aşıldı, ancak günümüzde kullanılan sığ derin ağlara kadar önemli bir adımdır.
* AlexNet'in uygulamasında LeNet'ten sadece birkaç satır daha var gibi görünse de, akademik topluluğun bu kavramsal değişimi benimsemesi ve mükemmel deneysel sonuçlarından yararlanması uzun yıllar aldı. Bu aynı zamanda verimli hesaplama araçlarının eksikliğinden kaynaklanıyordu.
* Dropout, ReLU ve ön işleme, bilgisayar görme görevlerinde mükemmel performans elde etmedeki diğer önemli adımlardı.

## Egzersizler

1. Çadırların sayısını artırmayı deneyin. LeNet ile karşılaştırıldığında, sonuçlar nasıl farklı? Neden?
1. AlexNet Moda-MNIST veri kümesi için çok karmaşık olabilir.
    1. Doğruluğun önemli ölçüde düşmemesini sağlarken, eğitimi daha hızlı yapmak için modeli basitleştirmeyi deneyin.
    1. Doğrudan $28 \times 28$ görüntülerde çalışan daha iyi bir model tasarlayın.
1. Toplu iş boyutunu değiştirin ve doğruluk ve GPU belleğindeki değişiklikleri gözlemleyin.
1. AlexNet'in hesaplama performansını analiz eder.
    1. AlexNet'in bellek ayak izi için baskın kısım nedir?
    1. AlexNet'te hesaplama için baskın kısım nedir?
    1. Sonuçları hesaplarken bellek bant genişliğine ne dersiniz?
1. LeNet-5'e bırakma ve ReLU uygulayın. İyileşiyor mu? Ön işleme ne dersin?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
