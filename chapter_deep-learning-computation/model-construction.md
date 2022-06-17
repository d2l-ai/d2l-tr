# Katmanlar ve Bloklar
:label:`sec_model_construction`

Sinir ağlarını ilk tanıttığımızda, tek çıktılı doğrusal modellere odaklandık. Burada tüm model sadece tek bir nörondan oluşuyor. Tek bir nöronun (i) bazı girdiler aldığını, (ii) karşılık gelen skaler bir çıktı ürettiğini ve (iii) ilgili bazı amaç işlevlerini optimize etmek için güncellenebilen bir dizi ilişkili parametreye sahip olduğunu unutmayın. Sonra, birden çok çıktıya sahip ağları düşünmeye başladığımızda, tüm bir nöron katmanını karakterize etmek için vektörleştirilmiş aritmetikten yararlandık. Tıpkı bireysel nöronlar gibi, katmanlar (i) bir dizi girdi alır, (ii) karşılık gelen çıktıları üretir ve (iii) bir dizi ayarlanabilir parametre ile açıklanır. Softmaks bağlanımı üzerinde çalıştığımızda, tek bir katmanın kendisi modeldi. Bununla birlikte, daha sonra MLP'leri devreye soktuğumuzda bile, modelin bu aynı temel yapıyı koruduğunu düşünebiliriz.

İlginç bir şekilde, MLPler için hem tüm model hem de kurucu katmanları bu yapıyı paylaşır. Tam model ham girdileri (öznitelikleri) alır, çıktılar (tahminler) üretir ve parametrelere (tüm kurucu katmanlardan birleşik parametreler) sahiptir. Benzer şekilde, her bir katman girdileri alır (önceki katman tarafından sağlanır) çıktılar (sonraki katmana girdiler) üretir ve sonraki katmandan geriye doğru akan sinyale göre güncellenen bir dizi ayarlanabilir parametreye sahiptir.

Nöronların, katmanların ve modellerin bize işimiz için yeterince soyutlama sağladığını düşünseniz de, genellikle tek bir katmandan daha büyük ancak tüm modelden daha küçük olan bileşenler hakkında konuşmayı uygun bulduğumuz ortaya çıkıyor. Örneğin, bilgisayarla görmede aşırı popüler olan ResNet-152 mimarisi, yüzlerce katmana sahiptir. Bu katmanlar, *katman gruplarının* tekrar eden desenlerinden oluşur. Böyle bir ağın her seferinde bir katman olarak uygulanması sıkıcı bir hal alabilir. Bu endişe sadece varsayımsal değildir---bu tür tasarım modelleri pratikte yaygındır. Yukarıda bahsedilen ResNet mimarisi, hem tanıma hem de tespit için 2015 ImageNet ve COCO bilgisayarla görme yarışmalarını kazandı :cite:`He.Zhang.Ren.ea.2016` ve birçok görme görevi için bir ilk tatbik edilen mimari olmaya devam ediyor. Katmanların çeşitli yinelenen desenlerde düzenlendiği benzer mimariler artık doğal dil işleme ve konuşma dahil olmak üzere diğer alanlarda da her yerde mevcuttur.

Bu karmaşık ağları uygulamak için, bir sinir ağı *bloğu* kavramını sunuyoruz. Bir blok, tek bir katmanı, birden çok katmandan oluşan bir bileşeni veya tüm modelin kendisini tanımlayabilir! Blok soyutlamayla çalışmanın bir yararı, bunların genellikle yinelemeli olarak daha büyük yapay nesnelerle birleştirilebilmeleridir. Bu, :numref:`fig_blocks` içinde gösterilmiştir. İsteğe bağlı olarak rastgele karmaşıklıkta bloklar oluşturmak için kod tanımlayarak, şaşırtıcı derecede sıkıştırılmış kod yazabilir ve yine de karmaşık sinir ağlarını uygulayabiliriz.

![Çoklu katmanlar bloklara birleştirilir, daha geniş modelleri oluşturan tekrar eden desenler oluştururlar](../img/blocks.svg)
:label:`fig_blocks`


Programlama açısından, bir blok *sınıf* ile temsil edilir. Onun herhangi bir alt sınıfı, girdisini çıktıya dönüştüren ve gerekli parametreleri depolayan bir ileri yayma yöntemi tanımlamalıdır. Bazı blokların herhangi bir parametre gerektirmediğini unutmayın. Son olarak, gradyanları hesaplamak için bir blok geriye yayma yöntemine sahip olmalıdır. Neyse ki, kendi bloğumuzu tanımlarken otomatik türev almanın sağladığı bazı perde arkası sihir sayesinde (:numref:`sec_autograd` içinde tanıtıldı), sadece parametreler ve ileri yayma işlevi hakkında endişelenmemiz gerekir.

[**Başlangıç olarak MLPleri uygulamak için kullandığımız kodları yeniden gözden geçiriyoruz**] (:numref:`sec_mlp_concise`). Aşağıdaki kod, 256 birim ve ReLU etkinleştirmesine sahip tam bağlı bir gizli katmana sahip bir ağ oluşturur ve ardından 10 birimle (etkinleştirme işlevi yok) tam bağlı çıktı katmanı gelir.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
Bu örnekte, modelimizi bir `nn.Sequential` örneğini oluşturarak döndürülen nesneyi `net` değişkenine atayarak oluşturduk. Daha sonra, katmanları çalıştırılmaları gereken sırayla eklemek için tekrar tekrar `add` işlevini çağırıyoruz. Kısaca, `nn.Sequential`, Gluon'da bir blok sunan sınıf olan özel bir `Block` türünü tanımlar. Oluşturucu `Block`'ların sıralı bir listesini tutar. `add` işlevi, basitçe her ardışık `Block`'un listeye eklenmesini kolaylaştırır. Her katmanın, kendisinin `Block`'un alt sınıfı olan `Dense` sınıfının bir örneği olduğuna dikkat edin. İleri yayma (`forward`) işlevi de oldukça basittir: Listedeki her `Block` birbirine zincirlenerek her birinin çıktısını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktılarını elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında `Block` sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.forward(X)` için kısaltmadır.
:end_tab:

:begin_tab:`pytorch`
Bu örnekte, modelimizi bir `nn.Sequential` örneğini oluşturarak, katmanları bağımsız değişken olarak iletilmeleri gereken sırayla oluşturduk. Kısaca, `nn.Sequential`, PyTorch'ta bir blok sunan özel bir sınıf `Module` türünü tanımlar. Kurucu `Module`'lerin sıralı bir listesini tutar. Tam bağlı iki katmanın her birinin, kendisi de `Module`'ün bir alt sınıfı olan `Linear` sınıfının bir örneği olduğuna dikkat edin. İleri yayma (`forward`) işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıktısını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktıları elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında `net.__call__(X)` için kısa yoldur.
:end_tab:

:begin_tab:`tensorflow`
Bu örnekte, modelimizi bir `keras.models.Sequential` örneğini oluşturarak, katmanları bağımsız değişken olarak iletilmeleri gereken sırayla oluşturduk. Kısaca `Sequential`, Keras'ta bir blok sunan özel bir sınıf `keras.Model`'i tanımlar, ki kurucu `Model`'lerin sıralı bir listesini tutar. İki tam bağlı katmanın her birinin, kendisi `Model`'in alt sınıfı olan `Dense` sınıfının bir örneği olduğuna dikkat edin. İleri yayma (`call`) işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıkışını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktılarını elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında `Block` sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.call(X)` için kısaltmadır.
:end_tab:

## [**Özel Yapım Blok**]

Bir bloğun nasıl çalıştığına dair sezgiyi geliştirmenin belki de en kolay yolu, bloğu kendimiz uygulamaktır. Kendi özel yapım bloğumuzu uygulamadan önce, her bloğun sağlaması gereken temel işlevleri kısaca özetliyoruz:

:begin_tab:`mxnet, tensorflow`

1. Girdi verilerini ileri yayma yöntemine bağımsız değişkenler olarak alın.
1. İleri yayma işlevi kullanarak bir değer döndürüp bir çıktı oluşturun. Çıktının girdiden farklı bir şekle sahip olabileceğini unutmayın. Örneğin, yukarıdaki modelimizdeki ilk tam bağlı katman, rastgele bir boyut girdisi alır, ancak 256 boyutunda bir çıktı verir.
1. Girdiye göre çıktısının gradyanını hesaplayın, ki bu da geriye yayma yöntemiyle erişilebilir. Genellikle bu otomatik olarak gerçekleşir.
1. İleri yaymayı hesaplamayı yürütmek için gerekli olan bu parametreleri saklayın ve bunlara erişim sağlayın.
1. Model parametrelerini gerektiği gibi ilkletin.

:end_tab:

:begin_tab:`pytorch`

1. Girdi verilerini ileri yayma yöntemine bağımsız değişkenler olarak alın.
1. İleri yayma işlevi kullanarak bir değer döndürüp bir çıktı oluşturun. Çıktının girdiden farklı bir şekle sahip olabileceğini unutmayın. Örneğin, yukarıdaki modelimizdeki ilk tam bağlı katman 20 boyutlu bir girdi alır, ancak 256 boyutunda bir çıktı verir.
1. Girdiye göre çıktısının gradyanını hesaplayın, ki bu da geriye yayma yöntemiyle erişilebilir. Genellikle bu otomatik olarak gerçekleşir.
1. İleri yaymayı hesaplamayı yürütmek için gerekli olan bu parametreleri saklayın ve bunlara erişim sağlayın.
1. Model parametrelerini gerektiği gibi ilkletin.

:end_tab:


Aşağıdaki kod parçacığında, 256 gizli birime sahip bir gizli katman ve 10 boyutlu bir çıktı katmanı ile bir MLP'ye karşılık gelen bir bloğu sıfırdan kodladık. Aşağıdaki `MLP` sınıfının bir bloğu temsil eden sınıftan kalıtsal çoğaltıldığını unutmayın. Yalnızca kendi kurucumuzu (Python'daki '__init__' işlevi) ve ileri yayma işlevini sağlayarak, büyük ölçüde ana sınıfın işlevlerine güveneceğiz.

```{.python .input}
class MLP(nn.Block):
    # Model parametreleriyle bir katman ilan edin. 
    # Burada, tam bağlı iki katman ilan ediyoruz.
    def __init__(self, **kwargs):
        # Gerekli ilklemeyi gerçekleştirmek için `MLP` üst sınıfının `Block` 
        # kurucusunu çağırın. Bu şekilde, sınıf örneği yaratma sırasında model
        # parametreleri, `params` (daha sonra açıklanacak) gibi diğer fonksiyon 
        # argümanları da belirtilebilir.
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Gizli katman
        self.out = nn.Dense(10)  # Çıktı katmanı

    # Modelin ileri yaymasını tanımla, yani `X` girdisine dayalı 
    # olarak gerekli model çıktısının nasıl döndürüleceğini tanımla.
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Model parametreleriyle bir katman tanımlayın. 
    # Burada tam bağlı iki katman tanımlıyoruz.
    def __init__(self):
        # Gerekli ilklemeyi gerçekleştirmek için `MLP` üst sınıfının `Module` 
        # kurucusunu çağırın. Bu şekilde, sınıf örneği yaratma sırasında model parametreleri, 
        # `params` (daha sonra açıklanacak) gibi diğer fonksiyon argümanları da belirtilebilir.
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Gizli katman
        self.out = nn.Linear(256, 10)  # Çıktı katmanı

    # Modelin ileri yaymasını tanımla, yani `X` girdisine dayalı 
    # olarak gerekli model çıktısının nasıl döndürüleceğini tanımla.
    def forward(self, X):
        # Burada, nn.function modülünde tanımlanan ReLU'nun fonksiyonel 
        # versiyonunu kullandığımıza dikkat edin.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Model parametreleriyle bir katman tanımlayın. 
    # Burada tam bağlı iki katman tanımlıyoruz.
    def __init__(self):
        # Gerekli ilklemeyi gerçekleştirmek için `MLP` üst sınıfının `Model` 
        # kurucusunu çağırın. Bu şekilde, sınıf örneği yaratma sırasında model parametreleri, 
        # `params` (daha sonra açıklanacak) gibi diğer fonksiyon argümanları da belirtilebilir.
        super().__init__()
        # Gizli katman
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Çıktı katmanı

    # Modelin ileri yaymasını tanımla, yani `X` girdisine dayalı 
    # olarak gerekli model çıktısının nasıl döndürüleceğini tanımla.
    def call(self, X):
        return self.out(self.hidden((X)))
```

İlk olarak ileri yayma işlevine odaklanalım. Girdi olarak `X`'i aldığını, gizli gösterimi etkinleştirme işlevi uygulanmış olarak hesapladığını ve logitlerini çıktı verdiğini unutmayın. Bu `MLP` uygulamasında, her iki katman da örnek değişkenlerdir. Bunun neden makul olduğunu anlamak için, iki MLP'yi (`net1` ve `net2`) somutlaştırdığınızı ve bunları farklı veriler üzerinde eğittiğinizi hayal edin. Doğal olarak, iki farklı öğrenilmiş modeli temsil etmelerini bekleriz.

[**MLP'nin katmanlarını**] kurucuda [**ilkliyoruz**] ve (**daha sonra bu katmanları**) her bir ileri yayma işlevi çağrısında (**çağırıyoruz**). Birkaç önemli ayrıntıya dikkat edin. İlk olarak, özelleştirilmiş `__init__` işlevimiz, `super().__init__()` aracılığıyla üst sınıfın `__init__` işlevini çağırır ve bizi çoğu bloğa uygulanabilen standart şablon kodunu yeniden biçimlendirme zahmetinden kurtarır. Daha sonra, tam bağlı iki katmanımızı `self.hidden` ve `self.out`'a atayarak somutlaştırıyoruz. Yeni bir operatör uygulamadığımız sürece, geriye yayma işlevi veya parametre ilkleme konusunda endişelenmemize gerek olmadığını unutmayın. Sistem bu işlevleri otomatik olarak üretecektir. Bunu deneyelim.

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

Blok soyutlamanın önemli bir özelliği çok yönlülüğüdür. Katmanlar (tam bağlı katman sınıfı gibi), tüm modeller (yukarıdaki `MLP` sınıfı gibi) veya ara karmaşıklığın çeşitli bileşenlerini oluşturmak için bir bloğu alt sınıflara ayırabiliriz. Bu çok yönlülüğü sonraki bölümlerde, mesela evrişimli sinir ağlarını, ele alırken kullanıyoruz.


## [**Dizili Blok**]

Artık `Sequential` (dizili) sınıfının nasıl çalıştığına daha yakından bakabiliriz. `Sequential`'nin diğer blokları birbirine zincirleme bağlamak için tasarlandığını hatırlayın. Kendi basitleştirilmiş `MySequential`'ımızı oluşturmak için, sadece iki anahtar işlev tanımlamamız gerekir:
1. Blokları birer birer listeye eklemek için bir işlev.
2. Bir girdiyi blok zincirinden, eklendikleri sırayla iletmek için bir ileri yayma işlevi.

Aşağıdaki `MySequential` sınıfı aynı işlevselliği varsayılan `Sequential` sınıfıyla sunar:

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Burada, `block`, `Block` alt sınıfının bir örneğidir ve eşsiz bir 
        # ada sahip olduğunu varsayıyoruz. Bunu `Block` sınıfının `_children` 
        # üye değişkenine kaydederiz ve türü OrderedDict'tir. `MySequential` 
        # örneği `initialize` işlevini çağırdığında, sistem tüm `_children` 
        # üyelerini otomatik olarak ilkler.

        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict, üyelerin eklendikleri sırayla işletileceğini garanti eder.
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Burada, `module` , `mMdule` alt sınıfının bir örneğidir. 
            # Bunu, `Module` sınıfının `_modules` üye değişkenine kaydederiz 
            # ve türü OrderedDict'tir.
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict, üyelerin eklendikleri sırayla işletileceğini garanti eder.
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Burada, `block`, bir `tf.keras.layers.Layer` alt sınıfının bir örneğidir
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` işlevi, sıralı `_children` sözlüğüne tek bir blok ekler. Neden her Gluon `Block`'unun bir `_children` özelliğine sahip olduğunu ve sadece bir Python listesi tanımlamak yerine bunu neden kullandığımızı merak edebilirsiniz. Kısacası, `_children`'in başlıca avantajı, bloğumuzun parametre ilklemesi sırasında Gluon'un, parametreleri de ilklemesi gereken alt blokları bulmak için `_children` sözlüğüne bakmayı bilmesidir.
:end_tab:

:begin_tab:`pytorch`
`__init__` yönteminde, her modülü sıralı `_modules` sözlüğüne tek tek ekliyoruz. Neden her `Module`'ün bir `_modules` özelliğine sahip olduğunu ve sadece bir Python listesi tanımlamak yerine bunu neden kullandığımızı merak edebilirsiniz. Kısacası, `_modules`'ün başlıca avantajı, modülün parametre ilklemesi sırasında, sistemin, parametreleri de ilklemesi gereken alt modülleri bulmak için `_modules` sözlüğüne bakmayı bilmesidir.
:end_tab:

`MySequential`'ımızın ileri yayma işlevi çağrıldığında, eklenen her blok eklendikleri sırayla yürütülür. Artık `MySequential` sınıfımızı kullanarak bir MLP'yi yeniden uygulayabiliriz.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

Bu `MySequential` kullanımının, daha önce `Sequential` sınıfı için yazdığımız kodla aynı olduğuna dikkat edin (:numref:`sec_mlp_concise` içinde açıklandığı gibi).

## [**İleri Yayma İşlevinde Kodu Yürütme**]

`Sequential` sınıfı, model yapımını kolaylaştırarak, kendi sınıfımızı tanımlamak zorunda kalmadan yeni mimarileri bir araya getirmemize olanak tanır. Bununla birlikte, tüm mimariler basit papatya zinciri değildir. Daha fazla esneklik gerektiğinde, kendi bloklarımızı tanımlamak isteyeceğiz. Örneğin, Python'un kontrol akışını ileri yayma işlevi ile yürütmek isteyebiliriz. Dahası, önceden tanımlanmış sinir ağı katmanlarına güvenmek yerine, keyfi matematiksel işlemler gerçekleştirmek isteyebiliriz.

Şimdiye kadar ağlarımızdaki tüm işlemlerin ağımızın etkinleştirmelerine ve parametrelerine göre hareket ettiğini fark etmiş olabilirsiniz. Ancak bazen, ne önceki katmanların sonucu ne de güncellenebilir parametrelerin sonucu olmayan terimleri dahil etmek isteyebiliriz. Bunlara *sabit parametreler* diyoruz. Örneğin, $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ işlevini hesaplayan bir katman istediğimizi varsayalım, burada $\mathbf{x}$ girdi, $\mathbf{w}$ bizim parametremiz ve $c$ optimizasyon sırasında güncellenmeyen belirli bir sabittir. Bu yüzden `FixedHiddenMLP` sınıfını aşağıdaki gibi uyguluyoruz.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # `get_constant` işleviyle oluşturulan rastgele ağırlık parametreleri 
        # eğitim sırasında güncellenmez (yani sabit parametreler)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Oluşturulan sabit parametrelerin yanı sıra `relu` ve `dot` 
        # işlevlerini kullanın
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Tam bağlı katmanı yeniden kullanın. Bu, parametreleri tamamen bağlı iki katmanla paylaşmaya eşdeğerdir.
        X = self.dense(X)
        # Kontrol akışı
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Gradyanları hesaplamayan ve bu nedenle eğitim sırasında sabit kalan 
        # rastgele ağırlık parametreleri
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Oluşturulan sabit parametrelerin yanı sıra `relu` ve `mm`işlevlerini kullanın
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Tam bağlı katmanı yeniden kullanın. Bu, parametreleri tamamen bağlı iki 
        # katmanla paylaşmaya eşdeğerdir.
        X = self.linear(X)
        # Kontrol akışı
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # `tf.constant` ile oluşturulan rastgele ağırlık parametreleri eğitim sırasında güncellenmez (yani sabit parametreler)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Oluşturulan sabit parametrelerin yanı sıra `relu` ve `matmul` 
        # işlevlerini kullanın
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Tam bağlı katmanı yeniden kullanın. Bu, parametreleri tamamen bağlı iki 
        # katmanla paylaşmaya eşdeğerdir.
        X = self.dense(X)
        # Kontrol akışı
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

Bu `FixedHiddenMLP` modelinde, ağırlıkları (`self.rand_weight`) örneklemede rastgele ilkletilen ve daha sonra sabit olan gizli bir katman uygularız. Bu ağırlık bir model parametresi değildir ve bu nedenle asla geri yayma ile güncellenmez. Ağ daha sonra bu "sabit" katmanın çıktısını tam bağlı bir katmandan geçirir.

Çıktıyı döndürmeden önce, modelimizin olağandışı bir şey yaptığını unutmayın. Bir while-döngüsü çalıştırdık, $L_1$ normunun $1$'den büyük olması koşulunu test ettik ve çıktı vektörümüzü bu koşulu karşılayana kadar $2$'ye böldük. Son olarak, `X`'deki girdilerin toplamını döndürdük. Bildiğimiz kadarıyla hiçbir standart sinir ağı bu işlemi gerçekleştirmez. Bu özel işlemin herhangi bir gerçek dünya sorununda yararlı olmayabileceğini unutmayın. Amacımız, yalnızca rastgele kodu sinir ağı hesaplamalarınızın akışına nasıl tümleştirebileceğinizi göstermektir.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

[**Blokları bir araya getirmenin çeşitli yollarını karıştırıp eşleştirebiliriz.**] Aşağıdaki örnekte, blokları birtakım yaratıcı yollarla iç içe yerleştiriyoruz.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## Verimlilik

:begin_tab:`mxnet` 
Hevesli okuyucu, bu işlemlerden bazılarının verimliliği konusunda endişelenmeye başlayabilir. Sonuçta, yüksek performanslı bir derin öğrenme kütüphanesinde yer alan çok sayıda sözlük arama, kod yürütme ve daha birçok Pythonik (Python dilince) şeyimiz var. Python'un [global yorumlayıcı kilidi](https://wiki.python.org/moin/GlobalInterpreterLock) sorunları iyi bilinmektedir. Derin öğrenme bağlamında, son derece hızlı GPU'larımızın, cılız bir CPU'nun Python kodunu çalıştırması için başka bir işe girmeden önce onu beklemeleri gerekebileceğinden endişeleniyoruz. Python'u hızlandırmanın en iyi yolu, ondan tamamen kaçınmaktır.

Gluon'un daha sonra tanımlayacağımız *melezleştirmeye* izin vermesi bunu yapmanın bir yoludur. Burada, Python yorumlayıcısı, ilk çalıştırıldığında bir blok yürütür. Gluon çalışma zamanında neler olduğunu kaydeder ve bir dahaki sefere kısa devre yaparak Python'a çağrı yapar. Bu, bazı durumlarda işleri önemli ölçüde hızlandırabilir, ancak kontrol akışı (yukarıdaki gibi) ağdan farklı geçişlerde farklı dallara yol açtığı zaman dikkatli olunması gerekir. İlgili okuyucunun, mevcut bölümü bitirdikten sonra derleme hakkında bilgi edinmek için melezleştirme bölümüne (:numref:`sec_hybridize`) bakmasını öneririz.
:end_tab:

:begin_tab:`pytorch`
Hevesli okuyucular, bu işlemlerin bazılarının verimliliği konusunda endişe duymaya başlayabilir. Ne de olsa, yüksek performanslı bir derin öğrenme kitaplığı olması gereken yerde çok sayıda sözlük araması, kod koşturma ve birçok başka Pythonik şey var.
Python'un [genel yorumlayıcı kilidi](https://wiki.python.org/moin/GlobalInterpreterLock) sorunları iyi bilinmektedir. Derin öğrenme bağlamında, son derece hızlı GPU'larımızın, başka bir işi çalıştırmadan önce cılız bir CPU'nun Python kodunu çalıştırmasını beklemesi gerekebileceğinden endişe duyabiliriz.
:end_tab:

:begin_tab:`tensorflow`
Hevesli okuyucular, bu işlemlerin bazılarının verimliliği konusunda endişe duymaya başlayabilir. Ne de olsa, yüksek performanslı bir derin öğrenme kitaplığı olması gereken yerde çok sayıda sözlük araması, kod koşturma ve birçok başka Pythonik şey var. 
Python'un [genel yorumlayıcı kilidi](https://wiki.python.org/moin/GlobalInterpreterLock) sorunları iyi bilinmektedir. Derin öğrenme bağlamında, son derece hızlı GPU'larımızın, başka bir işi çalıştırmadan önce cılız bir CPU'nun Python kodunu çalıştırmasını beklemesi gerekebileceğinden endişe duyabiliriz.
Python'u hızlandırmanın en iyi yolu, ondan tamamen kaçınmaktır.
:end_tab:

## Özet

* Katmanlar bloklardır.
* Birçok katman bir blok içerebilir.
* Bir blok birçok blok içerebilir.
* Bir blok kod içerebilir.
* Bloklar, parametre ilkleme ve geri yayma dahil olmak üzere birçok temel yürütme işlerini halleder.
* Katmanların ve blokların dizili birleştirmeleri `Sequential` blok tarafından gerçekleştirilir.


## Alıştırmalar

1. Blokları bir Python listesinde saklamak için `MySequential`'ı değiştirirseniz ne tür sorunlar ortaya çıkacaktır?
1. Bağımsız değişken olarak iki blok alan bir blok uygulayın, örneğin `net1` ve `net2` ve ileri yaymada her iki ağın birleştirilmiş çıktısını döndürsün. Buna paralel blok da denir.
1. Aynı ağın birden çok örneğini birleştirmek istediğinizi varsayın. Aynı bloğun birden çok örneğini oluşturan ve ondan daha büyük bir ağ oluşturan bir fabrika işlevi uygulayın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/264)
:end_tab:
