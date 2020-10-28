# Katmanlar ve Bloklar
:label:`sec_model_construction`

Sinir ağlarını ilk tanıttığımızda, tek çıktılı doğrusal modellere odaklandık. Burada tüm model sadece tek bir nörondan oluşuyor. Tek bir nöronun (i) bazı girdiler aldığını; (ii) karşılık gelen (*skaler*) bir çıktı ürettiğini; ve (iii) ilgili bazı amaç işlevlerini optimize etmek için güncellenebilen bir dizi ilişkili parametreye sahip olduğunu unutmayın. Sonra, birden çok çıktıya sahip ağları düşünmeye başladığımızda, tüm bir nöron *katmanını* karakterize etmek için vektörleştirilmiş aritmetikten yararlandık. Tıpkı bireysel nöronlar gibi, katmanlar (i) bir dizi girdi alır, (ii) karşılık gelen çıktıları üretir ve (iii) bir dizi ayarlanabilir parametre ile açıklanır. Softmaks bağlanımı üzerinde çalıştığımızda, tek bir *katmanın* kendisi *modeldi*. Bununla birlikte, daha sonra çok katmanlı algılayıcıları devreye soktuğumuzda bile, modelin bu aynı temel yapıyı koruduğunu düşünebiliriz.

İlginç bir şekilde, çok katmanlı algılayıcılar için hem *tüm model* hem de *kurucu katmanları* bu yapıyı paylaşır. (Tam) Model ham girdileri (öznitelikleri) alır, çıktılar (tahminler) üretir ve parametrelere (tüm kurucu katmanlardan birleşik parametreler) sahiptir. Benzer şekilde, her bir katman girdileri alır (önceki katman tarafından sağlanır) çıktılar (sonraki katmana girdiler) üretir ve sonraki katmandan geriye doğru akan sinyale göre güncellenen bir dizi ayarlanabilir parametreye sahiptir.

Nöronların, katmanların ve modellerin bize işimiz için yeterince soyutlama sağladığını düşünseniz de, genellikle tek bir katmandan daha büyük ancak tüm modelden daha küçük olan bileşenler hakkında konuşmayı uygun bulduğumuz ortaya çıkıyor. Örneğin, bilgisayarla görmede aşırı popüler olan ResNet-152 mimarisi, yüzlerce katmana sahiptir. Bu katmanlar, *katman gruplarının* tekrar eden desenlerinden oluşur. Böyle bir ağın her seferinde bir katman olarak uygulanması sıkıcı bir hal alabilir. Bu endişe sadece varsayımsal değildir---bu tür tasarım modelleri pratikte yaygındır. Yukarıda bahsedilen ResNet mimarisi, hem tanıma hem de tespit için 2015 ImageNet ve COCO bilgisayarlı görme yarışmalarını kazandı :cite:`He.Zhang.Ren.ea.2016` ve birçok görme görevi için bir ilk tatbik edilen mimari olmaya devam ediyor. Katmanların çeşitli yinelenen desenlerde düzenlendiği benzer mimariler artık doğal dil işleme ve konuşma dahil olmak üzere diğer alanlarda da her yerde mevcuttur.

Bu karmaşık ağları uygulamak için, bir sinir ağı *bloğu* kavramını sunuyoruz. Bir blok, tek bir katmanı, birden çok katmandan oluşan bir bileşeni veya tüm modelin kendisini tanımlayabilir! Blok soyutlamayla çalışmanın bir yararı, bunların genellikle yinelemeli olarak daha büyük yapay nesnelerle birleştirilebilmeleridir (şekile bakınız :numref:`fig_blocks`).

![Çoklu katmanlar bloklara birleştiriliyor](../img/blocks.svg)
:label:`fig_blocks`

İsteğe bağlı keyfi karmaşıklıkta bloklar oluşturmak için kod tanımlayarak şaşırtıcı derecede öz kod yazabilir ve yine de karmaşık sinir ağları uygulayabiliriz.

Yazılım açısından, bir blok *sınıf* ile temsil edilir. Onun herhangi bir alt sınıfı, girdisini çıktıya dönüştüren ve gerekli parametreleri depolayan bir ileriye doğru yöntem tanımlamalıdır. Bazı blokların herhangi bir parametre gerektirmediğini unutmayın! Son olarak, gradyanları hesaplamak için bir blok geriye doğru bir yönteme sahip olmalıdır. Neyse ki, kendi bloğumuzu tanımlarken otomatik türev almanın sağladığı bazı perde arkası sihir sayesinde (:numref:`sec_autograd`da tanıtıldı), sadece parametreler ve ileriye doğru işlev hakkında endişelenmemiz gerekir.

Başlamak için, çok katmanlı algılayıcıları uygulamak için kullandığımız kodları yeniden gözden geçiriyoruz (:numref:`sec_mlp_concise`). Aşağıdaki kod, 256 birim ve ReLU aktivasyonuna sahip tam bağlı bir gizli katmana sahip bir ağ oluşturur ve ardından 10 birimle (etkinleştirme işlevi yok) tam bağlı *çıktı katmanı* gelir.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.random.uniform(size=(2, 20))

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F


x = torch.randn(2,20)
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(x)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

x = tf.random.uniform((2, 20))
net(x)
```

:begin_tab:`mxnet`
This is actually just shorthand for `net.forward(X)`, a slick Python trick achieved via the Block class's `__call__` function.

Bu örnekte, modelimizi bir `nn.Sequential` örneğini oluşturarak, döndürülen nesneyi `net` değişkenine atayarak oluşturduk. Daha sonra, katmanları çalıştırılmaları gereken sırayla eklemek için tekrar tekrar `add` yöntemini çağırıyoruz. Kısaca, `nn.Sequential`, Gluon'da bir blok sunan sınıf olan özel bir `Block` türünü tanımlar. Oluşturucu `Block`ların sıralı bir listesini tutar. `add` yöntemi, basitçe her ardışık `Block`'un listeye eklenmesini kolaylaştırır. Her katmanın, kendisinin `Block`'un alt sınıfı olan `Dense` sınıfının bir örneği olduğuna dikkat edin. `Forward` işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıktısını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktılarını elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında Block sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.forward(X)` için kısaltmadır.
:end_tab:

:begin_tab:`pytorch`
Bu örnekte, modelimizi bir `nn.Sequential` örneğini oluşturarak, katmanları bağımsız değişken olarak iletilmeleri gereken sırayla oluşturduk. Kısaca, `nn.Sequential`, PyTorch'ta bir blok sunan özel bir sınıf `Module` türünü tanımlar, ki kurucu `Module`lerin sıralı bir listesini tutar. Tam bağlı iki katmanın her birinin, kendisi de `Module`ün bir alt sınıfı olan `Linear` sınıfının bir örneği olduğuna dikkat edin. `forward` işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıktısını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktıları elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında Block sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.forward(X)` için kısaltmadır.
:end_tab:

:begin_tab:`tensorflow`
Bu örnekte, modelimizi bir `keras.models.Sequential` örneğini oluşturarak, katmanları bağımsız değişken olarak iletilmeleri gereken sırayla oluşturduk. Kısaca `Sequential`, Keras'ta bir blok sunan özel bir sınıf `keras.Model`'i tanımlar, ki kurucu `Model`lerin sıralı bir listesini tutar. İki tam bağlı katmanın her birinin, kendisi `Model`'in alt sınıfı olan `Dense` sınıfının bir örneği olduğuna dikkat edin. `forward` işlevi de oldukça basittir: Listedeki her bloğu birbirine zincirleyerek her birinin çıkışını bir sonrakine girdi olarak iletir. Şimdiye kadar, çıktılarını elde etmek için modellerimizi `net(X)` yapısı aracılığıyla çağırdığımızı unutmayın. Bu aslında Block sınıfının `__call__` işlevi aracılığıyla elde edilen ustaca bir Python marifeti olan `net.call(X)` için kısaltmadır.
:end_tab:

## Özel Kesim Blok 

Bir bloğun nasıl çalıştığına dair sezgiyi geliştirmenin belki de en kolay yolu, bir bloğu kendimiz uygulamaktır. Kendi özel kesim bloğumuzu uygulamadan önce, her bloğun sağlaması gereken temel işlevleri kısaca özetliyoruz:

1. Girdi verilerini ileriye doğru yöntemine bağımsız değişkenler olarak alın.
1. İleriye doğru bir değer döndürerek bir çıktı oluşturun. Çıktının girdiden farklı bir şekle sahip olabileceğini unutmayın. Örneğin, yukarıdaki modelimizdeki ilk tam bağlı katman, rastgele bir boyut girdisi alır, ancak 256 boyutunda bir çıktı verir.
1. Girdiye göre çıktısının gradyanını hesaplayın,ki bu da geriye doğru yöntemiyle erişilebilir. Genellikle bu otomatik olarak gerçekleşir.
1. İleriye doğru hesaplamayı yürütmek için gerekli olan bu parametreleri saklayın ve bunlara erişim sağlayın.
1. Bu parametreleri gerektiği gibi ilkletin.

Aşağıdaki kod parçacığında, 256 gizli düğüme sahip bir gizli katman ve 10 boyutlu bir çıktı katmanı ile çok katmanlı bir algılayıcıya karşılık gelen bir bloğu sıfırdan kodladık. Aşağıdaki `MLP` sınıfının kalıtsal çoğaltıldığını ve bir bloğu temsil ettiğini unutmayın. Yalnızca kendi `__init__` ve ileriye doğru yöntemlerimizi sağlayarak, büyük ölçüde ana sınıfın yöntemlerine güveneceğiz.

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input `x`
    def forward(self, x):
        return self.out(self.hidden(x))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__()
        self.hidden = nn.Linear(20,256)  # Hidden layer
        self.out = nn.Linear(256,10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input `x`
    def forward(self, x):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(x)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input `x`
    def call(self, x):
        return self.out(self.hidden((x)))
```

Başlamak için ileri yönteme odaklanalım. Girdi olarak `x`i aldığını, gizli gösterimi (`self.hidden(x)`) etkinleştirme işlevi uygulanmış olarak hesapladığını ve logitlerini (`self.out(...)`) çıktı verdiğini unutmayın. Bu MLP uygulamasında, her iki katman da örnek değişkenlerdir. Bunun neden makul olduğunu anlamak için, iki MLP'yi (`net1` ve `net2`) somutlaştırdığınızı ve bunları farklı veriler üzerinde eğittiğinizi hayal edin. Doğal olarak, iki farklı öğrenilmiş modeli temsil etmelerini bekleriz.

MLP'nin katmanlarını `__init__` yönteminde (kurucu) başlatıyoruz ve daha sonra bu katmanları her bir ileri doğru yöntemi çağrısında çağırıyoruz. Birkaç önemli ayrıntıya dikkat edin. İlk olarak, özelleştirilmiş `__init__` yöntemimiz, `super() .__init__()` aracılığıyla üst sınıfın `__init__` yöntemini çağırır ve bizi çoğu bloğa uygulanabilen standart şablon kodunu yeniden biçimlendirme zahmetinden kurtarır. Daha sonra, tam bağlı iki katmanımızı `self.hidden` ve `self.out`'a atayarak somutlaştırıyoruz. Yeni bir operatör uygulamadığımız sürece, geriye yayma (geriye doğru yöntemi) veya parametre ilkleme konusunda endişelenmemize gerek olmadığını unutmayın. Sistem bu yöntemleri otomatik olarak üretecektir. Bunu deneyelim:

```{.python .input}
net = MLP()
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(x)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(x)
```

Blok soyutlamanın önemli bir özelliği çok yönlülüğüdür. Katmanlar (tam bağlı katman sınıfı gibi), tüm modeller (yukarıdaki `MLP` gibi) veya ara karmaşıklığın çeşitli bileşenlerini oluşturmak için blok sınıfını alt sınıflara ayırabiliriz. Bu çok yönlülüğü sonraki bölümlerde, özellikle evrişimli sinir ağlarını ele alırken kullanıyoruz.


## Dizili Blok

Artık `Sequential` (dizili) sınıfının nasıl çalıştığına daha yakından bakabiliriz. `Sequential`nın diğer blokları birbirine zincirleme bağlamak için tasarlandığını hatırlayın. Kendi basitleştirilmiş MySequential'ımızı oluşturmak için, sadece iki anahtar yöntem tanımlamamız gerekir:
1. Blokları birer birer listeye eklemek için bir yöntem.
2. Blok zincirinden bir girdiyi geçirmek için ileriye doğru bir yöntem (eklendikleri sırayla).

Aşağıdaki `MySequential` sınıfı aynı işlevselliği varsayılan `Sequential` sınıfıyla sunar:

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has
        # a unique name. We save it in the member variable _children of the
        # Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize function, the system automatically
        # initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            x = block(x)
        return x
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # Here, block is an instance of a Module subclass. We save it in
            # the member variable _modules of the Module class, and its type
            # is OrderedDict
            self._modules[block] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            x = block(x)
        return x
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, block is an instance of a tf.keras.layers.Layer subclass
            self.modules.append(block)

    def call(self, x):
        for module in self.modules:
            x = module(x)
        return x
```

:begin_tab:`mxnet`
`add` yöntemi, sıralı `_children` sözlüğüne tek bir blok ekler. Neden her Gluon `Block`'unun bir `_children` özelliğine sahip olduğunu ve sadece bir Python listesi tanımlamak yerine bunu neden kullandığımızı merak edebilirsiniz. Kısacası, `_children` in başlıca avantajı, bloğumuzun parametre ilklemesi sırasında Gluon'un, parametreleri de ilkelemesi gereken alt blokları bulmak için `_children` sözlüğüne bakmayı bilmesidir.
:end_tab:

:begin_tab:`pytorch`
`__init__` yönteminde, her bloğu sıralı `_modules` sözlüğüne tek tek ekliyoruz. Neden her `Module`ün bir `_modules` özelliğine sahip olduğunu ve sadece bir Python listesi tanımlamak yerine bunu neden kullandığımızı merak edebilirsiniz. Kısacası, `_modules` in başlıca avantajı, bloğumuzun parametre ilklemesi sırasında, sistemin, parametreleri de ilklemesi gereken alt blokları bulmak için `_modules` sözlüğüne bakmayı bilmesidir.
:end_tab:

:begin_tab:`tensorflow`
FIXME, `MySequential` seçeneğini uygulamak için `Sequential` kullanmayın.
:end_tab:

`MySequential`'ımızın ileriye doğru yöntemi çağrıldığında, eklenen her blok eklendikleri sırayla yürütülür. Artık `MySequential` sınıfımızı kullanarak bir MLP'yi yeniden uygulayabiliriz.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(x)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(x)
```

Bu `MySequential` kullanımının, daha önce `Sequential` sınıfı için yazdığımız kodla aynı olduğuna dikkat edin (burada açıklandığı gibi :numref:`sec_mlp_concise`).

## İleriye Doğru Yönteminde Kodu Yürütme

`Sequential` sınıf, model yapımını kolaylaştırarak, kendi sınıfımızı tanımlamak zorunda kalmadan yeni mimarileri bir araya getirmemize olanak tanır. Bununla birlikte, tüm mimariler basit papatya zinciri değildir. Daha fazla esneklik gerektiğinde, kendi bloklarımızı tanımlamak isteyeceğiz. Örneğin, Python'un kontrol akışını ileri doğru yöntemi ile yürütmek isteyebiliriz. Dahası, önceden tanımlanmış sinir ağı katmanlarına güvenmek yerine, keyfi matematiksel işlemler gerçekleştirmek isteyebiliriz.

Şimdiye kadar ağlarımızdaki tüm işlemlerin ağımızın etkinleştirmelerine ve parametrelerine göre hareket ettiğini fark etmiş olabilirsiniz. Ancak bazen, ne önceki katmanların sonucu ne de güncellenebilir parametrelerin sonucu olmayan terimleri dahil etmek isteyebiliriz. Bunlara *sabit* parametreler diyoruz. Örneğin, $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ işlevini hesaplayan bir katman istediğimizi varsayalım, burada $\mathbf{x}$ girdi, $\mathbf{w}$ bizim parametremiz ve $c$ optimizasyon sırasında güncellenmeyen belirli bir sabittir.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the get_constant are not
        # iterated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the relu and dot
        # functions
        x = npx.relu(np.dot(x, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while np.abs(x).sum() > 1:
            x /= 2
        return x.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # Use the constant parameters created, as well as the relu and dot
        # functions
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.linear(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while x.norm().item() > 1:
            x /= 2
        return x.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        x = self.flatten(inputs)
        # Use the constant parameters created, as well as the relu and dot
        # functions
        x = tf.nn.relu(tf.matmul(x, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while tf.norm(x) > 1:
            x /= 2
        return tf.reduce_sum(x)
```

Bu `FixedHiddenMLP` modelinde, ağırlıkları (`self.rand_weight`) örneklemede rastgele ilkletilen ve daha sonra sabit olan gizli bir katman uygularız. Bu ağırlık bir model parametresi değildir ve bu nedenle asla geri yayma ile güncellenmez. Ağ daha sonra bu *sabit* katmanın çıktısını tam bağlı bir katmandan geçirir.

Çıktıyı döndürmeden önce, modelimizin olağandışı bir şey yaptığını unutmayın. Bir `while` döngüsü çalıştırdık, normunun 1'den büyük olması koşulunu test ettik ve çıktı vektörümüzü bu koşulu karşılayana kadar $2$'ye böldük. Son olarak, `x`'deki girdilerin toplamını döndürdük. Bildiğimiz kadarıyla hiçbir standart sinir ağı bu işlemi gerçekleştirmez. Bu özel işlemin herhangi bir gerçek dünya sorununda yararlı olmayabileceğini unutmayın. Amacımız, yalnızca rastgele kodu sinir ağı hesaplamalarınızın akışına nasıl tümleştirebileceğinizi göstermektir.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(x)
```

Blokları bir araya getirmenin çeşitli yollarını karıştırıp eşleştirebiliriz. Aşağıdaki örnekte, blokları birtakım yaratıcı yollarla iç içe yerleştiriyoruz.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())

chimera.initialize()
chimera(x)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, x):
        return self.linear(self.net(x))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(x)
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
chimera(x)
```

## Derleme

:begin_tab:`mxnet, tensorflow` 
Hevesli okuyucu, bu işlemlerden bazılarının verimliliği konusunda endişelenmeye başlayabilir. Sonuçta, yüksek performanslı bir derin öğrenme kütüphanesinde yer alan çok sayıda sözlük arama, kod yürütme ve daha birçok Pythonic (Python dilince) şeyimiz var. Python'un [Global Yorumlayıcı Kilidi](https://wiki.python.org/moin/GlobalInterpreterLock) sorunları iyi bilinmektedir. Derin öğrenme bağlamında, son derece hızlı GPU'larımızın, cılız bir CPU'nun Python kodunu çalıştırması için başka bir işe girmeden önce onu beklemeleri gerekebileceğinden endişeleniyoruz. Python'u hızlandırmanın en iyi yolu, ondan tamamen kaçınmaktır.
:end_tab:

:begin_tab:`mxnet`
Gluon'un melezleştirmesine izin vermek bunu yapmanın bir yoludur (:numref:`sec_hybridize`). Burada, Python yorumlayıcısı, ilk çalıştırıldığında bir Blok yürütür. Gluon çalışma zamanında neler olduğunu kaydeder ve bir dahaki sefere kısa devre yaparak Python'a çağrı yapar. Bu, bazı durumlarda işleri önemli ölçüde hızlandırabilir, ancak kontrol akışı (yukarıdaki gibi) ağdan farklı geçişlerde farklı dallara yol açtığı zaman dikkatli olunması gerekir. İlgilenen okuyucunun, mevcut bölümü bitirdikten sonra derleme hakkında bilgi edinmek için melezleştirme bölümüne (:numref:`sec_hybridize`) bakmasını öneririz.
:end_tab:

## Özet

* Katmanlar bloklardır.
* Birçok katman bir blok içerebilir.
* Bir blok birçok blok içerebilir.
* Bir blok kod içerebilir.
* Bloklar, parametre ilkleme ve geri yayma dahil olmak üzere birçok idare işini halleder.
* Katmanların ve blokların dizili birleştirmeleri `Sequential` blok tarafından gerçekleştirilir.


## Alıştırmalar

1. Blokları bir Python listesinde saklamak için `MySequential`ı değiştirirseniz ne tür sorunlar ortaya çıkacaktır.
1. Bağımsız değişken olarak iki blok alan bir blok uygulayın, örneğin `net1` ve `net2` ve ileri doğru geçişte her iki ağın birleştirilmiş çıkışını döndürsün (buna paralel blok da denir).
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
