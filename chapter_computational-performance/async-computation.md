# Asenkron Hesaplama
:label:`sec_async`

Günümüzün bilgisayarları, birden fazla CPU çekirdeği (genellikle çekirdek başına birden fazla iş parçacığı), GPU başına birden çok işlem öğesi ve genellikle cihaz başına birden çok GPU'dan oluşan son derece paralel sistemlerdir. Kısacası, birçok farklı şeyi aynı anda, genellikle farklı cihazlarda işleyebiliriz. Ne yazık ki Python, en azından ekstra yardım almadan paralel ve asenkron kod yazmanın harika bir yolu değildir. Sonuçta, Python tek dişlidir ve gelecekte değişmesi pek olası değildir. MXNet ve TensorFlow gibi derin öğrenme çerçeveleri, performansı artırmak için*asenkron bir programlama* modeli benimser, PyTorch ise Python'un kendi zamanlayıcısını kullanarak farklı bir performans değişimine yol açar. PyTorch için varsayılan olarak GPU işlemleri eşzamansızdır. GPU kullanan bir işlev çağırdığınızda, işlemler belirli bir aygıta sıralanır, ancak daha sonra kadar zorunlu olarak yürütülmez. Bu, CPU veya diğer GPU'lardaki işlemler de dahil olmak üzere paralel olarak daha fazla hesaplama yürütmemize olanak tanır. 

Bu nedenle, zaman uyumsuz programlamanın nasıl çalıştığını anlamak, hesaplama gereksinimlerini ve karşılıklı bağımlılıkları proaktif olarak azaltarak daha verimli programlar geliştirmemize yardımcı olur. Bu, bellek yükünü azaltmamıza ve işlemci kullanımını artırmamıza olanak tanır.

```{.python .input}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## Arka Uç üzerinden eşzamanzamanlama

:begin_tab:`mxnet`
Bir ısınma için aşağıdaki oyuncak sorununu göz önünde bulundurun: rastgele bir matris oluşturmak ve çarpmak istiyoruz. Farkı görmek için NumPy ve `mxnet.np`'te bunu yapalım.
:end_tab:

:begin_tab:`pytorch`
Bir ısınma için aşağıdaki oyuncak sorununu göz önünde bulundurun: rastgele bir matris oluşturmak ve çarpmak istiyoruz. Farkı görmek için bunu hem NumPy hem de PyTorch tensorunda yapalım. PyTorch `tensor`'ün bir GPU'da tanımlandığını unutmayın.
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
MXNet üzerinden yapılan kıyaslama çıkışı, büyüklük siparişlerinden daha hızlıdır. Her ikisi de aynı işlemcide çalıştırıldığı için başka bir şey oluyor olmalı. MXNet'i geri dönmeden önce tüm arka uç hesaplamasını bitirmeye zorlamak daha önce ne olduğunu gösterir: hesaplama arka uç tarafından yürütülür ve ön yüz denetimi Python'a döndürür.
:end_tab:

:begin_tab:`pytorch`
PyTorch üzerinden yapılan kıyaslama çıkışı, büyüklük siparişlerinden daha hızlıdır. NumPy nokta ürünü CPU işlemcisinde yürütülür ve PyTorch matris çarpımı GPU'da yürütülür ve bu nedenle ikincisinin çok daha hızlı olması beklenir. Ama büyük zaman farkı, başka bir şeyin döndüğünü gösteriyor. Varsayılan olarak, PyTorch'ta GPU işlemleri eşzamansızdır. PyTorch'u geri dönmeden önce tüm hesaplamayı bitirmeye zorlamak daha önce neler olduğunu gösterir: hesaplama arka uç tarafından yürütülür ve ön yüz denetimi Python'a döndürür.
:end_tab:

```{.python .input}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
Genel olarak, MXNet, örneğin Python aracılığıyla kullanıcılarla doğrudan etkileşimler için bir ön yüze ve sistem tarafından hesaplamayı gerçekleştirmek için kullanılan bir arka ucuna sahiptir. :numref:`fig_frontends`'te gösterildiği gibi, kullanıcılar Python, R, Scala ve C++ gibi çeşitli ön yüz dillerinde MXNet programları yazabilir. Kullanılan ön yüz programlama dili ne olursa olsun, MXNet programlarının yürütülmesi öncelikle C++ uygulamalarının arka ucunda gerçekleşir. Ön yüz dili tarafından verilen işlemler yürütme için arka ucuna iletilir. Arka uç, sıraya alınmış görevleri sürekli olarak toplayan ve yürüten kendi iş parçacıklarını yönetir. Bunun çalışması için arka ucun hesaplama grafiğindeki çeşitli adımlar arasındaki bağımlılıkları takip edebilmesi gerektiğini unutmayın. Bu nedenle, birbirine bağlı işlemleri paralel hale getirmek mümkün değildir.
:end_tab:

:begin_tab:`pytorch`
Genel olarak, PyTorch kullanıcılarla doğrudan etkileşim için bir ön yüze sahiptir, örneğin Python üzerinden, hem de sistem tarafından hesaplama gerçekleştirmek için kullanılan bir arka uç. :numref:`fig_frontends`'te gösterildiği gibi, kullanıcılar Python ve C++ gibi çeşitli ön yüz dillerinde PyTorch programlarını yazabilirler. Kullanılan önyüz programlama dilinden bağımsız olarak, PyTorch programlarının yürütülmesi öncelikle C++ uygulamalarının arka ucunda gerçekleşir. Ön yüz dili tarafından verilen işlemler yürütme için arka ucuna iletilir. Arka uç, sıraya alınmış görevleri sürekli olarak toplayan ve yürüten kendi iş parçacıklarını yönetir. Bunun çalışması için arka ucun hesaplama grafiğindeki çeşitli adımlar arasındaki bağımlılıkları takip edebilmesi gerektiğini unutmayın. Bu nedenle, birbirine bağlı işlemleri paralel hale getirmek mümkün değildir.
:end_tab:

![Programming language frontends and deep learning framework backends.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

Bağımlılık grafiğini biraz daha iyi anlamak için başka bir oyuncak örneğine bakalım.

```{.python .input}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![The backend tracks dependencies between various steps in the computational graph.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

Yukarıdaki kod parçacığı da :numref:`fig_asyncgraph`'te gösterilmiştir. Python ön uç iş parçacığı ilk üç ifadeden birini çalıştırdığında, görevi arka uç kuyruğuna döndürür. Son deyimin sonuçları*yazdırılmış* olması gerektiğinde, Python ön uç iş parçacığı C++ arka uç iş parçacığının `z` değişkenin sonucunu hesaplamayı tamamlamasını bekler. Bu tasarımın bir avantajı, Python ön uç ipliğinin gerçek hesaplamaları gerçekleştirmesine gerek olmadığıdır. Böylece, Python'un performansından bağımsız olarak programın genel performansı üzerinde çok az etkisi yoktur. :numref:`fig_threading`, ön uç ve arka ucun nasıl etkileşime girdiğini gösterir. 

![Interactions of the frontend and backend.](../img/threading.svg)
:label:`fig_threading`

## Engeller ve Engelleyiciler

:begin_tab:`mxnet`
Python'u tamamlanmasını beklemeye zorlayacak bir dizi işlem vardır: 

* Açıkçası `npx.waitall()`, işlem yönergelerinin ne zaman verildiğine bakılmaksızın tüm hesaplama tamamlanana kadar bekler. Pratikte, kötü performansa yol açabileceğinden kesinlikle gerekli olmadıkça bu operatörü kullanmak kötü bir fikirdir.
* Belirli bir değişken kullanılabilir olana kadar beklemek istiyorsak `z.wait_to_read()`'i arayabiliriz. Bu durumda MXNet blokları, `z` değişkeni hesaplanıncaya kadar Python'a döner. Diğer hesaplamalar daha sonra devam edebilir.

Bunun pratikte nasıl çalıştığını görelim.
:end_tab:

```{.python .input}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
Her iki işlemin de tamamlanması yaklaşık aynı zaman alır. Bariz engelleme işlemlerinin yanı sıra, *örtülü engelleyicilerin farkında olmanızı öneririz. Bir değişkenin yazdırılması, değişkenin kullanılabilir olmasını gerektirir ve bu nedenle bir engelleyicidir. Son olarak, NumPy asenkron kavramı olmadığı için `z.item()` üzerinden `z.item()` üzerinden skalerlere dönüşümler ve `z.item()` üzerinden NumPy dönüşümleri engelliyor. `print` işlevi gibi değerlere erişmesi gerekir.  

MXNet'in kapsamından NumPy ve geri sık küçük miktarlarda verilerin kopyalanması, aksi takdirde verimli bir kodun performansını yok edebilir, çünkü bu tür her bir işlem, ilgili terimi elde etmek için gerekli tüm ara sonuçları değerlendirmek için hesaplama grafiği gerektirdiğinden, başka bir şey yapılabilir* önce*.
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## Hesaplama Geliştirme

:begin_tab:`mxnet`
Ağır çok iş parçacıklı bir sistemde (normal dizüstü bilgisayarlarda bile 4 iş parçacığı veya daha fazla ve çok soketli sunucularda bu sayı 256'yı aşabilir) zamanlama işlemlerinin yükü önemli hale gelebilir. Bu nedenle hesaplama ve zamanlamanın eşzamansız ve paralel olarak gerçekleşmesi son derece arzu edilir. Bunu yapmanın faydasını göstermek için, bir değişkeni hem sırayla hem de eşzamansız olarak birden fazla kez artırırsak ne olacağını görelim. Her ekleme arasına bir `wait_to_read` bariyer ekleyerek senkron yürütme simüle ediyoruz.
:end_tab:

```{.python .input}
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Python ön uç iş parçacığı ve C++ arka uç iş parçacığı arasındaki biraz basitleştirilmiş bir etkileşim aşağıdaki gibi özetlenebilir:
1. Ön uç, arka ucun hesaplama görevini `y = x + 1`'ü sıraya eklemesini emrediyor.
1. Arka uç daha sonra hesaplama görevlerini kuyruktan alır ve gerçek hesaplamaları gerçekleştirir.
1. Arka uç daha sonra hesaplama sonuçlarını ön uca döndürür.
Bu üç aşamanın sürelerinin sırasıyla $t_1, t_2$ ve $t_3$ olduğunu varsayalım. Eşzamansız programlama kullanmazsak, 10000 hesaplamaları gerçekleştirmek için alınan toplam süre yaklaşık $10000 (t_1+ t_2 + t_3)$'dir. Zaman uyumsuz programlama kullanılıyorsa, 10000 hesaplamaları gerçekleştirmek için alınan toplam süre $t_1 + 10000 t_2 + t_3$ ($10000 t_2 > 9999t_1$ varsayarak) azaltılabilir, çünkü ön uç her döngü için hesaplama sonuçlarını döndürmek için beklemek zorunda değildir.
:end_tab:

## Özet

* Derin öğrenme çerçeveleri, Python ön ucunu yürütme arka ucundan ayırabilir. Bu, komutların arka uca ve ilişkili paralellik içine hızlı zaman uyumsuz olarak eklenmesine olanak tanır.
* Eşzamanzamanlılık oldukça duyarlı bir ön yüze yol açar. Ancak, aşırı bellek tüketimine neden olabileceğinden görev sırasını aşırı doldurmamak için dikkatli olun. Ön ucu ve arka ucunu yaklaşık olarak senkronize etmek için her minibatch için senkronize edilmesi önerilir.
* Chip satıcıları, derin öğrenmenin verimliliği hakkında çok daha ince taneli bir içgörü elde etmek için sofistike performans analizi araçları sunar.

:begin_tab:`mxnet`
* MXNet'in bellek yönetiminden Python'a dönüşümlerin arka ucunu belirli değişken hazır olana kadar beklemeye zorlayacağını unutmayın. `print`, `asnumpy` ve `item` gibi işlevlerin hepsi bu etkiye sahiptir. Bu arzu edilebilir, ancak senkronizasyonun dikkatsiz kullanımı performansı mahvedebilir.
:end_tab:

## Egzersizler

:begin_tab:`mxnet`
1. Biz asenkron hesaplama kullanarak 10000 hesaplamaları $t_1 + 10000 t_2 + t_3$ için gerçekleştirmek için gereken toplam süreyi azaltabilir yukarıda belirtti. Neden burada $10000 t_2 > 9999 t_1$'ü varsaymak zorundayız?
1. `waitall` ve `wait_to_read` arasındaki farkı ölçün. İpucu: Bir dizi talimat uygulayın ve bir ara sonuç için senkronize edin.
:end_tab:

:begin_tab:`pytorch`
1. CPU'da, bu bölümdeki aynı matris çarpma işlemlerini karşılaştırın. Arka uç üzerinden hala asenkron gözlemleyebilir misiniz?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2564)
:end_tab:
