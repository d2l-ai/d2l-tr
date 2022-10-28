# Otomatik Paralellik
:label:`sec_auto_para`

Derin öğrenme çerçeveleri (örneğin, MXNet ve PyTorch) arka işlemcide otomatik olarak hesaplama çizgeleri oluşturur. Bir hesaplama çizgesi kullandığından, sistem tüm bağımlılıkların farkındadır ve hızı artırmak için paralel olarak birden çok birbirine bağımlı olmayan görevi seçici olarak yürütebilir. Örneğin, :numref:`sec_async` içinde :numref:`fig_asyncgraph` bağımsız olarak iki değişkeni ilkler. Sonuç olarak sistem bunları paralel olarak yürütmeyi seçebilir. 

Genellikle, tek bir operatör tüm CPU veya tek bir GPU'da tüm hesaplama kaynaklarını kullanır. Örneğin, `dot` operatörü, tek bir makinede birden fazla CPU işlemcisi olsa bile, tüm CPU'lardaki tüm çekirdekleri (ve iş parçacıklarını) kullanır. Aynısı tek bir GPU için de geçerlidir. Bu nedenle paralelleştirme, tek aygıtlı bilgisayarlar için o kadar yararlı değildir. Birden fazla cihazla işler daha önemli hale gelir. Paralelleştirme genellikle birden fazla GPU arasında en alakadar olmakla birlikte, yerel CPU'nun eklenmesi performansı biraz artıracaktır. Örneğin, bkz. :cite:`Hadjis.Zhang.Mitliagkas.ea.2016`, bir GPU ve CPU'yu birleştiren bilgisayarla görme modellerini eğitmeye odaklanır. Otomatik olarak paralelleştirilen bir çerçevenin kolaylığı sayesinde aynı hedefi birkaç satır Python kodunda gerçekleştirebiliriz. Daha geniş bir şekilde, otomatik paralel hesaplama konusundaki tartışmamız, hem CPU'ları hem de GPU'ları kullanarak paralel hesaplamaya ve aynı zamanda hesaplama ve iletişimin paralelleşmesine odaklanmaktadır. 

Bu bölümdeki deneyleri çalıştırmak için en az iki GPU'ya ihtiyacımız olduğunu unutmayın.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU'larda Paralel Hesaplama

Test etmek için bir referans iş yükü tanımlayarak başlayalım: Aşağıdaki `run` işlevi iki değişkene, `x_gpu1` ve `x_gpu2`, ayrılan verileri kullanarak seçeceğimiz cihazda 10 matris-matris çarpımı gerçekleştirir.

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
Şimdi işlevi verilere uyguluyoruz. Önbelleğe almanın sonuçlarda bir rol oynamadığından emin olmak için, ölçümden önce bunlardan herhangi birine tek bir geçiş yaparak cihazları ısıtırız.
:end_tab:

:begin_tab:`pytorch`
Şimdi işlevi verilere uyguluyoruz. Önbelleğe almanın sonuçlarda bir rol oynamadığından emin olmak için, ölçmeden önce her ikisine de tek bir geçiş yaparak cihazları ısıtırız. `torch.cuda.synchronize()`, CUDA cihazındaki tüm akışlardaki tüm çekirdeklerin tamamlaması için bekler. Senkronize etmemiz gereken bir `device` argümanı alır. Aygıt bağımsız değişkeni `None` (varsayılan) ise, `current_device()` tarafından verilen geçerli aygıtı kullanır.
:end_tab:

```{.python .input}
run(x_gpu1)  # iki cihazi da isindir
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # bütün cihazlari isindir
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Her iki görev arasındaki `waitall` ifadesini kaldırırsak, sistem her iki cihazda da otomatik olarak hesaplamayı paralel hale getirmekte serbesttir.
:end_tab:

:begin_tab:`pytorch`
Her iki görev arasındaki `synchronize` ifadesini kaldırırsak, sistem her iki cihazda da otomatik olarak hesaplamayı paralel hale getirmekte serbesttir.
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

Yukarıdaki durumda, toplam yürütme süresi, parçalarının toplamından daha azdır, çünkü derin öğrenme çerçevesi, kullanıcı adına gelişmiş kod gerektirmeden her iki GPU cihazında hesaplamayı otomatik olarak planlar. 

## Paralel Hesaplama ve İletişim

Çoğu durumda, CPU ve GPU arasında veya farklı GPU'lar arasında, yani farklı cihazlar arasında veri taşımamız gerekir. Örneğin, bu durum, gradyanların birden fazla hızlandırıcı kart üzerinden toplamamız gereken dağıtılmış optimizasyonu gerçekleştirmek istediğimizde ortaya çıkar. Bunu GPU'da hesaplayarak benzetim yapalım ve sonuçları CPU'ya geri kopyalayalım.

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Bu biraz verimsiz. Listenin geri kalanı hala hesaplanırken `y`'nin parçalarını CPU'ya kopyalamaya başlayabileceğimizi unutmayın. Bu durum, örneğin, bir minigrup işlemindeki gradyanı hesapladığımızda ortaya çıkar. Bazı parametrelerin gradyanları diğerlerinden daha erken kullanılabilir olacaktır. Bu nedenle, GPU hala çalışırken PCI-Express veri yolu bant genişliğini kullanmaya başlamak bize avantaj sağlar. Her iki parça arasında `waitall`'i kaldırmak, bu senaryoyu benzetmemize olanak tanır.
:end_tab:

:begin_tab:`pytorch`
Bu biraz verimsiz. Listenin geri kalanı hala hesaplanırken `y`'nin parçalarını CPU'ya kopyalamaya başlayabileceğimizi unutmayın. Bu durum, örneğin, bir minigrup işlemindeki (backprop) gradyanı hesapladığımızda ortaya çıkar. Bazı parametrelerin gradyanları diğerlerinden daha erken kullanılabilir olacaktır. Bu nedenle, GPU hala çalışırken PCI-Express veri yolu bant genişliğini kullanmaya başlamak ize avantaj sağlar. PyTorch'ta,  `to()` ve `copy_()` gibi çeşitli işlevler, gereksiz olduğunda çağrı yapanın senkronizasyonu atlamasını sağlayan açık bir "non_blocking" argümanını kabul eder. `non_blocking=True` ayarı bu senaryoyu benzetmemize izin verir.
:end_tab:

```{.python .input}
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Her iki işlem için gereken toplam süre (beklendiği gibi) parçalarının toplamından daha azdır. Farklı bir kaynak kullandığı için bu görevin paralel hesaplamadan farklı olduğunu unutmayın: CPU ve GPU'lar arasındaki veri yolu. Aslında, her iki cihazda da işlem yapabilir ve iletişim kurabiliriz, hepsini aynı anda. Yukarıda belirtildiği gibi, hesaplama ve iletişim arasında bir bağımlılık vardır: `y[i]` CPU'ya kopyalanmadan önce hesaplanmalıdır. Neyse ki, sistem toplam çalışma süresini azaltmak için `y[i]`'i hesaplarken `y[i-1]`'i kopyalayabilir. 

:numref:`fig_twogpu` içinde gösterildiği gibi, bir CPU ve iki GPU üzerinde eğitim yaparken basit bir iki katmanlı MLP için hesaplama çizgesinin ve bağımlılıklarının bir gösterimi ile sonlandırıyoruz. Bundan kaynaklanan paralel programı manuel olarak planlamak oldukça acı verici olurdu. Eniyileme için çizge tabanlı bir hesaplama arka işlemcisine sahip olmanın avantajlı olduğu yer burasıdır.

![Bir CPU ve iki GPU üzerindeki iki katmanlı MLP'nin hesaplama çizgesi ve bağımlılıkları.](../img/twogpu.svg)
:label:`fig_twogpu`

## Özet

* Modern sistemler, birden fazla GPU ve CPU gibi çeşitli cihazlara sahiptir. Paralel, eşzamansız olarak kullanılabilirler. 
* Modern sistemler ayrıca PCI Express, depolama (genellikle katı hal sürücüleri veya ağlar aracılığıyla) ve ağ bant genişliği gibi iletişim için çeşitli kaynaklara sahiptir. En yüksek verimlilik için paralel olarak kullanılabilirler. 
* Arka işlemci, otomatik paralel hesaplama ve iletişim yoluyla performansı artırabilir. 

## Alıştırmalar

1. Bu bölümde tanımlanan `run` işlevinde sekiz işlem gerçekleştirildi. Aralarında bağımlılık yoktur. Derin öğrenme çerçevesinin bunları otomatik olarak paralel yürüteceğini görmek için bir deney tasarlayın.
1. Tek bir operatörün iş yükü yeterince küçük olduğunda, paralelleştirmede tek bir CPU veya GPU'da bile yardımcı olabilir. Bunu doğrulamak için bir deney tasarlayın. 
1. CPU'ları, GPU'ları ve her iki aygıt arasındaki iletişimde paralel hesaplamayı kullanan bir deney tasarlayın.
1. Kodunuzun verimli olduğunu doğrulamak için NVIDIA [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) gibi bir hata ayıklayıcısı kullanın. 
1. Daha karmaşık veri bağımlılıkları içeren hesaplama görevleri tasarlayın ve performansı artırırken doğru sonuçları elde edip edemeyeceğinizi görmek için deneyler çalıştırın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/1681)
:end_tab:
