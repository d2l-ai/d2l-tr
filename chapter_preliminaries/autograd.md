# Otomatik Türev Alma
:label:`sec_autograd`

:numref:`sec_calculus` içinde açıkladığımız gibi, türev alma neredeyse tüm derin öğrenme optimizasyon algoritmalarında çok önemli bir adımdır.
Bu türevleri almak için gerekli hesaplamalar basittir ve sadece biraz temel kalkülüs gerektirirken, karmaşık modeller için güncellemeleri elle yapmak bir sancılı olabilir (ve genellikle hataya açık olabilir).

Derin öğrenme çerçeveleri, türevleri otomatik olarak hesaplayarak, yani *otomatik türev alma* yoluyla bu çalışmayı hızlandırır.
Uygulamada, tasarladığımız modele dayalı olarak sistem, çıktıyı üretmek için hangi verinin hangi işlemlerle birleştirildiğini izleyen bir *hesaplama grafiği (çizgesi)* oluşturur.
Otomatik türev alma, sistemin daha sonra gradyanları geri yaymasını (backpropagation) sağlar.
Burada *geri yayma*, her bir parametreye göre kısmi türevleri doldurarak, hesaplama grafiğini izlemek anlamına gelir.


## Basit Bir Örnek

Bir oyuncak örnek olarak, (**$y = 2\mathbf{x}^{\top}\mathbf{x}$ fonksiyonunun $\mathbf{x}$ sütun vektörüne göre türevini almakla**) ilgilendiğimizi söyleyelim.
Başlamak için, `x` değişkenini oluşturalım ve ona bir başlangıç değeri atayalım.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**$y$'nin $\mathbf{x}$'e göre gradyanını hesaplamadan önce, onu saklayabileceğimiz bir yere ihtiyacımız var.**]
Bir parametreye göre her türev aldığımızda yeni bir bellek ayırmamamız önemlidir çünkü aynı parametreleri binlerce kez veya milyonlarca kez güncelleyeceğiz ve belleğimiz hızla tükenebilir.
Skaler değerli bir fonksiyonun bir $\mathbf{x}$ vektörüne göre gradyanının kendisinin vektör değerli olduğuna ve $\mathbf{x}$ ile aynı şekle sahip olduğuna dikkat edin.

```{.python .input}
# `attach_grad` çağırarak tensörün gradyanı için bellek tahsis ederiz.
x.attach_grad()
# `x`'e göre alınan bir gradyanı hesapladıktan sonra, değerleri 0'a ilklenmiş 
# `grad` özelliği ile ona erişebileceğiz.
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # `x = torch.arange(4.0, requires_grad=True)` ile aynıdır
x.grad  # None varsayılan değerdir
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

(**Şimdi $y$'yi hesaplayalım.**)

```{.python .input}
# Hesaplama grafiğini oluşturmak için kodumuzu bir `autograd.record` kapsamına yerleştirin
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Bütün hesaplamaları teybe kaydet
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

`x`, 4 uzunluklu bir vektör olduğu için, `x` ve `x`'in iç çarpımı gerçekleştirilir ve `y`'ye atadığımız skaler çıktı elde edilir.
Daha sonra, [**geri yayma için işlevi çağırarak ve gradyanı yazdırarak `x`'in her bir bileşenine göre `y` gradyanını otomatik olarak hesaplayabiliriz.**]

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**$y = 2\mathbf{x}^{\top}\mathbf{x}$ fonksiyonunun $\mathbf{x}$'e göre gradyanı $4\mathbf{x}$ olmalıdır.**)
İstenilen gradyanın doğru hesaplandığını hızlıca doğrulayalım.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

[**Şimdi başka bir `x` fonksiyonunu hesaplayalım.**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Yeni hesaplanan gradyan tarafından üzerine yazılır
```

```{.python .input}
#@tab pytorch
# PyTorch, gradyanı varsayılan olarak biriktirir, önceki değerleri temizlememiz gerekir.
x.grad.zero_() 
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Yeni hesaplanan gradyan tarafından üzerine yazılır
```

## Skaler Olmayan Değişkenler için Geriye Dönüş

Teknik olarak, `y` skaler olmadığında, `y` vektörünün `x` vektörüne göre türevinin en doğal yorumu bir matristir.
Daha yüksek kademeli ve daha yüksek boyutlu `y` ve `x` için, türevin sonucu yüksek kademeli bir tensör olabilir.

Bununla birlikte, bu daha egzotik nesneler gelişmiş makine öğrenmesinde ([**derin öğrenmedekiler**] dahil) ortaya çıkarken, daha sıklıkla [**bir vektör üzerinde geriye doğru dönük çağırdığımızda**], bir *grup* eğitim örneğinde kayıp fonksiyonlarının her bir bileşeni için türevlerini hesaplamaya çalışıyoruz.
Burada (**amacımız**), türev matrisini hesaplamak değil, (**gruptaki her örnek için ayrı ayrı hesaplanan kısmi türevlerin toplamını**) hesaplamaktır.

```{.python .input}
# Vektör değerli bir değişken olan `y` (`x`'in işlevi) üzerinde `geridönüş`'ü çağırdığımızda, 
# `y`'deki öğeleri toplayarak yeni bir skaler değişken oluşturulur. 
# Daha sonra bu skaler değişkenin `x`'e göre gradyanı hesaplanır.
with autograd.record():
    y = x * x  # `y` bir vektördür
y.backward()
x.grad  # y = sum(x * x)'ye eşittir
```

```{.python .input}
#@tab pytorch
# Skaler olmayan üzerinde `geridönüş` çağırmak, farklılaştırılmış işlevin kendine göre 
# `self` gradyanını belirten bir `gradyan` argümanının iletilmesini gerektirir. Bizim durumumuzda, 
# sadece kısmi türevleri toplamak istiyoruz, bu nedenle birlerden oluşan gradyanı iletmek uygundur.
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) aşağısı ile aynıdır
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # `y = tf.reduce_sum(x * x)` ile aynı
```

## Hesaplamanın Ayrılması

Bazen, [**bazı hesaplamaları kaydedilen hesaplama grafiğinin dışına taşımak**] isteriz.
Örneğin, `y`'nin `x`'in bir fonksiyonu olarak hesaplandığını ve daha sonra `z`'nin hem `y`'nin hem de `x`'in bir fonksiyonu olarak hesaplandığını varsayalım.
Şimdi, `z`'nin gradyanını `x`'e göre hesaplamak istediğimizi, ancak nedense `y`'yi bir sabit olarak kabul etmek istediğimizi ve `x`'in `y` hesaplandıktan sonra oynadığı rolü hesaba kattığımızı hayal edin.

Burada, `y` ile aynı değere sahip yeni bir `u` değişkeni döndürmek için `y`'yi ayırabiliriz, ancak bu `y`'nin hesaplama grafiğinde nasıl hesaplandığına dair tüm bilgileri yok sayar.
Başka bir deyişle, gradyan `u`'dan `x`'e geriye doğru akmayacaktır.
Bu nedenle, aşağıdaki geri yayma işlevi, `z = x * x * x`'in `x`'e göre kısmi türevi hesaplamak yerine, `z = u * x`'in `x`'e göre kısmi türevini `u` sabitmiş gibi davranarak hesaplar. 

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# `t.gradient`i birden fazla çalıştırmak için `persistent=True` şekilde ayarlayın
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

`y` hesaplaması kaydedildiğinden, `y = x * x`'nin  `x`'e göre türevi olan `2 * x`'i elde etmek için `y` üzerinden geri yaymayı başlatabiliriz.

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Python Kontrol Akışının Gradyanını Hesaplama

Otomatik türev almayı kullanmanın bir yararı, bir Python kontrol akışı labirentinden (örneğin, koşullu ifadeler, döngüler ve rastgele fonksiyon çağrıları) geçen (**bir fonksiyondan hesaplama çizgesi oluşturup**), (**elde edilen değişkenin gradyanını yine de hesaplayabilmemizdir**).
Aşağıdaki kod parçacığında, `while` döngüsünün yineleme sayısının ve `if` ifadesinin değerlendirmesinin `a` girdisinin değerine bağlı olduğuna dikkat edin.

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Hadi gradyanı hesaplayalım.

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

Şimdi yukarıda tanımlanan `f` fonksiyonunu analiz edebiliriz.
Fonksiyonun `a` girdisinde parçalı doğrusal olduğuna dikkat edin.
Diğer bir deyişle, herhangi bir `a` için, `k` değerinin `a` girdisine bağlı olduğu `f(a) = k * a` olacak şekilde sabit bir `k` vardır.
Sonuç olarak `d / a`, gradyanın doğru olduğunu doğrulamamıza izin verir.

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## Özet

* Derin öğrenme çerçeveleri türevlerin hesaplanmasını otomatikleştirebilir. Kullanmak için, önce kısmi türevleri bulmak istediğimiz değişkenlere gradyanlar ekleriz. Daha sonra hedef değerimizin hesaplamasını kaydeder, geri yayma için işlevi uygularız ve elde edilen gradyanlara erişiriz.


## Alıştırmalar

1. İkinci türevi hesaplamak neden birinci türeve göre çok daha pahalıdır?
1. Geri yayma için işlevi çalıştırdıktan sonra, hemen tekrar çalıştırın ve ne olduğunu görün.
1. `d`'nin `a`'ya göre türevini hesapladığımız kontrol akışı örneğinde, `a` değişkenini rastgele bir vektör veya matris olarak değiştirirsek ne olur. Bu noktada, `f(a)` hesaplamasının sonucu artık skaler değildir. Sonuca ne olur? Bunu nasıl analiz ederiz?
1. Kontrol akışının gradyanını bulmak için yeniden bir örnek tasarlayın. Sonucu çalıştırın ve analiz edin.
1. $f(x) = \sin(x)$ olsun. $f(x)$ ve $\frac{df(x)}{dx}$ grafiklerini çizin, buradaki ikinci terim $f'(x) = \cos(x)$ kullanılmadan hesaplansın.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/200)
:end_tab:
