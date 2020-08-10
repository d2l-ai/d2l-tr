# Veri ile Oynama Yapmak
:label:`sec_ndarray`

Bir şeylerin yapılabilmesi için, verileri depolamak ve oynama yapmak (manipule etmek) için bazı yollar bulmamız gerekir.
Genellikle verilerle ilgili iki önemli şey vardır: (i) bunları elde etmek; ve (ii) bilgisayar içine girdikten sonra bunları işlemek. Veri depolamanın bir yolu olmadan onu elde etmenin bir anlamı yok, bu yüzden önce sentetik (yapay) verilerle oynayarak ellerimizi kirletelim. Başlamak için, *gerey (tensör)* olarak da adlandırılan $n$ boyutlu diziyi tanıtalım.

Python'da en çok kullanılan bilimsel hesaplama paketi olan NumPy ile çalıştıysanız, bu bölümü tanıdık bulacaksınız.
Hangi çerçeveyi kullanırsanız kullanın, *tensör sınıfı* (MXNet'teki `ndarray`, hem PyTorch hem de TensorFlow'daki `Tensor`), fazladan birkaç öldürücü özellik ile NumPy'nin ndarray'ına benzer.
İlk olarak, GPU hesaplamayı hızlandırmak için iyi desteklenirken, NumPy sadece CPU hesaplamasını destekler.
İkincisi, tensör sınıfı otomatik türev almayı destekler.
Bu özellikler, tensör sınıfını derin öğrenme için uygun hale getirir.
Kitap boyunca, tensörler dediğimizde, aksi belirtilmedikçe tensör sınıfının örneklerinden bahsediyoruz.

## Başlangıç

Bu bölümde, sizi ayaklandırıp koşturmayı, kitapta ilerledikçe üstüne koyarak geliştireceğiniz temel matematik ve sayısal hesaplama araçlarıyla donatılmayı amaçlıyoruz.
Bazı matematiksel kavramları veya kütüphane işlevlerini içselleştirme de zorlanıyorsanız, endişelenmeyin.
Aşağıdaki bölümlerde bu konular pratik örnekler bağlamında tekrar ele alınacak ve oturacaktır.
Öte yandan, zaten biraz bilgi birikiminiz varsa ve matematiksel içeriğin daha derinlerine inmek istiyorsanız, bu bölümü atlamanız yeterlidir.

:begin_tab:`mxnet`
Başlarken, MXNet'ten `np` (`numpy`) ve `npx` (`numpy_extension`) modüllerini içe aktarıyoruz (import).
Burada, np modülü, NumPy tarafından desteklenen işlevleri içerirken, npx modülü, NumPy benzeri bir ortamda derin öğrenmeyi güçlendirmek için geliştirilmiş bir dizi uzantı içerir.
Tensörleri kullanırken neredeyse her zaman `set_np` işlevini çağırırız: bu, tensör işlemenin MXNet'in diğer bileşenleri tarafından uyumluluğu içindir.
:end_tab:

:begin_tab:`pytorch`
Başlamak için `torch`u içe aktarıyoruz. PyTorch olarak adlandırılsa da, `pytorch` yerine `torch`u  içe aktarmamız gerektiğini unutmayın.
:end_tab:

:begin_tab:`tensorflow`
Başlamak için, `tensorflow`u içe aktarıyoruz. İsim biraz uzun olduğu için, genellikle kısa bir takma ad olan `tf` ile içe aktarırız.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

Bir tensör, (muhtemelen çok boyutlu) bir sayısal değerler dizisini temsil eder.
Bir eksende, bir tensör (matematikte) bir *vektöre* karşılık gelir.
İki eksende, bir tensör *matrise* karşılık gelir.
İkiden fazla ekseni olan tensörlerin özel matematik isimleri yoktur.

Başlamak için, varsayılan olarak yüzer sayı (float) olarak oluşturulmuş olmalarına rağmen, 0 ile başlayan ilk 12 tamsayıyı içeren bir satır vektörü, `x`, oluşturmak için `arange` i kullanabiliriz.
Bir tensördeki değerlerin her birine tensörün *eleman*ı denir.
Örneğin, tensör `x`'de 12 eleman vardır.
Aksi belirtilmedikçe, yeni bir tensör ana bellekte saklanacak ve CPU tabanlı hesaplama için kullanılacaktır.

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

Bir tensörün *şekli*ne (her eksen boyunca uzunluk) `shape` özelliğini inceleyerek erişebiliriz.

```{.python .input}
#@tab all
x.shape
```

Bir tensördeki toplam eleman sayısını, yani tüm şekil elemanlarının çarpımını bilmek istiyorsak, boyutunu inceleyebiliriz.
Burada bir vektörle uğraştığımız için, `shape` (şeklinin) tek elemanı, vektör boyutu ile aynıdır.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

Eleman sayısını veya değerlerini değiştirmeden bir tensörün şeklini değiştirmek için `reshape` işlevini çağırabiliriz.
Örneğin, `x` tensörümüzü, (12,) şekilli bir satır vektöründen (3, 4) şekilli bir matrise dönüştürebiliriz .
Bu yeni tensör tam olarak aynı değerleri içerir, ancak onları 3 satır ve 4 sütun olarak düzenlenmiş bir matris olarak görür.
Yinelemek gerekirse, şekil değişmiş olsa da, `x`'deki elemanlar değişmemiştir.
Boyutun yeniden şekillendirilerek değiştirilmediğine dikkat edin.

```{.python .input}
#@tab mxnet, pytorch
x = x.reshape(3, 4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(x, (3, 4))
x
```

Her boyutu manuel olarak belirterek yeniden şekillendirmeye gerek yoktur.
Hedef şeklimiz (yükseklik, genişlik) şekline sahip bir matrisse, o zaman genişliği öğrendiysek sonra, yükseklik üstü kapalı olarak verilmiştir.
Neden bölünmeyi kendimiz yapmak zorunda olalım ki?
Yukarıdaki örnekte, 3 satırlı bir matris elde etmek için, hem 3 satır hem de 4 sütun olması gerektiğini belirttik.
Neyse ki, tensörler, bir boyut eksik geri kalanlar verildiğinde, kalan bir boyutu otomatik olarak çıkarabilir.
Bu özelliği, tensörlerin otomatik olarak çıkarımını istediğimiz boyuta `-1` yerleştirerek çağırıyoruz.
Bizim durumumuzda, `x.reshape(3, 4)` olarak çağırmak yerine, eşit biçimde `x.reshape(-1, 4)` veya `x.reshape(3, -1)` olarak çağırabilirdik.

Tipik olarak, matrislerimizin sıfırlar, birler, diğer bazı sabitler veya belirli bir dağılımdan rastgele örneklenmiş sayılarla başlatılmasını isteriz.
Tüm elemanları 0 olarak ayarlanmış ve (2, 3, 4) şeklindeki bir tensörü temsil eden bir tensörü aşağıdaki şekilde oluşturabiliriz:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros(2, 3, 4)
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

Benzer şekilde, her bir eleman 1'e ayarlanmış şekilde tensörler oluşturabiliriz:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

Genellikle, bir tensördeki her eleman için değerleri bir olasılık dağılımından rastgele örneklemek isteriz.
Örneğin, bir sinir ağında parametre görevi görecek dizileri oluşturduğumuzda, değerlerini genellikle rastgele başlatırız.
Aşağıdaki kod parçası (3, 4) şekilli bir tensör oluşturur .
Elemanlarının her biri ortalaması 0 ve standart sapması 1 olan standart Gauss (normal) dağılımından rastgele örneklenir.

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

Sayısal değerleri içeren bir Python listesi (veya liste listesi) sağlayarak istenen tensördeki her eleman için kesin değerleri de belirleyebiliriz.
Burada, en dıştaki liste 0. eksene, içteki liste ise 1. eksene karşılık gelir.

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## İşlemler

Bu kitap yazılım mühendisliği ile ilgili değildir.
İlgi alanlarımız basitçe dizilerden/dizilere veri okumak ve yazmakla sınırlı değildir.
Bu diziler üzerinde matematiksel işlemler yapmak istiyoruz.
En basit ve en kullanışlı işlemlerden bazıları *eleman-yönlü (elementwise)* işlemlerdir.
Bunlar bir dizinin her elemanına standart bir sayıl (skaler) operasyon uygular.
İki diziyi girdi olarak alan işlevler için, eleman-yönlü işlemler iki diziden karşılık gelen her bir elaman çiftine standart bir ikili operatör uygular.
Sayıldan sayıla (skalerden skalere) eşleşen herhangi bir fonksiyondan eleman-yönlü bir fonksiyon oluşturabiliriz.

Matematiksel gösterimde, böyle bir *tekli* skaler işlemi (bir girdi alarak) $f: \mathbb{R} \rightarrow \mathbb{R}$ imzasıyla ifade ederiz.
Bu, işlevin herhangi bir gerçel sayıdan ($\mathbb{R}$) diğerine eşlendiği anlamına gelir.
Benzer şekilde, $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ imzası ile bir *ikili* skaler operatörü (iki gerçel girdi alarak ve bir çıktı verir) belirtiriz.
*Aynı şekilli* iki  $\mathbf{u}$ ve $\mathbf{v}$ vektörü ve $f$ ikili operatörü verildiğinde, tüm $i$ler için $c_i \gets f(u_i, v_i)$ ayarlayarak $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ vektörünü üretebiliriz; burada $c_i, u_i$ ve $v_i$, $\mathbf{c}, \mathbf{u} $ ve $\mathbf{v}$ vektörlerinin $i.$ elemanlarıdır.
Burada, skaler fonksiyonu eleman-yönlü bir vektör işlemini *yükselterek* vektör değerli $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ ürettik.

Ortak standart aritmetik operatörler (`+`, `-`,` * `,`/` ve `**`), rastgele şekile sahip herhangi bir benzer şekilli tansörler için eleman-yönlü işlemlere *yükseltilmiştir*.
Aynı şekle sahip herhangi iki tansör üzerinde eleman-yönlü işlemleri çağırabiliriz.
Aşağıdaki örnekte, 5 öğeli bir grubu formüle etmek için virgül kullanıyoruz, her öğe eleman-yönlü bir işlemin sonucudur.

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Üs alma gibi tekli operatörler de dahil olmak üzere, çok daha fazla işlem eleman-yönlü olarak uygulanabilir.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

Eleman-yönlü hesaplamalara ek olarak, vektör iç çarpımı ve matris çarpımı dahil olmak üzere doğrusal cebir işlemleri de gerçekleştirebiliriz.
Doğrusal cebirin önemli parçalarını (varsayılmış hiçbir ön bilgi olmadan) şu şekilde açıklayacağız :numref:`sec_linear-algebra`.

Ayrıca birden fazla tensörü bir araya getirip daha büyük bir tensör oluşturmak için uçtan uca *istifleyebiliriz*.
Sadece tensörlerin bir listesini vermeli ve sisteme hangi eksende birleştireceklerini söylemeliyiz.
Aşağıdaki örnek, satırlar (eksen 0, şeklin ilk öğesi) ile sütunlar (eksen 1, şeklin ikinci öğesi) boyunca iki matrisi birleştirdiğimizde ne olacağını gösterir.
İlk çıktı tensörünün eksen-0 uzunluğunun ($6$) iki girdi tensörünün eksen-0 uzunluklarının ($3 + 3$) toplamı olduğunu görebiliriz; ikinci çıktı tensörünün eksen-1 uzunluğu ($8$) iki girdi tensörünün eksen-1 uzunluklarının ($4 + 4$) toplamıdır.

```{.python .input}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([x, y], axis=0), tf.concat([x, y], axis=1)
```

Bazen, *mantıksal ifadeler* aracılığıyla bir ikili tensör oluşturmak isteriz.
Örnek olarak `x == y`yi ele alalım.
Her konum için, eğer `x` ve `y` bu konumda eşitse, yeni tensördeki karşılık gelen girdi 1 değerini alır, yani mantıksal ifade `x == y` o konumda doğrudur; aksi halde bu pozisyon 0 değerini alır.

```{.python .input}
#@tab all
x == y
```

Tensördeki tüm elemanların toplanması, sadece bir elemanlı bir tensör verir.

```{.python .input}
#@tab mxnet, pytorch
x.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x)
```

## Yayma Mekanizması
:label:`subsec_broadcasting`

Yukarıdaki bölümde, aynı şekle sahip iki tensör üzerinde eleman-yönlü işlemlerin nasıl yapıldığını gördük. Belli koşullar altında, şekiller farklı olsa bile, *yayma mekanizmasını* çağırarak yine de eleman-yönlü işlemler gerçekleştirebiliriz.
Bu mekanizma şu şekilde çalışır: İlk olarak, bir veya her iki diziyi elemanları uygun şekilde kopyalayarak genişletin, böylece bu dönüşümden sonra iki tensör aynı şekle sahip olur.
İkincisi, sonuç dizileri üzerinde eleman-yönlü işlemleri gerçekleştirin.

Çoğu durumda, bir dizinin başlangıçta yalnızca 1 uzunluğuna sahip olduğu bir eksen boyunca yayın yaparız, aşağıdaki gibi:

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

`a` ve `b` sırasıyla $3\times1$ ve $1\times2$ matrisler olduğundan, onları toplamak istiyorsak şekilleri uyuşmaz.
Her iki matrisin girdilerini aşağıdaki gibi daha büyük bir $3\times2$ matrisine *yayınlıyoruz*: Her ikisini de eleman-yönlü eklemeden önce `a` matrisi için sütunlar çoğaltılır ve `b` matrisi için satırlar çoğaltılır.

```{.python .input}
#@tab all
a + b
```

## İndeksleme ve Dilimleme

Diğer tüm Python dizilerinde olduğu gibi, bir tensördeki öğelere indeksle erişilebilir.
Herhangi bir Python dizisinde olduğu gibi, ilk öğenin dizini 0'dır ve aralıklar ilk öğeyi içerecek ancak son öğeden *öncesi* eklenecek şekilde belirtilir.
Standart Python listelerinde olduğu gibi, öğelere, negatif endeksler kullanarak listenin sonuna göreceli konumlarına göre erişebiliriz.

Böylece, `[-1]` son elemanı seçer ve `[1:3]` ikinci ve üçüncü elemanları aşağıdaki gibi seçer:

```{.python .input}
#@tab all
x[-1], x[1:3]
```

Okumanın ötesinde, indisleri belirterek bir matrisin elemanlarını da yazabiliriz.

```{.python .input}
#@tab mxnet, pytorch
x[1, 2] = 9
x
```

```{.python .input}
#@tab tensorflow
x = tf.convert_to_tensor(tf.Variable(x)[1, 2].assign(9))
x
```

Birden fazla öğeye aynı değeri atamak istiyorsak, hepsini indeksleriz ve sonra da değer atarız.
Örneğin, `[0:2, :]` birinci ve ikinci satırlara erişir, burada `:` eksen 1 (sütun) boyunca tüm elemanları alır.
Biz burada matrisler için indekslemeyi tartışırken, anlatılanlar açıkça vektörler ve 2'den fazla boyuttaki tensörler için de geçerlidir.

```{.python .input}
#@tab mxnet, pytorch
x[0:2, :] = 12
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[1:2,:].assign(tf.ones(x_var[1:2,:].shape, dtype = tf.float32)*12)
x = tf.convert_to_tensor(x_var)
x
```

## Belleği Kaydetme

Koşturma işlemleri, ana makine sonuçlarına yeni bellek ayrılmasına neden olabilir.
Örneğin, `y = x + y` yazarsak, `y`yi göstermek için kullanılan tensörden vazgeçer ve bunun yerine yeni verilen bellekteki `y`yi işaret ederiz.
Aşağıdaki örnekte, bunu, bize bellekteki referans edilen nesnenin tam adresini veren Python'un `id()` fonksiyonu ile gösteriyoruz.
`y = y + x` komutunu çalıştırdıktan sonra, `id(y)` ifadesinin farklı bir yeri gösterdiğini göreceğiz.
Bunun nedeni, Python'un önce sonuç için yeni bellek ayırarak `y + x` değerini hesaplayıp ardından `y`'yi bellekteki bu yeni konuma işaret etmesidir.

```{.python .input}
#@tab all
before = id(y)
y = y + x
id(y) == before
```

Bu iki nedenden dolayı istenmez olabilir.
Birincisi, her zaman gereksiz yere bellek ayırmaya çalışmak istemiyoruz.
Makine öğrenmesinde yüzlerce megabayt parametreye sahip olabilir ve hepsini saniyede birkaç kez güncelleyebiliriz.
Genellikle, bu güncellemeleri *yerinde* yapmak isteyeceğiz.
İkinci olarak, birden çok değişkenden aynı parametrelere işaret edebiliriz.
Yerinde güncelleme yapmazsak, diğer referanslar hala eski bellek konumuna işaret eder ve bu da kodumuzun bazı bölümlerinin yanlışlıkla eski parametrelere başvurmasını olası kılar.

Neyse ki, MXNet'te yerinde işlemler yapmak kolaydır.
Bir işlemin sonucunu daha önce ayrılmış bir diziye dilim gösterimi ile atayabiliriz, örneğin, `y[:] = <ifade>`.
Bu kavramı göstermek için, önce başka bir `y` ile aynı şekle sahip yeni bir `z` matrisi yaratıyoruz ve bir blok $0$ girdisi tahsis etmek üzere `zeros_like`yi kullanıyoruz.

```{.python .input}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab pytorch
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab tensorflow
z = tf.Variable(tf.zeros_like(y))
print('id(z):', id(z))
z[:].assign(x + y)
print('id(z):', id(z))
```

Sonraki hesaplamalarda `x` değeri yeniden kullanılmazsa, işlemin bellek yükünü azaltmak için `x[:] = x + y` veya `x += y` kullanabiliriz.

```{.python .input}
#@tab mxnet, pytorch
before = id(x)
x += y
id(x) == before
```

```{.python .input}
#@tab tensorflow
before = id(x)
tf.Variable(x).assign(x + y)
id(x) == before
```

## Diğer Python Nesnelerine Dönüştürme

NumPy tensörüne dönüştürmek veya tam tersi kolaydır.
Dönüştürülen sonuç belleği paylaşmaz.
Bu küçük sıkıntı aslında oldukça önemlidir: CPU veya GPU'larda işlem yaparken, Python'un NumPy paketinin aynı bellek yığınıyla başka bir şey yapmak isteyip istemediğini görmek için hesaplamayı durdurmak istemezsiniz.

```{.python .input}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

```{.python .input}
#@tab pytorch
a = x.numpy()
b = torch.tensor(a)
type(a), type(b)
```

```{.python .input}
#@tab tensorflow
a = x.numpy()
b = tf.constant(a)
type(a), type(b)
```

1-boyutlu tensörünü bir Python skalerine (sayılına) dönüştürmek için `item` işlevini veya Python'un yerleşik işlevlerini çağırabiliriz.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## `d2l` Paketi

Bu kitabın çevrimiçi sürümü boyunca, birden çok çerçevenin uygulanmasını sağlayacağız.
Bununla birlikte, farklı çerçevelerin API adlarında veya kullanımlarında farklı olabilir.
Aynı kod bloğunu birden çok çerçevede daha iyi kullanmak için, `d2l` paketinde yaygın olarak kullanılan birkaç işlevi birleştiriyoruz.
`#@save` yorumu (comment), takip eden işlevin, sınıfın veya ifadelerin `d2l` paketine kaydedildiği özel bir işarettir.
Örneğin, daha sonra, desteklenen herhangi bir çerçevede tanımlanabilen bir tensör `a`'yı bir NumPy tensörüne dönüştürmesi için doğrudan `d2l.numpy(a)`yi çağırabiliriz.

```{.python .input}
#@save
numpy = lambda a: a.asnumpy()
size = lambda a: a.size
reshape = lambda a, *args: a.reshape(*args)
ones = np.ones
zeros = np.zeros
```

```{.python .input}
#@tab pytorch
#@save
numpy = lambda a: a.detach().numpy()
size = lambda a: a.numel()
reshape = lambda a, *args: a.reshape(*args)
ones = torch.ones
zeros = torch.zeros
```

```{.python .input}
#@tab tensorflow
#@save
numpy = lambda a: a.numpy()
size = lambda a: tf.size(a).numpy()
reshape = tf.reshape
ones = tf.ones
zeros = tf.zeros
```

Kitabın geri kalanında genellikle daha karmaşık fonksiyonlar veya sınıflar tanımlarız.
Daha sonra kullanılabilecekleri `d2l` paketine kaydedeceğiz, böylece daha sonra yeniden tanımlanmadan doğrudan çağrılabilirler.

## Özet

* Derin öğrenme için veri depolamada ve oynama yapmada ana arayüz tensördür ($n$-boyutlu dizi). Temel matematik işlemleri, yayınlama, indeksleme, dilimleme, bellek tasarrufu ve diğer Python nesnelerine dönüştürme gibi çeşitli işlevler sağlar.

## Alıştırmalar

1. Bu bölümdeki kodu çalıştırın. Bu bölümdeki `x == y` koşullu ifadesini, `x < y` veya `x > y` olarak değiştirin ve sonra ne tür bir tensör alabileceğinizi görün.
1. Yayın mekanizmasındaki öğeye göre çalışan iki tensörü, diğer şekillerle, örneğin 3 boyutlu tensörler ile değiştirin. Sonuç beklendiği gibi mi?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/187)
:end_tab:
