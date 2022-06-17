# Veri ile Oynama Yapmak
:label:`sec_ndarray`

Bir şeylerin yapılabilmesi için, verileri depolamak ve oynama yapmak (manipule etmek) için bazı yollar bulmamız gerekiyor.
Genellikle verilerle ilgili iki önemli şey vardır: (i) Bunları elde etmek; ve (ii) bilgisayar içine girdikten sonra bunları işlemek. Veri depolamanın bir yolu olmadan onu elde etmenin bir anlamı yok, bu yüzden önce sentetik (yapay) verilerle oynayarak ellerimizi kirletelim. Başlamak için, *gerey (tensör)* olarak da adlandırılan $n$ boyutlu diziyi tanıtalım.

Python'da en çok kullanılan bilimsel hesaplama paketi olan NumPy ile çalıştıysanız, bu bölümü tanıdık bulacaksınız.
Hangi çerçeveyi kullanırsanız kullanın, *tensör sınıfı* (MXNet'teki `ndarray`, hem PyTorch hem de TensorFlow'daki `Tensor`), fazladan birkaç vurucu özellik ile NumPy'nin `ndarray`'ına benzer.
İlk olarak, GPU hesaplama hızlandırmayı iyi desteklerken, NumPy sadece CPU hesaplamasını destekler.
İkincisi, tensör sınıfı otomatik türev almayı destekler.
Bu özellikler, tensör sınıfını derin öğrenme için uygun hale getirir.
Kitap boyunca, tensörler dediğimizde, aksi belirtilmedikçe tensör sınıfının örneklerinden bahsediyoruz.

## Başlangıç

Bu bölümde, sizi ayaklandırıp koşturmayı, kitapta ilerledikçe üstüne koyarak geliştireceğiniz temel matematik ve sayısal hesaplama araçlarıyla donatmayı amaçlıyoruz.
Bazı matematiksel kavramları veya kütüphane işlevlerini içselleştirmede zorlanıyorsanız, endişelenmeyin.
Aşağıdaki bölümlerde bu konular pratik örnekler bağlamında tekrar ele alınacak ve yerine oturacak.
Öte yandan, zaten biraz bilgi birikiminiz varsa ve matematiksel içeriğin daha derinlerine inmek istiyorsanız, bu bölümü atlayabilirsiniz.

:begin_tab:`mxnet`
Başlarken, MXNet'ten `np` (`numpy`) ve `npx` (`numpy_extension`) modüllerini içe aktarıyoruz (import).
Burada, np modülü, NumPy tarafından desteklenen işlevleri içerirken, npx modülü, NumPy benzeri bir ortamda derin öğrenmeyi güçlendirmek için geliştirilmiş bir dizi uzantıyı içerir.
Tensörleri kullanırken neredeyse her zaman `set_np` işlevini çağırırız: Bu, tensör işlemenin MXNet'in diğer bileşenlerine uyumluluğu içindir.
:end_tab:

:begin_tab:`pytorch`
(**Başlamak için `torch`u içe aktarıyoruz. PyTorch olarak adlandırılsa da, `pytorch` yerine `torch`u  içe aktarmamız gerektiğini unutmayın.**)
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

[**Bir tensör, (muhtemelen çok boyutlu) bir sayısal değerler dizisini temsil eder.**]
Bir eksende, tensöre *vektör* denir.
İki eksende, tensöre *matris* denir.
İkiden fazla ekseni olan tensörlerin özel matematik isimleri yoktur.
$k > 2$ eksende, tensörlerin özel adları yoktur, bu nesnelere $k.$ *dereceli tensörler* deriz.

:begin_tab:`mxnet`
MXNet, değerlerle önceden doldurulmuş yeni tensörler oluşturmak için çeşitli işlevler sağlar. Örneğin, `arange(n)`'yi çağırarak, 0'dan başlayarak (dahil) ve `n` ile biten (dahil değil) eşit aralıklı değerlerden oluşan bir vektör oluşturabiliriz.
Varsayılan olarak, aralık boyutu $1$'dir. Aksi belirtilmedikçe, yeni tensörler ana bellekte depolanır ve CPU tabanlı hesaplama için atanır.
:end_tab:

:begin_tab:`pytorch`
PyTorch, değerlerle önceden doldurulmuş yeni tensörler oluşturmak için çeşitli işlevler sağlar. Örneğin, `arange(n)`'yi çağırarak, 0'dan başlayarak (dahil) ve `n` ile biten (dahil değil) eşit aralıklı değerlerden oluşan bir vektör oluşturabiliriz.
Varsayılan olarak, aralık boyutu $1$'dir. Aksi belirtilmedikçe, yeni tensörler ana bellekte depolanır ve CPU tabanlı hesaplama için atanır.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow, değerlerle önceden doldurulmuş yeni tensörler oluşturmak için çeşitli işlevler sağlar. Örneğin, `arange(n)`'yi çağırarak, 0'dan başlayarak (dahil) ve `n` ile biten (dahil değil) eşit aralıklı değerlerden oluşan bir vektör oluşturabiliriz.
Varsayılan olarak, aralık boyutu $1$'dir. Aksi belirtilmedikçe, yeni tensörler ana bellekte depolanır ve CPU tabanlı hesaplama için atanır.
:end_tab:

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

(**Bir tensörün *şekli*ne**) (~~ve toplam eleman sayısına~~) (her eksen boyunca uzunluk) `shape` özelliğini inceleyerek erişebiliriz.

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

[**Eleman sayısını veya değerlerini değiştirmeden bir tensörün şeklini değiştirmek**] için `reshape` işlevini çağırabiliriz.
Örneğin, `x` tensörümüzü, (12,) şekilli bir satır vektöründen (3, 4) şekilli bir matrise dönüştürebiliriz .
Bu yeni tensör tam olarak aynı değerleri içerir, ancak onları 3 satır ve 4 sütun olarak düzenlenmiş bir matris olarak görür.
Yinelemek gerekirse, şekil değişmiş olsa da, elemanlar değişmemiştir.
Boyutun yeniden şekillendirilme ile değiştirilmediğine dikkat edin.

```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Her boyutu manuel olarak belirterek yeniden şekillendirmeye gerek yoktur.
Hedef şeklimiz (yükseklik, genişlik) şekilli bir matrisse, o zaman genişliği bilirsek, yükseklik üstü kapalı olarak verilmiş olur.
Neden bölmeyi kendimiz yapmak zorunda olalım ki?
Yukarıdaki örnekte, 3 satırlı bir matris elde etmek için, hem 3 satır hem de 4 sütun olması gerektiğini belirttik.
Neyse ki, tensörler, bir boyut eksik geri kalanlar verildiğinde, kalan bir boyutu otomatik olarak çıkarabilir.
Bu özelliği, tensörlerin otomatik olarak çıkarımını istediğimiz boyuta `-1` yerleştirerek çağırıyoruz.
Bizim durumumuzda, `x.reshape(3, 4)` olarak çağırmak yerine, eşit biçimde `x.reshape(-1, 4)` veya `x.reshape(3, -1)` olarak çağırabilirdik.

Tipik olarak, matrislerimizin sıfırlar, birler, diğer bazı sabitler veya belirli bir dağılımdan rastgele örneklenmiş sayılarla başlatılmasını isteriz.
[**Tüm elemanları 0 (~~veya 1~~) olarak ayarlanmış**] ve (2, 3, 4) şeklindeki bir tensörü temsil eden bir tensörü aşağıdaki şekilde oluşturabiliriz:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
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

Genellikle, [**bir tensördeki her eleman için değerleri bir olasılık dağılımından rastgele örneklemek**] isteriz.
Örneğin, bir sinir ağında parametre görevi görecek dizileri oluşturduğumuzda, değerlerini genellikle rastgele ilkletiriz.
Aşağıdaki kod parçası (3, 4) şekilli bir tensör oluşturur. Elemanlarının her biri ortalaması 0 ve standart sapması 1 olan standart Gauss (normal) dağılımından rastgele örneklenir.

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

Sayısal değerleri içeren bir Python listesi (veya liste listesi) sağlayarak istenen tensördeki [**her eleman için kesin değerleri de belirleyebiliriz.**]
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
En basit ve en kullanışlı işlemlerden bazıları *eleman yönlü (elementwise)* işlemlerdir.
Bunlar bir dizinin her elemanına standart bir sayıl işlem uygular.
İki diziyi girdi olarak alan işlevler için, eleman yönlü işlemler iki diziden karşılık gelen her bir eleman çiftine standart bir ikili operatör uygular.
Sayıldan sayıla eşleşen herhangi bir fonksiyondan eleman yönlü bir fonksiyon oluşturabiliriz.

Matematiksel gösterimde, böyle bir *tekli* skaler işlemi (bir girdi alarak) $f: \mathbb{R} \rightarrow \mathbb{R}$ imzasıyla ifade ederiz.
Bu, işlevin herhangi bir gerçel sayıdan ($\mathbb{R}$) diğerine eşlendiği anlamına gelir.
Benzer şekilde, $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ imzası ile bir *ikili* skaler operatörü (iki gerçel girdi alarak ve bir çıktı verir) belirtiriz.
*Aynı şekilli* iki  $\mathbf{u}$ ve $\mathbf{v}$ vektörü ve $f$ ikili operatörü verildiğinde, tüm $i$ler için $c_i \gets f(u_i, v_i)$ diye ayarlayarak $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ vektörünü üretebiliriz; burada $c_i, u_i$ ve $v_i$, $\mathbf{c}, \mathbf{u}$ ve $\mathbf{v}$ vektörlerinin $i.$ elemanlarıdır.
Burada, skaler fonksiyonu eleman yönlü bir vektör işlemini *yükselterek* vektör değerli $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ ürettik.

Ortak standart aritmetik operatörler (`+`, `-`, `*`, `/` ve `**`), rastgele şekile sahip herhangi bir benzer şekilli tensörler için eleman yönlü işlemlere *yükseltilmiştir*.
Aynı şekle sahip herhangi iki tensör üzerinde eleman yönlü işlemleri çağırabiliriz.
Aşağıdaki örnekte, 5 öğeli bir grubu formüle etmek için virgül kullanıyoruz, her öğe eleman yönlü bir işlemin sonucudur.

### İşlemler

[**Genel standart aritmatik işlemler
(`+`, `-`, `*`, `/`, ve `**`)
eleman yönlü işlemlere *yükseltilmiştir*.**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # ** işlemi kuvvet almadır.
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # ** işlemi kuvvet almadır.
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # ** işlemi kuvvet almadır.
```

Kuvvet alma gibi tekli operatörler de dahil olmak üzere, (**çok daha fazla işlem eleman yönlü olarak uygulanabilir.**)

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

Eleman yönlü hesaplamalara ek olarak, vektör iç çarpımı ve matris çarpımı dahil olmak üzere doğrusal cebir işlemleri de gerçekleştirebiliriz.
Doğrusal cebirin önemli parçalarını (varsayılmış hiçbir ön bilgi olmadan) :numref:`sec_linear-algebra` içinde açıklayacağız.

Ayrıca birden fazla tensörü bir araya getirip daha büyük bir tensör oluşturmak için uçtan uca *istifleyebiliriz*.
Sadece tensörlerin bir listesini vermeli ve sisteme hangi eksende birleştireceklerini söylemeliyiz.
Aşağıdaki örnek, satırlar (eksen 0, şeklin ilk öğesi) ile sütunlar (eksen 1, şeklin ikinci öğesi) boyunca iki matrisi birleştirdiğimizde ne olacağını gösterir.
İlk çıktı tensörünün eksen-0 uzunluğunun ($6$) iki girdi tensörünün eksen-0 uzunluklarının ($3 + 3$) toplamı olduğunu görebiliriz; ikinci çıktı tensörünün eksen-1 uzunluğu ($8$) iki girdi tensörünün eksen-1 uzunluklarının ($4 + 4$) toplamıdır.

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

Bazen, *mantıksal ifadeler* aracılığıyla bir ikili tensör oluşturmak isteriz.
Örnek olarak `X == Y`'yi ele alalım.
Her konum için, eğer `X` ve `Y` bu konumda eşitse, yeni tensördeki karşılık gelen girdi 1 değerini alır, yani mantıksal ifade `X == Y` o konumda doğrudur; aksi halde o pozisyon 0 değerini alır.

```{.python .input}
#@tab all
X == Y
```

[**Tensördeki tüm elemanların toplanması**], sadece bir elemanlı bir tensör verir.

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## Yayma Mekanizması
:label:`subsec_broadcasting`

Yukarıdaki bölümde, aynı şekle sahip iki tensör üzerinde eleman yönlü işlemlerin nasıl yapıldığını gördük. Belli koşullar altında, şekiller farklı olsa bile,[** *yayma mekanizmasını* çağırarak yine de eleman yönlü işlemler gerçekleştirebiliriz.**]
Bu mekanizma şu şekilde çalışır: İlk olarak, bir veya her iki diziyi elemanları uygun şekilde kopyalayarak genişletin, böylece bu dönüşümden sonra iki tensör aynı şekle sahip olur.
İkincisi, sonuç dizileri üzerinde eleman yönlü işlemleri gerçekleştirin.

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
Her iki matrisin girdilerini aşağıdaki gibi daha büyük bir $3\times2$ matrisine *yayınlıyoruz*: Her ikisini de eleman yönlü eklemeden önce `a` matrisi için sütunlar çoğaltılır ve `b` matrisi için satırlar çoğaltılır.

```{.python .input}
#@tab all
a + b
```

## İndeksleme ve Dilimleme

Diğer tüm Python dizilerinde olduğu gibi, bir tensördeki öğelere indeksle erişilebilir.
Herhangi bir Python dizisinde olduğu gibi, ilk öğenin dizini 0'dır ve aralıklar ilk öğeyi içerecek ancak son öğeden *öncesi* eklenecek şekilde belirtilir.
Standart Python listelerinde olduğu gibi, öğelere, negatif endeksler kullanarak listenin sonuna göreceli konumlarına göre erişebiliriz.

Böylece, [** `[-1]` son elemanı ve `[1:3]` ikinci ve üçüncü elemanları**] aşağıdaki gibi seçer:

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Okumanın ötesinde, (**indeksleri belirterek matrisin elemanlarını da yazabiliriz.**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow'daki `tensörler` (`Tensors`) değişmezdir, ve atanamazlar.
TensorFlow'daki `değişkenler` (`Variables`) atanmayı destekleyen durumların değiştirilebilir kapsayıcılardır. TensorFlow'daki gradyanların `değişken` (`Variable`) atamaları aracılığıyla geriye doğru akmadığını unutmayın.

Tüm `değişkenler`e (`Variable`) bir değer atamanın ötesinde, bir `değişken`in (`Variable`) öğelerini indeksler belirterek yazabiliriz.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

Birden fazla öğeye aynı değeri atamak istiyorsak, hepsini indeksleriz ve sonra da değer atarız.
Örneğin, `[0:2, :]` birinci ve ikinci satırlara erişir, burada `:` Eksen 1 (sütun) boyunca tüm elemanları alır.
Biz burada matrisler için indekslemeyi tartışırken, anlatılanlar açıkça vektörler ve 2'den fazla boyuttaki tensörler için de geçerlidir.

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## Belleği Kaydetme

[**Koşulan işlemler, sonuçların tutulması için yeni bellek ayrılmasına neden olabilir.**]
Örneğin, `Y = X + Y` yazarsak, `Y`'yi göstermek için kullanılan tensörden vazgeçer ve bunun yerine yeni ayrılan bellekteki `Y`'yi işaret ederiz.
Aşağıdaki örnekte, bunu, bize bellekteki referans edilen nesnenin tam adresini veren Python'un `id()` fonksiyonu ile gösteriyoruz.
`Y = Y + X` komutunu çalıştırdıktan sonra, `id(Y)` ifadesinin farklı bir yeri gösterdiğini göreceğiz.
Bunun nedeni, Python'un önce `Y + X` değerini hesaplayarak sonuç için yeni bellek ayırması ve ardından `Y`'yi bellekteki bu yeni konuma işaret etmesidir.

```{.python .input}
#@tab all
onceki = id(Y)
Y = Y + X
id(Y) == onceki
```

Bu iki nedenden dolayı istenmez olabilir.
Birincisi, her zaman gereksiz yere bellek ayırmaya çalışmak istemiyoruz.
Makine öğrenmesinde yüzlerce megabayt parametreye sahip olabilir ve hepsini saniyede birkaç kez güncelleyebiliriz.
Genellikle, bu güncellemeleri *yerinde* yapmak isteyeceğiz.
İkinci olarak, birden çok değişkenden aynı parametrelere işaret edebiliriz.
Yerinde güncelleme yapmazsak, diğer referanslar hala eski bellek konumuna işaret eder ve bu da kodumuzun bazı bölümlerinin yanlışlıkla eski parametrelere atıfta bulunmasını olası kılar.

:begin_tab:`mxnet, pytorch`
Neyse ki, (**yerinde işlemler yapmak**) kolaydır.
Bir işlemin sonucunu daha önce ayrılmış bir diziye dilim gösterimi ile atayabiliriz, örneğin, `Y[:] = <ifade>`.
Bu kavramı göstermek için, önce başka bir `Y` ile aynı şekle sahip yeni bir `Z` matrisi yaratıyoruz ve bir blok $0$ girdisi tahsis etmek üzere `zeros_like`'i kullanıyoruz.
:end_tab:

:begin_tab:`tensorflow`
`Değişkenler` (`Variable`), TensorFlow'daki değişken durum kapsayıcılarıdır. Model parametrelerinizi saklamanın bir yolunu sağlarlar. Bir işlemin sonucunu `atama` ile bir `değişken`e atayabiliriz. Bu kavramı göstermek için, $0$ girdilerinden oluşan bir blok ayırmak için `zeros_like` kullanarak, başka bir `Y` tensörü ile aynı şekle sahip bir `değişken` `Z` oluşturuyoruz.
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**`X` değeri sonraki hesaplamalarda yeniden kullanılmazsa, işlemin bellek yükünü azaltmak için `X[:] = X + Y` veya `X += Y` kullanabiliriz.**]
:end_tab:

:begin_tab:`tensorflow`
Durumu bir `değişken`de kalıcı olarak sakladığınızda bile, model parametreleriniz olmayan tensörler için aşırı bellek tahsislerden kaçınarak bellek kullanımınızı daha da azaltmak isteyebilirsiniz.

TensorFlow `Tensör`leri değişmez olduğundan ve gradyanlar `değişken` atamalarından akmadığından, TensorFlow, yerinde tek bir işlemi yürütmek için açıktan bir yol sağlamaz.

Ancak TensorFlow, hesaplamayı çalıştırmadan önce derlenen ve optimize edilen bir TensorFlow çizgesinin içine sarmak için `tf.function` dekoratörü sağlar. Bu, TensorFlow'un kullanılmayan değerleri budamasına ve artık gerekmeyen önceki tahsisleri yeniden kullanmasına olanak tanır. Bu, TensorFlow hesaplamalarının bellek yükünü en aza indirir.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
onceki = id(X)
X += Y
id(X) == onceki
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # Bu kullanılmayan değer budanacaktır
    A = X + Y  # Daha fazla gerekmediğinde tahsisler yeniden kullanılacaktır 
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Diğer Python Nesnelerine Dönüştürme
:begin_tab:`mxnet, tensorflow`
[**NumPy tensörüne (`ndarray`) dönüştürmek**] veya tam tersi kolaydır.
Dönüştürülen sonuç belleği paylaşmaz.
Bu küçük sıkıntı aslında oldukça önemlidir: CPU veya GPU'larda işlem yaparken, Python'un NumPy paketinin aynı bellek yığınıyla başka bir şey yapmak isteyip istemediğini görmek için hesaplamayı durdurmak istemezsiniz.
:end_tab:

:begin_tab:`pytorch`
[**NumPy tensörüne (`ndarray`) dönüştürmek**] veya tam tersi kolaydır.
Torch Tensörü ve numpy dizisi, temeldeki bellek konumlarını paylaşacak ve yerinde bir işlemle birini değiştirmek diğerini de değiştirecektir.
:end_tab:

```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

(**1-boyutlu tensörünü bir Python skalerine (sayılına) dönüştürmek için**) `item` işlevini veya Python'un yerleşik işlevlerini çağırabiliriz.

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

## Özet

* Derin öğrenme için veri depolamada ve oynama yapmada ana arayüz tensördür ($n$-boyutlu dizi). Temel matematik işlemleri, yayınlama, indeksleme, dilimleme, bellek tasarrufu ve diğer Python nesnelerine dönüştürme gibi çeşitli işlevler sağlar.

## Alıştırmalar

1. Bu bölümdeki kodu çalıştırın. Bu bölümdeki `X == Y` koşullu ifadesini, `X < Y` veya `X > Y` olarak değiştirin ve sonra ne tür bir tensör alabileceğinizi görün.
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
