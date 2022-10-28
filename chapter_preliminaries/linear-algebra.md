# Doğrusal Cebir
:label:`sec_linear-algebra`


Şimdi verileri saklayabileceğinize ve oynama yapabileceğinize göre, bu kitapta yer alan modellerin çoğunu anlamanız ve uygulamanızda gereken temel doğrusal cebir bilgilerini kısaca gözden geçirelim.
Aşağıda, doğrusal cebirdeki temel matematiksel nesneleri, aritmetiği ve işlemleri tanıtarak, bunların her birini matematiksel gösterim ve koddaki ilgili uygulama ile ifade ediyoruz.

## Sayıllar

Doğrusal cebir veya makine öğrenmesi üzerine hiç çalışmadıysanız, matematikle ilgili geçmiş deneyiminiz muhtemelen bir seferde bir sayı düşünmekten ibaretti.
Ayrıca, bir çek defterini dengelediyseniz veya hatta bir restoranda akşam yemeği için ödeme yaptıysanız, bir çift sayıyı toplama ve çarpma gibi temel şeyleri nasıl yapacağınızı zaten biliyorsunuzdur.
Örneğin, Palo Alto'daki sıcaklık $52$ Fahrenheit derecedir.
Usul olarak, sadece bir sayısal miktar içeren değerlere *sayıl (skaler)* diyoruz.
Bu değeri Celsius'a (metrik sistemin daha anlamlı sıcaklık ölçeği) dönüştürmek istiyorsanız, $f$'i $52$ olarak kurup $c = \frac{5}{9}(f - 32)$ ifadesini hesaplarsınız.
Bu denklemde ---$5$, $9$ ve $32$--- terimlerinin her biri skaler değerlerdir.
$c$ ve $f$ göstermelik ifadelerine (placeholders) *değişkenler* denir ve bilinmeyen skaler değerleri temsil ederler.

Bu kitapta, skaler değişkenlerin normal küçük harflerle (ör. $x$, $y$ ve $z$) gösterildiği matematiksel gösterimi kabul ediyoruz.
Tüm (sürekli) *gerçel değerli* skalerlerin alanını $\mathbb{R}$ ile belirtiyoruz.
Amaca uygun olarak, tam olarak *uzayın* ne olduğunu titizlikle tanımlayacağız, ancak şimdilik sadece $x \in \mathbb{R}$ ifadesinin $x$ değerinin gerçel değerli olduğunu söylemenin usula uygun bir yolu olduğunu unutmayın.
$\in$ sembolü "içinde" olarak telaffuz edilebilir ve sadece bir kümeye üyeliği belirtir.
Benzer şekilde, $x$ ve $y$'nin değerlerinin yalnızca $0$ veya $1$ olabilen rakamlar olduğunu belirtmek için $x, y \in \{0, 1 \}$ yazabiliriz.

(**Skaler, sadece bir elemente sahip bir tensör ile temsil edilir.**)
Sıradaki kod parçasında, iki skalere değer atıyoruz ve onlarla toplama, çarpma, bölme ve üs alma gibi bazı tanıdık aritmetik işlemleri gerçekleştiriyoruz.

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## Vektörler (Yöneyler)

[**Bir vektörü basitçe skaler değerlerin bir listesi olarak düşünebilirsiniz.**]
Bu değerlere vektörün *elemanları* (*giriş değerleri* veya *bileşenleri*) diyoruz.
Vektörlerimiz veri kümemizdeki örnekleri temsil ettiğinde, değerleri gerçek dünyadaki önemini korur.
Örneğin, bir kredinin temerrüde düşme (ödenmeme) riskini tahmin etmek için bir model geliştiriyorsak, her başvuru sahibini, gelirine, istihdam süresine, önceki temerrüt sayısına ve diğer faktörlere karşılık gelen bileşenleri olan bir vektörle ilişkilendirebiliriz.
Hastanedeki hastaların potansiyel olarak yaşayabilecekleri kalp krizi riskini araştırıyor olsaydık, her hastayı bileşenleri en güncel hayati belirtilerini, kolesterol seviyelerini, günlük egzersiz dakikalarını vb. içeren bir vektörle temsil edebilirdik.
Matematiksel gösterimlerde, genellikle vektörleri kalın, küçük harfler (örneğin, $\mathbf{x}$, $\mathbf{y}$ ve $\mathbf{z})$ olarak göstereceğiz.

Vektörlerle tek boyutlu tensörler aracılığıyla çalışırız.
Genelde tensörler, makinenizin bellek sınırlarına tabi olarak keyfi uzunluklara sahip olabilir.

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

Bir vektörün herhangi bir öğesini bir altindis kullanarak tanımlayabiliriz.
Örneğin, $\mathbf{x}$'in $i.$ elemanını $x_i$ ile ifade edebiliriz.
$x_i$ öğesinin bir skaler olduğuna dikkat edin, bu nedenle ondan bahsederken fontta kalın yazı tipi kullanmıyoruz.
Genel literatür sütun vektörlerini vektörlerin varsayılan yönü olarak kabul eder, bu kitap da öyle kabullenir.
Matematikte, $\mathbf{x}$ vektörü şu şekilde yazılabilir:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

burada $x_1, \ldots, x_n$ vektörün elemanlarıdır.
Kod olarak, (**herhangi bir öğeye onu tensöre indeksleyerek erişiriz.**)

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### Uzunluk, Boyutluluk ve Şekil

Bazı kavramları tekrar gözden geçirelim :numref:`sec_ndarray`.
Bir vektör sadece bir sayı dizisidir.
Ayrıca her dizinin bir uzunluğu olduğu gibi, her vektör de bir uzunluğa sahiptir.
Matematiksel gösterimde, $\mathbf{x}$ vektörünün $n$ gerçel değerli skalerlerden oluştuğunu söylemek istiyorsak, bunu $\mathbf{x} \in \mathbb{R}^n$ olarak ifade edebiliriz.
Bir vektörün uzunluğuna genel olarak vektörün *boyutu* denir.

Sıradan bir Python dizisinde olduğu gibi, Python'un yerleşik (built-in) `len()` işlevini çağırarak [**bir tensörün uzunluğuna erişebiliriz.**]

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

Bir tensör bir vektörü (tam olarak bir ekseni ile) temsil ettiğinde, uzunluğuna `.shape` özelliği ile de erişebiliriz.
Şekil, tensörün her ekseni boyunca uzunluğu (boyutsallığı) listeleyen bir gruptur.
(**Sadece bir ekseni olan tensörler için, şeklin sadece bir elemanı vardır.**)

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

"Boyut" kelimesinin bu bağlamlarda aşırı yüklenme eğiliminde olduğunu ve bunun da insanları şaşırtma yöneliminde olduğunu unutmayın.
Açıklığa kavuşturmak için, bir *vektörün* veya *ekseninin* boyutluluğunu onun uzunluğuna atıfta bulunmak için kullanırız; yani bir vektörün veya eksenin eleman sayısı.
Halbuki, bir tensörün boyutluluğunu, bir tensörün sahip olduğu eksen sayısını ifade etmek için kullanırız.
Bu anlamda, bir tensörün bazı eksenlerinin boyutluluğu, bu eksenin uzunluğu olacaktır.

## Matrisler

Vektörler, skalerleri sıfırdan birinci dereceye kadar genelleştirirken, matrisler de vektörleri birinci dereceden ikinci dereceye genelleştirir.
Genellikle kalın, büyük harflerle (örn., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$) göstereceğimiz matrisler, kodda iki eksenli tensörler olarak temsil edilir.

Matematiksel gösterimde, $\mathbf{A}$ matrisinin gerçel değerli skaler $m$ satır ve $n$ sütundan oluştuğunu ifade etmek için $\mathbf{A} \in \mathbb{R}^{m \times n}$'i kullanırız.
Görsel olarak, herhangi bir $\mathbf{A} \in \mathbb{R}^{m \times n}$ matrisini $a_{ij}$ öğesinin $i.$ satıra ve $j.$ sütuna ait olduğu bir tablo olarak gösterebiliriz:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

Herhangi bir $\mathbf{A}\in \mathbb{R}^{m\times n}$ için, $\mathbf{A}$ ($m$, $n$) veya $m\times n$ şeklindedir.
Özellikle, bir matris aynı sayıda satır ve sütuna sahip olduğunda, şekli bir kareye dönüşür; dolayısıyla buna *kare matris* denir.

Bir tensörü örneği yaratırken, en sevdiğimiz işlevlerden herhangi birini çağırıp $m$ ve $n$ iki bileşeninden oluşan bir şekil belirterek [**$m \times n$ matrisi oluşturabiliriz.**]

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

Satır ($i$) ve sütun ($j$) indekslerini belirterek $\mathbf{A}$ matrisinin $a_{ij}$ skaler öğesine erişebiliriz, $[\mathbf{A}]_{ij}$ gibi.
Bir $\mathbf{A}$ matrisinin skaler elemanları verilmediğinde, örneğin :eqref:`eq_matrix_def` gibi, basitçe $\mathbf{A}$ matrisinin küçük harfli altindislisi $a_{ij}$'yi kullanarak $[\mathbf{A}]_{ij}$'ye atıfta bulunuruz.
Gösterimi basit tutarken indeksleri ayırmak için virgüller yalnızca gerekli olduğunda eklenir, örneğin $a_{2, 3j}$ ve $[\mathbf{A}]_{2i-1, 3}$ gibi.


Bazen eksenleri ters çevirmek isteriz.
Bir matrisin satırlarını ve sütunlarını değiştirdiğimizde çıkan sonuç matrisine *devrik (transpose)* adı verilir.
Usul olarak, bir $\mathbf{A}$'nin devriğini $\mathbf{A}^\top$ ile gösteririz ve eğer $\mathbf{B} = \mathbf{A}^\top$ ise herhangi bir $i$ and $j$ için $b_{ij} = a_{ji}$'dir.
Bu nedenle, :eqref:`eq_matrix_def`'deki $\mathbf{A}$'nin devriği bir $n\times m$ matristir:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Şimdi kodda bir (**matrisin devriğine**) erişiyoruz.

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

Kare matrisin özel bir türü olarak, [**bir *simetrik matris* $\mathbf{A}$, devriğine eşittir: $\mathbf{A} = \mathbf{A}^\top$.**]
Burada simetrik bir matrisi `B` diye tanımlıyoruz.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

Şimdi `B`yi kendi devriğiyle karşılaştıralım.

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

Matrisler yararlı veri yapılarıdır: Farklı değişim (varyasyon) kiplerine (modalite) sahip verileri düzenlememize izin verirler.
Örneğin, matrisimizdeki satırlar farklı evlere (veri örneklerine) karşılık gelirken, sütunlar farklı özelliklere karşılık gelebilir.
Daha önce elektronik tablo yazılımı kullandıysanız veya şurayı okuduysanız :numref:`sec_pandas`, bu size tanıdık gelecektir.
Bu nedenle, tek bir vektörün varsayılan yönü bir sütun vektörü olmasına rağmen, bir tablo veri kümesini temsil eden bir matriste, her veri örneğini matristeki bir satır vektörü olarak ele almak daha gelenekseldir.
Ve sonraki bölümlerde göreceğimiz gibi, bu düzen ortak derin öğrenme tatbikatlarını mümkün kılacaktır.
Örneğin, bir tensörün en dış ekseni boyunca, veri örneklerinin minigruplarına veya  minigrup yoksa yalnızca veri örneklerine erişebilir veya bir bir sayabiliriz.

## Tensörler

Vektörlerin skalerleri genellemesi ve matrislerin vektörleri genellemesi gibi, daha fazla eksenli veri yapıları oluşturabiliriz.
[**Tensörler**] (bu alt bölümdeki "tensörler" cebirsel nesnelere atıfta bulunur) (**bize rastgele sayıda ekseni olan $n$-boyutlu dizileri tanımlamanın genel bir yolunu verir.**) Vektörler, örneğin, birinci dereceden tensörlerdir ve matrisler ikinci dereceden tensörlerdir.
Tensörler, özel bir yazı tipinin büyük harfleriyle (ör. $\mathsf{X}$, $\mathsf{Y}$ ve $\mathsf{Z}$) ve indeksleme mekanizmalarıyla (ör. $X_{ijk}$ ve $[\mathsf{X}]_{1, 2i-1, 3}$), matrislerinkine benzer gösterilir.

Renk kanallarını (kırmızı, yeşil ve mavi) istiflemek için yükseklik, genişlik ve bir *kanal* eksenine karşılık gelen 3 eksene sahip $n$-boyutlu dizi olarak gelen imgelerle çalışmaya başladığımızda tensörler daha önemli hale gelecektir.
Şimdilik, daha yüksek dereceli tensörleri atlayacağız ve temellere odaklanacağız.

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## Tensör Aritmetiğinin Temel Özellikleri

Skalerler, vektörler, matrisler ve rasgele sayıda eksenli tensörler (bu alt bölümdeki "tensörler" cebirsel nesnelere atıfta bulunur), çoğu zaman kullanışlı olan bazı güzel özelliklere sahiptir.
Örneğin, bir eleman yönlü işlemin tanımından, herhangi bir eleman yönlü tekli işlemin işlenen nesnenin şeklini değiştirmediğini fark etmiş olabilirsiniz.
Benzer şekilde, (**aynı şekle sahip herhangi bir iki tensör göz önüne alındığında, herhangi bir ikili elemanlı işlemin sonucu, gene aynı şekle sahip bir tensör olacaktır.**)
Örneğin, aynı şekle sahip iki matris toplama, bu iki matrisin üzerinde eleman yönlü toplama gerçekleştirir.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # `A`nın kopyasını yeni bellek tahsis ederek `B`ye atayın
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # `A`nın kopyasını yeni bellek tahsis ederek `B`ye atayın
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # Yeni bellek tahsis ederek `A`yı `B`ye klonlamak yok
A, A + B
```

Özellikle, [**iki matrisin eleman yönlü çarpımına *Hadamard çarpımı***] (matematik gösterimi $\odot$) denir.
$i.$ satır ve $j.$ sütununun öğesi $b_{ij}$ olan $\mathbf{B}\in\mathbb{R}^{m\times n}$ matrisini düşünün. $\mathbf{A}$ (:eqref:`eq_matrix_def`'da tanımlanmıştır) ve $\mathbf{B}$ matrislerinin Hadamard çarpımı:

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

[**Bir tensörün skaler ile çarpılması veya toplanması**], işlenen tensörün her elemanını skaler ile toplayacağından veya çarpacağından tensörün şeklini de değiştirmez.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## İndirgeme
:label:`subseq_lin-alg-reduction`

Keyfi tensörlerle gerçekleştirebileceğimiz faydalı işlemlerden biri [**elemanlarının toplamını**] hesaplamaktır.
Matematiksel gösterimde, $\sum$ sembolünü kullanarak toplamları ifade ederiz.
$d$ uzunluğa sahip $\mathbf{x}$ vektöründeki öğelerin toplamını ifade etmek için $\sum_{i=1}^d x_i$ yazarız.
Kodda, toplamı hesaplamak için sadece ilgili işlevi çağırabiliriz.

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

[**Rasgele şekilli tensörlerin elamanları üzerindeki toplamları**] ifade edebiliriz.
Örneğin, $m\times n$ matris $\mathbf{A}$ öğelerinin toplamı $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ diye yazılabilir.

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

Varsayılan olarak, toplamı hesaplama işlevini çağırmak, tüm eksenleri boyunca bir tensörü skalere *indirger*.
[**Toplama yoluyla tensörün indirgendiği eksenleri de belirtebiliriz.**]
Örnek olarak matrisleri alın.
Tüm satırların öğelerini toplayarak satır boyutunu (eksen 0) indirgemek için, işlevi çağırırken `axis=0` değerini belirtiriz.
Girdi matrisi, çıktı vektörü oluşturmak için eksen 0 boyunca indirgendiğinden, girdinin eksen 0 boyutu, çıktının şeklinde kaybolur.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

`axis=1` olarak belirtmek, tüm sütunların öğelerini toplayarak sütun boyutunu (eksen 1) azaltacaktır.
Böylece, girdinin eksen 1 boyutu, çıktının şeklinde kaybolur.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Bir matrisin toplama yoluyla hem satırlar hem de sütunlar boyunca indirgenmesi, matrisin tüm öğelerinin toplanmasıyla eşdeğerdir.

```{.python .input}
A.sum(axis=[0, 1])  # `A.sum()` ile aynı
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # `A.sum()` ile aynı
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  #  `tf.reduce_sum(A)` ile aynı
```

[**İlgili bir miktar da *ortalama*dır.**]
Ortalamayı, toplamı toplam eleman sayısına bölerek hesaplıyoruz.
Kod olarak, keyfi şekildeki tensörlerdeki ortalamanın hesaplanmasında ilgili işlevi çağırabiliriz.

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Benzer şekilde, ortalama hesaplama fonksiyonu, belirtilen eksenler boyunca bir tensörü de indirgeyebilir.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### İndirgemesiz Toplama
:label:`subseq_lin-alg-non-reduction`

Gene de, bazen toplamı veya ortalamayı hesaplamak için işlevi çağırırken [**eksen sayısını değiştirmeden**] tutmak yararlı olabilir.

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

Örneğin, `sum_A` her satırı topladıktan sonra hala iki eksenini koruduğundan,(**`A`'yı yayınlayarak `sum_A` ile bölebiliriz.**)

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

[**Bir eksen boyunca `A`'nın öğelerinin biriktirilmiş (kümülatif) toplamını hesaplamak**] istiyorsak, `axis=0` diyelim (satır satır), `cumsum` işlevini çağırabiliriz. Bu işlev girdi tensörünü herhangi bir eksen boyunca indirgemez.

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## Nokta Çarpımları

Şimdiye kadar sadece eleman yönlü işlemler, toplamalar ve ortalamalar gerçekleştirdik. Ayrıca tüm yapabileceğimiz bu olsaydı, doğrusal cebir muhtemelen kendi bölümünü hak etmeyecekti. Bununla birlikte, en temel işlemlerden biri iç çarpımdır. İki vektör $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ verildiğinde, *iç çarpımları* $\mathbf{x}^\top \mathbf{y}$ (veya $\langle \mathbf{x}, \mathbf{y}  \rangle$), aynı konumdaki öğelerin çarpımlarının toplamıdır: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~İki vektörün *nokta çarpımı*, aynı konumdaki elemanların çarpımlarının toplamıdır.~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

(**İki vektörün nokta çarpımlarını, eleman yönlü bir çarpma ve ardından bir toplam gerçekleştirerek eşit şekilde ifade edebileceğimizi**) unutmayın:

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

Nokta çarpımları çok çeşitli bağlamlarda kullanışlıdır.
Örneğin, $\mathbf{x} \in \mathbb{R}^d$ vektörü ve $\mathbf{w} \in \mathbb{R}^d$ ile belirtilen bir ağırlık kümesi verildiğinde, $\mathbf{x}$ içindeki değerlerin $\mathbf{w}$ ağırlıklarına göre ağırlıklı toplamı $\mathbf{x}^\top \mathbf{w}$ nokta çarpımı olarak ifade edilebilir.
Ağırlıklar negatif olmadığında ve bire (örn., $\left(\sum_{i=1}^{d} {w_i} = 1\right)$) toplandığında, nokta çarpımı *ağırlıklı ortalama*yı ifade eder.
İki vektörü birim uzunluğa sahip olacak şekilde normalleştirdikten sonra, nokta çarpımlar arasındaki açının kosinüsünü ifade eder.
*Uzunluk* kavramını bu bölümün ilerleyen kısımlarında usüle uygun tanıtacağız.

## Matris-Vektör Çarpımları

Artık nokta çarpımlarını nasıl hesaplayacağımızı bildiğimize göre, *matris-vektör çarpımları*nı anlamaya başlayabiliriz.
$\mathbf{A} \in \mathbb{R}^{m \times n}$ matrisini ve $\mathbf{x} \in \mathbb{R}^n$ vektörünü sırasıyla tanımladık ve :eqref:`eq_matrix_def` ve :eqref:`eq_vec_def`'de görselleştirdik.
$\mathbf{A}$ matrisini satır vektörleriyle görselleştirerek başlayalım.

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

burada her $\mathbf{a}^\top_{i} \in \mathbb{R}^n$, $\mathbf{A}$ matrisinin $i .$ satırını temsil eden bir satır vektörüdür.
[**Matris-vektör çarpımı $\mathbf{A}\mathbf{x}$, basitçe $i.$ elemanı $\mathbf{a}^\top_i \mathbf{x}$ iç çarpımı olan $m$ uzunluğunda bir sütun vektörüdür.**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

$\mathbf{A}\in \mathbb{R}^{m \times n}$ matrisi ile çarpmayı vektörleri $\mathbb{R}^{n}$'den $\mathbb{R}^{m}$'e yansıtan bir dönüşüm olarak düşünebiliriz.
Bu dönüşümler oldukça faydalı oldu.
Örneğin, döndürmeleri bir kare matrisle çarpma olarak gösterebiliriz.
Sonraki bölümlerde göreceğimiz gibi, bir önceki katmanın değerleri göz önüne alındığında, bir sinir ağındaki her bir katman hesaplanırken gereken en yoğun hesaplamaları tanımlamak için matris-vektör çarpımlarını da kullanabiliriz.

:begin_tab:`mxnet`
Matris-vektör çarpımlarını tensörlerle kodda ifade ederken, nokta çarpımlarındaki aynı `dot` işlevini kullanırız.
`A` matrisi ve `x` vektörü ile `np.dot(A, x)` dediğimizde matris-vektör çarpımı gerçekleştirilir.
`A` sütun boyutunun (eksen 1 boyunca uzunluğu) `x` boyutuyla (uzunluğu) aynı olması gerektiğini unutmayın.
:end_tab:

:begin_tab:`pytorch`
Matris-vektör çarpımlarını tensörlerle kodda ifade ederken, `mv` işlevini kullanırız.
`A` matrisi ve `x` vektörü ile `torch.mv(A, x)` dediğimizde matris-vektör çarpımı gerçekleştirilir.
`A` sütun boyutunun (eksen 1 boyunca uzunluğu) `x` boyutuyla (uzunluğu) aynı olması gerektiğini unutmayın.
:end_tab:

:begin_tab:`tensorflow`
Matris-vektör çarpımlarını tensörlerle kodda ifade ederken, `matvec` işlevini kullanırız.
`A` matrisi ve `x` vektörü ile `tf.linalg.matvec(A, x)` dediğimizde matris-vektör çarpımı gerçekleştirilir.
`A` sütun boyutunun (eksen 1 boyunca uzunluğu) `x` boyutuyla (uzunluğu) aynı olması gerektiğini unutmayın.
:end_tab:

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## Matris-Matris Çarpımı

Eğer nokta ve matris-vektör çarpımlarını anladıysanız, *matris-matris çarpımı* açık olmalıdır.

$\mathbf{A} \in \mathbb{R}^{n \times k}$  ve $\mathbf{B} \in \mathbb{R}^{k \times m}$ diye iki matrisimizin olduğunu diyelim:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

$\mathbf{A}$ matrisinin $i.$ satırını temsil eden satır vektörünü $\mathbf{a}^\top_{i}$ ile belirtelim ve $\mathbf{B}$ matrisinin $j.$ sütunu da $\mathbf{b}_{j}$ olsun.
$\mathbf{C} = \mathbf{A}\mathbf{B}$ matris çarpımını üretmek için $\mathbf{A}$'yı satır vektörleri ve $\mathbf{B}$'yi sütun vektörleri ile düşünmek en kolay yoldur:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

Daha sonra, $\mathbf{C} \in \mathbb{R}^{n \times m}$ matris çarpımı her bir $c_{ij}$ öğesi $\mathbf{a}^\top_i \mathbf{b}_j$ nokta çarpımı hesaplanarak üretilir:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Matris-matris çarpımı $\mathbf{AB}$'yi sadece $m$ tane matris-vektör çarpımı gerçekleştirmek ve sonuçları $n\times m$ matrisi oluşturmak için birleştirmek olarak düşünebiliriz.**]
Aşağıdaki kod parçasında, `A` ve `B` üzerinde matris çarpımı yapıyoruz.
Burada `A`, 5 satır ve 4 sütunlu bir matristir ve `B` 4 satır ve 3 sütunlu bir matristir.
Çarpma işleminden sonra 5 satır ve 3 sütun içeren bir matris elde ederiz.

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

Matris-matris çarpımı basitçe *matris çarpımı* olarak adlandırılabilir ve Hadamard çarpımı ile karıştırılmamalıdır.


## Normlar (Büyüklükler)
:label:`subsec_lin-algebra-norms`

Doğrusal cebirde en kullanışlı operatörlerden bazıları *normlardır*.
Gayri resmi olarak, bir vektörün normu bize bir vektörün ne kadar *büyük* olduğunu söyler.
Burada ele alınan *ebat* kavramı boyutsallık değil, bileşenlerin büyüklüğü ile ilgilidir.

Doğrusal cebirde, bir vektör normu, bir vektörü bir skalere eşleştiren ve bir avuç dolusu özelliği karşılayan bir $f$ fonksiyonudur.
Herhangi bir $\mathbf{x}$ vektörü verildiğinde, ilk özellik, bir vektörün tüm elemanlarını sabit bir $\alpha$ çarpanına göre ölçeklendirirsek, normunun da aynı sabit çarpanın *mutlak değerine* göre ölçeklendiğini söyler:

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$


İkinci özellik, tanıdık üçgen eşitsizliğidir:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$


Üçüncü özellik, normun negatif olmaması gerektiğini söyler:

$$f(\mathbf{x}) \geq 0.$$

Çoğu bağlamda herhangi bir şey için en küçük *ebat* 0 olduğu için bu mantıklıdır.
Nihai özellik, en küçük normun sadece ve sadece tüm sıfırlardan oluşan bir vektör tarafından elde edilmesini gerektirir.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

Normların mesafe ölçülerine çok benzediğini fark edebilirsiniz.
Ayrıca eğer ilkokuldan Öklid mesafesini hatırlarsanız (Pisagor teoremini düşünün), o zaman negatif olmama ve üçgen eşitsizlik kavramları zihininizde bir zil çalabilir.
Aslında Öklid mesafesi bir normdur: Özellikle $L_2$ normudur.
$n$ boyutlu vektör, $\mathbf{x}$, içindeki öğelerin $x_1,\ldots,x_n$ olduğunu varsayalım.
[**$\mathbf{x}$'in $L_2$ *normu*, vektör öğelerinin karelerinin toplamının kareköküdür:**]

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**)

burada $2$ altindisi genellikle $L_2$ normlarında atlanır, yani, $\|\mathbf{x}\|$, $\|\mathbf{x}\|_2$ ile eşdeğerdir. Kodda, bir vektörün $L_2$ normunu aşağıdaki gibi hesaplayabiliriz.

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

Derin öğrenmede, kare $L_2$ normuyla daha sık çalışırız.
Ayrıca, vektör öğelerinin mutlak değerlerinin toplamı olarak ifade edilen [**$L_1$ *normu***] ile de sık karşılaşacaksınız:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

$L_2$ normuna kıyasla, sıradışı (aykırı) değerlerden daha az etkilenir.
$L_1$ normunu hesaplamak için, elemanların toplamı üzerinde mutlak değer fonksiyonunu oluştururuz.

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

Hem $L_2$ normu hem de $L_1$ normu, daha genel $L_p$ *normu*nun özel durumlarıdır:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

$L_2$ vektör normlarına benzer bir şekilde, [**$\mathbf{X} \in \mathbb{R}^{m \times n}$ matrisinin *Frobenius normu***], matris elemanlarının karelerin toplamının kare köküdür:

[**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

Frobenius normu, vektör normlarının tüm özelliklerini karşılar.
Matris şeklindeki bir vektörün bir $L_2$ normu gibi davranır.
Aşağıdaki işlevi çağırmak, bir matrisin Frobenius normunu hesaplar.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### Normlar ve Amaç Fonksiyonları
:label:`subsec_norms_and_objectives`

Kendimizi aşmak istemesek de, şimdiden bu kavramların neden faydalı olduğuna dair bazı sezgiler ekleyebiliriz.
Derin öğrenmede, genellikle optimizasyon (eniyileme) sorunlarını çözmeye çalışıyoruz: Gözlenen verilere atanan olasılığı *en üst düzeye çıkar*; tahminler ve gerçek-doğru gözlemler arasındaki mesafeyi *en aza indir*.
Benzer öğeler arasındaki mesafe en aza indirilecek ve benzer olmayan öğeler arasındaki mesafe en üst düzeye çıkarılacak şekilde öğelere (kelimeler, ürünler veya haber makaleleri gibi) vektör ifadeleri ata.
Çoğu zaman, amaç fonksiyonları, ki belki de derin öğrenme algoritmalarının (verilerin yanı sıra) en önemli bileşenleridir, normlar cinsinden ifade edilir.


## Doğrusal Cebir Hakkında Daha Fazlası

Sadece bu bölümde, modern derin öğrenmenin dikkate değer bir bölümünü anlamak için ihtiyaç duyacağınız tüm doğrusal cebiri öğrettik.
Doğrusal cebirde çok daha fazlası vardır ve daha fazla matematik makine öğrenmesi için yararlıdır.
Örneğin, matrisler faktörlere ayrıştırılabilir ve bu ayrışmalar gerçek dünya veri kümelerinde düşük boyutlu yapıları ortaya çıkarabilir.
Veri kümelerindeki yapıyı keşfetmek ve tahmin problemlerini çözmek için matris ayrıştırmalarına ve onların yüksek dereceli tensörlere genellemelerini kullanmaya odaklanan koca makine öğrenmesi alt alanları vardır.
Ancak bu kitap derin öğrenmeye odaklanmaktadır.
Gerçek veri kümelerinde faydalı makine öğrenmesi modelleri uygulayarak ellerinizi kirlettikten sonra daha fazla matematik öğrenmeye çok daha meyilli olacağınıza inanıyoruz.
Bu nedenle, daha sonra daha fazla matematiği öne sürme hakkımızı saklı tutarken, bu bölümü burada toparlayacağız.

Doğrusal cebir hakkında daha fazla bilgi edinmek istiyorsanız, şunlardan birine başvurabilirsiniz: [Doğrusal cebir işlemleri üzerine çevrimiçi ek](https://tr.d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) veya diğer mükemmel kaynaklar :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.


## Özet

* Skalerler, vektörler, matrisler ve tensörler doğrusal cebirdeki temel matematiksel nesnelerdir.
* Vektörler skaleri genelleştirir ve matrisler vektörleri genelleştirir.
* Skalerler, vektörler, matrisler ve tensörler sırasıyla sıfır, bir, iki ve rastgele sayıda eksene sahiptir.
* Bir tensör, belirtilen eksenler boyunca `toplam` ve `ortalama` ile indirgenebilir.
* İki matrisin eleman yönlü olarak çarpılmasına Hadamard çarpımı denir. Matris çarpımından farklıdır.
* Derin öğrenmede, genellikle $L_1$ normu, $L_2$ normu ve Frobenius normu gibi normlarla çalışırız.
* Skalerler, vektörler, matrisler ve tensörler üzerinde çeşitli işlemler gerçekleştirebiliriz.

## Alıştırmalar
1. $\mathbf{A}$'nın devrik bir matrisinin devrik işleminin $\mathbf{A}$, yani $(\mathbf{A}^\top)^\top = \mathbf{A}$ olduğunu kanıtlayın.
1. $\mathbf{A}$ ve $\mathbf{B}$ matrisleri verildiğinde, devriklerin toplamının bir toplamın devriğine eşit olduğunu gösterin: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Herhangi bir kare matris $\mathbf{A}$ verildiğinde, $\mathbf{A} + \mathbf{A}^\top$ her zaman simetrik midir? Neden?
1. Bu bölümde (2, 3, 4) şeklinde `X` tensörünü tanımladık. `len(X)` çıktısı nedir?
1. Rasgele şekilli bir tensör `X` için, `len(X)` her zaman belirli bir `X` ekseninin uzunluğuna karşılık gelir mi? Bu eksen nedir?
1. `A / A.sum(axis=1)` komutunu çalıştırın ve ne olduğuna bakın. Sebebini analiz edebilir misiniz?
1. Manhattan'da iki nokta arasında seyahat ederken, koordinatlar açısından, yani cadde ve sokak cinsinden, kat etmeniz gereken mesafe nedir? Çaprazlama seyahat edebilir misiniz?
1. (2, 3, 4) şekilli bir tensörü düşünün. 0, 1 ve 2 ekseni boyunca toplam çıktılarının şekilleri nelerdir?
1. 3 veya daha fazla eksenli bir tensörü `linalg.norm` fonksiyonuna besleyin ve çıktısını gözlemleyin. Bu işlev keyfi şekilli tansörler için ne hesaplar?

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/196)
:end_tab:
