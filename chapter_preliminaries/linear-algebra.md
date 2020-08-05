# Doğrusal Cebir
:label:`sec_linear-algebra`

Şimdi verileri saklayabileceğinize ve oynama yapabileceğinize göre, bu kitapta yer alan modellerin çoğunu anlamanız ve uygulamanızda gereken temel doğrusal cebir bilgilerini kısaca gözden geçirelim.
Aşağıda, doğrusal cebirdeki temel matematiksel nesneleri, aritmetiği  ve işlemleri tanıtarak, bunların her birini matematiksel gösterim ve koddaki ilgili uygulama ile ifade ediyoruz.

## Sayıllar (Skalerler)

Doğrusal cebir veya makine öğrenmesi üzerine hiç çalışmadıysanız, matematikle ilgili geçmiş deneyiminiz muhtemelen bir seferde bir sayı düşünmekten ibaretti.
Ayrıca, bir çek defterini dengelediyseniz veya hatta bir restoranda akşam yemeği için ödeme yaptıysanız, zaten bir çift sayıyı toplama ve çarpma gibi temel şeyleri nasıl yapacağınızı zaten biliyorsunuzdur.
Örneğin, Palo Alto'daki sıcaklık $52$ Fahrenheit derecedir.
Usul olarak, sadece bir sayısal miktar içeren değerlere *sayıl (skaler)* diyoruz.
Bu değeri Celsius'a (metrik sistemin daha hassas sıcaklık ölçeği) dönüştürmek istiyorsanız, $f$i $52$ olarak ayarlayarak $c = \frac{5}{9}(f - 32)$ ifadesini hesaplarsınız.
Bu denklemde --- $5$, $9$ ve $32$ --- terimlerinin her biri skaler değerlerdir.
$c$ ve $f$ yer tutucularına (placeholders) *değişkenler* denir ve bilinmeyen skaler değerleri temsil ederler.

Bu kitapta, skaler değişkenlerin normal küçük harflerle (ör. $x$, $y$ ve $z$) gösterildiği matematiksel gösterimi kabul ediyoruz.
Tüm (sürekli) *gerçel değerli* skalerlerin alanını $\mathbb{R}$ ile belirtiyoruz.
Amaca uygun olarak, tam olarak *uzayın* ne olduğunu titizlikle tanımlayacağız, ancak şimdilik sadece $x \in \mathbb{R}$ ifadesinin $x$ değerinin gerçel değerli olduğunu söylemenin usula uygun bir yolu olduğunu unutmayın.
$\in$ sembolü "içinde" olarak telaffuz edilebilir ve sadece bir kümeye üyeliği belirtir.
Benzer şekilde, $x$ ve $y$'nin değerlerinin yalnızca $0$ veya $1$ olabilen rakamlar olduğunu belirtmek için $x, y \in \{0, 1 \}$ yazabiliriz.

Skaler, sadece bir elemente sahip bir tensör ile temsil edilir.
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

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant([3.0])
y = tf.constant([2.0])

x + y, x * y, x / y, x**y
```

## Vektörler (Yöneyler)

Bir vektörü basitçe skaler değerlerin bir listesi olarak düşünebilirsiniz.
Bu değerlere vektörün *elemanları* (*giriş değerleri* veya *bileşenleri*) diyoruz.
Vektörlerimiz veri kümemizdeki örnekleri temsil ettiğinde, değerleri gerçek dünyadaki önemini korur.
Örneğin, bir kredinin temerrüde düşme (ödenmeme) riskini tahmin etmek için bir model geliştiriyorsak, her başvuru sahibini, gelirine, istihdam süresine, önceki temerrüt (ödenmeme) sayısına ve diğer faktörlere karşılık gelen bileşenleri olan bir vektörle ilişkilendirebiliriz.
Hastanedeki hastaların potansiyel olarak yaşayabilecekleri kalp krizi riskini araştırıyor olsaydık, her hastayı bileşenleri en güncel hayati belirtilerini, kolesterol seviyelerini, günlük egzersiz dakikalarını vb. içeren bir vektörle temsil edebiliriz.
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
Örneğin, $\mathbf{x}$in $i.$ elemanını $x_i$ ile ifade edebiliriz.
$x_i$ öğesinin bir skaler olduğuna dikkat edin, bu nedenle ondan bahsederken fontta kalın yazı tipi kullanmıyoruz.
Genel literatür sütun vektörlerini vektörlerin varsayılan yönü olarak kabul eder, bu kitap da öyle.
Matematikte, $\mathbf{x}$ vektörü şu şekilde yazılabilir:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

burada $x_1, \ldots, x_n$ vektörün elemanlarıdır.
Kod olarak, herhangi bir öğeye onu tensöre indeksleyerek erişiriz.

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
Ve her dizinin bir uzunluğu olduğu gibi, her vektör de bir uzunluğa sahiptir.
Matematiksel gösterimde, $\mathbf{x}$ vektörünün $n$ gerçel değerli skalerlerden oluştuğunu söylemek istiyorsak, bunu $\mathbf{x} \in \mathbb{R}^n$ olarak ifade edebiliriz.
Bir vektörün uzunluğuna genel olarak vektörün *boyutu* denir.

Sıradan bir Python dizisinde olduğu gibi, Python'un yerleşik (built-in) `len()` işlevini çağırarak bir tensörün uzunluğuna erişebiliriz.

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
Sadece bir ekseni olan tensörler için, şeklin sadece bir elemanı vardır.

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

"Boyut" kelimesinin bu bağlamlarda aşırı yüklenme eğiliminde olduğunu ve bunun insanları şaşırtma eğiliminde olduğunu unutmayın.
Açıklığa kavuşturmak için, bir *vektörün*  veya *ekseninin* boyutluluğunu onun uzunluğuna atıfta bulunmak için kullanırız; yani bir vektörün veya eksenin eleman sayısı.
Bununla birlikte, bir tensörün boyutluluğunu, bir tensörün sahip olduğu eksen sayısını ifade etmek için kullanırız.
Bu anlamda, bir tensörün bazı eksenlerinin boyutluluğu, bu eksenin uzunluğu olacaktır.

## Matrisler

Vektörler, skalerleri sıfırdan birinci dereceye kadar genelleştirirken, matrisler de vektörleri birinci dereceden ikinci dereceye genelleştirir.
Genellikle kalın, büyük harflerle (örn., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$) göstereceğimiz matrisler, kodda iki eksenli tensörler olarak temsil edilir.

Matematiksel gösterimde, $\mathbf{A}$ matrisinin gerçel değerli skaler $m$ satır ve $n$ sütundan oluştuğunu ifade etmek için $\mathbf{A} \in \mathbb{R}^{m \times n}$i kullanırız .
Görsel olarak, herhangi bir $\mathbf{A} \in \mathbb{R}^{m \times n}$ matrisini $a_{ij}$ öğesinin $i.$ satıra ve $j.$ sütuna ait olduğu tablo olarak gösterebiliriz  :

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

Herhangi bir $\mathbf{A} \in \mathbb{R}^{m \times n} için, $\mathbf{A}$ ($m$, $n$) veya $m \times n$ şeklindedir.
Özellikle, bir matris aynı sayıda satır ve sütuna sahip olduğunda, şekli bir kareye dönüşür; dolayısıyla buna *kare matris* denir.

Bir tensörü örneklerken, en sevdiğimiz işlevlerden herhangi birini çağırıp $m$ ve $n$ iki bileşeninden oluşan bir şekil belirterek $m \times n$ matrisi oluşturabiliriz.

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
Bir $\mathbf{A}$ matrisinin skaler elemanları verilmediğinde, örneğin :eqref:`eq_matrix_def` gibi, basitçe $\mathbf{A}$ matrisinin küçük harfli altindislisi $a_{ij}$yi kullanarak $[\mathbf{A}]_{ij}$'ye atıfta bulunuruz.
Gösterimi basit tutarken indeksleri ayırmak için virgüller yalnızca gerekli olduğunda eklenir, örneğin $a_{2, 3j}$ ve $[\mathbf{A}]_{2i-1, 3}$ gibi.


Bazen eksenleri ters çevirmek isteriz.
Bir matrisin satırlarını ve sütunlarını değiştirdiğimizde çıkan sonuç matrisine *devrik (transpose)* adı verilir.
Usul olarak, bir $\mathbf{A}$'nin devriğini $\mathbf{A}^\top$ ile gösteririz ve eğer $\mathbf{B} = \mathbf{A}^\top$ ise herhangi bir $i$ and $j$ için $b_{ij} = a_{ji}$'dir.
Bu nedenle, :eqref:`eq_matrix_def`deki $\mathbf{A}$'in devriği bir $n\times m$ matristir:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Şimdi kodda bir matrisin devriğine erişiyoruz.

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

Kare matrisin özel bir türü olarak, bir *simetrik matris* $\mathbf{A}$, devriğine eşittir: $\mathbf{A} = \mathbf{A}^\top$.
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

Matrisler yararlı veri yapılarıdır: farklı değişim (varyasyon) kiplerine (modalite) sahip verileri düzenlememize izin verirler.
Örneğin, matrisimizdeki satırlar farklı evlere (veri örneklerine veya veri noktalarına) karşılık gelirken, sütunlar farklı özniteliklere karşılık gelebilir.
Daha önce elektronik tablo yazılımı kullandıysanız veya şunu okuduysanız :numref:`sec_pandas`, bu size tanıdık gelecektir .
Bu nedenle, tek bir vektörün varsayılan yönü bir sütun vektörü olmasına rağmen, bir tablo veri kümesini temsil eden bir matriste, her veri örneğini matristeki bir satır vektörü olarak ele almak daha gelenekseldir.
Ve sonraki bölümlerde göreceğimiz gibi, bu düzen ortak derin öğrenme tatbikatlarını mümkün kılacaktır.
Örneğin, bir tensörün en dış ekseni boyunca, veri örneklerinin mini-grup'larına veya  mini-grup yoksa yalnızca veri örneklerine erişebilir veya bir bir sayabiliriz.

## Tensörler (Gereyler)

Vektörlerin skalerleri genellemesi ve matrislerin vektörleri genellemesi gibi, daha fazla eksenli veri yapıları oluşturabiliriz.
Tensörler (bu alt bölümdeki "tensörler" cebirsel nesnelere atıfta bulunur) bize rastgele sayıda ekseni olan $n$-boyutlu dizileri tanımlamanın genel bir yolunu verir. Vektörler, örneğin, birinci dereceden tensörlerdir ve matrisler ikinci dereceden tensörlerdir.
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

## Basic Properties of Tensor Arithmetic

Scalars, vectors, matrices, and tensors ("tensors" in this subsection refer to algebraic objects) of an arbitrary number of axes have some nice properties that often come in handy.
For example, you might have noticed from the definition of an elementwise operation that any elementwise unary operation does not change the shape of its operand.
Similarly, given any two tensors with the same shape, the result of any binary elementwise operation will be a tensor of that same shape.
For example, adding two matrices of the same shape performs elementwise addition over these two matrices.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

Specifically, elementwise multiplication of two matrices is called their *Hadamard product* (math notation $\odot$).
Consider matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ whose element of row $i$ and column $j$ is $b_{ij}$. The Hadamard product of matrices $\mathbf{A}$ (defined in :eqref:`eq_matrix_def`) and $\mathbf{B}$

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

Multiplying or adding a tensor by a scalar also does not change the shape of the tensor, where each element of the operand tensor will be added or multiplied by the scalar.

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

## Reduction
:label:`subseq_lin-alg-reduction`

One useful operation that we can perform with arbitrary tensors is to calculate the sum of their elements.
In mathematical notation, we express sums using the $\sum$ symbol.
To express the sum of the elements in a vector $\mathbf{x}$ of length $d$, we write $\sum_{i=1}^d x_i$.
In code, we can just call the function for calculating the sum.

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

We can express sums over the elements of tensors of arbitrary shape.
For example, the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

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

By default, invoking the function for calculating the sum *reduces* a tensor along all its axes to a scalar.
We can also specify the axes along which the tensor is reduced via summation.
Take matrices as an example.
To reduce the row dimension (axis 0) by summing up elements of all the rows, we specify `axis=0` when invoking the function.
Since the input matrix reduces along axis 0 to generate the output vector, the dimension of axis 0 of the input is lost in the output shape.

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

Specifying `axis=1` will reduce the column dimension (axis 1) by summing up elements of all the columns.
Thus, the dimension of axis 1 of the input is lost in the output shape.

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

Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix.

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

A related quantity is the *mean*, which is also called the *average*.
We calculate the mean by dividing the sum by the total number of elements.
In code, we could just call the function for calculating the mean on tensors of arbitrary shape.

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

Likewise, the function for calculating the mean can also reduce a tensor along the specified axes.

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

### Non-Reduction Sum
:label:`subseq_lin-alg-non-reduction`

However, sometimes it can be useful to keep the number of axes unchanged when invoking the function for calculating the sum or mean.

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

For instance, since `sum_A` still keeps its two axes after summing each row, we can divide `A` by `sum_A` with broadcasting.

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

If we want to calculate the cumulative sum of elements of `A` along some axis, say `axis=0` (row by row), we can call the `cumsum` function. This function will not reduce the input tensor along any axis.

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

## Dot Products

So far, we have only performed elementwise operations, sums, and averages. And if this was all we could do, linear algebra probably would not deserve its own section. However, one of the most fundamental operations is the dot product. Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their *dot product* $\mathbf{x}^\top \mathbf{y}$ (or $\langle \mathbf{x}, \mathbf{y}  \rangle$) is a sum over the products of the elements at the same position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

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

Note that we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:

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

Dot products are useful in a wide range of contexts.
For example, given some set of values, denoted by a vector $\mathbf{x}  \in \mathbb{R}^d$ and a set of weights denoted by $\mathbf{w} \in \mathbb{R}^d$, the weighted sum of the values in $\mathbf{x}$ according to the weights $\mathbf{w}$
could be expressed as the dot product $\mathbf{x}^\top \mathbf{w}$.
When the weights are non-negative and sum to one (i.e., $\left(\sum_{i=1}^{d} {w_i} = 1\right)$), the dot product expresses a *weighted average*.
After normalizing two vectors to have the unit length, the dot products express the cosine of the angle between them.
We will formally introduce this notion of *length* later in this section.


## Matrix-Vector Products

Now that we know how to calculate dot products, we can begin to understand *matrix-vector products*.
Recall the matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and the vector $\mathbf{x} \in \mathbb{R}^n$ defined and visualized in :eqref:`eq_matrix_def` and :eqref:`eq_vec_def` respectively.
Let us start off by visualizing the matrix $\mathbf{A}$ in terms of its row vectors

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ is a row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$.
The matrix-vector product $\mathbf{A}\mathbf{x}$ is simply a column vector of length $m$, whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \mathbf{x}$:

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

We can think of multiplication by a matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$ as a transformation that projects vectors from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.
These transformations turn out to be remarkably useful.
For example, we can represent rotations as multiplications by a square matrix.
As we will see in subsequent chapters, we can also use matrix-vector products to describe the most intensive calculations required when computing each layer in a neural network given the values of the previous layer.

Expressing matrix-vector products in code with tensors, we use the same `dot` function as for dot products.
When we call `np.dot(A, x)` with a matrix `A` and a vector `x`, the matrix-vector product is performed.
Note that the column dimension of `A` (its length along axis 1) must be the same as the dimension of `x` (its length).

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

## Matrix-Matrix Multiplication

If you have gotten the hang of dot products and matrix-vector products, then *matrix-matrix multiplication* should be straightforward.

Say that we have two matrices $\mathbf{A} \in \mathbb{R}^{n \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times m}$:

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


Denote by $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ the row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$, and let $\mathbf{b}_{j} \in \mathbb{R}^k$ be the column vector from the $j^\mathrm{th}$ column of the matrix $\mathbf{B}$.
To produce the matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$, it is easiest to think of $\mathbf{A}$ in terms of its row vectors and $\mathbf{B}$ in terms of its column vectors:

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


Then the matrix product $\mathbf{C} \in \mathbb{R}^{n \times m}$ is produced as we simply compute each element $c_{ij}$ as the dot product $\mathbf{a}^\top_i \mathbf{b}_j$:

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

We can think of the matrix-matrix multiplication $\mathbf{AB}$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \times m$ matrix.
In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with 5 rows and 4 columns, and `B` is a matrix with 4 rows and 3 columns.
After multiplication, we obtain a matrix with 5 rows and 3 columns.

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

Matrix-matrix multiplication can be simply called *matrix multiplication*, and should not be confused with the Hadamard product.


## Norms

Some of the most useful operators in linear algebra are *norms*.
Informally, the norm of a vector tells us how *big* a vector is.
The notion of *size* under consideration here concerns not dimensionality but rather the magnitude of the components.

In linear algebra, a vector norm is a function $f$ that maps a vector to a scalar, satisfying a handful of properties.
Given any vector $\mathbf{x}$, the first property says that if we scale all the elements of a vector by a constant factor $\alpha$, its norm also scales by the *absolute value* of the same constant factor:

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$


The second property is the familiar triangle inequality:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$


The third property simply says that the norm must be non-negative:

$$f(\mathbf{x}) \geq 0.$$

That makes sense, as in most contexts the smallest *size* for anything is 0.
The final property requires that the smallest norm is achieved and only achieved by a vector consisting of all zeros.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

You might notice that norms sound a lot like measures of distance.
And if you remember Euclidean distances (think Pythagoras' theorem) from grade school, then the concepts of non-negativity and the triangle inequality might ring a bell.
In fact, the Euclidean distance is a norm: specifically it is the $\ell_2$ norm.
Suppose that the elements in the $n$-dimensional vector $\mathbf{x}$ are $x_1, \ldots, x_n$.
The $\ell_2$ *norm* of $\mathbf{x}$ is the square root of the sum of the squares of the vector elements:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

where the subscript $2$ is often omitted in $\ell_2$ norms, i.e., $\|\mathbf{x}\|$ is equivalent to $\|\mathbf{x}\|_2$. In code, we can calculate the $\ell_2$ norm of a vector as follows.

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

In deep learning, we work more often
with the squared $\ell_2$ norm.
You will also frequently encounter the $\ell_1$ *norm*,
which is expressed as the sum of the absolute values of the vector elements:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

As compared with the $\ell_2$ norm, it is less influenced by outliers.
To calculate the $\ell_1$ norm, we compose the absolute value function with a sum over the elements.

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

Both the $\ell_2$ norm and the $\ell_1$ norm are special cases of the more general $\ell_p$ *norm*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Analogous to $\ell_2$ norms of vectors, the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the square root of the sum of the squares of the matrix elements:

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

The Frobenius norm satisfies all the properties of vector norms.
It behaves as if it were an $\ell_2$ norm of a matrix-shaped vector.
Invoking the following function will calculate the Frobenius norm of a matrix.

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

### Norms and Objectives
:label:`subsec_norms_and_objectives`

While we do not want to get too far ahead of ourselves, we can plant some intuition already about why these concepts are useful.
In deep learning, we are often trying to solve optimization problems: *maximize* the probability assigned to observed data; *minimize* the distance between predictions and the ground-truth observations.
Assign vector representations to items (like words, products, or news articles) such that the distance between similar items is minimized, and the distance between dissimilar items is maximized.
Oftentimes, the objectives, perhaps the most important components of deep learning algorithms (besides the data), are expressed as norms.



## More on Linear Algebra

In just this section, we have taught you all the linear algebra that you will need to understand a remarkable chunk of modern deep learning.
There is a lot more to linear algebra and a lot of that mathematics is useful for machine learning.
For example, matrices can be decomposed into factors, and these decompositions can reveal low-dimensional structure in real-world datasets.
There are entire subfields of machine learning that focus on using matrix decompositions and their generalizations to high-order tensors to discover structure in datasets and solve prediction problems.
But this book focuses on deep learning.
And we believe you will be much more inclined to learn more mathematics once you have gotten your hands dirty deploying useful machine learning models on real datasets.
So while we reserve the right to introduce more mathematics much later on, we will wrap up this section here.

If you are eager to learn more about linear algebra, you may refer to either :numref:`sec_geometry-linear-algebraic-ops` or other excellent resources :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.



## Summary

* Scalars, vectors, matrices, and tensors are basic mathematical objects in linear algebra.
* Vectors generalize scalars, and matrices generalize vectors.
* Scalars, vectors, matrices, and tensors have zero, one, two, and an arbitrary number of axes, respectively.
* A tensor can be reduced along the specified axes by `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
* In deep learning, we often work with norms such as the $\ell_1$ norm, the $\ell_2$ norm, and the Frobenius norm.
* We can perform a variety of operations over scalars, vectors, matrices, and tensors.

## Exercises

1. Prove that the transpose of a matrix $\mathbf{A}$'s transpose is $\mathbf{A}$: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that the sum of transposes is equal to the transpose of a sum: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Why?
1. We defined the tensor `X` of shape (2, 3, 4) in this section. What is the output of `len(X)`?
1. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
1. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the reason?
1. When traveling between two points in Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
1. Consider a tensor with shape (2, 3, 4). What are the shapes of the summation outputs along axis 0, 1, and 2?
1. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for tensors of arbitrary shape?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
