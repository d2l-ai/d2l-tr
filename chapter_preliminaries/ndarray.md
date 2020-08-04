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

Similarly, we can create tensors with each element set to 1 as follows:

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

Often, we want to randomly sample the values for each element in a tensor from some probability distribution.
For example, when we construct arrays to serve as parameters in a neural network, we will typically initialize their values randomly.
The following snippet creates a tensor with shape (3, 4).
Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of 0 and a standard deviation of 1.

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

We can also specify the exact values for each element in the desired tensor by supplying a Python list (or list of lists) containing the numerical values.
Here, the outermost list corresponds to axis 0, and the inner list to axis 1.

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

## Operations

This book is not about software engineering.
Our interests are not limited to simply reading and writing data from/to arrays.
We want to perform mathematical operations on those arrays.
Some of the simplest and most useful operations are the *elementwise* operations.
These apply a standard scalar operation to each element of an array.
For functions that take two arrays as inputs, elementwise operations apply some standard binary operator on each pair of corresponding elements from the two arrays.
We can create an elementwise function from any function that maps from a scalar to a scalar.

In mathematical notation, we would denote such a *unary* scalar operator (taking one input) by the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
This just means that the function is mapping from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator (taking two real inputs, and yielding one output) by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, and a binary operator $f$, we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting $c_i \gets f(u_i, v_i)$ for all $i$, where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ by *lifting* the scalar function to an elementwise vector operation.

The common standard arithmetic operators (`+`, `-`, `*`, `/`, and `**`) have all been *lifted* to elementwise operations for any identically-shaped tensors of arbitrary shape.
We can call elementwise operations on any two tensors of the same shape.
In the following example, we use commas to formulate a 5-element tuple, where each element is the result of an elementwise operation.

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

Many more operations can be applied elementwise, including unary operators like exponentiation.

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

In addition to elementwise computations, we can also perform linear algebra operations, including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra (with no assumed prior knowledge) in :numref:`sec_linear-algebra`.

We can also *concatenate* multiple tensors together, stacking them end-to-end to form a larger tensor.
We just need to provide a list of tensors and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate two matrices along rows (axis 0, the first element of the shape) vs. columns (axis 1, the second element of the shape).
We can see that the first output tensor's axis-0 length ($6$) is the sum of the two input tensors' axis-0 lengths ($3 + 3$); while the second output tensor's axis-1 length ($8$) is the sum of the two input tensors' axis-1 lengths ($4 + 4$).

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

Sometimes, we want to construct a binary tensor via *logical statements*.
Take `x == y` as an example.
For each position, if `x` and `y` are equal at that position, the corresponding entry in the new tensor takes a value of 1, meaning that the logical statement `x == y` is true at that position; otherwise that position takes 0.

```{.python .input}
#@tab all
x == y
```

Summing all the elements in the tensor yields a tensor with only one element.

```{.python .input}
#@tab mxnet, pytorch
x.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x)
```

## Broadcasting Mechanism
:label:`subsec_broadcasting`

In the above section, we saw how to perform elementwise operations on two tensors of the same shape. Under certain conditions, even when shapes differ, we can still perform elementwise operations by invoking the *broadcasting mechanism*.
This mechanism works in the following way: First, expand one or both arrays by copying elements appropriately so that after this transformation, the two tensors have the same shape.
Second, carry out the elementwise operations on the resulting arrays.

In most cases, we broadcast along an axis where an array
initially only has length 1, such as in the following example:

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

Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively, their shapes do not match up if we want to add them.
We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows: for matrix `a` it replicates the columns and for matrix `b` it replicates the rows before adding up both elementwise.

```{.python .input}
#@tab all
a + b
```

## Indexing and Slicing

Just as in any other Python array, elements in a tensor can be accessed by index.
As in any Python array, the first element has index 0 and ranges are specified to include the first but *before* the last element.
As in standard Python lists, we can access elements according to their relative position to the end of the list by using negative indices.

Thus, `[-1]` selects the last element and `[1:3]` selects the second and the third elements as follows:

```{.python .input}
#@tab all
x[-1], x[1:3]
```

Beyond reading, we can also write elements of a matrix by specifying indices.

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

If we want to assign multiple elements the same value, we simply index all of them and then assign them the value.
For instance, `[0:2, :]` accesses the first and second rows, where `:` takes all the elements along axis 1 (column).
While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than 2 dimensions.

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

## Saving Memory

Running operations can cause new memory to be allocated to host results.
For example, if we write `y = x + y`, we will dereference the tensor that `y` used to point to and instead point `y` at the newly allocated memory.
In the following example, we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory.
After running `y = y + x`, we will find that `id(y)` points to a different location.
That is because Python first evaluates `y + x`, allocating new memory for the result and then makes `y` point to this new location in memory.

```{.python .input}
#@tab all
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons.
First, we do not want to run around allocating memory unnecessarily all the time.
In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second.
Typically, we will want to perform these updates *in place*.
Second, we might point at the same parameters from multiple variables.
If we do not update in place, other references will still point to the old memory location, making it possible for parts of our code to inadvertently reference stale parameters.

Fortunately, performing in-place operations in MXNet is easy.
We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`.
To illustrate this concept, we first create a new matrix `z` with the same shape as another `y`, using `zeros_like` to allocate a block of $0$ entries.

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

If the value of `x` is not reused in subsequent computations,
we can also use `x[:] = x + y` or `x += y`
to reduce the memory overhead of the operation.

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

## Conversion to Other Python Objects

Converting to a NumPy tensor, or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important: when you perform operations on the CPU or on GPUs, you do not want to halt computation, waiting to see whether the NumPy package of Python might want to be doing something else with the same chunk of memory.

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

To convert a size-1 tensor to a Python scalar, we can invoke the `item` function or Python's built-in functions.

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

## The `d2l` Package

Throughout the online version of this book, we will provide implementations of multiple frameworks.
However, different frameworks may be different in their API names or usage.
To better reuse the same code block across multiple frameworks, we unify a few commonly-used functions in the `d2l` package.
The comment `#@save` is a special mark where the following function, class, or statements are saved in the `d2l` package.
For instance, later we can directly invoke `d2l.numpy(a)` to convert a tensor `a`, which can be defined in any supported framework, into a NumPy tensor.

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

In the rest of the book, we often define more complicated functions or classes.
For those that can be used later, we will also save them in the `d2l` package so later they can be directly invoked without being redefined.


## Summary

* The main interface to store and manipulate data for deep learning is the tensor ($n$-dimensional array). It provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of tensor you can get.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab: