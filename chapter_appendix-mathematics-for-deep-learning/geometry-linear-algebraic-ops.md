# Geometri ve Doğrusal Cebirsel İşlemler
:label:`sec_geometry-linear-algebraic-ops`

:numref:`sec_linear-cebebra`da, doğrusal cebirin temelleriyle karşılaştık ve verilerimizi dönüştürürken genel işlemleri ifade etmek için nasıl kullanılabileceğini gördük. Doğrusal cebir, derin öğrenmede ve daha geniş anlamda makine öğrenmesinde yaptığımız işlerin çoğunun altında yatan temel matematiksel sütunlardan biridir. :numref:`sec_linear-cebebra`, modern derin öğrenme modellerinin mekaniğini iletmek için yeterli mekanizmayı içerirken, konuyla ilgili daha çok şey var.
Bu bölümde daha derine ineceğiz, doğrusal cebir işlemlerinin bazı geometrik yorumlarını vurgulayacağız ve özdeğerler (eigenvalues) ve özvektörler (eigenvectors) dahil birkaç temel kavramı tanıtacağız.

## Vektörlerin Geometrisi
İlk olarak, uzaydaki noktalar veya yönler olarak vektörlerin iki ortak geometrik yorumunu tartışmamız gerekiyor.
Temel olarak bir vektör, aşağıdaki Python listesi gibi bir sayı listesidir.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

Matematikçiler bunu genellikle bir *sütun* veya *satır* vektörü olarak yazarlar;

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

veya

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$ 

Bunlar genellikle veri noktalarının sütun vektörleri ve ağırlıklı toplamları oluşturmada kullanılan ağırlıkların satır vektörleri olduğu farklı yorumlara sahiptir.
Ancak esnek olmak faydalı olabilir.
Matrisler yararlı veri yapılarıdır: Değişimlerinde farklı model modlarına (gösterimlerine) sahip verileri düzenlememize izin verirler. Örneğin, matrisimizdeki satırlar farklı evlere (veri noktalarına) karşılık gelirken, sütunlar farklı özelliklere karşılık gelebilir. Daha önce elektronik tablo yazılımı kullandıysanız veya :numref:`sec_pandas`'i okuduysanız, bu size tanıdık gelecektir. Bu nedenle, tek bir vektörün varsayılan yönü bir sütun vektörü olmasına rağmen, bir çizelgesel veri kümesini temsil eden bir matriste, her veri noktasını matristeki bir satır vektörü olarak ele almak daha gelenekseldir. Sonraki bölümlerde göreceğimiz gibi, bu düzen ortak derin öğrenme uygulamalarını mümkün kılacaktır. Örneğin, bir tensörün en dış ekseni boyunca, veri noktalarının mini gruplarına veya mini grup yoksa sadece veri noktalarına erişebilir veya bunları numaralandırabiliriz.

Bir vektör verildiğinde, ona vermemiz gereken ilk yorum uzayda bir nokta olduğudur.
İki veya üç boyutta, bu noktaları, *köken (orijin)* adı verilen sabit bir referansa kıyasla uzaydaki konumlarını belirtmek için vektör bileşenlerini kullanarak görselleştirebiliriz. Bu, şurada görülebilir :numref:`fig_grid`.

![Vektörleri düzlemdeki noktalar olarak görselleştirmenin bir örneği. Vektörün ilk bileşeni $x$ koordinatını verir, ikinci bileşen $y$ koordinatını verir. Görselleştirilmesi çok daha zor olsa da, daha yüksek boyutlar da benzerdir.](../img/GridPoints.svg)
:label:`fig_grid`

Bu geometrik bakış açısı, sorunu daha soyut bir düzeyde ele almamızı sağlar.
Artık resimleri kedi veya köpek olarak sınıflandırmak gibi başa çıkılmaz görünen bir problemle karşılaşmadığımızdan, görevleri soyut olarak uzaydaki nokta toplulukları olarak değerlendirmeye ve görevi iki farklı nokta kümesini nasıl ayıracağımızı keşfetmek olarak resmetmeye başlayabiliriz.

Buna paralel olarak, insanların genellikle vektörleri aldıkları ikinci bir bakış açısı vardır: Uzayda yönler olarak.
$\mathbf{v} = [2,3]^\top$ vektörünü başlangıç noktasından $2$ birim sağda ve $3$ birim yukarıda bir konum olarak düşünmekle kalmayabiliriz, aynı zamanda onu sağa doğru $2$ adım ve yukarı doğru $3$ adım şekilde yönün kendisi olarak da düşünebiliriz.
Bu şekilde, şekildeki tüm vektörleri aynı kabul ederiz :numref:`fig_arrow`.

![Herhangi bir vektör, düzlemde bir ok olarak görselleştirilebilir. Bu durumda, çizilen her vektör $(2,3)$ vektörünün bir temsilidir.](../img/ParVec.svg)
:label:`fig_arrow`

Bu değisik gösterimin faydalarından biri, vektör toplama işlemini görsel olarak anlamlandırabilmemizdir.
Özellikle, bir vektör tarafından verilen yönleri izliyoruz ve sonra diğerinin verdiği yönleri takip ediyoruz, şekilde görüldüğü gibi :numref:`fig_add-vec`.

![Önce bir vektörü, sonra diğerini takip ederek vektör toplamayı görselleştirebiliriz.](../img/VecAdd.svg)
:label:`fig_add-vec`

Vektör çıkarma işleminin benzer bir yorumu vardır.
$\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$ özdeşliğini göz önünde bulundurursak, $\mathbf{u} - \mathbf{v}$ vektörü, bizi $\mathbf{u}$ noktasından $\mathbf{v}$ noktasına götüren yöndür.


## Nokta (İç) Çarpımları ve Açılar
:numref:`sec_linear-algebra`da gördüğümüz gibi, $\mathbf{u}$ ve $\mathbf{v}$ gibi iki sütun vektörü alırsak, bunların nokta çarpımını aşağıdaki işlemi hesaplayarak oluşturabiliriz:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

:eqref:`eq_dot_def` simetrik olduğundan, klasik çarpmanın gösterimini kopyalayacağız ve şöyle yazacağız:

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

Böylece vektörlerin sırasını değiştirmenin aynı cevabı vereceği gerçeğini vurgulamış olacağız.

İç çarpım :eqref:`eq_dot_def` ayrıca geometrik bir yorumu da kabul eder: O da iki vektör arasındaki açı ile yakından ilgilidir. :numref:`fig_angle`da gösterilen açıyı düşünün.

![Düzlemdeki herhangi iki vektör arasında iyi tanımlanmış bir $\theta$ açısı vardır. Bu açının iç çarpıma yakından bağlı olduğunu göreceğiz.](../img/VecAngle.svg)
:label:`fig_angle`

Başlamak için iki belli vektörü ele alalım:

$$
\mathbf{v} = (r,0) \; \text{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

$\mathbf{v}$ vektörü $r$ uzunluğundadır ve $x$ eksenine paralel uzanır, $\mathbf{w}$ vektörü $s$ uzunluğundadır ve $x$ ekseni ile arasında $\theta$ açısı vardır.
Bu iki vektörün iç çarpımını hesaplarsak, şunu görürüz:

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

Bazı basit cebirsel işlemlerle, terimleri yeniden düzenleyebiliriz.

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

Kısacası, bu iki belli vektör için, normlarla birleştirilmiş iç çarpım bize iki vektör arasındaki açıyı söyler. Aynı gerçek genel olarak doğrudur. Burada ifade türetmeyeceğiz, ancak $\|\mathbf{v} - \mathbf{w}\|^2$'ı iki şekilde yazmayı düşünürsek, biri nokta çarpımı ile, diğeri geometrik olarak kosinüsler yasasının kullanımı ile, tam ilişkiyi elde edebiliriz.
Gerçekten de, herhangi iki vektör, $\mathbf{v}$ ve $\mathbf{w}$ için, aralarındaki açı:

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

Hesaplamadaki hiçbir şey iki boyutluluğu referans almadığı için bu güzel bir sonuçtur.
Aslında bunu üç veya üç milyon boyutta da sorunsuz olarak kullanabiliriz.

Basit bir örnek olarak, bir çift vektör arasındaki açıyı nasıl hesaplayacağımızı görelim:

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

Şu anda kullanmayacağız, ancak açılarının $\pi/2$ (veya eşdeğer olarak $90^{\circ}$) olduğu vektörleri *dik* olarak isimlendireceğimizi bilmekte fayda var.
Yukarıdaki denklemi inceleyerek, bunun $\theta = \pi/2$ olduğunda gerçekleştiğini görürüz, bu $\cos(\theta) = 0$ ile aynı şeydir.
Bunun gerçekleşmesinin tek yolu, nokta çarpımın kendisinin sıfır olmasıdır ve ancak ve ancak $\mathbf{v}\cdot\mathbf{w} = 0$ ise iki vektörün dik olur.
Bu, nesneleri geometrik olarak anlarken faydalı bir formül olacaktır.  

Şu soruyu sormak mantıklıdır: Açıyı hesaplamak neden yararlıdır?
Cevap, verinin sahip olmasını beklediğimiz türden değişmezlikten gelir.
Bir görüntü ve her piksel değerinin aynı, ancak parlaklığın $\% 10$ olduğu kopya bir görüntü düşünün.
Tek tek piksellerin değerleri genel olarak asıl değerlerden uzaktır.
Bu nedenle, hakiki görüntü ile daha karanlık olan arasındaki mesafe hesaplanırsa, mesafe büyük olabilir.
Gene de, çoğu makine öğrenmesi uygulaması için *içerik* aynıdır---kedi/köpek sınıflandırıcısı söz konusu olduğunda yine de bir kedinin görüntüsüdür.
Ancak, açıyı düşünürsek, herhangi bir $\mathbf{v}$ vektörü için $\mathbf{v}$ ve $0.1\cdot\mathbf{v}$ arasındaki açının sıfır olduğunu görmek zor değildir.
Bu, ölçeklemenin vektörlerin yönlerini koruduğu ve sadece uzunluğu değiştirdiği gerçeğine karşılık gelir.
Açı, koyu görüntüyü aynı kabul edecektir.

Buna benzer örnekler her yerdedir.
Metinde, aynı şeyleri söyleyen iki kat daha uzun bir belge yazarsak tartışılan konunun değişmemesini isteyebiliriz.
Bazı kodlamalar için (herhangi sözcük haznesindeki kelimelerin kaç kere geçtiğinin sayılması gibi), bu, belgeyi kodlayan vektörün ikiye çarpılmasına karşılık gelir ki, böylece yine açıyı kullanabiliriz.

### Cosine Similarity
In ML contexts where the angle is employed to measure the closeness of two vectors, practitioners adopt the term *cosine similarity* to refer to the portion 
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

The cosine takes a maximum value of $1$ when the two vectors point in the same direction, a minimum value of $-1$ when they point in opposite directions, and a value of $0$ when the two vectors are orthogonal.
Note that if the components of high-dimensional vectors are sampled randomly with mean $0$, their cosine will nearly always be close to $0$.


## Hyperplanes

In addition to working with vectors, another key object that you must understand to go far in linear algebra is the *hyperplane*, a generalization to higher dimensions of a line (two dimensions) or of a plane (three dimensions).
In an $d$-dimensional vector space, a hyperplane has $d-1$ dimensions and divides the space into two half-spaces. 

Let us start with an example.
Suppose that we have a column vector $\mathbf{w}=[2,1]^\top$. We want to know, "what are the points $\mathbf{v}$ with $\mathbf{w}\cdot\mathbf{v} = 1$?"
By recalling the connection between dot products and angles above :eqref:`eq_angle_forumla`, we can see that this is equivalent to 
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![Recalling trigonometry, we see the formula $\|\mathbf{v}\|\cos(\theta)$ is the length of the projection of the vector $\mathbf{v}$ onto the direction of $\mathbf{w}$](../img/ProjVec.svg)
:label:`fig_vector-project`

If we consider the geometric meaning of this expression, we see that this is equivalent to saying that the length of the projection of $\mathbf{v}$ onto the direction of $\mathbf{w}$ is exactly $1/\|\mathbf{w}\|$, as is shown in :numref:`fig_vector-project`. 
The set of all points where this is true is a line at right angles to the vector $\mathbf{w}$.
If we wanted, we could find the equation for this line and see that it is $2x + y = 1$ or equivalently $y = 1 - 2x$.

If we now look at what happens when we ask about the set of points with $\mathbf{w}\cdot\mathbf{v} > 1$ or $\mathbf{w}\cdot\mathbf{v} < 1$, we can see that these are cases where the projections are longer or shorter than $1/\|\mathbf{w}\|$, respectively.
Thus, those two inequalities define either side of the line.
In this way, we have found a way to cut our space into two halves, where all the points on one side have dot product below a threshold, and the other side above as we see in :numref:`fig_space-division`.

![If we now consider the inequality version of the expression, we see that our hyperplane (in this case: just a line) separates the space into two halves.](../img/SpaceDivision.svg)
:label:`fig_space-division`

The story in higher dimension is much the same.
If we now take $\mathbf{w} = [1,2,3]^\top$ and ask about the points in three dimensions with $\mathbf{w}\cdot\mathbf{v} = 1$, we obtain a plane at right angles to the given vector $\mathbf{w}$.
The two inequalities again define the two sides of the plane as is shown in :numref:`fig_higher-division`.

![Hyperplanes in any dimension separate the space into two halves.](../img/SpaceDivision3D.svg)
:label:`fig_higher-division`

While our ability to visualize runs out at this point, nothing stops us from doing this in tens, hundreds, or billions of dimensions.
This occurs often when thinking about machine learned models.
For instance, we can understand linear classification models like those from :numref:`sec_softmax`, as methods to find hyperplanes that separate the different target classes.
In this context, such hyperplanes are often referred to as *decision planes*.
The majority of deep learned classification models end with a linear layer fed into a softmax, so one can interpret the role of the deep neural network to be to find a non-linear embedding such that the target classes can be separated cleanly by hyperplanes.

To give a hand-built example, notice that we can produce a reasonable model to classify tiny images of t-shirts and trousers from the Fashion MNIST dataset (seen in :numref:`sec_fashion_mnist`) by just taking the vector between their means to define the decision plane and eyeball a crude threshold.  First we will load the data and compute the averages.

```{.python .input}
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

It can be informative to examine these averages in detail, so let us plot what they look like.  In this case, we see that the average indeed resembles a blurry image of a t-shirt.

```{.python .input}
#@tab all
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

In the second case, we again see that the average resembles a blurry image of trousers.

```{.python .input}
#@tab all
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

In a fully machine learned solution, we would learn the threshold from the dataset. In this case, I simply eyeballed a threshold that looked good on the training data by hand.

```{.python .input}
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean(predictions.type(y_test.dtype) == y_test, dtype=torch.float64)
```

## Geometry of Linear Transformations

Through :numref:`sec_linear-algebra` and the above discussions, we have a solid understanding of the geometry of vectors, lengths, and angles. 
However, there is one important object we have omitted discussing, and that is a geometric understanding of linear transformations represented by matrices. Fully internalizing what matrices can do to transform data between two potentially different high dimensional spaces takes significant practice, and is beyond the scope of this appendix. 
However, we can start building up intuition in two dimensions.

Suppose that we have some matrix:

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

If we want to apply this to an arbitrary vector 
$\mathbf{v} = [x, y]^\top$, 
we multiply and see that

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

This may seem like an odd computation, where something clear became somewhat impenetrable.
However, it tells us that we can write the way that a matrix transforms *any* vector in terms of how it transforms *two specific vectors*: $[1,0]^\top$ and $[0,1]^\top$. 
This is worth considering for a moment. 
We have essentially reduced an infinite problem (what happens to any pair of real numbers) to a finite one (what happens to these specific vectors).
These vectors are an example a *basis*, where we can write any vector in our space as a weighted sum of these *basis vectors*.

Let us draw what happens when we use the specific matrix

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

If we look at the specific vector $\mathbf{v} = [2, -1]^\top$, we see this is $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$, and thus we know that the matrix $A$ will send this to $2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
If we follow this logic through carefully, say by considering the grid of all integer pairs of points, we see that what happens is that the matrix multiplication can skew, rotate, and scale the grid, but the grid structure must remain as you see in :numref:`fig_grid-transform`.

![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.](../img/GridTransform.svg)
:label:`fig_grid-transform`

This is the most important intuitive point to internalize about linear transformations represented by matrices.
Matrices are incapable of distorting some parts of space differently than others.
All they can do is take the original coordinates on our space and skew, rotate, and scale them.

Some distortions can be severe.  For instance the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

compresses the entire two-dimensional plane down to a single line.
Identifying and working with such transformations are the topic of a later section, but geometrically we can see that this is fundamentally different from the types of transformations we saw above. 
For instance, the result from matrix $\mathbf{A}$ can be "bent back" to the original grid. The results from matrix $\mathbf{B}$ cannot because we will never know where the vector $[1,2]^\top$ came from---was it $[1,1]^\top$ or $[0, -1]^\top$?

While this picture was for a $2\times2$ matrix, nothing prevents us from taking the lessons learned into higher dimensions.
If we take similar basis vectors like $[1,0, \ldots,0]$ and see where our matrix sends them, we can start to get a feeling for how the matrix multiplication distorts the entire space in whatever dimension space we are dealing with.

## Linear Dependence

Consider again the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

This compresses the entire plane down to live on the single line $y = 2x$. The question now arises: is there some way we can detect this just looking at the matrix itself?
The answer is that indeed we can.
Let us take $\mathbf{b}_1 = [2,4]^\top$ and $\mathbf{b}_2 = [-1, -2]^\top$ be the two columns of $\mathbf{B}$.
Remember that we can write everything transformed by the matrix $\mathbf{B}$ as a weighted sum of the columns of the matrix: like $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
We call this a *linear combination*. 
The fact that $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ means that we can write any linear combination of those two columns entirely in terms of say $\mathbf{b}_2$ since

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

This means that one of the columns is, in a sense, redundant because it does not define a unique direction in space. 
This should not surprise us too much  since we already saw that this matrix collapses the entire plane down into a single line.
Moreover, we see that the linear dependence $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ captures this. 
To make this more symmetrical between the two vectors, we will write this as

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

In general, we will say that a collection of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are *linearly dependent* if there exist coefficients $a_1, \ldots, a_k$ *not all equal to zero* so that

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

In this case, we can solve for one of the vectors in terms of some combination of the others, and effectively render it redundant.
Thus, a linear dependence in the columns of a matrix is a witness to the fact that our matrix is compressing the space down to some lower dimension.
If there is no linear dependence we say the vectors are *linearly independent*. 
If the columns of a matrix are linearly independent, no compression occurs and the operation can be undone.

## Rank

If we have a general $n\times m$ matrix, it is reasonable to ask what dimension space the matrix maps into.
A concept known as the *rank* will be our answer.
In the previous section, we noted that a linear dependence bears witness to compression of space into a lower dimension and so we will be able to use this to define the notion of rank. 
In particular, the rank of a matrix $\mathbf{A}$ is the largest number of linearly independent columns amongst all subsets of columns. For example, the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

has $\mathrm{rank}(B)=1$, since the two columns are linearly dependent, but either column by itself is not linearly dependent.
For a more challenging example, we can consider

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

and show that $\mathbf{C}$ has rank two since, for instance, the first two columns are linearly independent, however any of the four collections of three columns are dependent.  

This procedure, as described, is very inefficient. 
It requires looking at every subset of the columns of our given matrix, and thus is potentially exponential in the number of columns.
Later we will see a more computationally efficient way to compute the rank of a matrix, but for now, this is sufficient to see that the concept is well defined and understand the meaning.

## Invertibility

We have seen above that multiplication by a matrix with linearly dependent columns cannot be undone, i.e., there is no inverse operation that can always recover the input. However, multiplication by a full-rank matrix (i.e., some $\mathbf{A}$ that is $n \times n$ matrix with rank $n$), we should always be able to undo it. Consider the matrix

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1 
\end{bmatrix}.
$$

which is the matrix with ones along the diagonal, and zeros elsewhere. 
We call this the *identity* matrix. 
It is the matrix which leaves our data unchanged when applied. 
To find a matrix which undoes what our matrix $\mathbf{A}$ has done, we want to find a matrix $\mathbf{A}^{-1}$ such that

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

If we look at this as a system, we have $n \times n$ unknowns (the entries of $\mathbf{A}^{-1}$) and $n \times n$ equations (the equality that needs to hold between every entry of the product $\mathbf{A}^{-1}\mathbf{A}$ and every entry of $\mathbf{I}$) so we should generically expect a solution to exist. 
Indeed, in the next section we will see a quantity called the *determinant*, which has the property that as long as the determinant is not zero, we can find a solution. We call such a matrix $\mathbf{A}^{-1}$ the *inverse* matrix.
As an example, if $\mathbf{A}$ is the general $2 \times 2$ matrix 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d 
\end{bmatrix},
$$

then we can see that the inverse is

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a 
\end{bmatrix}.
$$

We can test to see this by seeing that multiplying by the inverse given by the formula above works in practice.

```{.python .input}
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

### Numerical Issues
While the inverse of a matrix is useful in theory, we must say that most of the time we do not wish to *use* the matrix inverse to solve a problem in practice. 
In general, there are far more numerically stable algorithms for solving linear equations like

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

than computing the inverse and multiplying to get

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

Just as division by a small number can lead to numerical instability, so can inversion of a matrix which is close to having low rank.

Moreover, it is common that the matrix $\mathbf{A}$ is *sparse*, which is to say that it contains only a small number of non-zero values. 
If we were to explore examples, we would see that this does not mean the inverse is sparse. 
Even if $\mathbf{A}$ was a $1$ million by $1$ million matrix with only $5$ million non-zero entries (and thus we need only store those $5$ million), the inverse will typically have almost every entry non-negative, requiring us to store all $1\text{M}^2$ entries---that is $1$ trillion entries!

While we do not have time to dive all the way into the thorny numerical issues frequently encountered when working with linear algebra, we want to provide you with some intuition about when to proceed with caution, and generally avoiding inversion in practice is a good rule of thumb.

## Determinant
The geometric view of linear algebra gives an intuitive way to interpret a a fundamental quantity known as the *determinant*.
Consider the grid image from before, but now with a highlighted region (:numref:`fig_grid-filled`).

![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/GridTransformFilled.svg)
:label:`fig_grid-filled`

Look at the highlighted square.  This is a square with edges given by $(0, 1)$ and $(1, 0)$ and thus it has area one. After $\mathbf{A}$ transforms this square, we see that it becomes a parallelogram.
There is no reason this parallelogram should have the same area that we started with, and indeed in the specific case shown here of

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

it is an exercise in coordinate geometry to compute the area of this parallelogram and obtain that the area is $5$.

In general, if we have a matrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

we can see with some computation that the area of the resulting parallelogram is $ad-bc$.
This area is referred to as the *determinant*.

Let us check this quickly with some example code.

```{.python .input}
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

The eagle-eyed amongst us will notice that this expression can be zero or even negative.
For the negative term, this is a matter of convention  taken generally in mathematics: if the matrix flips the figure, we say the area is negated.
Let us see now that when the determinant is zero, we learn more.

Let us consider

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

If we compute the determinant of this matrix, we get $2\cdot(-2 ) - 4\cdot(-1) = 0$.
Given our understanding above, this makes sense. 
$\mathbf{B}$ compresses the square from the original image down to a line segment, which has zero area.
And indeed, being compressed into a lower dimensional space is the only way to have zero area after the transformation.
Thus we see the following result is true: a matrix $A$ is invertible if and only if the determinant is not equal to zero.

As a final comment, imagine that we have any figure drawn on the plane.
Thinking like computer scientists, we can decompose that figure into a collection of little squares so that the area of the figure is in essence just the number of squares in the decomposition.
If we now transform that figure by a matrix, we send each of these squares to parallelograms, each one of which has area given by the determinant.
We see that for any figure, the determinant gives the (signed) number that a matrix scales the area of any figure.

Computing determinants for larger matrices can be laborious, but the  intuition is the same.
The determinant remains the factor that $n\times n$ matrices scale $n$-dimensional volumes.

## Tensors and Common Linear Algebra Operations

In :numref:`sec_linear-algebra` the concept of tensors was introduced.
In this section, we will dive more deeply into tensor contractions (the tensor equivalent of matrix multiplication), and see how it can provide a unified view on a number of matrix and vector operations.  

With matrices and vectors we knew how to multiply them to transform data.
We need to have a similar definition for tensors if they are to be useful to us.
Think about matrix multiplication:

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

or equivalently

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

This pattern is one we can repeat for tensors.
For tensors, there is no one case of what to sum over that can be universally chosen, so we need specify exactly which indices we want to sum over.
For instance we could consider

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

Such a transformation is called a *tensor contraction*.
It can represent a far more flexible family of transformations that matrix multiplication alone. 

As a often-used notational simplification, we can notice that the sum is over exactly those indices that occur more than once in the expression, thus people often work with *Einstein notation*, where the summation is implicitly taken over all repeated indices.
This gives the compact expression:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Common Examples from Linear Algebra

Let us see how many of the linear algebraic definitions 
we have seen before can be expressed in this compressed tensor notation:

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

In this way, we can replace a myriad of specialized notations with short tensor expressions.

### Expressing in Code
Tensors may flexibly be operated on in code as well.
As seen in :numref:`sec_linear-algebra`, we can create tensors as is shown below.

```{.python .input}
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

Einstein summation has been implemented directly  via ```np.einsum```. 
The indices that occurs in the Einstein summation can be passed as a string,  followed by the tensors that are being acted upon.
For instance, to implement matrix multiplication, we can consider the Einstein summation seen above ($\mathbf{A}\mathbf{v} = a_{ij}v_j$) and strip out the indices themselves to get the implementation:

```{.python .input}
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

This is a highly flexible notation.
For instance if we want to compute what would be traditionally written as

$$
c_{kl} = \sum_{ij} \mathbf{B}_{ijk}\mathbf{A}_{il}v_j.
$$

it can be implemented via Einstein summation as:

```{.python .input}
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

This notation is readable and efficient for humans, however bulky if for whatever reason we need to generate a tensor contraction programmatically.
For this reason, `einsum` provides an alternative notation by providing integer indices for each tensor.
For example, the same tensor contraction can also be written as:

```{.python .input}
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't support this type of notation.
```

Either notation allows for concise and efficient representation of tensor contractions in code.

## Summary
* Vectors can be interpreted geometrically as either points or directions in space.
* Dot products define the notion of angle to arbitrarily high-dimensional spaces.
* Hyperplanes are high-dimensional generalizations of lines and planes.  They can be used to define decision planes that are often used as the last step in a classification task.
* Matrix multiplication can be geometrically interpreted as uniform distortions of the underlying coordinates. They represent a very restricted, but mathematically clean, way to transform vectors.
* Linear dependence is a way to tell when a collection of vectors are in a lower dimensional space than we would expect (say you have $3$ vectors living in a $2$-dimensional space). The rank of a matrix is the size of the largest subset of its columns that are linearly independent.
* When a matrix's inverse is defined, matrix inversion allows us to find another matrix that undoes the action of the first. Matrix inversion is useful in theory, but requires care in practice owing to numerical instability.
* Determinants allow us to measure how much a matrix expands or contracts a space. A nonzero determinant implies an invertible (non-singular) matrix and a zero-valued determinant means that the matrix is non-invertible (singular).
* Tensor contractions and Einstein summation provide for a neat and clean notation for expressing many of the computations that are seen in machine learning.

## Exercises
1. What is the angle between
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. True or false: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ and $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ are inverses of one another?
3. Suppose that we draw a shape in the plane with area $100\mathrm{m}^2$.  What is the area after transforming the figure by the matrix
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Which of the following sets of vectors are linearly independent?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Suppose that you have a matrix written as $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ for some choice of values $a, b, c$, and $d$.  True or false: the determinant of such a matrix is always $0$?
6. The vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ are orthogonal.  What is the condition on a matrix $A$ so that $Ae_1$ and $Ae_2$ are orthogonal?
7. How can you write $\mathrm{tr}(\mathbf{A}^4)$ in Einstein notation for an arbitrary matrix $A$?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab: