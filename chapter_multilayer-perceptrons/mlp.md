# Çok Katmanlı Algılayıcılar
:label:`sec_mlp`

:numref:`chap_linear` içinde, softmaks regresyonunu (:numref:`sec_softmax`), algoritmayı sıfırdan uygulayarak (:numref:`sec_softmax_scratch`) ve yüksek seviyeli API'leri (:numref:`sec_softmax_concise`) kullanarak ve düşük çözünürlüklü görüntülerden 10 giysi kategorisini tanımak için sınıflandırıcıları eğiterek tanıttık. Yol boyunca, verileri nasıl karıştıracağımızı, çıktılarımızı geçerli bir olasılık dağılımına nasıl zorlayacağımızı, uygun bir kayıp fonksiyonunu nasıl uygulayacağımızı ve modelimizin parametrelerine göre onu nasıl en aza indireceğimizi öğrendik. Şimdi bu mekaniği basit doğrusal modeller bağlamında öğrendiğimize göre, bu kitabın öncelikli olarak ilgilendiği nispeten zengin modeller sınıfı olan derin sinir ağlarını keşfetmeye başlayabiliriz.


## Gizli Katmanlar

:numref:`subsec_linear_model`içinde, bir ek girdi eklenen doğrusal bir dönüşüm olan afin dönüşümünü tanımladık. Başlangıç olarak,  :numref:`fig_softmaxreg`'da gösterilen softmaks regresyon örneğimize karşılık gelen model mimarisini hatırlayalım. Bu model, girdilerimizi tek bir afin dönüşüm ve ardından bir softmaks işlemi aracılığıyla doğrudan çıktılarımıza eşledi. Etiketlerimiz gerçekten bir afin dönüşüm yoluyla girdi verilerimizle ilişkili olsaydı, bu yaklaşım yeterli olurdu. Ancak afin dönüşümlerdeki doğrusallık *güçlü* bir varsayımdır.


### Doğrusal Modeller Yanlış Gidebilir

Örneğin, doğrusallık *daha zayıf* *monotonluk* varsayımını ifade eder: Özniteliğimizdeki herhangi bir artışın ya modelimizin çıktısında her zaman bir artışa (karşılık gelen ağırlık pozitifse) ya da modelimizin çıktısında her zaman bir düşüşe neden olması gerektiği (karşılık gelen ağırlık negatifse). Genelde bu mantıklı gelir. Örneğin, bir bireyin bir krediyi geri ödeyip ödemeyeceğini tahmin etmeye çalışıyor olsaydık, makul olarak ve her şeyi eşit tutarak, daha yüksek gelire sahip bir başvuru sahibinin, daha düşük gelirli bir başvuru sahibine göre geri ödeme olasılığının her zaman daha yüksek olacağını hayal edebilirdik. Monoton olsa da, bu ilişki muhtemelen geri ödeme olasılığıyla doğrusal olarak ilişkili değildir. Gelirdeki 0'dan 50 bine bir artış, geri ödeme olabilirliğinde kapsamında gelirdeki 1 milyondan 1.05 milyona artıştan daha büyük bir artmaya karşılık gelir. Bunu halletmenin bir yolu, verilerimizi, örneğin özniteliğimiz olan gelirin logaritmasını kullanarak doğrusallığın daha makul hale geleceği şekilde, önceden işlemek olabilir.

Monotonluğu ihlal eden örnekleri kolayca bulabileceğimizi unutmayın. Örneğin, vücut ısısına bağlı olarak ölüm olasılığını tahmin etmek istediğimizi varsayalım. Vücut ısısı 37°C'nin (98.6°F) üzerinde olan kişiler için, daha yüksek sıcaklıklar daha büyük riski gösterir. Bununla birlikte, vücut ısısı 37°C'nin altında olan kişiler için, daha yüksek sıcaklıklar daha düşük riski gösterir! Bu durumda da sorunu akıllıca bir ön işlemle çözebiliriz. Yani 37°C'den uzaklığı özniteliğimiz olarak kullanabiliriz.

Peki ya kedi ve köpeklerin resimlerini sınıflandırmaya ne dersiniz? (13, 17) konumundaki pikselin yoğunluğunu artırmak, görüntünün bir köpeği tasvir etme olabilirliğini her zaman artırmalı mı (yoksa her zaman azaltmalı mı)? Doğrusal bir modele güvenmek, kedileri ve köpekleri ayırt etmek için tek gerekliliğin tek tek piksellerin parlaklığını değerlendirmek olduğu şeklindeki örtülü varsayıma karşılık gelir. Bu yaklaşım, bir görüntünün tersine çevrilmesinin kategoriyi koruduğu bir dünyada başarısızlığa mahkumdur.

Yine de, burada doğrusallığın aşikar saçmalığına rağmen, önceki örneklerimizle karşılaştırıldığında, sorunu basit bir ön işleme düzeltmesiyle çözebileceğimiz daha az açıktır. Bunun nedeni, herhangi bir pikselin anlamının karmaşık yollarla bağlamına (çevreleyen piksellerin değerlerine) bağlı olmasıdır. Verilerimizin özniteliklerimiz arasındaki ilgili etkileşimleri hesaba katan bir temsili olabilir, bunun üzerine doğrusal bir model uygun olabilir, ancak biz bunu elle nasıl hesaplayacağımızı bilmiyoruz. Derin sinir ağlarıyla, hem gizli katmanlar aracılığıyla bir gösterimi hem de bu gösterim üzerinde işlem yapan doğrusal bir tahminciyi birlikte öğrenmek için gözlemsel verileri kullandık.


### Gizli Katmanları Birleştirme

Doğrusal modellerin bu sınırlamalarının üstesinden gelebiliriz ve bir veya daha fazla gizli katman ekleyerek daha genel işlev sınıflarıyla başa çıkabiliriz. Bunu yapmanın en kolay yolu, tam-bağlı birçok katmanı birbirinin üzerine yığmaktır. Her katman, biz çıktılar üretene kadar üstündeki katmana beslenir. İlk $L-1$ katmanı temsilimiz ve son katmanı da doğrusal tahmincimiz olarak düşünebiliriz. Bu mimariye genellikle *çok katmanlı algılayıcı (multilayer perceptron)* denir ve genellikle *MLP* olarak kısaltılır. Aşağıda, bir MLP'yi şematik olarak tasvir ediyoruz (:numref:`fig_nlp`).

![5 gizli birimli bir gizli katmana sahip bir MLP ](../img/mlp.svg)
:label:`fig_nlp`

Bu MLP'nin 4 girdisi, 3 çıktısı vardır ve gizli katmanı 5 gizli birim içerir. Girdi katmanı herhangi bir hesaplama içermediğinden, bu ağ ile çıktıların üretilmesi hem gizli hem de çıktı katmanları için hesaplamaların gerçekleştirilmesini gerektirir; dolayısıyla, bu MLP'deki katman sayısı 2'dir. Bu katmanların her ikisinin de tam-bağlı olduğuna dikkat edin. Her girdi, gizli katmandaki her nöronu etkiler ve bunların her biri de çıktı katmanındaki her nöronu etkiler.


### Doğrusaldan Doğrusal Olmayana

Daha önce olduğu gibi, $\mathbf{X} \in \mathbb{R}^{n \times d}$ matrisiyle, her örneğin $d$ girdisine (öznitelikler) sahip olduğu $n$ örneklerden oluşan bir mini grubu gösteriyoruz. Gizli katmanı $h$ gizli birimlere sahip olan tek gizli katmanlı bir MLP için, gizli katmanın çıktılarını $\mathbf{H} \in \mathbb{R}^{n \times h}$ ile belirtelim. Burada, $\mathbf{H}$ aynı zamanda *gizli katman değişkeni* veya *gizli değişken* olarak da bilinir. Gizli ve çıktı katmanlarının her ikisi de tam-bağlı olduğundan, $\mathbf{W}_1 \in \mathbb{R}^{d \times h}$ ve  $\mathbf{b}_1 \in \mathbb{R}^{1 \times h}$ içinde sırasıyla gizli katman ağırlıkları ve ek girdileri, $$\mathbf{W}_2 \in \mathbb{R}^{h \times q}$ ve $\mathbf{b}_2 \in \mathbb{R}^{1 \times q}$ içinde de çıktı katmanı ağırlıkları ve ek girdileri var. Biçimsel olarak, tek gizli katmanlı MLP'nin $\mathbf{X} \in \mathbb{R}^{n \times q}$ çıktılarını şu şekilde hesaplarız:


$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}_1 + \mathbf{b}_1, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}_2 + \mathbf{b}_2.
\end{aligned}
$$

Gizli katmanı ekledikten sonra, modelimizin artık ek parametre kümelerini izlememizi ve güncellememizi gerektirdiğini unutmayın. Peki karşılığında ne kazandık? Yukarıda tanımlanan modelde *sorunlarımız için hiçbir şey kazanmadığımızı* öğrenmek sizi şaşırtabilir! Nedeni açık. Yukarıdaki gizli birimler, girdilerin bir afin işlevi tarafından verilir ve çıktılar (softmaks-öncesi), gizli birimlerin yalnızca bir afin işlevidir. Bir afin fonksiyonun afin fonksiyonu, kendi başına bir afin fonksiyonudur. Dahası, doğrusal modelimiz zaten herhangi bir afin işlevi temsil edebiliyordu.

Ağırlıkların herhangi bir değeri için gizli katmanı daraltarak parametrelerle eşdeğer bir tek katmanlı model oluşturabileceğimizi kanıtlayarak eşdeğerliği biçimsel olarak görebiliriz.
$\mathbf{W} = \mathbf{W}_1\mathbf{W}_2$ and $\mathbf{b} = \mathbf{b}_1 \mathbf{W}_2 + \mathbf{b}_2$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2 = \mathbf{X} \mathbf{W}_1\mathbf{W}_2 + \mathbf{b}_1 \mathbf{W}_2 + \mathbf{b}_2 = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

Çok katmanlı mimarilerin potansiyelini gerçekleştirmek için, bir anahtar bileşene daha ihtiyacımız var: Afin dönüşümün ardından her gizli birime uygulanacak doğrusal olmayan $\sigma$ *etkinleştirme (aktivasyon) fonksiyonu*. Genel olarak, etkinleştirme işlevleri yürürlükte olduğunda, MLP'mizi doğrusal bir modele indirgemek artık mümkün değildir:

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}_2 + \mathbf{b}_2.\\
\end{aligned}
$$

$\mathbf{X}$ içindeki her satır, minigruptaki bir örneğe karşılık geldiğinden, birazcık gösterimi kötüye kullanarak, $\sigma$ doğrusal olmama durumunu kendi girdilerine satır-yönlü, yani her seferinde bir örnek olarak uygulanacak şekilde tanımlarız. Softmaks için gösterimi, aynı şekilde, bir satırsal işlemi belirtmek için kullandığımıza dikkat edin :numref:`subsec_softmax_vectorization`. Çoğu zaman, bu bölümde olduğu gibi, gizli katmanlara uyguladığımız etkinleştirme işlevleri yalnızca satır bazında değil, bazen eleman-yönlüdür. Bu, katmanın doğrusal bölümünü hesapladıktan sonra, diğer gizli birimler tarafından alınan değerlere bakmadan her bir etkinleştirme sonucunu hesaplayabileceğimiz anlamına gelir. Bu, çoğu etkinleştirme işlevi için geçerlidir.

Daha genel MLP'ler oluşturmak için, bu tür gizli katmanları yığmaya devam edebiliriz, örneğin, $\mathbf{H}_1 = \sigma_1(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)$ ve $\mathbf{H}_2 = \sigma_2(\mathbf{H}_1 \mathbf{W}_2 + \mathbf{b}_2)$, birbiri ardına, her zamankinden daha kuvvetli ifade edici modeller üretiyor.

### Universal Approximators

MLPs can capture complex interactions among our inputs via their hidden neurons, which depend on the values of each of the inputs. We can easily design hidden nodes to perform arbitrary computation, for instance, basic logic operations on a pair of inputs. Moreover, for certain choices of the activation function, it is widely known that MLPs are universal approximators. Even with a single-hidden-layer network, given enough nodes (possibly absurdly many), and the right set of weights, we can model any function, though actually learning that function is the hard part. You might think of your neural network as being a bit like the C programming language. The language, like any other modern language, is capable of expressing any computable program. But actually coming up with a program that meets your specifications is the hard part.

Moreover, just because a single-hidden-layer network *can* learn any function does not mean that you should try to solve all of your problems with single-hidden-layer networks. In fact, we can approximate many functions much more compactly by using deeper (vs. wider) networks. We will touch upon more rigorous arguments in subsequent chapters.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Activation Functions

Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it. They are differentiable operators to transform input signals to outputs, while most of them add non-linearity. Because activation functions are fundamental to deep learning, let us briefly survey some common activation functions.

### ReLU Function

The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks, is the *rectified linear unit* (*ReLU*). ReLU provides a very simple nonlinear transformation. Given an element $x$, the function is defined as the maximum of that element and $0$:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Informally, the ReLU function retains only positive elements and discards all negative elements by setting the corresponding activations to 0. To gain some intuition, we can plot the function. As you can see, the activation function is piecewise linear.

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

When the input is negative, the derivative of the ReLU function is 0, and when the input is positive, the derivative of the ReLU function is 1. Note that the ReLU function is not differentiable when the input takes value precisely equal to 0. In these cases, we default to the left-hand-side derivative and say that the derivative is 0 when the input is 0. We can get away with this because the input may never actually be zero. There is an old adage that if subtle boundary conditions matter, we are probably doing (*real*) mathematics, not engineering. That conventional wisdom may apply here. We plot the derivative of the ReLU function plotted below.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

The reason for using ReLU is that its derivatives are particularly well behaved: either they vanish or they just let the argument through. This makes optimization better behaved and it mitigated the well-documented problem of vanishing gradients that plagued previous versions of neural networks (more on this later).

Note that there are many variants to the ReLU function, including the *parameterized ReLU* (*pReLU*) function :cite:`He.Zhang.Ren.ea.2015`. This variation adds a linear term to ReLU, so some information still gets through, even when the argument is negative:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoid Function

The *sigmoid function* transforms its inputs, for which values lie in the domain $\mathbb{R}$, to outputs that lie on the interval (0, 1). For that reason, the sigmoid is often called a *squashing function*: it squashes any input in the range (-inf, inf) to some value in the range (0, 1):

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

In the earliest neural networks, scientists were interested in modeling biological neurons which either *fire* or *do not fire*. Thus the pioneers of this field, going all the way back to McCulloch and Pitts, the inventors of the artificial neuron, focused on thresholding units. A thresholding activation takes value 0 when its input is below some threshold and value 1 when the input exceeds the threshold.


When attention shifted to gradient based learning, the sigmoid function was a natural choice because it is a smooth, differentiable approximation to a thresholding unit. Sigmoids are still widely used as activation functions on the output units, when we want to interpret the outputs as probabilities for binary classification problems (you can think of the sigmoid as a special case of the softmax). However, the sigmoid has mostly been replaced by the simpler and more easily trainable ReLU for most use in hidden layers. In later chapters on recurrent neural networks, we will describe architectures that leverage sigmoid units to control the flow of information across time.

Below, we plot the sigmoid function. Note that when the input is close to 0, the sigmoid function approaches a linear transformation.

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

The derivative of the sigmoid function is given by the following equation:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


The derivative of the sigmoid function is plotted below. Note that when the input is 0, the derivative of the sigmoid function reaches a maximum of 0.25. As the input diverges from 0 in either direction, the derivative approaches 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### Tanh Function

Like the sigmoid function, the tanh (hyperbolic tangent) function also squashes its inputs, transforming them into elements on the interval between -1 and 1:

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

We plot the tanh function below. Note that as the input nears 0, the tanh function approaches a linear transformation. Although the shape of the function is similar to that of the sigmoid function, the tanh function exhibits point symmetry about the origin of the coordinate system.

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

The derivative of the tanh function is:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

The derivative of tanh function is plotted below. As the input nears 0, the derivative of the tanh function approaches a maximum of 1. And as we saw with the sigmoid function, as the input moves away from 0 in either direction, the derivative of the tanh function approaches 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

In summary, we now know how to incorporate nonlinearities to build expressive multilayer neural network architectures. As a side note, your knowledge already puts you in command of a similar toolkit to a practitioner circa 1990. In some ways, you have an advantage over anyone working in the 1990s, because you can leverage powerful open-source deep learning frameworks to build models rapidly, using only a few lines of code. Previously, training these networks required researchers to code up thousands of lines of C and Fortran.

## Summary

* MLP adds one or multiple fully-connected hidden layers between the output and input layers and transforms the output of the hidden layer via an activation function.
* Commonly-used activation functions include the ReLU function, the sigmoid function, and the tanh function.


## Exercises

1. Compute the derivative of the pReLU activation function.
1. Show that an MLP using only ReLU (or pReLU) constructs a continuous piecewise linear function.
1. Show that $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
1. Assume that we have a nonlinearity that applies to one minibatch at a time. What kinds of problems do you expect this to cause?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
