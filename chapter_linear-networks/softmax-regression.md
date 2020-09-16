# Eşiksiz En Büyük İşlev Bağlanımı  (Softmax Regression)
:label:`sec_softmax`

:numref:`sec_linear_regression`'de, doğrusal bağlanımı tanıttık, :numref:`sec_linear_scratch`'da sıfırdan uygulamalar üzerinde çalıştık ve ağır işi yapmak için :numref:`sec_linear_concise`da derin öğrenme çerçevesinin yüksek seviyeli API'lerini tekrar kullandık.

Bağlanım (regresyon), *ne kadar?* veya *kaç?* sorularını yanıtlamak istediğimizde ulaştığımız çekiçtir. Bir evin kaç dolara satılacağını (fiyatı) veya bir beyzbol takımının kazanabileceği galibiyet sayısını veya bir hastanın taburcu edilmeden önce hastanede kalacağı gün sayısını tahmin etmek istiyorsanız, o zaman muhtemelen bir regresyon modeli arıyorsunuz.

Uygulamada, daha çok *sınıflandırma* ile ilgileniyoruz. "Ne kadar" değil, "hangisi" sorusu sorarak:

* Bu e-posta spam klasörüne mi yoksa gelen kutusuna mı ait?
* Bu müşterinin bir abonelik hizmetine *kaydolma* veya *kaydolmama* olasılığı mı daha yüksek?
* Bu resim bir eşeği, köpeği, kediyi veya horozu mu tasvir ediyor?
* Aston'ın bir sonraki izleyeceği film hangisi?

Konuşma dilinde, makine öğrenimi uygulayıcıları iki ince farklı sorunu tanımlamak için *sınıflandırma* kelimesini aşırı yüklüyorlar: (i) yalnızca örneklerin kategorilere (sınıflara) zor olarak atanmasıyla ilgilendiğimiz konular ve (ii) yumuşak atamalar yapmak istediğimiz yerler, yani her bir kategorinin geçerli olma olasılığını değerlendirme. Ayrım, kısmen bulanıklaşma eğilimindedir, çünkü çoğu zaman, yalnızca zor görevleri önemsediğimizde bile, yine de yumuşak atamalar yapan modeller kullanıyoruz.

## Sınıflandırma Problemi

Ayaklarımızı ısındırmak için basit bir görüntü sınıflandırma problemiyle başlayalım. Buradaki her girdi $2\times2$ gri tonlamalı bir resimden oluşur. Her piksel değerini tek bir skaler ile temsil edebiliriz ve bu da  bize dört özellik, $x_1, x_2, x_3, x_4$, verir. Ayrıca, her görüntünün "kedi", "tavuk" ve "köpek" kategorilerinden birine ait olduğunu varsayalım.

Daha sonra, etiketleri nasıl temsil edeceğimizi seçmeliyiz. İki bariz seçeneğimiz var. Belki de en doğal dürtü, tam sayıların sırasıyla {köpek, kedi, tavuk}'u temsil ettiği $\{1, 2, 3 \}$ içinden $y$'yi seçmek olacaktır. Bu, bu tür bilgileri bir bilgisayarda *saklamanın* harika bir yoludur. Kategoriler arasında doğal bir sıralama varsa, örneğin {bebek, yürümeye başlayan çocuk, ergen, genç yetişkin, yetişkin, yaşlı} tahmin etmeye çalışıyor olsaydık, bu sorunu bağlanım olarak kabul etmek ve etiketleri bu biçimde tutmak mantıklı bile olabilirdi.

Ancak genel sınıflandırma sorunları, sınıflar arasında doğal sıralamalarla gelmez. Neyse ki, istatistikçiler uzun zaman önce kategorik verileri göstermenin basit bir yolunu keşfettiler: *bire bir kodlama*. Bire bir kodlama, kategorilerimiz kadar bileşen içeren bir vektördür. Belirli bir örneğin kategorisine karşılık gelen bileşen $1$'e ve diğer tüm bileşenler $0$'a ayarlanmıştır. Bizim durumumuzda, $y$ etiketi üç boyutlu bir vektör olacaktır: $(1, 0, 0)$ - "kedi", $(0, 1, 0)$ - "tavuk" ve $(0, 0, 1)$ - "köpek":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## Ağ mimarisi

Tüm olası sınıflarla ilişkili koşullu olasılıkları tahmin etmek için, sınıf başına bir tane olmak üzere birden çok çıktıya sahip bir modele ihtiyacımız var. Doğrusal modellerle sınıflandırmayı ifade etmek için, çıktılarımız olduğu kadar çok sayıda afin (affine) fonksiyona ihtiyacımız olacak. Her çıktı kendi afin işlevine karşılık gelecektir. Bizim durumumuzda, 4 özniteliğimiz ve 3 olası çıktı kategorimiz olduğundan, ağırlıkları temsil etmek için 12 skalere (sayıla) ($w$ gösterimli) ve ek girdileri temsil etmek için 3 skalere ($b$ gösterimli) ihtiyacımız olacak. Her girdi için şu üç *logit*i, $o_1, o_2$ ve $o_3$, hesaplıyoruz:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Bu hesaplamayı :numref:`fig_softmaxreg`'da gösterilen sinir ağı diyagramı ile tasvir edebiliriz. Doğrusal regresyonda olduğu gibi, softmaks regresyon da tek katmanlı bir sinir ağıdır. Ayrıca her çıktının, $o_1, o_2$ ve $o_3$, hesaplanması,  tüm girdilere, $x_1$, $x_2$, $x_3$ ve $x_4$, bağlı olduğundan, eşiksiz en büyük işlev bağlanımının (softmaks regresyonunun) çıktı katmanı da tamamen bağlı katman olarak tanımlanır.

![Eşiksiz en büyük işlev bağlanımı bir tek katmanlı sinir ağıdır.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Modeli daha özlü bir şekilde ifade etmek için doğrusal cebir gösterimini kullanabiliriz. Vektör biçiminde, hem matematik hem de kod yazmak için daha uygun bir biçim olan $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$'ye ulaşıyoruz. Tüm ağırlıklarımızı bir $3 \times 4$ matriste topladığımızı ve belirli bir $\mathbf{x}$ veri örneğinin öznitelikleri için, çıktılarımızın bizim öznitelikler girdimiz ile ağırlıklarımızın bir matris-vektör çarpımı artı ek girdilerimiz $\mathbf{b}$ olarak verildiğini unutmayın.

# Eşiksiz En Büyük İşlev İşlemi 

Burada ele alacağımız ana yaklaşım, modelimizin çıktılarını olasılıklar olarak yorumlamaktır. Gözlemlenen verilerin olabilirliğini en üst düzeye çıkaran olasılıklar üretmek için parametrelerimizi optimize edeceğiz (eniyileceğiz). Ardından, tahminler üretmek için bir eşik belirleyeceğiz, örneğin maksimum tahmin edilen olasılığa sahip etiketi seçeceğiz.

Biçimsel olarak ifade edersek, herhangi bir $\hat{y}_j$ çıktısının belirli bir öğenin $j$ sınıfına ait olma olasılığı olarak yorumlanmasını istiyoruz. Sonra en büyük çıktı değerine sahip sınıfı tahminimiz $\operatorname*{argmax}_j y_j$ olarak seçebiliriz. Örneğin, $\hat{y}_1$, $\hat{y}_2$ ve $\hat{y}_3$ sırasıyla 0.1, 0.8 ve 0.1 ise, o zaman (örneğimizde) "tavuğu" temsil eden kategori 2'yi tahmin ederiz.

Logit $O$'yu doğrudan ilgilendiğimiz çıktılarımız olarak yorumlamamızı önermek isteyebilirsiniz. Bununla birlikte, doğrusal katmanın çıktısının doğrudan bir olasılık olarak yorumlanmasında bazı sorunlar vardır. Bir yandan, hiçbir şey bu sayıların toplamını 1'e sınırlamıyor. Diğer yandan, girdilere bağlı olarak negatif değerler alabilirler. Bunlar, :numref:`sec_prob`da sunulan temel olasılık aksiyomlarını ihlal ediyor.

Çıktılarımızı olasılıklar olarak yorumlamak için, (yeni verilerde bile) bunların negatif olmayacağını ve toplamlarının 1 olacağını garanti etmeliyiz. Dahası, modeli gerçeğe uygun olasılıkları tahmin etmeye teşvik eden bir eğitim amaç fonksiyonuna ihtiyacımız var. Bir sınıflandırıcı tüm örneklerden 0.5 çıktısını verdiğinde, bu örneklerin yarısının gerçekte tahmin edilen sınıfa ait olacağını umuyoruz. Bu, *kalibrasyon* adı verilen bir özelliktir.

1959'da sosyal bilimci R. Duncan Luce tarafından *seçim modelleri* bağlamında icat edilen *eşiksiz en büyük işlevi (softmaks)* tam olarak bunu yapar. Logitlerimizi negatif olmayacak ve toplamı 1 olacak şekilde dönüştürmek için, modelin turevlenebilir kalmasını gerekliyken, önce her logiti üsleriz (negatif olmamasını sağlar) ve sonra toplamlarına böleriz (toplamlarının 1 olmasını sağlar):

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{öyle ki}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

Tüm $j$ için $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$'ı $0 \leq \hat{y}_j \leq 1$ ile görmek kolaydır. Dolayısıyla, $\hat{\mathbf{y}}$, eleman değerleri uygun şekilde yorumlanabilen uygun bir olasılık dağılımıdır. Softmaks işleminin, her sınıfa atanan olasılıkları belirleyen basit softmaks-öncesi değerler olan $\mathbf{o}$ logitleri arasındaki sıralamayı değiştirmediğini unutmayın. Bu nedenle, tahmin sırasında yine en olası sınıfı seçebiliriz:

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

Softmaks doğrusal olmayan bir fonksiyon olmasına rağmen, softmaks regresyonunun çıktıları hala girdi özntellklerinin afin dönüşümü ile *belirlenir*; dolayısıyla, softmaks regresyon doğrusal bir modeldir.

## Minigruplar için Vektörleştirme
:label:`subsec_softmax_vectorization`

Hesaplama verimliliğini artırmak ve GPU'lardan yararlanmak için, genellikle veri minigrupları için vektör hesaplamaları yapıyoruz. Öznitelik boyutsallığı (girdi sayısı) $d$ ve parti boyutu $n$ içeren bir minigrup $\mathbf{X}$ verildiğini varsayalım. Üstelik, çıktıda $q$ kategorimizin olduğunu varsayalım. Sonra minigrup öznitelikleri $\mathbf{X}$, $\mathbb{R}^{n \times d}$ içinde, ağırlıkları $\mathbf{W} \in \mathbb{R}^{d \times q}$ ve ek girdiyi, $\mathbf{b} \in \mathbb{R}^{1\times q}$ olarak karşılar.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Bu, baskın olan işlemi, bir matris-matris çarpımı $\mathbf{X} \mathbf{W}$ ile her seferinde bir örnek işleseydik yürüteceğimiz matris-vektör çarpımlarına göreceli olarak, hızlandırır. $\mathbf{X}$ içindeki her satır bir veri örneği olduğundan, softmaks işleminin kendisi *satır bazında* hesaplanabilir: her $\mathbf{O}$ satırı için, tüm girdileri üsleyin ve sonra bunları toplayarak normalleştirin. :eqref:`eq_minibatch_softmax_reg`'deki $\mathbf{X} \mathbf{W} + \mathbf{b}$ toplamı sırasında yayınlamayı tetikleriz, hem minigrup logitleri $\mathbf{O}$ hem de çıktı olasılıkları $\hat{\mathbf{Y}}$, $n \times q$ matrislerdir.

## Loss Function

Next, we need a loss function to measure the quality of our predicted probabilities. We will rely on maximum likelihood estimation, the very same concept that we encountered when providing a probabilistic justification for the mean squared error objective in linear regression (:numref:`subsec_normal_distribution_and_squared_loss`).


### Log-Likelihood

The softmax function gives us a vector $\hat{\mathbf{y}}$, which we can interpret as estimated conditional probabilities of each class given any input $\mathbf{x}$, e.g., $\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$. Suppose that the entire dataset $\{\mathbf{X}, \mathbf{Y}\}$ has $n$ examples, where the example indexed by $i$ consists of a feature vector $\mathbf{x}^{(i)}$ and a one-hot label vector $\mathbf{y}^{(i)}$. We can compare the estimates with reality by checking how probable the actual classes are according to our model, given the features:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

According to maximum likelihood estimation, we maximize $P(\mathbf{Y} \mid \mathbf{X})$, which is equivalent to minimizing the negative log-likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

where for any pair of label $\mathbf{y}$ and model prediction $\hat{\mathbf{y}}$ over $q$ classes, the loss function $l$ is

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

For reasons explained later on, the loss function in :eqref:`eq_l_cross_entropy` is commonly called the *cross-entropy loss*. Since $\mathbf{y}$ is a one-hot vector of length $q$, the sum over all its coordinates $j$ vanishes for all but one term. Since all $\hat{y}_j$ are predicted probabilities, their logarithm is never larger than $0$. Consequently, the loss function cannot be minimized any further if we correctly predict the actual label with *certainty*, i.e., if the predicted probability $P(\mathbf{y} \mid \mathbf{x}) = 1$ for the actual label $\mathbf{y}$. Note that this is often impossible. For example, there might be label noise in the dataset (some examples may be mislabeled). It may also not be possible when the input features are not sufficiently informative to classify every example perfectly.

### Softmax and Derivatives

Since the softmax and the corresponding loss are so common, it is worth understanding a bit better how it is computed. Plugging :eqref:`eq_softmax_y_and_o` into the definition of the loss in :eqref:`eq_l_cross_entropy` and using the definition of the softmax we obtain:

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

To understand a bit better what is going on, consider the derivative with respect to any logit $o_j$. We get

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

In other words, the derivative is the difference between the probability assigned by our model, as expressed by the softmax operation, and what actually happened, as expressed by elements in the one-hot label vector. In this sense, it is very similar to what we saw in regression, where the gradient was the difference between the observation $y$ and estimate $\hat{y}$. This is not coincidence. In any [exponential family](https://en.wikipedia.org/wiki/Exponential_family) model, the gradients of the log-likelihood are given by precisely this term. This fact makes computing gradients easy in practice.

### Cross-Entropy Loss

Now consider the case where we observe not just a single outcome but an entire distribution over outcomes. We can use the same representation as before for the label $\mathbf{y}$. The only difference is that rather than a vector containing only binary entries, say $(0, 0, 1)$, we now have a generic probability vector, say $(0.1, 0.2, 0.7)$. The math that we used previously to define the loss $l$ in :eqref:`eq_l_cross_entropy` still works out fine, just that the interpretation is slightly more general. It is the expected value of the loss for a distribution over labels. This loss is called the *cross-entropy loss* and it is one of the most commonly used losses for classification problems. We can demystify the name by introducing just the basics of information theory. If you wish to understand more details of information theory, you may further refer to the [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html).

## Information Theory Basics

*Information theory* deals with the problem of encoding, decoding, transmitting, and manipulating information (also known as data) in as concise form as possible.


### Entropy

The central idea in information theory is to quantify the information content in data. This quantity places a hard limit on our ability to compress the data. In information theory, this quantity is called the *entropy* of a distribution $p$, and it is captured by the following equation:

$$H[p] = \sum_j - p(j) \log p(j).$$
:eqlabel:`eq_softmax_reg_entropy`

One of the fundamental theorems of information theory states that in order to encode data drawn randomly from the distribution $p$, we need at least $H[p]$ "nats" to encode it. If you wonder what a "nat" is, it is the equivalent of bit but when using a code with base $e$ rather than one with base 2. Thus, one nat is $\frac{1}{\log(2)} \approx 1.44$ bit.


### Surprisal

You might be wondering what compression has to do with prediction. Imagine that we have a stream of data that we want to compress. If it is always easy for us to predict the next token, then this data is easy to compress! Take the extreme example where every token in the stream always takes the same value. That is a very boring data stream! And not only it is boring, but it is also easy to predict. Because they are always the same, we do not have to transmit any information to communicate the contents of the stream. Easy to predict, easy to compress.

However if we cannot perfectly predict every event, then we might sometimes be surprised. Our surprise is greater when we assigned an event lower probability. Claude Shannon settled on $\log \frac{1}{P(j)} = -\log P(j)$ to quantify one's *surprisal* at observing an event $j$ having assigned it a (subjective) probability $P(j)$. The entropy defined in :eqref:`eq_softmax_reg_entropy` is then the *expected surprisal* when one assigned the correct probabilities that truly match the data-generating process.


### Cross-Entropy Revisited

So if entropy is level of surprise experienced by someone who knows the true probability, then you might be wondering, what is cross-entropy? The cross-entropy *from* $p$ *to* $q$, denoted $H(p, q)$, is the expected surprisal of an observer with subjective probabilities $q$ upon seeing data that were actually generated according to probabilities $p$. The lowest possible cross-entropy is achieved when $p=q$. In this case, the cross-entropy from $p$ to $q$ is $H(p, p)= H(p)$.

In short, we can think of the cross-entropy classification objective in two ways: (i) as maximizing the likelihood of the observed data; and (ii) as minimizing our surprisal (and thus the number of bits) required to communicate the labels.


## Model Prediction and Evaluation

After training the softmax regression model, given any example features, we can predict the probability of each output class. Normally, we use the class with the highest predicted probability as the output class. The prediction is correct if it is consistent with the actual class (label). In the next part of the experiment,
we will use *accuracy* to evaluate the model’s performance. This is equal to the ratio between the number of correct predictions and the total number of predictions.


## Summary

* The softmax operation takes a vector and maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output class in the softmax operation.
* Cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.

## Exercises

1. We can explore the connection between exponential families and the softmax in some more depth.
    * Compute the second derivative of the cross-entropy loss $l(\mathbf{y},\hat{\mathbf{y}})$ for the softmax.
    * Compute the variance of the distribution given by $\mathrm{softmax}(\mathbf{o})$ and show that it matches the second derivative computed above.
1. Assume that we have three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    * What is the problem if we try to design a binary code for it?
    * Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
1. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    * Prove that $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    * Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
    * Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    * What does the soft-min look like?
    * Extend this to more than two numbers.

[Tartışmalar](https://discuss.d2l.ai/t/46)
