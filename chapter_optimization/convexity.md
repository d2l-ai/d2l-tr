# Dışdışeylik
:label:`sec_convexity`

Konveksite optimizasyon algoritmalarının tasarımında hayati bir rol oynamaktadır. Bunun nedeni, algoritmaları böyle bir bağlamda analiz etmek ve test etmek çok daha kolay olmasından kaynaklanmaktadır. Başka bir deyişle, algoritma dışbükey ayarda bile kötü performans gösteriyorsa, genellikle başka türlü harika sonuçlar görmeyi ummamalıyız. Dahası, derin öğrenmedeki optimizasyon problemleri genellikle dışbükey olmamasına rağmen, genellikle yerel minima yakın dışbükey olanların bazı özelliklerini sergiliyorlar. Bu, :cite:`Izmailov.Podoprikhin.Garipov.ea.2018` gibi heyecan verici yeni optimizasyon varyantlarına yol açabilir.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## Tanımlar

Dışbükey analizden önce, *dışbükey setler* ve *dışbükey fonksiyonları* tanımlamamız gerekir. Makine öğrenimine yaygın olarak uygulanan matematiksel araçlara yol açarlar. 

### Dışbükey Setler

Setler dışbükeyliğin temelini oluşturur. Basitçe söylemek gerekirse, bir vektör uzayında bir set $\mathcal{X}$ $a, b \in \mathcal{X}$ için $a$ ve $b$'yi bağlayan hat segmenti de $\mathcal{X}$'de ise $\mathcal{X}$'de* dışbüküm* olur. Matematiksel anlamda bu, tüm $\lambda \in [0, 1]$ için sahip olduğumuz anlamına gelir 

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

Kulağa biraz soyut geliyor. :numref:`fig_pacman`'ü düşünün. İçinde bulunmayan çizgi segmentleri bulunduğundan ilk set dışbükey değildir. Diğer iki takım böyle bir sorun yaşamaz. 

![The first set is nonconvex and the other two are convex.](../img/pacman.svg)
:label:`fig_pacman`

Onlarla bir şeyler yapamazsanız, kendi başlarına tanımlar özellikle yararlı değildir. Bu durumda :numref:`fig_convex_intersect`'te gösterildiği gibi kavşaklara bakabiliriz. $\mathcal{X}$ ve $\mathcal{Y}$ dışbükey kümeler olduğunu varsayalım. Sonra $\mathcal{X} \cap \mathcal{Y}$ de dışbükey. Bunu görmek için herhangi bir $a, b \in \mathcal{X} \cap \mathcal{Y}$'yi düşünün. $\mathcal{X}$ ve $\mathcal{Y}$ dışbükey olduğundan $a$ ve $b$'u bağlayan hat segmentleri hem $\mathcal{X}$ hem de $\mathcal{Y}$'da bulunur. Bu göz önüne alındığında, $\mathcal{X} \cap \mathcal{Y}$'te de yer almaları gerekiyor, böylece teoremimizi kanıtlıyor. 

![The intersection between two convex sets is convex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Bu sonucu az çaba ile güçlendirebiliriz: dışbükey setler $\mathcal{X}_i$ göz önüne alındığında, kesişme $\cap_{i} \mathcal{X}_i$ dışbükey. Converse doğru olmadığını görmek için, iki ayrık kümesi düşünün $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Şimdi $a \in \mathcal{X}$ ve $b \in \mathcal{Y}$'yı seçin. $a$ ve $b$ bağlayan :numref:`fig_nonconvex`'teki hat segmenti $\mathcal{X}$'te ne de $\mathcal{Y}$'de $\mathcal{Y}$'de $\mathcal{Y}$'de $\mathcal{Y}$'yi kabul ettiğimiz için $\mathcal{X} \cap \mathcal{Y} = \emptyset$'yi kabul ettiğimiz bir kısmı içermelidir. Dolayısıyla hat segmenti $\mathcal{X} \cup \mathcal{Y}$ ya da değil, böylece dışbükey kümelerin genel birlikleri dışbükey olması gerekmez kanıtlayan. 

![The union of two convex sets need not be convex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

Derin öğrenmedeki problemler genellikle dışbükey kümeler üzerinde tanımlanır. Örneğin, 73229363617, reel sayıların $d$ boyutlu vektörleri kümesi dışbükey bir kümedir (sonuçta, $\mathbb{R}^d$'de iki nokta arasındaki çizgi $\mathbb{R}^d$'de kalır). Bazı durumlarda $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\| \leq r\}$ tarafından tanımlanan yarıçap $r$ topları gibi sınırlanmış uzunlukta değişkenlerle çalışırız. 

### Dışbükey Fonksiyonlar

Artık dışbükey setlere sahip olduğumuza*dışbükey fonksiyonları* $f$'ü tanıtabiliriz. Bir dışbükey set $\mathcal{X}$ göz önüne alındığında, $f: \mathcal{X} \to \mathbb{R}$ bir işlev $f: \mathcal{X} \to \mathbb{R}$, tüm $x, x' \in \mathcal{X}$ için ve tüm $\lambda \in [0, 1]$ için 

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Bunu göstermek için birkaç işlev çizelim ve hangilerinin gereksinimi karşıladığını kontrol edelim. Aşağıda hem dışbükey hem de dışbükey olmayan birkaç fonksiyon tanımlıyoruz.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Beklendiği gibi kosinüs fonksiyonu*dışbükey* olmayan*, parabol ve üstel fonksiyon ise. $\mathcal{X}$'in dışbükey bir set olması koşulunun mantıklı olması için gerekli olduğunu unutmayın. Aksi takdirde $f(\lambda x + (1-\lambda) x')$'ün sonucu iyi tanımlanmış olmayabilir. 

### Jensen'ın Eşitsizliği

Bir dışbükey fonksiyon $f$ göz önüne alındığında, en kullanışlı matematiksel araçlardan biri *Jensen eşitsizliği*. Dışbükeylik tanımının genelleştirilmesi anlamına gelir: 

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

burada $\alpha_i$ negatif olmayan reel sayılar $\sum_i \alpha_i = 1$ ve $X$ rasgele bir değişkendir. Başka bir deyişle, dışbükey bir fonksiyonun beklentisi, ikincisinin genellikle daha basit bir ifade olduğu bir beklentinin dışbükey işlevinden daha az değildir. İlk eşitsizliği kanıtlamak için dışbükeylik tanımını defalarca bir defada toplamda bir terime uygularız. 

Jensen'ın eşitsizliğinin yaygın uygulamalarından biri daha karmaşık bir ifadeyi daha basit bir ifadeyle bağlamaktır. Örneğin, uygulaması kısmen gözlenen rastgele değişkenlerin günlük olasılığı ile ilgili olabilir. Yani, kullandığımız 

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

beri $\int P(Y) P(X \mid Y) dY = P(X)$. Bu, varyasyonel yöntemlerde kullanılabilir. Burada $Y$ genellikle gözlenmeyen rastgele değişkendir, $P(Y)$ dağıtılabilir nasıl en iyi tahmindir, ve $P(X)$ ile dağıtımıdır $Y$ entegre. Örneğin, kümeleme $Y$ küme etiketleri olabilir ve $P(X \mid Y)$ küme etiketleri uygularken oluşturucu modeldir. 

## Özellikler

Dışbükey işlevler birçok yararlı özelliğe sahiptir. Aşağıda yaygın olarak kullanılan birkaç tanesini tanımlıyoruz. 

### Yerel Minima Global Minima

Her şeyden önce, dışbükey işlevlerin yerel minimumu da küresel minimadır. Bunu aşağıdaki gibi çelişkilerle kanıtlayabiliriz. 

Bir dışbükey fonksiyon düşünün $f$ dışbükey bir küme üzerinde tanımlanan $\mathcal{X}$. $x^{\ast} \in \mathcal{X}$'ün yerel bir minimum olduğunu varsayalım: $p$ için $0 < |x - x^{\ast}| \leq p$'u tatmin eden $x \in \mathcal{X}$ için $f(x^{\ast}) < f(x)$'ye sahip olmak için küçük bir pozitif değer vardır. 

Yerel minimum $x^{\ast}$ $x^{\ast}$ küresel en az $f$ olmadığını varsayalım: $x' \in \mathcal{X}$ hangi $f(x') < f(x^{\ast})$ için var. Ayrıca var $\lambda \in [0, 1)$ gibi $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ böylece $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$.  

Ancak, dışbükey fonksiyonların tanımına göre, 

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

hangi $x^{\ast}$ yerel bir asgari olduğu ifadesinde çelişmektedir. Bu nedenle, $x' \in \mathcal{X}$ için $f(x') < f(x^{\ast})$ yok. Yerel minimum $x^{\ast}$ da küresel asgari değerdir. 

Örneğin, dışbükey işlev $f(x) = (x-1)^2$ yerel minimum $x=1$ değerine sahiptir ve bu da küresel minimum değerdir.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Dışbükey fonksiyonlar için yerel minima da küresel minima olması çok uygundur. Fonksiyonları en aza indirirsek “sıkışamayız” anlamına gelir. Bununla birlikte, bunun birden fazla küresel minimum olamayacağı veya bir tane bile var olabileceği anlamına gelmediğini unutmayın. Örneğin, $f(x) = \mathrm{max}(|x|-1, 0)$ işlevi $[-1, 1]$ aralığında minimum değerini alır. Tersine, $f(x) = \exp(x)$ işlevi $\mathbb{R}$ üzerinde minimum bir değere ulaşmaz: $x \to -\infty$ için $0$ için asimtotlar, ancak $x$ için $f(x) = 0$ yoktur. 

### Dışbükey Fonksiyonların Altında Dışbükey Fonksiyonlar

Dışbükey fonksiyonların *aşağıdaki setleri* aracılığıyla dışbükey setleri rahatça tanımlayabiliriz. Beton, dışbükey bir fonksiyon verilen $f$ dışbükey bir set üzerinde tanımlanan $\mathcal{X}$, herhangi bir aşağıdaki set 

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

dışbükey.  

Bunu çabucak kanıtlayalım. Herhangi bir $x, x' \in \mathcal{S}_b$ için $\lambda \in [0, 1]$ kadar $\lambda \in [0, 1]$ olarak $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ olduğunu göstermemiz gerektiğini hatırlayın. $f(x) \leq b$ ve $f(x') \leq b$'den bu yana, dışbükeyliğin tanımı gereği  

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### Konvekslik ve İkinci Türevler

$f: \mathbb{R}^n \rightarrow \mathbb{R}$ fonksiyonunun ikinci türevi varsa, $f$'in dışbükey olup olmadığını kontrol etmek çok kolaydır. Tek yapmamız gereken $f$ Hessian pozitif yarı sonlu olup olmadığını kontrol etmektir: $\nabla^2f \succeq 0$, yani $\nabla^2f$ tarafından $\mathbf{H}$, $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$, $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ için $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ tarafından $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$. Örneğin, $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ işlevi $\nabla^2 f = \mathbf{1}$'ten beri dışbükeydir, yani Hessian bir kimlik matrisidir. 

Resmi olarak, çift diferansiyellenebilir tek boyutlu bir fonksiyon $f: \mathbb{R} \rightarrow \mathbb{R}$ dışbükey ise ve sadece ikinci türevi $f'' \geq 0$ ise. Herhangi bir çift diferansiyellenebilir çok boyutlu fonksiyon $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ için, eğer ve sadece Hessian $\nabla^2f \succeq 0$ ise dışbükey olur. 

İlk olarak, tek boyutlu davayı kanıtlamamız gerekiyor. $f$'ün dışbükeyliğinin $f'' \geq 0$'i ima ettiğini görmek için 

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

İkinci türev sonlu farklar üzerindeki limitle verildiğinden 

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

$f'' \geq 0$'i görmek için $f$'nin dışbükey olduğu anlamına gelir. $f'' \geq 0$'in $f'$'un monoton olarak azalmayan bir işlev olduğu anlamına gelmektedir. Let $a < x < b$ üç puan olmak $\mathbb{R}$, nerede $x = (1-\lambda)a + \lambda b$ ve $\lambda \in (0, 1)$. Ortalama değer teoremine göre, $\alpha \in [a, x]$ ve $\beta \in [x, b]$ gibi var 

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

Monotonicity $f'(\beta) \geq f'(\alpha)$, dolayısıyla 

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

beri $x = (1-\lambda)a + \lambda b$, biz 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

böylece dışbükeyliği kanıtlıyor. 

İkincisi, çok boyutlu davayı kanıtlamadan önce bir lemmaya ihtiyacımız var: $f: \mathbb{R}^n \rightarrow \mathbb{R}$ dışbükey ise ve yalnızca $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

dışbükey. 

$f$'ün dışbükeyliğinin $g$'in dışbükey olduğunu ima ettiğini kanıtlamak için, tüm $a, b, \lambda \in [0, 1]$ için (böylece $0 \leq \lambda a + (1-\lambda) b \leq 1$) 

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

Converse kanıtlamak için, tüm $\lambda \in [0, 1]$ için bunu gösterebiliriz  

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$

Son olarak, yukarıdaki lemma ve tek boyutlu davanın sonucunu kullanarak, çok boyutlu kasa aşağıdaki gibi kanıtlanabilir. Çok boyutlu bir fonksiyon $f: \mathbb{R}^n \rightarrow \mathbb{R}$ dışbükey ise ve yalnızca $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, burada $z \in [0,1]$ dışbükey ise dışbükey olur. Tek boyutlu duruma göre, eğer ve sadece $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) tüm $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ için ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$), pozitif yarı sonsuz matrislerin tanımı başına $\mathbf{H} \succeq 0$ eşdeğerdir. 

## Kısıtlamalar

Dışbükey optimizasyonun güzel özelliklerinden biri, kısıtlamaları verimli bir şekilde ele almamıza izin vermesidir. Yani, formun*kısıtlı optimizasyon* sorunlarını çözmemize izin verir: 

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

burada $f$ hedeftir ve $c_i$ fonksiyonları kısıtlama işlevleridir. Bu durumda dikkate ne görmek için nerede $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. Bu durumda $\mathbf{x}$ parametreleri birim topuyla sınırlıdır. İkinci bir kısıtlama $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$ ise, bu yarım alanda yatan $\mathbf{x}$'in tümüne karşılık gelir. Her iki kısıtlama da aynı anda tatmin etmek, bir topun bir dilim seçilmesi anlamına gelir. 

### Lagrangian

Genel olarak, kısıtlı bir optimizasyon problemini çözmek zordur. Bunu ele almanın bir yolu fizikten oldukça basit bir sezgiyle kaynaklanıyor. Bir kutunun içinde bir top hayal et. Top en düşük yere yuvarlanacak ve yerçekimi kuvvetleri, kutunun kenarlarının topa empoze edebileceği kuvvetlerle dengelenecektir. Kısacası, objektif fonksiyonun degrade (yani, yerçekimi), kısıtlama fonksiyonunun gradyanı ile dengelenecektir (topun duvarların “geri iterek” kutunun içinde kalması gerekir). Bazı kısıtlamaların aktif olmayabileceğini unutmayın: topun dokunmadığı duvarlar topa herhangi bir güç uygulayamayacaktır. 

*Lagrangian* $L$'ün türetilmesi üzerinden atlandığında, yukarıdaki akıl yürütme, aşağıdaki eyer noktası optimizasyonu problemi ile ifade edilebilir: 

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

Burada $\alpha_i$ ($i=1,\ldots,n$) değişkenleri, kısıtlamaların doğru şekilde uygulanmasını sağlayan *Lagrange çarpanları* olarak adlandırılır. Onlar sağlamak için yeterince büyük seçilir $c_i(\mathbf{x}) \leq 0$ tüm $i$. Örneğin, $c_i(\mathbf{x}) < 0$'in doğal olarak $c_i(\mathbf{x}) < 0$'in $\alpha_i = 0$'i seçtiği herhangi bir $\mathbf{x}$ için. Dahası, bu, $\alpha_i$ ile ilgili olarak $L$ ve aynı anda $\mathbf{x}$ ile ilgili olarak en üst düzeye çıkarma* $\mathbf{x}$ ile ilgili olarak en üst düzeye çıkarmak istediği bir eyer noktası optimizasyon sorunudur. $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$ fonksiyonuna nasıl ulaşılacağını açıklayan zengin bir edebiyat gövdesi var. Amacımız için, orijinal kısıtlı optimizasyon probleminin en iyi şekilde çözüldüğü $L$ eyer noktasının nerede olduğunu bilmek yeterlidir. 

### Cezalar

Kısıtlı optimizasyon sorunlarını en azından*yaklaşık* tatmin etmenin bir yolu Lagrangian $L$'yi uyarlamaktır. $c_i(\mathbf{x}) \leq 0$'yı tatmin etmek yerine $\alpha_i c_i(\mathbf{x})$'i objektif fonksiyona $f(x)$'ü ekliyoruz. Bu, kısıtlamaların çok kötü ihlal edilmesini sağlar. 

Aslında, başından beri bu numarayı kullanıyorduk. :numref:`sec_weight_decay`'te ağırlık çürümesini düşünün. İçinde $\frac{\lambda}{2} \|\mathbf{w}\|^2$'i nesnel işleve $\mathbf{w}$'in çok büyük olmamasını sağlamak için ekliyoruz. Görüş kısıtlı optimizasyon açısından biz bu sağlayacağını görebilirsiniz $\|\mathbf{w}\|^2 - r^2 \leq 0$ Bazı yarıçap $r$. $\lambda$ değerinin ayarlanması, $\mathbf{w}$ boyutunu değiştirmemize izin verir. 

Genel olarak, ceza eklemek, yaklaşık kısıtlama memnuniyetini sağlamanın iyi bir yoludur. Pratikte bu, tam memnuniyetten çok daha sağlam olduğu ortaya çıkıyor. Dahası, dışbükey olmayan problemler için, dışbükey durumda (örn., optimalite) kesin yaklaşımı çok çekici hale getiren özelliklerin çoğu artık tutmuyor. 

### Projeksiyonlar

Kısıtlamaları tatmin etmek için alternatif bir strateji projeksiyonlardır. Yine, daha önce onlarla karşılaştık, örneğin, :numref:`sec_rnn_scratch`'te degrade kırpma ile uğraşırken. Orada bir degrade $\theta$ ile sınırlanmış uzunluğa sahip olduğunu sağlanmıştır 

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

Bu, $\theta$ yarıçap topu üzerine $\mathbf{g}$'ün bir *projeksiyonu* olduğu ortaya çıkıyor. Daha genel olarak, bir dışbükey kümedeki bir projeksiyon $\mathcal{X}$ 

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

hangi $\mathcal{X}$ için $\mathbf{x}$ en yakın noktasıdır.  

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

Projeksiyonların matematiksel tanımı biraz soyut gelebilir. :numref:`fig_projections` biraz daha net bir şekilde açıklıyor. İçinde iki dışbükey set, bir daire ve bir elmas var. Her iki kümedeki noktalar (sarı) projeksiyonlar sırasında değişmeden kalır. Her iki kümenin dışındaki noktalar (siyah), kümelerin içindeki noktalara (kırmızı) orijinal noktalara (siyah) gömme noktalara yansıtılır. $L_2$ topları için bu yönü değişmeden bırakırken, elmas durumunda görülebileceği gibi genel olarak böyle olması gerekmez. 

Dışbükey projeksiyonların kullanımlarından biri seyrek ağırlık vektörlerini hesaplamaktır. Bu durumda ağırlık vektörlerini $L_1$ topuna yansıtıyoruz, bu da :numref:`fig_projections`'te elmas kılıfın genelleştirilmiş bir versiyonu olan bir $L_1$ topuna yansıtıyoruz. 

## Özet

Derin öğrenme bağlamında dışbükey fonksiyonların temel amacı optimizasyon algoritmalarını motive etmek ve bunları ayrıntılı olarak anlamamıza yardımcı olmaktır. Aşağıda degrade iniş ve stokastik degrade iniş buna göre türetilebilir nasıl göreceksiniz. 

* Dışbükey kümelerin kesişimleri dışbükeydir. Sendikalar değil.
* Dışbükey bir fonksiyonun beklentisi, bir beklentinin dışbükey işlevinden daha az değildir (Jensen eşitsizliği).
* İki kez diferansiyellenebilir bir fonksiyon, eğer ve sadece Hessian (ikinci türevlerin bir matrisi) pozitif yarı sonsuz ise dışbükeydir.
* Dışbükey kısıtlamalar Lagrangian aracılığıyla eklenebilir. Uygulamada, onları sadece objektif işleve bir ceza ile ekleyebiliriz.
* Projeksiyonlar, orijinal noktalara en yakın dışbükey kümedeki noktalarla eşleştirilir.

## Egzersizler

1. Kümedeki noktalar arasındaki tüm çizgileri çizerek ve çizgilerin içerip içermediğini kontrol ederek bir kümenin dışbükeyliğini doğrulamak istediğimizi varsayalım.
    1. Sadece sınırdaki noktaları kontrol etmenin yeterli olduğunu kanıtlayın.
    1. Sadece kümenin köşelerini kontrol etmenin yeterli olduğunu kanıtlayın.
1. $p$-normunu kullanarak $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ yarıçap topu $r$ ile belirtin. Kanıtlayın $\mathcal{B}_p[r]$ hepsi için dışbükey olduğunu $p \geq 1$.
1. Verilen dışbükey fonksiyonlar $f$ ve $g$, $\mathrm{max}(f, g)$ de dışbükey olduğunu göstermektedir. $\mathrm{min}(f, g)$ dışbükey olmadığını kanıtlayın.
1. Softmax fonksiyonunun normalleştirilmesinin dışbükey olduğunu kanıtlayın. Daha spesifik olarak $f(x) = \log \sum_i \exp(x_i)$'ün dışbükeyliğini kanıtlayın.
1. Doğrusal alt uzayların, yani $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$'ün dışbükey kümeler olduğunu kanıtlayın.
1. $\mathbf{b} = \mathbf{0}$ ile doğrusal alt uzaylarda $\mathrm{Proj}_\mathcal{X}$ projeksiyon $\mathrm{Proj}_\mathcal{X}$ bazı matris $\mathbf{M}$ için $\mathbf{M} \mathbf{x}$ olarak yazılabileceğini kanıtlayın.
1. İki kez diferansiyellenebilir dışbükey fonksiyonlar için $f$ $f$ bazı $\xi \in [0, \epsilon]$ için $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ yazabileceğimizi gösterin.
1. Bir vektör verilen $\mathbf{w} \in \mathbb{R}^d$ ile $\|\mathbf{w}\|_1 > 1$ üzerinde projeksiyon hesaplamak $L_1$ birim top.
    1. Bir ara adım olarak cezalandırılmış hedefi yazmak $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$ ve belirli bir $\lambda > 0$ için çözüm hesaplamak.
    1. $\lambda$'ün “doğru” değerini çok fazla deneme yanılma olmadan bulabilir misiniz?
1. Bir dışbükey set $\mathcal{X}$ ve iki vektör $\mathbf{x}$ ve $\mathbf{y}$ göz önüne alındığında, projeksiyonların mesafeleri asla artırmadığını kanıtlayın, yani $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$.

[Discussions](https://discuss.d2l.ai/t/350)
