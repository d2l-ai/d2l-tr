# Dışbükeylik
:label:`sec_convexity`

Dışbükeylik optimizasyon algoritmalarının tasarımında hayati bir rol oynamaktadır. Bunun nedeni, algoritmaları böyle bir bağlamda analiz etmenin ve test etmenin çok daha kolay olmasından kaynaklanmaktadır. Başka bir deyişle, algoritma dışbükey ayarda bile kötü başarım gösteriyorsa, genellikle başka türlü harika sonuçlar görmeyi ummamalıyız. Dahası, derin öğrenmedeki optimizasyon problemleri çoğunlukla dışbükey olmamasına rağmen, genellikle yerel minimumun yakınında dışbükey olanların bazı özelliklerini sergilerler. Bu, :cite:`Izmailov.Podoprikhin.Garipov.ea.2018` çalışmasındaki gibi heyecan verici yeni optimizasyon türlerine yol açabilir.

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

Dışbükey analizden önce, *dışbükey kümeleri* ve *dışbükey fonksiyonları* tanımlamamız gerekir. Makine öğrenmesinde yaygın olarak uygulanan matematiksel araçlara yön verirler. 

### Dışbükey Kümeler

Kümeler dışbükeyliğin temelini oluşturur. Basitçe söylemek gerekirse, bir vektör uzayında bir $\mathcal{X}$ kümesi,  $a, b \in \mathcal{X}$ için $a$ ve $b$'yi bağlayan doğru parçası da $\mathcal{X}$'de ise *dışbüküm* olur. Matematiksel anlamda bu, tüm $\lambda \in [0, 1]$ için aşağıdaki ifadeye sahip olduğumuz anlamına gelir 

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ ne zaman ki } a, b \in \mathcal{X}.$$

Kulağa biraz soyut geliyor. :numref:`fig_pacman` figürünü düşünün. İçinde tamamı kapsanmayan doğru parçaları bulunduğundan ilk küme dışbükey değildir. Diğer iki kümede böyle bir sorun yaşanmaz. 

![İlk küme dışbükey değildir ve diğer ikisi dışbükeydir.](../img/pacman.svg)
:label:`fig_pacman`

Onlarla bir şeyler yapamazsanız, kendi başlarına tanımlar özellikle yararlı değildir. Bu durumda :numref:`fig_convex_intersect` şeklinde gösterildiği gibi kesişimlere bakabiliriz. $\mathcal{X}$ ve $\mathcal{Y}$'nin dışbükey kümeler olduğunu varsayalım. O halde $\mathcal{X} \cap \mathcal{Y}$ de dışbükeydir. Bunu görmek için herhangi bir $a, b \in \mathcal{X} \cap \mathcal{Y}$'yi düşünün. $\mathcal{X}$ ve $\mathcal{Y}$ dışbükey olduğundan $a$ ve $b$'yi bağlayan doğru parçaları hem $\mathcal{X}$ hem de $\mathcal{Y}$'de bulunur. Bu göz önüne alındığında, $\mathcal{X} \cap \mathcal{Y}$'de de yer almaları gerekiyor, böylece teoremimizi kanıtlıyor. 

![İki dışbükey kümenin kesişimi dışbükeydir.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Bu sonucu az çaba ile güçlendirebiliriz: Dışbükey kümeler $\mathcal{X}_i$ göz önüne alındığında, kesişmeleri $\cap_{i} \mathcal{X}_i$ dışbükeydir. Tersinin doğru olmadığını görmek için, iki ayrık küme $\mathcal{X} \cap \mathcal{Y} = \emptyset$ düşünün. Şimdi $a \in \mathcal{X}$ ve $b \in \mathcal{Y}$'yi seçin. $a$ ve $b$'yi birbirine bağlayan :numref:`fig_nonconvex` içindeki doğru parçasının, $\mathcal{X}$'da veya $\mathcal{Y}$'da olmayan bir kısım içermesi gerekir, çünkü $\mathcal{X} \cap \mathcal{Y} = \emptyset$ olduğunu varsaydık. Dolayısıyla doğru parçası $\mathcal{X} \cup \mathcal{Y}$'da da değildir, bu da genel olarak dışbükey kümelerin birleşimlerinin dışbükey olması gerekmediğini kanıtlar.

![İki dışbükey kümenin birleşiminin dışbükey olması gerekmez.](../img/nonconvex.svg)
:label:`fig_nonconvex`

Derin öğrenmedeki problemler genellikle dışbükey kümeler üzerinde tanımlanır. Örneğin, $\mathbb{R}^d$, gerçek sayıların $d$ boyutlu vektörleri kümesi, bir dışbükey kümedir (sonuçta, $\mathbb{R}^d$ içindeki herhangi iki nokta arasındaki çizgi $\mathbb{R}^d$ içinde kalır). Bazı durumlarda $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ ve } \|\mathbf{x}\| \leq r\}$ tarafından tanımlanan $r$ yarıçap topları gibi sınırlanmış uzunlukta değişkenlerle çalışırız. 

### Dışbükey İşlevler

Artık dışbükey kümelere sahip olduğumuza göre *dışbükey fonksiyonları* $f$'yi tanıtabiliriz. Bir dışbükey $\mathcal{X}$ kümesi verildiğinde, eğer tüm $x, x' \in \mathcal{X}$ için ve elimizdeki tüm $\lambda \in [0, 1]$ için aşağıdaki ifade tutarsa $f: \mathcal{X} \to \mathbb{R}$ işlevi *dışbükeydir*:
$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Bunu göstermek için birkaç işlev çizelim ve hangilerinin gereksinimi karşıladığını kontrol edelim. Aşağıda hem dışbükey hem de dışbükey olmayan birkaç fonksiyon tanımlıyoruz.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Dışbükey
g = lambda x: d2l.cos(np.pi * x)  # Dışbükey olmayan
h = lambda x: d2l.exp(0.5 * x)  # Dışbükey

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Beklendiği gibi kosinüs fonksiyonu *dışbükey* olmayan*, parabol ve üstel fonksiyon ise dışbükeydir. $\mathcal{X}$'in dışbükey bir küme olması koşulunun anlamlı olması için gerekli olduğunu unutmayın. Aksi takdirde $f(\lambda x + (1-\lambda) x')$'in sonucu iyi tanımlanmış olmayabilir. 

### Jensen'ın Eşitsizliği

Bir dışbükey $f$ fonksiyonu göz önüne alındığında, en kullanışlı matematiksel araçlardan biri *Jensen eşitsizliği*dir. Dışbükeylik tanımının genelleştirilmesi anlamına gelir: 

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ ve }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

burada $\alpha_i$ negatif olmayan gerçel sayılar $\sum_i \alpha_i = 1$ ve $X$ rasgele bir değişkenlerdir. Başka bir deyişle, dışbükey bir fonksiyonun beklentisi, ikincisinin genellikle daha basit bir ifade olduğu bir beklentinin dışbükey işlevinden daha az değildir. İlk eşitsizliği kanıtlamak için, dışbükeylik tanımını her seferinde toplamdaki bir terime tekrar tekrar uygularız.

Jensen'in eşitsizliğinin yaygın uygulamalarından biri daha karmaşık bir ifadeyi daha basit bir ifadeyle bağlamaktır. Örneğin, uygulaması kısmen gözlenen rastgele değişkenlerin log olabilirliği ile ilgili olabilir. Yani, kullandığımız 

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

çünkü $\int P(Y) P(X \mid Y) dY = P(X)$. Bu, varyasyonel yöntemlerde kullanılabilir. Burada $Y$ tipik olarak gözlemlenmeyen rastgele değişkendir, $P(Y)$ bunun nasıl dağılabileceğine dair en iyi tahmindir ve $P(X)$, $Y$'nin integral uygulandığı dağılımdır. Örneğin, öbeklemede $Y$ küme etiketleri olabilir ve $P(X \mid Y)$ öbek etiketleri uygularken üretici modeldir. 

## Özellikler

Dışbükey işlevler birçok yararlı özelliğe sahiptir. Aşağıda yaygın olarak kullanılan birkaç tanesini tanımlıyoruz. 

### Yerel Minimum Küresel Minimum Olduğunda

Her şeyden önce, dışbükey işlevlerin yerel minimumu da küresel minimumdur. Bunu aşağıdaki gibi tezatlıklarla kanıtlayabiliriz. 

Bir dışbükey küme $\mathcal{X}$ üzerinde tanımlanmış bir dışbükey fonksiyon $f$ düşünün. $x^{\ast} \in \mathcal{X}$'ın yerel bir minimum olduğunu varsayalım: Küçük bir pozitif $p$ değeri vardır, öyle ki $x \in \mathcal{X}$ için $0 < |x - x^{\ast}| \leq p$ elimizde $f(x^{\ast}) < f(x)$.

Yerel minimum $x^{\ast}$'in $f$'nin küresel minimumu olmadığını varsayalım: $f(x') < f(x^{\ast})$ için $x' \in \mathcal{X}$ vardır. Ayrıca $\lambda \in [0, 1)$ vardır, örneğin $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ öyle ki $0 < |\lambda x^ {\ast} + (1-\lambda) x' - x^{\ast}| \leq p$.

Ancak, dışbükey fonksiyonların tanımına göre, 

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

bu $x^{\ast}$'in yerel bir minimum olduğu ifadesinde çelişmektedir. Bu nedenle, $x' \in \mathcal{X}$ için $f(x') < f(x^{\ast})$ yok. Yerel minimum $x^{\ast}$ da küresel minimum değerdir. 

Örneğin, dışbükey işlev $f(x) = (x-1)^2$ yerel minimum $x=1$ değerine sahiptir ve bu da küresel minimum değerdir.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Dışbükey fonksiyonlar için yerel minimum da küresel minimum olması çok uygundur. Fonksiyonları en aza indirirsek “takılıp kalmayız” anlamına gelir. Bununla birlikte, bunun birden fazla küresel minimum olamayacağı veya bir tane bile var olabileceği anlamına gelmediğini unutmayın. Örneğin, $f(x) = \mathrm{max}(|x|-1, 0)$ işlevi $[-1, 1]$ aralığında minimum değerini alır. Tersine, $f(x) = \exp(x)$ işlevi $\mathbb{R}$ üzerinde minimum bir değere ulaşmaz: $x \to -\infty$ için $0$'da asimtota dönüşür, ancak $x$ için $f(x) = 0$ yoktur. 

### Dışbükey Fonksiyonların Aşağı Kümeleri Dışbükey Fonksiyonlardır

Dışbükey fonksiyonların *aşağıdaki kümeleri* aracılığıyla dışbükey kümeleri rahatça tanımlayabiliriz. Somut olarak, dışbükey bir $\mathcal{X}$ kümesi üzerinde tanımlanan dışbükey bir $f$  fonksiyonu verildiğinde, herhangi bir aşağı küme 

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ ve } f(x) \leq b\}$$

dışbükeydir.  

Bunu çabucak kanıtlayalım. Herhangi bir $x, x' \in \mathcal{S}_b$ için  $\lambda \in [0, 1]$ oldukça $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ olduğunu göstermemiz gerektiğini hatırlayın. $f(x) \leq b$ ve $f(x') \leq b$'den dolayı, dışbükeyliğin tanımı gereği aşağıdaki ifadeye sahip oluruz:

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### Dışbükeylik ve İkinci Türevler

$f: \mathbb{R}^n \rightarrow \mathbb{R}$ fonksiyonunun ikinci türevi varsa, $f$'in dışbükey olup olmadığını kontrol etmek çok kolaydır. Tek yapmamız gereken $f$'ın Hessian'ının pozitif yarı-kesin olup olmadığını kontrol etmektir: $\nabla^2f \succeq 0$, yani, $\nabla^2f$ Hessian matrisini $\mathbf{H}$ ile ifade edersek $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ $\mathbf{x} \in \mathbb{R}^n$. Örneğin, $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ işlevi $\nabla^2 f = \mathbf{1}$'ten dolayı dışbükeydir, yani Hessian bir birim matrisdir. 

Biçimsel olarak, çift türevlenebilen tek boyutlu bir fonksiyon $f: \mathbb{R} \rightarrow \mathbb{R}$, sadece ve sadece ikinci türevi $f'' \geq 0$ ise dışbükeydir. Herhangi bir çift türevlenebilen çok boyutlu fonksiyon $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$, sadece ve sadece Hessian $\nabla^2f \succeq 0$ ise dışbükey olur. 

İlk olarak, tek boyutlu durumu kanıtlamamız gerekiyor. $f$'nin dışbükeyliğinin $f'' \geq 0$'ı ima ettiğini görmek için 

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

İkinci türev sonlu farklar üzerindeki limitle verildiğinde: 

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

$f'' \geq 0$'ı görmek $f$'nin dışbükey olduğu anlamına gelir, burada $f'' \geq 0$'in $f'$'nin monoton olarak azalmayan bir işlev olduğunu gösterdiğini kullandık. $a < x < b$ $\mathbb{R}$'de üç nokta olsun, öyleki $x = (1-\lambda)a + \lambda b$ ve $\lambda \in (0, 1)$ olsun. Ortalama değer teoremine göre, öyle $\alpha \in [a, x]$ ve $\beta \in [x, b]$ vardır ki:

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

Monotonluktan dolayı $f'(\beta) \geq f'(\alpha)$, dolayısıyla 

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Çünkü $x = (1-\lambda)a + \lambda b$, 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

böylece dışbükeyliği kanıtlıyoruz. 

İkincisi, çok boyutlu durumu kanıtlamadan önce bir önsava (lemma) ihtiyacımız var: $f: \mathbb{R}^n \rightarrow \mathbb{R}$, sadece ve sadece $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ öyle ki } z \in [0,1]$$ 

dışbükey ise dışbükeydir. 

$f$'nin dışbükeyliğinin $g$'nin dışbükey olduğunu ima ettiğini kanıtlamak için, tüm $a, b, \lambda \in [0, 1]$ için (böylece $0 \leq \lambda a + (1-\lambda) b \leq 1$) 

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

Tersini kanıtlamak için, tüm $\lambda \in [0, 1]$ ise şunu gösterebiliriz:

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$

Son olarak, yukarıdaki önsav ve tek boyutlu durumun sonucunu kullanarak, çok boyutlu durum aşağıdaki gibi kanıtlanabilir. Çok boyutlu bir fonksiyon $f: \mathbb{R}^n \rightarrow \mathbb{R}$, sadece ve sadece $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, burada $z \in [0,1]$, dışbükey ise dışbükey olur. Tek boyutlu duruma göre, bu sadece ve sadece $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) tüm $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ için ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) olursa tutar, ki bu da pozitif yarı-kesin matrislerin tanımı $\mathbf{H} \succeq 0$ ile eşdeğerdir. 

## Kısıtlamalar

Dışbükey optimizasyonun güzel özelliklerinden biri, kısıtlamaları verimli bir şekilde ele almamıza izin vermesidir. Yani, *kısıtlı optimizasyon* biçimdeki sorunlarını çözmemize izin verir: 

$$\begin{aligned} \mathop{\mathrm{minimize~et~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ tabi } & c_i(\mathbf{x}) \leq 0 \text{ her } i \in \{1, \ldots, n\} \text{ için },
\end{aligned}$$

burada $f$ amaç işlevi ve $c_i$ fonksiyonları kısıtlama işlevleridir. Bunun ne yaptığını görmek için $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$ durumunu düşünün. Bu durumda $\mathbf{x}$ parametreleri birim topla sınırlıdır. İkinci bir kısıtlama $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$ ise, bu yarı uzayda yatan $\mathbf{x}$'lerin tümüne karşılık gelir. Her iki kısıtlama da aynı anda tatmin etmek, bir topun bir diliminin seçilmesi anlamına gelir. 

### Lagrangian

Genel olarak, kısıtlı bir optimizasyon problemini çözmek zordur. Bunu ele almanın bir yolu fizikten kaynaklanan basit bir sezgidir. Bir kutunun içinde bir topu hayal edin. Top en düşük yere yuvarlanacak ve yerçekimi kuvvetleri, kutunun kenarlarının topa uygulayabileceği kuvvetlerle dengelenecektir. Kısacası, amaç fonksiyonun gradyanı (yani, yerçekimi), kısıtlama fonksiyonunun gradyanı ile dengelenecektir (topun duvarlarca “geri iterek” kutunun içinde kalması gerekir). Bazı kısıtlamaların aktif olmayabileceğini unutmayın: Topun dokunmadığı duvarlar topa herhangi bir güç uygulayamayacaktır. 

*Lagrangian* $L$'nin türetilmesi es geçersek, yukarıdaki akıl yürütme, aşağıdaki eyer noktası optimizasyonu problemi ile ifade edilebilir: 

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ burada } \alpha_i \geq 0.$$

Burada $\alpha_i$ ($i=1,\ldots,n$) değişkenleri, kısıtlamaların doğru şekilde uygulanmasını sağlayan *Lagrange çarpanları* olarak adlandırılır. Tüm $i$'lerde $c_i(\mathbf{x}) \leq 0$  sağlamak için yeterince büyük seçilir. Örneğin, $c_i(\mathbf{x}) < 0$'ın doğal olarak $c_i(\mathbf{x}) < 0$'ın $\alpha_i = 0$'ı seçtiği herhangi bir $\mathbf{x}$ için. Üstelik, bu, $L$'nin tüm $\alpha_i$'ya göre *maksimize edilmek* ve aynı anda $\mathbf{x}$'a göre *minimize edilmek* istendiği bir eyer noktası optimizasyon problemidir. $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$ fonksiyonuna nasıl ulaşılacağını açıklayan zengin bir yazın birikimi vardır. Amacımız için, asıl kısıtlı optimizasyon probleminin en iyi şekilde çözüldüğü $L$ eyer noktasının nerede olduğunu bilmek yeterlidir. 

### Cezalar

Kısıtlı optimizasyon sorunlarını en azından *yaklaşık* tatmin etmenin bir yolu Lagrangian $L$'yi uyarlamaktır. $c_i(\mathbf{x}) \leq 0$'ı tatmin etmek yerine $\alpha_i c_i(\mathbf{x})$'i amaç fonksiyonu $f(x)$'e ekliyoruz. Bu, kısıtlamaların aşırı ihlal edilmemesini sağlar. 

Aslında, başından beri bu hileyi kullanıyorduk. :numref:`sec_weight_decay` içindeki ağırlık sönümünü düşünün. İçinde $\frac{\lambda}{2} \|\mathbf{w}\|^2$'i amaç işlevine $\mathbf{w}$'nin çok büyük olmamasını sağlamak için ekliyoruz. Kısıtlı optimizasyon bakış açısından bunun bize bazı $r$ yarıçapı için $\|\mathbf{w}\|^2 - r^2 \leq 0$ sağlayacağını görebilirsiniz. $\lambda$ değerinin ayarlanması, $\mathbf{w}$'nin boyutunu değiştirmemize izin verir. 

Genel olarak, ceza eklemek, yaklaşık kısıtlama tatminini sağlamanın iyi bir yoludur. Pratikte bunun tam tatminden çok daha gürbüz olduğu ortaya çıkıyor. Dahası, dışbükey olmayan problemler için, dışbükey durumdaki (örn., eniyilik) kesin yaklaşımı çok çekici hale getiren özelliklerin çoğu artık tutmuyor. 

### İzdüşümler

Kısıtlamaları tatmin etmek için alternatif bir strateji izdüşümlerdir. Yine, daha önce onlarla karşılaştık, örneğin, :numref:`sec_rnn_scratch` içinde gradyan kırpma ile uğraşırken. Orada bir gradyan $\theta$ ile sınırlanmış uzunluğa sahip olmasını sağladık 

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

Bu, $\theta$ yarıçaplı top üzerine $\mathbf{g}$'nin bir *izdüşümü* olduğu ortaya çıkarıyor. Daha genel olarak, bir dışbükey kümedeki bir izdüşüm $\mathcal{X}$ şöyle tanımlanır:

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

ki buda $\mathcal{X}$ içinde $\mathbf{x}$'e en yakın noktadır.  

![Dışbükey izdüşümler.](../img/projections.svg)
:label:`fig_projections`

İzdüşümlerin matematiksel tanımı biraz soyut gelebilir. :numref:`fig_projections` bunu biraz daha net bir şekilde açıklıyor. İçinde iki dışbükey küme, bir daire ve bir elmas var. Her iki kümedeki noktalar (sarı) izidüşümler esnasında değişmeden kalır. Her iki kümenin (siyah) dışındaki noktalar, asıl noktalara (siyah) yakın olan kümelerin içindeki (kırmızı) noktalara yansıtılır. $L_2$ topları için yön değişmeden kalırken, elmas durumunda görülebileceği gibi genel olarak böyle olması gerekmez. 

Dışbükey izdüşümlerin kullanımlarından biri seyrek ağırlık vektörlerini hesaplamaktır. Bu durumda ağırlık vektörlerini $L_1$ topuna izdüşürüyoruz, bu da :numref:`fig_projections` içinde elmas durumunun genelleştirilmiş bir versiyonu olan bir $L_1$ topuna izdüşürüyoruz. 

## Özet

Derin öğrenme bağlamında dışbükey fonksiyonların temel amacı optimizasyon algoritmalarını özendirmek ve bunları ayrıntılı olarak anlamamıza yardımcı olmaktır. Sonrasında gradyan iniş ve rasgele gradyan iniş buna göre nasıl türetilebilir göreceksiniz. 

* Dışbükey kümelerin kesişimleri dışbükeydir. Birleşimleri değil.
* Dışbükey bir fonksiyonun beklentisi, bir beklentinin dışbükey işlevinden daha az değildir (Jensen eşitsizliği).
* İki kez türevlenebilen bir fonksiyon, sadece ve sadece Hessian (ikinci türevlerin bir matrisi) pozitif yarı kesin ise dışbükeydir.
* Dışbükey kısıtlamalar Lagrangian aracılığıyla eklenebilir. Uygulamada, onları sadece amaç işlevine bir ceza ile ekleyebiliriz.
* İzdüşümler, orijinal noktaları dışbükey kümedeki en yakın noktalarla eşleştirirler.

## Alıştırmalar

1. Kümedeki noktalar arasındaki tüm çizgileri çizerek ve çizgilerin içerilip içerilmediğine kontrol ederek bir kümenin dışbükeyliğini doğrulamak istediğimizi varsayalım.
    1. Sadece sınırdaki noktaları kontrol etmenin yeterli olduğunu kanıtlayın.
    1. Sadece kümenin köşelerini kontrol etmenin yeterli olduğunu kanıtlayın.
1. $p$-normunu kullanarak $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ ve } \|\mathbf{x}\|_p \leq r\}$ yarıçap topu $r$ ile belirtin. Tüm $p \geq 1$ için $\mathcal{B}_p[r]$'in dışbükey olduğunu kanıtlayın.
1. Verilen dışbükey fonksiyonlar $f$ ve $g$ için, $\mathrm{max}(f, g)$'nin de dışbükey olduğunu gösterin. $\mathrm{min}(f, g)$'nin dışbükey olmadığını kanıtlayın.
1. Softmax fonksiyonunun normalleştirilmesinin dışbükey olduğunu kanıtlayın. Daha özel olarak $f(x) = \log \sum_i \exp(x_i)$'in dışbükeyliğini kanıtlayın.
1. Doğrusal alt uzayların, yani $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$'nin dışbükey kümeler olduğunu kanıtlayın.
1. $\mathbf{b} = \mathbf{0}$ ile doğrusal alt uzaylarda $\mathrm{Proj}_\mathcal{X}$ izdüşümünün bazı $\mathbf{M}$ matrisi için $\mathbf{M} \mathbf{x}$ olarak yazılabileceğini kanıtlayın.
1. İki kez türevlenebilir dışbükey $f$ fonksiyonları için, bazı $\xi \in [0, \epsilon]$ için $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ yazabileceğimizi gösterin.
1. Verilen $\mathbf{w} \in \mathbb{R}^d$ vektörü ve $\|\mathbf{w}\|_1 > 1$ ile $L_1$ birim topu üzerindeki izdüşümü hesaplayın.
    1. Bir ara adım olarak $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$  cezalandırılmış hedefini yazın ve belirli bir $\lambda > 0$ için çözümü hesaplayın.
    1. $\lambda$'nin “doğru” değerini çok fazla deneme yanılma olmadan bulabilir misiniz?
1. Bir dışbükey küme $\mathcal{X}$ ve iki vektör $\mathbf{x}$ ve $\mathbf{y}$ göz önüne alındığında, izdüşümlrin mesafeleri asla artırmadığını kanıtlayın, yani $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$.

[Tartışmalar](https://discuss.d2l.ai/t/350)
