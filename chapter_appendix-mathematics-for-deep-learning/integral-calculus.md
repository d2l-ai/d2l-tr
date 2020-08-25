# Integral (Tümlev) Kalkülüsü
:label:`sec_integral_calculus`

Türev alma, geleneksel bir matematik eğitiminin içeriğinin yalnızca yarısını oluşturur. Diğer sütun, integral alma, oldukça ayrık bir soru gibi görünerek başlar, "Bu eğrinin altındaki alan nedir?" Görünüşte alakasız olsa da, intgeral alma, *kalkülüsün temel teoremi* olarak bilinen ilişki aracılığıyla türev alma ile sıkı bir şekilde iç içe geçmiştir.

Bu kitapta tartıştığımız makine öğrenmesi düzeyinde, derin bir türev alma anlayışına ihtiyacımız olmayacak. Gene de, daha sonra karşılaşacağımız diğer uygulamalara zemin hazırlamak için kısa bir giriş sağlayacağız.

## Geometrik Yorum
Bir $f(x)$ fonksiyonumuz olduğunu varsayalım. Basit olması için, $f(x)$'nin negatif olmadığını varsayalım (asla sıfırdan küçük bir değer almıyor). Denemek ve anlamak istediğimiz şudur: $f(x)$ ile $x$ ekseni arasındaki alan nedir?

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

Çoğu durumda, bu alan sonsuz veya tanımsız olacaktır ($f(x) = x^{2}$ altındaki alanı düşünün), bu nedenle insanlar genellikle bir çift uç arasındaki alan hakkında konuşacaklar, örneğin $a$ ve $b$.

```{.python .input}
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

Bu alanı aşağıdaki integral (tümlev) sembolü ile göstereceğiz:

$$
\mathrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

İç değişken, bir $\sum$ içindeki bir toplamın indisine çok benzeyen bir yapay değişkendir ve bu nedenle bu, istediğimiz herhangi bir iç değerle eşdeğer olarak yazılabilir:

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

Bu tür integralleri nasıl yaklaşık olarak tahmin etmeye çalışabileceğimizin ve anlayabileceğimizin geleneksel bir yolu var: $a$ ile $b$ arasındaki bölgeyi alıp onu $N$ dikey dilimlere böldüğümüzü hayal edebiliriz. $N$ büyükse, her dilimin alanını bir dikdörtgenle tahmin edebilir ve ardından eğrinin altındaki toplam alanı elde etmek için alanları toplayabiliriz. Bunu kodla yapan bir örneğe bakalım. Gerçek değeri nasıl elde edeceğimizi daha sonraki bir bölümde göreceğiz.

```{.python .input}
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([2.])) / 2

d2l.set_figsize()
d2l.plt.bar(x.numpy(), f.numpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

Sorun şu ki, sayısal (numerik) olarak yapılabilse de, bu yaklaşımı analitik olarak sadece aşağıdaki gibi en basit işlevler için yapabiliriz.

$$
\int_a^b x \;dx.
$$

Yukarıdaki koddaki örneğimiz gibi biraz daha karmaşık bir şey:

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$

Bu böyle doğrudan bir yöntemle çözebileceğimizin ötesinde bir örnektir.

Bunun yerine farklı bir yaklaşım benimseyeceğiz. Alan kavramıyla sezgisel olarak çalışacağız ve integralleri bulmak için kullanılan ana hesaplama aracını öğreneceğiz: *Kalkülüsün temel teoremi*. Bu, integral alma (tümleme) çalışmamızın temeli olacaktır.

## Kalkülüsün Temel Teoremi

Integral alma teorisinin derinliklerine inmek için bir fonksiyon tanıtalım

$$
F(x) = \int_0^x f(y) dy.
$$

This function measures the area between $0$ and $x$ depending on how we change $x$.  Notice that this is everything we need since
Bu işlev, $x$'i nasıl değiştirdiğimize bağlı olarak $0$ ile $x$ arasındaki alanı ölçer. İhtiyacımız olan her şeyin burada olduğuna dikkat edin:

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

Bu, alanı en uzak uç noktaya kadar ölçebildiğimiz ve ardından yakın uç noktaya kadaki alanı çıkarabileceğimiz gerçeğinin matematiksel bir kodlamasıdır :numref:`fig_area-subtract`.

![İki nokta arasındaki bir eğrinin altındaki alanı hesaplama sorununu bir noktanın solundaki alanı hesaplamaya neden indirgeyeceğimizi görselleştirelim.](../img/SubArea.svg)
:label:`fig_area-subtract`

Dolayısıyla, herhangi bir aralıktaki integralin ne olduğunu $F(x)$'in ne olduğunu bularak bulabiliriz.

Bunun için bir deney yapalım. Kalkülüste sık sık yaptığımız gibi, değeri çok az değiştirdiğimizde ne olduğunu görelim. Yukarıdaki yorumdan biliyoruz ki

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

Bu bize, fonksiyonun küçük bir fonksiyon şeridinin altındaki alana göre değiştiğini söyler.

Bu, bir yaklaşıklama yaptığımız noktadır. Bunun gibi küçük bir alana bakarsak, bu alan, yüksekliği $f(x)$ ve taban genişliği $\epsilon$ olan dikdörtgenin alanına yakın görünüyor. Aslında, $\epsilon \rightarrow 0$ olunca bu yaklaşıklamanın gittikçe daha iyi hale geldiği gösterilebilir. Böylece şu sonuca varabiliriz:

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

Bununla birlikte, şimdi fark edebiliriz ki $F$'nin türevini hesaplıyor olsaydık, tam olarak beklediğimiz kalıp bu olurdu! Böylece şu oldukça şaşırtıcı gerçeği görüyoruz:

$$
\frac{dF}{dx}(x) = f(x).
$$

Bu, *kalkülüsün temel teoremidir*. Bunu genişletilmiş biçimde şöyle yazabiliriz:
$$\frac{d}{dx}\int_{-\infty}^x f(y) \; dy = f(x).$$
:eqlabel:`eq_ftc`

Alan bulma kavramını alır (ki muhtemelen oldukça zordur) ve onu bir ifade türevine (çok daha anlaşılmış bir şey) indirger. Yapmamız gereken son bir yorum, bunun bize $F(x)$'nin tam olarak ne olduğunu söylemediğidir. Aslında, herhangi bir $C$ için $F(x) + C$ aynı türeve sahiptir. Bu, integral alma teorisinde hayatın gerçeğidir. Belirli integrallerle çalışırken sabitlerin düştüğüne ve dolayısıyla sonuçla ilgisiz olduklarına dikkat edin.

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

Bu soyut olarak bir anlam ifade etmiyor gibi görünebilir, ancak bunun bizde integral hesaplamaya yepyeni bir bakış açısı kazandırdığını anlamak için biraz zaman ayıralım. Amacımız artık bir çeşit parçala ve topla işlemi yapmak ve alanı geri kurtarmayı denemek değil; bunun yerine sadece türevi sahip olduğumuz fonksiyon olan başka bir fonksiyon bulmamız gerekiyor! Bu inanılmaz bir olay çünkü artık pek çok zorlu integrali sadece :numref:`sec_derivative_table`dan tabloyu ters çevirerek listeleyebiliriz. Örneğin, $x^{n}$'nin türevinin $nx^{n-1}$ olduğunu biliyoruz. Böylece, temel teoremi kullanarak şunu söyleyebiliriz :eqref:`eq_ftc`

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

Benzer şekilde, $e^{x}$'nin türevinin kendisi olduğunu biliyoruz, yani

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

Bu şekilde, diferansiyel kalkülüsten gelen fikirlerden özgürce yararlanarak bütün integral alma teorisini geliştirebiliriz. Her integral kuralı bu olgudan türetilir.

## Değişkenlerin Değişimi
:label:`integral_example`

Aynen türev almada olduğu gibi, integrallerin hesaplanmasını daha izlenebilir kılan bir dizi kural vardır. Aslında, diferansiyel kalkülüsün her kuralı (çarpım kuralı, toplam kuralı ve zincir kuralı gibi), integral hesaplamada karşılık gelen bir kurala sahiptir (formül sırasıyla, parçalayarak integral alma, integralin doğrusallığı ve değişkenler değişimi). Bu bölümde, listeden tartışmasız en önemli olanı inceleyeceğiz: değişkenlerin değişimi formülü.

İlk olarak, kendisi bir integral olan bir fonksiyonumuz olduğunu varsayalım:

$$
F(x) = \int_0^x f(y) \; dy.
$$ 

Diyelim ki, $F(u(x))$ elde etmek için onu başka bir fonksiyonla oluşturduğumuzda bu fonksiyonun nasıl göründüğünü bilmek istediğimizi varsayalım. Zincir kuralı ile biliyoruz ki:

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{dx}(u(x))\cdot \frac{du}{dx}.
$$

Yukarıdaki gibi, temel teoremi kullanarak bunu integral alma ilgili bir ifadeye dönüştürebiliriz :eqref:`eq_ftc`. Böylece:

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{dx}(u(y))\cdot \frac{du}{dy} \;dy.
$$

$F$'nin kendisinin bir integral olduğunu hatırlamak, sol tarafın yeniden yazılabilmesine olanak verir.

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{dx}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Benzer şekilde, $F$'nin bir integral olduğunu hatırlamak, temel teoremi kullanarak $\frac{dF}{dx} = f$'i tanımamıza izin verir :eqref:`eq_ftc` ve böylece şu sonuca varabiliriz:

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

Bu, *değişkenlerin değişimi* formülüdür.

Daha sezgisel bir türetme için, $x$ ile $x+\epsilon$ arasında bir $f(u(x))$ integralini aldığımızda ne olacağını düşünün. Küçük bir $\epsilon$ için, bu integral, yaklaşık olarak $\epsilon f(u(x))$, yani ilişkili dikdörtgenin alanıdır. Şimdi bunu $u(x) $ ile $u(x+\epsilon)$ arasındaki $f(y)$ integraliyle karşılaştıralım. $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$ olduğunu biliyoruz, bu nedenle bu dikdörtgenin alanı yaklaşık $\epsilon \frac{du}{dx}(x)f(u(x))$'dir. Bu nedenle, bu iki dikdörtgenin alanını uyuşacak hale getirmek için, ilkini $\frac{du}{dx}(x)$ ile çarpmamız gerekiyor :numref:`fig_rect-transform`.

![Değişkenlerin değişimi altında tek bir ince dikdörtgenin dönüşümünü görselleştirelim.](../img/RectTrans.svg)
:label:`fig_rect-transform`

Bu bize şunu söyler:

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$  

Bu, tek bir küçük dikdörtgen için ifade edilen değişken değişimi formülüdür.

$u(x)$ ve $f(x)$ doğru seçildirse, bu inanılmaz derecede karmaşık integrallerin hesaplanmasına izin verebilir. Örneğin, $f(y) = 1$ ve $u(x) = e^{-x^{2}}$ seçersek (bu $\frac{du}{dx}(x) = -2xe^{-x^{2}}$ anlamına gelir), bu örnek şunu gösterebilir:

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

Tekrar düzenlersek:

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## İşaret Gösterimlerine Bir Yorum

Keskin gözlü okuyucular yukarıdaki hesaplamalarla ilgili tuhaf bir şey gözlemleyecekler. Yani, aşağıdaki gibi hesaplamalar

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

negatif sayılar üretebilirler. Alanlar hakkında düşünürken, negatif bir değer görmek garip olabilir ve bu nedenle bu gösterimin ne olduğunu araştırmaya değer.

Matematikçiler işaretli alanlar kavramını benimser. Bu kendini iki şekilde gösterir. İlk olarak, bazen sıfırdan küçük olan $f(x)$ fonksiyonunu düşünürsek, alan da negatif olacaktır. Yani örneğin

$$
\int_0^{1} (-1)\;dx = -1.
$$

Benzer şekilde, soldan sağa değil sağdan sola ilerleyen integraller de negatif alan olarak tanımlanır.

$$
\int_0^{-1} 1\; dx = -1.
$$

Standart alan (pozitif bir fonksiyonda soldan sağa) her zaman pozitiftir. Ters çevirerek elde edilen herhangi bir şey (örneğin, negatif bir sayının integralini elde etmek için $x$ eksenini ters çevirmek veya bir integrali yanlış sırada elde etmek için $y$ eksenini ters çevirmek gibi) negatif bir alan üretecektir. Aslında, iki kez ters yüz etme, pozitif alana sahip olmak için birbirini götüren bir çift negatif işaret verecektir.

$$
\int_0^{-1} (-1)\;dx =  1.
$$

Bu tartışma size tanıdık geliyorsa, normaldir! :numref:`sec_geometry-linear-algebraic-ops`'de determinantın işaretli alanı nasıl aynı şekilde temsil ettiğini tartıştık.


## Çoklu İntegraller
Bazı durumlarda daha yüksek boyutlarda çalışmamız gerekecektir. Örneğin, $f(x, y)$ gibi iki değişkenli bir fonksiyonumuz olduğunu ve $x$'nin $[a, b]$ ve $y$'nin ise $[c, d]$ arasında değiştiğinde $f$ altındaki hacim nedir bilmek istediğimizi varsayalım. .

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy())
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

Şöyle yazabiliriz:

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Bu integrali hesaplamak istediğimizi varsayalım. Bizim iddiamız, bunu önce $x$'deki integrali yinelemeli olarak hesaplayarak ve sonra da $y$'deki integrale kayarak yapabileceğimizdir, yani

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

Bunun neden olduğunu görelim.

Fonksiyonu $\epsilon \times \epsilon$ karelere böldüğümüzü ve $i, j$ tamsayı koordinatlarıyla indekslediğimizi düşünün. Bu durumda integralimiz yaklaşık olarak

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

Problemi ayrıklaştırdığımızda, bu karelerdeki değerleri istediğimiz sırayla toplayabiliriz ve değerleri değiştirme konusunda endişelenmeyiz. Bu, şu şekilde gösterilmektedir :numref:`fig_sum-order`. Özellikle şunu söyleyebiliriz:

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![İlk önce sütunlar üzerinden bir toplam olarak birçok karede bir toplamın nasıl ayrıştırılacağını (1), ardından sütun toplamlarının nasıl toplanacağını (2) gösterelim.](../img/SumOrder.svg)
:label:`fig_sum-order`
 
İç kısımdaki toplam, tam olarak integralin ayrıklaştırılmasıdır.

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

Son olarak, bu iki ifadeyi birleştirirsek şunu elde ettiğimize dikkat edin: 

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Böylece hepsini bir araya getirirsek, buna sahip oluruz:

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

Dikkat edin, bir kez ayrıklaştırıldı mı, yaptığımız tek şey, bir sayı listesi eklediğimiz sırayı yeniden düzenlemek oldu. Bu, burada hiçbir zorluk yokmuş gibi görünmesine neden olabilir, ancak bu sonuç (*Fubini Teoremi* olarak adlandırılır) her zaman doğru değildir! Makine öğrenmesi (sürekli fonksiyonlar) yapılırken karşılaşılan matematik türü için herhangi bir endişe yoktur, ancak başarısız olduğu durumlardan örnekler oluşturmak mümkündür (örneğin $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$ fonksiyonunu $[0,2]\times[0,1]$ dikdörtgenin üzerinde deneyin).

İntegrali önce $x$ cinsinden ve ardından $y$ cinsinden yapma seçeneğinin keyfi olduğuna dikkat edin. Önce $y$ için, sonra da $x$ için yapmayı eşit derecede iyi seçebilirdik:

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

Çoğu zaman, vektör gösterimine yoğunlaşacağız ve $U = [a, b]\times [c, d]$ için bunun şöyle olduğunu söyleyeceğiz:

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## Change of Variables in Multiple Integrals
As we with single variables in :eqref:`eq_change_var`, the ability to change variables inside a higher dimensional integral is a key tool.  Let us summarize the result without derivation.  

We need a function that reparameterizes our domain of integration.  We can take this to be $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$, that is any function which takes in $n$ real variables and returns another $n$.  To keep the expressions clean, we will assume that $\phi$ is *injective* which is to say it never folds over itself ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$).  

In this case, we can say that

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

where $D\phi$ is the *Jacobian* of $\phi$, which is the matrix of partial derivatives of $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$,

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

Looking closely, we see that this is similar to the single variable chain rule :eqref:`eq_change_var`, except we have replaced the term $\frac{du}{dx}(x)$ with $\left|\det(D\phi(\mathbf{x}))\right|$.  Let us see how we can to interpret this term.  Recall that the $\frac{du}{dx}(x)$ term existed to say how much we stretched our $x$-axis by applying $u$.  The same process in higher dimensions is to determine how much we stretch the area (or volume, or hyper-volume) of a little square (or little *hyper-cube*) by applying $\boldsymbol{\phi}$.  If $\boldsymbol{\phi}$ was the multiplication by a matrix, then we know how the determinant already gives the answer.  

With some work, one can show that the *Jacobian* provides the best approximation to a multivariable function $\boldsymbol{\phi}$ at a point by a matrix in the same way we could approximate by lines or planes with derivatives and gradients. Thus the determinant of the Jacobian exactly mirrors the scaling factor we identified in one dimension.

It takes some work to fill in the details to this, so do not worry if they are not clear now.  Let us see at least one example we will make use of later on.  Consider the integral

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

Playing with this integral directly will get us no-where, but if we change variables, we can make significant progress.  If we let $\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$ (which is to say that $x = r \cos(\theta)$, $y = r \sin(\theta)$), then we can apply the change of variable formula to see that this is the same thing as

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

where 

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

Thus, the integral is

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

where the final equality follows by the same computation that we used in section :numref:`integral_example`.  

We will meet this integral again when we study continuous random variables in :numref:`sec_random_variables`.

## Summary

* The theory of integration allows us to answer questions about areas or volumes.
* The fundamental theorem of calculus allows us to leverage knowledge about derivatives to compute areas via the observation that the derivative of the area up to some point is given by the value of the function being integrated.
* Integrals in higher dimensions can be computed by iterating single variable integrals.

## Exercises
1. What is $\int_1^2 \frac{1}{x} \;dx$?
2. Use the change of variables formula to integrate $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$.
3. What is $\int_{[0,1]^2} xy \;dx\;dy$?
4. Use the change of variables formula to compute $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ and $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$ to see they are different.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:
