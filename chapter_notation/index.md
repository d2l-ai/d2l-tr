# Notasyon
:label:`chap_notation`

Bu kitap boyunca, aşağıdaki gösterim kurallarına bağlı kalacağız. Bu sembollerden bazılarının göstermelik değişken olduğunu, bazılarının ise belirli nesnelere atıfta bulunduğunu unutmayın. Genel bir kural olarak, belirsiz "a" nesnesi, sembolün bir göstermelik değişken olduğunu ve benzer şekilde biçimlendirilmiş sembollerin aynı tipteki diğer nesneleri gösterebileceğini belirtir. Örneğin, "$x$: bir sayıl", küçük harflerin genellikle sayıl değerleri temsil ettiği anlamına gelir.



## Sayısal Nesneler


* $x$: skalar (sayıl)
* $\mathbf{x}$: vektör (Yöney)
* $\mathbf{X}$: matris (Dizey)
* $\mathsf{X}$: bir genel tensör (Gerey)
* $\mathbf{I}$: birim dizeyi -- köşegen girdileri $1$ köşegen-olmayan girdileri $0$ olan kare dizey
* $x_i$, $[\mathbf{x}]_i$: $\mathbf{x}$ dizeyinin $i.$ elemanı
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: $\mathbf{X}$ dizeyinin $i.$ satır $j.$ sütundaki elemanı



## Küme Kuramı


* $\mathcal{X}$: küme
* $\mathbb{Z}$: tam sayılar kümesi
* $\mathbb{Z}^+$: pozitif tam sayılar kümesi
* $\mathbb{R}$: gerçel sayılar kümesi
* $\mathbb{R}^n$: $n$ boyutlu gerçel sayılı yöneyler kümesi
* $\mathbb{R}^{a\times b}$: $a$ satır ve $b$ sütunlu gerçek sayılı matrisler kümesi
* $|\mathcal{X}|$: $\mathcal{X}$ kümesinin kardinalitesi (eleman sayısı)
* $\mathcal{A}\cup\mathcal{B}$: $\mathcal{A}$ ve $\mathcal{B}$ kümelerinin bileşkesi
* $\mathcal{A}\cap\mathcal{B}$: $\mathcal{A}$ ve $\mathcal{B}$ kümelerinin kesişimi
* $\mathcal{A}\setminus\mathcal{B}$: $\mathcal{B}$ kümesinin $\mathcal{A}$ kümesinden çıkarılması (sadece $\mathcal{A}$ kümesinde olup $\mathcal{B}$ kümesinde olmayan elemanları içerir)


## Fonksiyonlar ve Operatörler


* $f(\cdot)$: işlev (fonksiyon)
* $\log(\cdot)$: doğal logaritma
* $\log_2(\cdot)$: $2$lik tabanda logaritma
* $\exp(\cdot)$: üstel fonksiyon
* $\mathbf{1}(\cdot)$: gösterge fonksiyonu, mantık argümanı doğru ise $1$ ve değil ise $0$
* $\mathbf{1}_{\mathcal{X}}(z)$: küme-üyeliği gösterge işlevi, eğer $z$ elemanı $\mathcal{X}$ kümesine ait ise $1$ ve değil ise $0$
* $\mathbf{(\cdot)}^\top$: bir vektörün veya matrisin devriği
* $\mathbf{X}^{-1}$: $\mathbf{X}$ matrisinin tersi
* $\odot$: Hadamard (eleman-yönlü) çarpımı
* $[\cdot, \cdot]$: bitiştirme
* $\|\cdot\|_p$: $L_p$ büyüklüğü (Norm)
* $\|\cdot\|$: $L_2$ büyüklüğü (Norm)
* $\langle \mathbf{x}, \mathbf{y} \rangle$: $\mathbf{x}$ ve $\mathbf{y}$ vektörlerinin iç (nokta) çarpımı
* $\sum$: bir elemanlar topluluğu üzerinde toplam
* $\prod$: bir elemanlar topluluğu üzerinde toplam çarpımı
* $\stackrel{\mathrm{def}}{=}$: sol tarafta bir sembol tanımlarken kullanılan eşitlik 

## Hesaplama (Kalkülüs)

* $\frac{dy}{dx}$: $y$'nin $x$'e göre türevi
* $\frac{\partial y}{\partial x}$: $y$'nin $x$'e göre kısmi türevi
* $\nabla_{\mathbf{x}} y$:   $y$'nin $\mathbf{x}$'e göre eğimi (Gradyan)
* $\int_a^b f(x) \;dx$: $f$'in $x$'e göre $a$'dan $b$'ye belirli bir tümlevi (integrali)
* $\int f(x) \;dx$:  $f$'in $x$'e göre belirsiz bir tümlevi

## Olasılık ve Bilgi Kuramı

* $X$: rasgele değişken
* $P$: olasılık dağılımı
* $X \sim P$: $X$ rasgele değişkeni $P$ olasılık dağılımına sahiptir
* $P(X=x)$: $X$ rasgele değişkeninin $x$ değerini alma olayının olasılığı
* $P(X \mid Y)$: $Y$ bilindiğinde $X$'in koşullu olasılık dağılımı
* $p(\cdot)$: P dağılımı ile ilişkili koşullu olasılık yoğunluk fonksiyonu
* ${E}[X]$: $X$ rasgele değişkeninin beklentisi
* $X \perp Y$: $X$ ve $Y$ rasgele değişkenleri bağımsızdır
* $X \perp Y \mid Z$: $X$ ve $Y$ rasgele değişkenleri, $Z$ göz önüne alındığında (verildiğinde) koşullu olarak bağımsızdır
* $\sigma_X$: $X$ rasgele değişkeninin standart sapması
* $\mathrm{Var}(X)$: $X$ rasgele değişkeninin değişintisi (varyansı), $\sigma^2_X$'e eşittir
* $\mathrm{Cov}(X, Y)$: $X$ ve $Y$ rasgele değişkenlerinin eşdeğişintisi (kovaryansı)
* $\rho(X, Y)$: $X$ ve $Y$ rasgele değişkenleri arasındaki Pearson ilinti katsayısı (korrelasyonu), $\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$'ye eşittir
* $H(X)$: $X$ rasgele değişkeninin düzensizliği (entropisi)
* $D_{\mathrm{KL}}(P\|Q)$: $Q$ dağılımından $P$ dağılımına KL-Iraksaması (veya göreceli düzensizlik)


[Tartışmalar](https://discuss.d2l.ai/t/25)
