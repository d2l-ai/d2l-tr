#  Ön Hazırlık
:label:`chap_preliminaries`

Derin öğrenmeye başlamak için birkaç temel beceri geliştirmemiz gerekecek.
Tüm makine öğrenmesi verilerden bilgi çıkarmakla ilgilidir.
Bu nedenle, verileri depolamak, oynama yapmak (manipule etmek) ve ön işlemek için pratik becerileri öğrenerek başlayacağız.

Dahası, makine öğrenmesi tipik olarak satırların örneklere ve sütunların niteliklere karşılık geldiği tablolar olarak düşünebileceğimiz büyük veri kümeleriyle çalışmayı gerektirir.
Doğrusal cebir, tablo verileriyle çalışmak için bize bir dizi güçlü teknik sunar.
Boyumuzu aşan sulara çok fazla girmeyeceğiz, daha ziyade dizey (matris) işlemlerinin temeline ve bunların uygulanmasına odaklanacağız.

Ek olarak, derin öğrenme tamamen eniyileme (optimizasyon) ile ilgilidir.
Bazı parametrelere sahip bir modelimiz var ve verilerimize *en uygun olanları* bulmak istiyoruz.
Bir algoritmanın her adımında her bir parametreyi hangi şekilde hareket ettireceğini karar vermek için, burada kısaca bahsedeceğiz, bir miktar hesaplama (kalkülüs) gerekir.
Neyse ki, `autograd` paketi bizim için otomatik olarak türevleri hesaplar; bunu daha sonra işleyeceğiz.

Dahası, makine öğrenmesi tahminlerde bulunmakla ilgilidir: Gözlemlediğimiz bilgiler göz önüne alındığında, bazı bilinmeyen özelliklerin olası değeri nedir?
Belirsizlik altında titizlikle çıkarsama yapabilmek için olasılık dilini hatırlamamız gerekecek.

Hakikatinde, asli kaynaklar bu kitabın ötesinde birçok açıklama ve örnek sunmaktadır.
Bölümü bitirken size gerekli bilgiler için kaynaklara nasıl bakacağınızı göstereceğiz.

Bu kitap, derin öğrenmeyi doğru bir şekilde anlamak için gerekli olan matematiksel içeriği en azda tutmuştur.
Ancak, bu, bu kitabın matematik içermediği anlamına gelmez.
Bu nedenle, bu bölüm, herhangi bir kişinin, kitabın matematiksel içeriğinin en azından *çoğunu* anlayabilmesi için temel ve sık kullanılan matematiğe hızlı bir giriş yapmasını sağlar.
Matematiksel içeriğin *tümünü* anlamak istiyorsanız, [matematik üzerine çevrimiçi ek](https://tr.d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)i derinlemesine gözden geçirmek yeterli olacaktır.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

