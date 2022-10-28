# Parametre Sunucuları
:label:`sec_parameterserver`

Tek bir GPU'dan birden çok GPU'ya ve ardından, muhtemelen tümü birden çok rafa ve ağ anahtarına yayılmış birden çok GPU içeren birden çok sunucuya geçerken, dağıtılmış ve paralel eğitim için algoritmalarımızın çok daha karmaşık hale gelmesi gerekiyor. Farklı ara bağlantılarının çok farklı bant genişliğine sahip olması nedeniyle ayrıntılar önemlidir (örn. NVLink uygun bir ortamda 6 bağlantı için 100 GB/s'ye kadar sunabilir, PCIe 4.0 (16 şeritli) 32 GB/s, yüksek hızlı 100GbE Ethernet bile sadece 10 GB/s'ye ulaşır). Aynı zamanda istatistiksel bir modelleyicinin ağ ve sistemlerde uzman olmasını beklemek mantıksızdır. 

Parametre sunucusunun temel fikri, dağıtılmış gizli değişken modeller bağlamında :cite:`Smola.Narayanamurthy.2010`'da tanıtıldı. İtme ve çekme anlambiliminin bir açıklaması, :cite:`Ahmed.Aly.Gonzalez.ea.2012`, ardından onu izleyen :cite:`Ahmed.Aly.Gonzalez.ea.2012` ve sistem ve açık kaynak kütüphanesinin bir açıklaması ve ardından onu izleyen :cite:`Li.Andersen.Park.ea.2014` bu konudaki çalışmalardır. Aşağıda verimlilik için gerekli bileşenleri motive edeceğiz. 

## Veri-Paralel Eğitim

Dağıtılmış eğitime veri paralel eğitim yaklaşımını inceleyelim. Pratikte uygulanması önemli ölçüde daha basit olduğu için, bu bölümdeki diğerlerini hariç tutmak için bunu kullanacağız. GPU'ların günümüzde çok fazla belleği olduğundan, paralellik için başka herhangi bir stratejinin tercih edildiği (grafikler üzerinde derin öğrenmenin yanı sıra) neredeyse hiçbir kullanım durumu yoktur. :numref:`sec_multi_gpu` içinde uyguladığımız veri paralelliğinin varyantını açıklamaktadır. Bunun en önemli yönü, güncelleştirilmiş parametreler tüm GPU'lara yeniden yayınlanmadan önce gradyanların toplanmasının GPU 0'da gerçekleşmesidir. 

![Sol: Tek GPU eğitimi. Sağda: Çoklu GPU eğitiminin bir çeşidi: (1) kaybı ve gradyanı hesaplarız, (2) tüm gradyanlar tek bir GPU'da toplanır, (3) parametre güncellemesi gerçekleşir ve parametreler tüm GPU'lara yeniden dağıtılır.](../img/ps.svg)
:label:`fig_parameterserver`

Geriye dönüp bakıldığında, GPU 0'da toplama kararı oldukça geçici görünüyor. Sonuçta, CPU üzerinde de bir araya getirebiliriz. Aslında, bazı parametreleri bir GPU'da ve bazılarını diğerinde toplamaya bile karar verebiliriz. Optimizasyon algoritmasının bunu desteklemesi şartıyla, yapamamamızın gerçek bir nedeni yoktur. Örneğin, $\mathbf{g}_1, \ldots, \mathbf{g}_4$ ile ilişkili gradyanları olan dört parametre vektörümüz varsa, gradyanları her $\mathbf{g}_i$ ($i = 1, \ldots, 4$) için bir GPU'da toplayabiliriz. 

Bu muhakeme keyfi ve anlamsız görünüyor. Sonuçta, matematik baştan sona aynıdır. Bununla birlikte, :numref:`sec_hardware` içinde tartışıldığı gibi farklı veri yollarının farklı bant genişliğine sahip olduğu gerçek fiziksel donanımlarla uğraşıyoruz. :numref:`fig_bw_hierarchy` şeklinde açıklandığı gibi gerçek bir 4 yönlü GPU sunucusunu düşünün. Özellikle iyi bağlıysa, 100 GbE ağ kartına sahip olabilir. Daha tipik sayılar, 100 MB/sn ila 1 GB/sn arasında etkili bant genişliğine sahip 1—-10 GbE aralığındadır. İşlemcilerin tüm GPU'lara doğrudan bağlanmak için çok az PCIe şeridi olduğundan (örneğin, tüketici sınıfı Intel CPU'ların 24 şeritli olması) [çoklayıcı](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches)ya ihtiyacımız var. 16x Gen3 bağlantısındaki CPU'dan gelen bant genişliği 16 GB/sn'dir. Bu aynı zamanda GPU'ların *her birinin* anahtara bağlanma hızıdır. Bu, cihazlar arasında iletişim kurmanın daha etkili olduğu anlamına gelir. 

![A 4-yönlü GPU sunucusu.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

Tartışma uğruna gradyanların 160 MB olduğunu varsayalım. Bu durumda, gradyanları kalan 3 GPU'dan dördüncüye göndermek için 30 ms gerekir (her aktarım 10 ms = 160 MB/16 GB/s alır). Ağırlık vektörlerini iletmek için 30 ms daha eklersek toplam 60 ms'ye ulaşırız. Tüm verileri CPU'ya gönderirsek, dört GPU'nun *her birinin* verileri CPU'ya göndermesi gerektiğinden, toplam 80 ms verimle 40 ms'lik bir cezaya maruz kalırız. Son olarak gradyanları her biri 40 MB'lık 4 parçaya ayırabileceğimizi varsayalım. PCIe anahtarı tüm bağlantılar arasında tam bant genişliği işlemi sunduğundan, artık parçaların her birini farklı bir GPU'da *eşzamanlı olarak* toplayabiliriz. 30 ms yerine bu 7.5 ms sürer ve bir eşzamanlama işlemi toplam 15 ms sürer. Kısacası, parametreleri nasıl senkronize ettiğimize bağlı olarak aynı işlem 15 ms'den 80 ms'ye kadar herhangi bir zaman sürebilir. :numref:`fig_ps_distributed`, parametrelerin değişimi için farklı stratejileri gösterir. 

![Parametre eşzamanlama stratejileri.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

Performansı artırmaya gelince elimizde başka bir araç daha olduğunu unutmayın: Derin bir ağda yukarıdan aşağıya tüm gradyanları hesaplamak biraz zaman alır. Bazı parametre grupları için gradyanları, diğerleri için hesaplamakla meşgulken bile senkronize etmeye başlayabiliriz. Örneğin bkz. :cite:`Sergeev.Del-Balso.2018` ve bunun nasıl yapılacağına ilişkin ayrıntılar için [Horovod](https://github.com/horovod/horovod) adresine bakınız.

## Halka Eşzamanlaması

Modern derin öğrenme donanımı üzerinde eşzamanlama söz konusu olduğunda genellikle önemli ölçüde ısmarlama ağ bağlantısıyla karşılaşırız. Örneğin, AWS p3.16xlarge ve NVIDIA DGX-2 örnekleri :numref:`fig_nvlink` şeklindeki bağlantı yapısını paylaşır. Her GPU, en iyi 16 GB/s hızında çalışan bir PCIe bağlantısı üzerinden bir ana işlemciye bağlanır. Ayrıca her bir GPU'nun 6 NVLink bağlantısı vardır ve bunların her biri çift yönlü 300 Gbit/s'yi aktarabilir. Bu, yön başına bağlantı başına 18 GB/s civarındadır. Kısacası, toplam NVLink bant genişliği PCIe bant genişliğinden önemli ölçüde daha yüksektir. Soru, onu en verimli şekilde nasıl kullanacağıdır. 

![8 tane V100 GPU sunucularında NVLink bağlantısı (İmge NVIDIA'nın izniyle verilmektedir).](../img/nvlink.svg)
:label:`fig_nvlink`

En uygun eşzamanlama stratejisinin ağı iki halkaya ayırmak ve bunları doğrudan verileri senkronize etmek için kullanmak olduğu ortaya çıkıyor :cite:`Wang.Li.Liberty.ea.2018`. :numref:`fig_nvlink_twoloop`, ağın çift NVLink bant genişliği ile bir (1-2-3-4-5-6-7-8-1) halkasına ve düzenli bant genişliği ile bir (1-4-6-3-5-8-2-7-1) halkasına ayrılabileceğini gösterir. Bu durumda verimli bir eşzamanlama protokolü tasarlamak önemsizdir. 


![NVLink ağının iki halkaya ayrıştırılması.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`

Aşağıdaki düşünce deneyini düşünün: $n$ tane bilgi işlem düğümlü (veya GPU'lu) bir halka göz önüne alındığında, ilk düğümden ikinci düğüme gradyanlar gönderebiliriz. Orada yerel gradyana eklenir ve üçüncü düğüme gönderilir, vb. $n-1$ adımından sonra toplam gradyan son ziyaret edilen düğümde bulunabilir. Yani, gradyanları toplama zamanı, düğüm sayısı ile doğrusal olarak büyür. Ama bunu yaparsak algoritma oldukça verimsiz olur. Sonuçta, herhangi bir zamanda iletişim kuran sadece bir düğüm vardır. Gradyanları $n$ parçaya bölersek ve $i$ düğümünden başlayarak $i$ öbeğini senkronize etmeye başlarsak ne olur? Her yığın boyutu $1/n$ olduğundan toplam süre artık $(n-1)/n \approx 1$'dir. Başka bir deyişle, gradyanları birleştirmek için harcanan süre, halkanın boyutunu arttırdığımızdan *büyümez*. Bu oldukça şaşırtıcı bir sonuçtur. :numref:`fig_ringsync`, $n=4$ düğümün adımların sırasını göstermektedir. 

![4 düğüm arasında halka senkronizasyonu. Her düğüm, birleştirilmiş gradyan sağ komşusunda bulunana kadar gradyan parçalarını sol komşusuna iletmeye başlar.](../img/ringsync.svg)
:label:`fig_ringsync`

8 V100 GPU'da 160 MB senkronize etme örneğini kullanırsak yaklaşık $2 \cdot 160 \mathrm{MB} / (3 \cdot 18 \mathrm{GB/s}) \approx 6 \mathrm{ms}$'e ulaşırız. Bu, şu anda 8 GPU kullanıyor olsak da PCIe veri yolu kullanmaktan daha iyidir. Pratikte bu sayıların biraz daha kötü olduğunu unutmayın, çünkü derin öğrenme çerçeveleri genellikle iletişimi büyük çoğuşma transferlerinde birleştirmeyi başaramaz.

Halka eşzamanlama diğer eşzamanlama algoritmalarından temelde farklı olmasının yaygın bir yanlış anlaşılma olduğunu unutmayın. Tek fark, eşzamanlama yolunun basit bir ağaçla karşılaştırıldığında biraz daha ayrıntılı olmasıdır. 

## Çoklu Makine Eğitimi

Birden fazla makinede dağıtılmış eğitim ek bir zorluk getirir: Yalnızca, bazı durumlarda çok daha yavaş olabilen, nispeten daha düşük bant genişliğine sahip bir yapı üzerinden bağlanan sunucularla iletişim kurmamız gerekiyor. Cihazlar arasında eşzamanlama çetrefillidir. Sonuçta, eğitim kodu çalıştıran farklı makineler çok farklı hızlara sahip olacaktır. Bu nedenle eşzamanlı dağıtılmış optimizasyonu kullanmak istiyorsak bunları senkronize etmeliyiz. :numref:`fig_ps_multimachine` dağıtılmış paralel eğitimin nasıl gerçekleştiğini göstermektedir. 

1. Her makinede bir (farklı) veri grubu okunur, birden fazla GPU'ya bölünür ve GPU belleğine aktarılır. Tahminler ve gradyanlar her GPU toplu işleminde ayrı ayrı hesaplanır.
2. Tüm yerel GPU'lardan gelen gradyanlar tek bir GPU'da toplanır (veya bunların bir kısmı farklı GPU'lar üzerinde toplanır).
3. Gradyanlar CPU'lara gönderilir.
4. CPU'lar gradyanları tüm gradyanları toplayan bir merkezi parametre sunucusuna gönderir.
5. Toplanan gradyanlar daha sonra parametreleri güncelleştirmek için kullanılır ve güncelleştirilmiş parametreler tek tek CPU'lara geri yayınlanır.
6. Bilgiler bir (veya birden çok) GPU'ya gönderilir.
7. Güncelleştirilmiş parametreler tüm GPU'lara yayılır.

![Çoklu makine çoklu GPU dağıtılmış paralel eğitim.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Bu operasyonların her biri oldukça basit görünüyor. Aslında, tek bir makinede verimli bir şekilde gerçekleştirilebilirler. Birden fazla makineye baktığımızda, merkezi parametre sunucusunun darboğaz haline geldiğini görebiliriz. Sonuçta, sunucu başına bant genişliği sınırlıdır, bu nedenle $m$ işçi makine için tüm gradyanları sunucuya göndermek için gereken süre $\mathcal{O}(m)$'dir. Sunucu sayısını $n$'e yükselterek bu engeli aşabiliriz. Bu noktada her sunucunun yalnızca parametrelerin $\mathcal{O}(1/n)$'unu depolaması gerekir, bu nedenle güncellemeler ve optimizasyon için toplam süre $\mathcal{O}(m/n)$ olur. Her iki sayının da eşleştirilmesi, kaç işçi makine ile uğraştığımıza bakılmaksızın sürekli ölçeklendirme sağlar. Uygulamada, *aynı* makineleri hem işçi makine hem de sunucu olarak kullanıyoruz. :numref:`fig_ps_multips` tasarımı göstermektedir (ayrıntılar için ayrıca bkz. :cite:`Li.Andersen.Park.ea.2014`). Özellikle, birden fazla makinenin makul olmayan gecikmeler olmadan çalışmasını sağlamak önemsizdir. Engellerle ilgili ayrıntıları atlıyoruz ve aşağıda eşzamanlı olan ve olmayan güncellemelere kısaca değineceğiz. 

![Üst: Bant genişliği sınırlı olduğundan tek parametreli bir sunucu bir darboğazdır. Alt: Birden çok parametre sunucusu, toplam bant genişliğine sahip parametrelerin bölümlerini depolar.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## Anahtar--Değer Depoları

Pratikte dağıtılmış çoklu GPU eğitimi için gerekli adımların uygulanması önemsiz değildir. Bu nedenle, yeniden tanımlanmış güncelleme semantiğine sahip bir *anahtar-değer deposu* gibi ortak bir soyutlama kullanmak faydalı olur. 

Birçok işçi makine ve birçok GPU arasında $i$ gradyan hesaplaması aşağıdadır:

$$\mathbf{g}_{i} = \sum_{k \in \text{işciler}} \sum_{j \in \text{GPUs}} \mathbf{g}_{ijk},$$

burada $\mathbf{g}_{ijk}$, işçi $k$ GPU $j$'da bölünmüş $i$ gradyanının bir parçasıdır. Bu işlemdeki kilit nokta, bunun bir *değişken indirgeme* olmasıdır, yani birçok vektörü bire çevirir ve işlemin uygulanma sırası önemli değildir. Hangi gradyanın ne zaman alındığı üzerinde hassas kontrole sahip olmadığımızdan bu, amaçlarımız için mükemmeldir. Ayrıca, bu işlemin farklı $i$'ler arasında bağımsız olduğunu unutmayın. 

Bu, aşağıdaki iki işlemi tanımlamamıza olanak sağlar: Gradyanları biriken *itme* ve toplam gradyanları alan *çekme*. Birçok farklı gradyan kümesine sahip olduğumuzdan (sonuçta birçok katmana sahibiz), gradyanları bir $i$ anahtarı ile indekslememiz gerekiyor. Dynamo'da :cite:`DeCandia.Hastorun.Jampani.ea.2007` tanıtıldığı gibi, anahtar--değer depolarına olan bu benzerlik tesadüf değildir. Bunlar da, özellikle parametreleri birden çok sunucuya dağıtmak söz konusu olduğunda birçok benzer özelliği karşılar. 

Anahtar-değer depoları için itme ve çekme işlemleri aşağıdaki gibi açıklanır: 

* **itme (anahtar, değer)** bir işçiden ortak bir depolamaya belirli bir gradyan (değer) gönderir. Orada değer biriktirilir, örneğin, toplanarak.
* **çekme (anahtar, değer)** ortak depodan, örneğin tüm işçi öğelerinin gradyanlarını birleştirdikten sonra toplam değeri alır.

Basit bir itme ve çekme işleminin arkasındaki eşzamanlama ile ilgili tüm karmaşıklığı gizleyerek, optimizasyonu basit terimlerle ifade edebilmek isteyen istatistiksel modelleyicilerin endişelerini ve dağıtılmış senkronizasyonun doğasında olan karmaşıklıkla başa çıkması gereken sistem mühendislerinin endişelerini ayırabiliriz. 

## Özet

* Eşzamanlamanın, bir sunucu içindeki belirli ağ altyapısına ve bağlantısına son derece uyarlanabilir olması gerekir. Bu, eşzamanlama için gereken sürede önemli bir fark yaratabilir.
* Halka senkronizasyonu p3 ve DGX-2 sunucuları için en uygun olabilir. Diğerleri için muhtemelen o kadar değildir.
* Artırılmış bant genişliği için birden fazla parametre sunucusu eklerken hiyerarşik eşzamanlama stratejisi iyi çalışır.

## Alıştırmalar

1. Halka eşzamanlamasını daha da artırabilir misin? İpucu: Her iki yönde de mesaj gönderebilirsiniz.
1. Eşzamansız iletişime izin vermek mümkün mü (hesaplama hala devam ederken)? Performansı nasıl etkiler?
1. Uzun süren bir hesaplama sırasında bir sunucuyu kaybedersek ne olur? Hesaplamanın tamamen yeniden başlatılmasını önlemek için bir *hata toleransı* mekanizmasını nasıl tasarlayabiliriz?

[Tartışmalar](https://discuss.d2l.ai/t/366)
