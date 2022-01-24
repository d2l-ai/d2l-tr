# Parametre Sunucuları
:label:`sec_parameterserver`

Tek bir GPU'dan birden fazla GPU'ya ve daha sonra birden fazla GPU içeren birden çok sunucuya geçerken, muhtemelen hepsi birden fazla raf ve ağ anahtarlarına yayıldıkça, dağıtılmış ve paralel eğitim algoritmalarımızın çok daha karmaşık hale gelmesi gerekiyor. Farklı ara bağlantıların çok farklı bant genişliğine sahip olması nedeniyle ayrıntılar önemlidir (örn. NVLink uygun bir ortamda 6 bağlantı için 100 GB/s'ye kadar sunabilir, PCIe 4.0 (16 şeritli) 32 GB/s, yüksek hızlı 100GbE Ethernet bile sadece 10 GB/s'ye ulaşır). Aynı zamanda istatistiksel bir modelleyicinin ağ ve sistemlerde uzman olmasını beklemek mantıksızdır. 

Parametre sunucusunun temel fikri, dağıtılmış gizli değişken modeller bağlamında :cite:`Smola.Narayanamurthy.2010`'da tanıtıldı. İtme ve çekme semantiğinin bir açıklaması ardından :cite:`Ahmed.Aly.Gonzalez.ea.2012` yılında izlenen :cite:`Ahmed.Aly.Gonzalez.ea.2012` ve sistem ve açık kaynak kütüphanesinin bir açıklaması ve ardından :cite:`Li.Andersen.Park.ea.2014`. Aşağıda verimlilik için gerekli bileşenleri motive edeceğiz. 

## Veri-Paralel Eğitim

Dağıtılmış eğitime veri paralel eğitim yaklaşımını inceleyelim. Pratikte uygulanması önemli ölçüde daha basit olduğu için bunu bu bölümdeki diğerlerinin hariç tutulması için kullanacağız. GPU'ların günümüzde bol miktarda belleğe sahip olduğu için paralellik için başka bir stratejinin tercih edildiği hemen hemen hiç kullanım vakası yoktur (grafikler üzerinde derin öğrenmenin yanı sıra). :numref:`sec_multi_gpu` yılında uyguladığımız veri paralelliğinin varyantını açıklamaktadır. Bunun en önemli yönü, güncelleştirilmiş parametreler tüm GPU'lara yeniden yayınlanmadan önce degradelerin toplanmasının GPU 0'da gerçekleşmesidir. 

![Left: single GPU training. Right: a variant of multi-GPU training: (1) we compute loss and gradient, (2) all gradients are aggregated on one GPU, (3) parameter update happens and the parameters are re-distributed to all GPUs.](../img/ps.svg)
:label:`fig_parameterserver`

Geçmişe bakıldığında, GPU 0 üzerinde toplamaya karar oldukça önemli görünüyor. Sonuçta, CPU üzerinde de bir araya gelebiliriz. Aslında, bazı parametreleri bir GPU'daki ve bazılarını diğerinde toplamaya bile karar verebiliriz. Optimizasyon algoritmasının bunu desteklediği şartıyla, yapamamamızın gerçek bir nedeni yoktur. Örneğin, $\mathbf{g}_1, \ldots, \mathbf{g}_4$ ilişkili degradeleri olan dört parametre vektörümüz varsa, degradeleri her $\mathbf{g}_i$ ($i = 1, \ldots, 4$) için bir GPU'da toplayabiliriz. 

Bu muhakeme keyfi ve anlamsız görünüyor. Sonuçta, matematik boyunca aynıdır. Bununla birlikte, :numref:`sec_hardware`'te tartışıldığı gibi farklı otobüslerin farklı bant genişliğine sahip olduğu gerçek fiziksel donanımlarla uğraşıyoruz. :numref:`fig_bw_hierarchy`'te açıklandığı gibi gerçek bir 4 yönlü GPU sunucusunu düşünün. Özellikle iyi bağlıysa, 100 GbE ağ kartına sahip olabilir. Daha tipik sayılar, 100 MB/s ila 1 GB/s arasında etkili bant genişliğine sahip 1—10 GbE aralığındadır. İşlemcilerin tüm GPU'lara doğrudan bağlanmak için çok az PCIe şeridi olduğundan (örneğin, tüketici sınıfı Intel CPU'ların 24 şeritli olması) [multiplexer](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches)'ya ihtiyacımız var. 16x Gen3 bağlantısındaki CPU'dan gelen bant genişliği 16 GB/s'dir. Bu aynı zamanda GPU'ların her biri* anahtara bağlı olduğu hızdır. Bu, cihazlar arasında iletişim kurmanın daha etkili olduğu anlamına gelir. 

![A 4-way GPU server.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

Argüman uğruna degradelerin 160 MB olduğunu varsayalım. Bu durumda, degradeleri kalan 3 GPU'dan dördüncüye göndermek için 30 ms gerekir (her aktarım 10 ms = 160 MB/16 GB/s alır). Ağırlık vektörlerini iletmek için 30 ms daha eklersek toplam 60 ms'ye ulaşırız. Tüm verileri CPU'ya gönderirsek, dört GPU'nun her biri* veriyi CPU'ya göndermesi gerektiğinden 40 ms ceza alırız ve toplam 80 ms elde edilir. Son olarak degradeleri her biri 40 MB'lık 4 parçaya ayırabileceğimizi varsayalım. PCIe anahtarı tüm bağlantılar arasında tam bant genişliğine sahip bir çalışma sunduğundan, artık parçaların her birini farklı bir GPU'da*eşzamanlı olarak toplayabiliriz. Bunun yerine 30 ms bir senkronizasyon işlemi için toplam 15 ms verir 7,5 ms alır. Kısacası, parametreleri nasıl senkronize ettiğimize bağlı olarak aynı işlem 15 ms'den 80 ms'ye kadar herhangi bir yere gidebilir. :numref:`fig_ps_distributed`, parametrelerin değişimi için farklı stratejileri gösterir. 

![Parameter synchronization strategies.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

[Horovod](https://github.com/horovod/horovod)'te bunun nasıl yapılacağı hakkında ayrıntılar için performance: in a deep network it takes some time to compute all gradients from the top to the bottom. We can begin synchronizing gradients for some parameter groups even while we are still busy computing them for others. See e.g., :cite:`Sergeev.Del-Balso.2018` iyileştirilmesi söz konusu olduğunda elimizde başka bir araç olduğunu unutmayın. 

## Halka Senkronizasyonu

Modern derin öğrenme donanımı üzerinde senkronizasyon söz konusu olduğunda genellikle önemli ölçüde ısmarlama ağ bağlantısıyla karşılaşırız. Örneğin, AWS p3.16xlarge ve NVIDIA DGX-2 örnekleri :numref:`fig_nvlink`'ün bağlantı yapısını paylaşır. Her GPU, en iyi 16 GB/s hızında çalışan bir PCIe bağlantısı üzerinden bir ana işlemciye bağlanır. Ayrıca her bir GPU'nun 6 NVLink bağlantısı vardır ve bunların her biri çift yönlü 300 Gbit/s'yi aktarabilir. Bu, yön başına bağlantı başına 18 GB/s civarındadır. Kısacası, toplam NVLink bant genişliği PCIe bant genişliğinden önemli ölçüde daha yüksektir. Soru, onu en verimli şekilde nasıl kullanacağıdır. 

![NVLink connectivity on 8  V100 GPU servers (image courtesy of NVIDIA).](../img/nvlink.svg)
:label:`fig_nvlink`

Optimum senkronizasyon stratejisinin ağı iki halkaya ayırmak ve bunları doğrudan :cite:`Wang.Li.Liberty.ea.2018` verileri senkronize etmek için kullanmak olduğu ortaya çıkıyor. :numref:`fig_nvlink_twoloop`, ağın çift NVLink bant genişliği ile tek bir halkaya (1-2-3-4-5-6-7-8-1) ayrıştırılabileceğini göstermektedir. Normal bant genişliği. Bu durumda verimli bir senkronizasyon protokolü tasarlamak önemsizdir. 

![Decomposition of the NVLink network into two rings.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`

Aşağıdaki düşünce deneyini düşünün: $n$ bilgi işlem düğümlerinin (veya GPU'ların) bir halka göz önüne alındığında, ilk düğümden ikinci düğüme degradeler gönderebiliriz. Orada yerel degradeye eklenir ve üçüncü düğüme gönderilir, vb. $n-1$ adımından sonra toplam degrade son ziyaret edilen düğümde bulunabilir. Yani, degradeleri toplama zamanı, düğüm sayısı ile doğrusal olarak büyür. Ama bunu yaparsak algoritma oldukça verimsiz olur. Sonuçta, herhangi bir zamanda iletişim kuran düğümlerden sadece biri vardır. Eğer degradeleri $n$ parçalarına ayırırsak ve $i$ numaralı düğümden başlayarak $i$ yığın senkronize etmeye başladıysak ne olur? Her yığın boyutu $1/n$ olduğundan toplam süre artık $(n-1)/n \approx 1$'dir. Başka bir deyişle, degradeleri birleştirmek için harcanan süre*, yüzüğün boyutunu arttırdığımızdan* büyümez. Bu oldukça şaşırtıcı bir sonuçtur. :numref:`fig_ringsync`, $n=4$ düğümlerinde adımların sırasını göstermektedir. 

![Ring synchronization across 4 nodes. Each node starts transmitting parts of gradients to its left neighbor until the assembled gradient can be found in its right neighbor.](../img/ringsync.svg)
:label:`fig_ringsync`

8 V100 GPU'da 160 MB senkronize etme örneğini kullanırsak yaklaşık $2 \cdot 160 \mathrm{MB} / (3 \cdot 18 \mathrm{GB/s}) \approx 6 \mathrm{ms}$'e ulaşırız. Bu, şu anda 8 GPU kullanıyor olsak da PCIe veri yolu kullanmaktan daha iyidir. Pratikte bu sayıların biraz daha kötü olduğunu unutmayın, çünkü derin öğrenme çerçeveleri genellikle iletişimi büyük patlama transferlerine monte edememektedir.  

Halka senkronizasyonunun diğer eşitleme algoritmalarından temelde farklı olduğu ortak bir yanlış anlaşılma olduğunu unutmayın. Tek fark, senkronizasyon yolunun basit bir ağaçla karşılaştırıldığında biraz daha ayrıntılı olmasıdır. 

## Çoklu Makine Eğitimi

Birden fazla makinede dağıtılmış eğitim daha da zorluk getirir: Bazı durumlarda daha yavaş bir büyüklük sırası üzerinde olabilen nispeten daha düşük bant genişliğindeki bir yapıya bağlı sunucularla iletişim kurmamız gerekir. Cihazlar arasında senkronizasyon zor. Sonuçta, eğitim kodunu çalıştıran farklı makineler ustaca farklı bir hıza sahip olacaktır. Bu nedenle senkron dağıtılmış optimizasyonu kullanmak istiyorsak bunları senkronize etmeliyiz. :numref:`fig_ps_multimachine` dağıtılmış paralel eğitimin nasıl gerçekleştiğini göstermektedir. 

1. Her makinede bir (farklı) veri grubu okunur, birden fazla GPU'ya bölünür ve GPU belleğine aktarılır. Tahminler ve degradeler her GPU toplu işleminde ayrı ayrı hesaplanır.
2. Tüm yerel GPU'lardan gelen degradeler tek bir GPU'da toplanır (veya bunların bir kısmı farklı GPU'lar üzerinde toplanır).
3. Degradeler CPU'lara gönderilir.
4. CPU'lar degradeleri tüm degradeleri toplayan bir merkezi parametre sunucusuna gönderir.
5. Toplam degradeler daha sonra parametreleri güncelleştirmek için kullanılır ve güncelleştirilmiş parametreler tek tek CPU'lara geri yayınlanır.
6. Bilgiler bir (veya birden çok) GPU'ya gönderilir.
7. Güncelleştirilmiş parametreler tüm GPU'lara yayılır.

![Multi-machine multi-GPU distributed parallel training.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Bu operasyonların her biri oldukça basit görünüyor. Ve aslında, tek bir makine ile* verimli bir şekilde gerçekleştirilebilirler. Birden fazla makineye baktığımızda, merkezi parametre sunucusunun darboğaz olduğunu görebiliriz. Sonuçta, sunucu başına bant genişliği sınırlıdır, bu nedenle $m$ işçi için tüm degradeleri sunucuya göndermek için gereken süre $\mathcal{O}(m)$'dir. Sunucu sayısını $n$'e yükselterek bu engeli aşabiliriz. Bu noktada her sunucunun yalnızca parametrelerin $\mathcal{O}(1/n)$'unu depolaması gerekir, bu nedenle güncellemeler ve optimizasyon için toplam süre $\mathcal{O}(m/n)$ olur. Her iki sayının da eşleştirilmesi, kaç işçi ile uğraştığımıza bakılmaksızın sürekli ölçeklendirme sağlar. Uygulamada, *aynı* makineleri hem işçi hem de sunucu olarak kullanıyoruz. :numref:`fig_ps_multips` tasarımı göstermektedir (ayrıntılar için ayrıca bkz. :cite:`Li.Andersen.Park.ea.2014`). Özellikle, birden fazla makinenin makul olmayan gecikmeler olmadan çalışmasını sağlamak önemsizdir. Engellerle ilgili ayrıntıları atlıyoruz ve aşağıda senkronize olmayan ve senkronize olmayan güncellemelere kısa bir süre değineceğiz. 

![Top: a single parameter server is a bottleneck since its bandwidth is finite. Bottom: multiple parameter servers store parts of the parameters with aggregate bandwidth.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## Anahtar-Değerli Mağazalar

Dağıtılmış çoklu GPU eğitimi için gerekli adımları uygulamada uygulamak önemsiz değildir. Bu nedenle, yeniden tanımlanmış güncelleme semantiği olan bir *anahtar-değer deposunun* ortak bir soyutlama kullanmak için ödeme yapar.  

Birçok işçi ve birçok GPU arasında $i$ degrade hesaplaması 

$$\mathbf{g}_{i} = \sum_{k \in \text{workers}} \sum_{j \in \text{GPUs}} \mathbf{g}_{ijk},$$

nerede $\mathbf{g}_{ijk}$ degrade parçasıdır $i$ GPU bölünmüş $j$ işçi $k$. Bu işlemdeki en önemli husus, bir *değişken indirgeme*, yani birçok vektörü bir hale getirmesi ve işlemin uygulandığı sıranın önemli olmadığıdır. Hangi degradenin alındığı zaman üzerinde ince taneli kontrole sahip olmadığımız için bu, amaçlarımız için mükemmeldir. Ayrıca, bu işlemin farklı $i$ arasında bağımsız olduğunu unutmayın. 

Bu, aşağıdaki iki işlemi tanımlamamıza olanak sağlar: degradeleri biriken *itme* ve toplam degradeleri alan *pull*. Birçok farklı degrade kümesine sahip olduğumuzdan (sonuçta birçok katmana sahibiz), degradeleri bir anahtarla endekslememiz gerekiyor $i$. Dynamo :cite:`DeCandia.Hastorun.Jampani.ea.2007`'te tanıtılan anahtar değerli depolarla olan bu benzerlik tesadüf eseri değildir. Bunlar da, özellikle parametreleri birden çok sunucuya dağıtmak söz konusu olduğunda birçok benzer özelliği tatmin ederler. 

Anahtar değer depoları için itme ve çekme işlemleri aşağıdaki gibi açıklanmıştır: 

* **push (anahtar, değer) ** bir worker (işçi) öğesinden ortak bir depolamaya belirli bir degrade (değer) gönderir. Orada değer toplanır, örneğin, özetlenerek.
* **pull (anahtar, değer) ** ortak depolamadan, örneğin tüm işçi öğelerinin degradelerini birleştirdikten sonra toplam bir değer alır.

Basit bir itme ve çekme işleminin arkasındaki senkronizasyonla ilgili tüm karmaşıklığı gizleyerek, optimizasyonu basit terimlerle ifade edebilmek isteyen istatistiksel modelleyicilerin endişelerini ve dağıtılmış senkronizasyonun doğasında olan karmaşıklıkla başa çıkması gereken sistem mühendislerinin endişelerini ayırabiliriz. 

## Özet

* Senkronizasyonun, bir sunucu içindeki belirli ağ altyapısına ve bağlantısına son derece uyarlanabilir olması gerekir. Bu, senkronize etmek için gereken süreye göre önemli bir fark yaratabilir.
* Halka senkronizasyonu p3 ve DGX-2 sunucuları için en uygun olabilir. Diğerleri için muhtemelen o kadar değil.
* Artırılmış bant genişliği için birden fazla parametre sunucusu eklerken hiyerarşik eşitleme stratejisi iyi çalışır.

## Egzersizler

1. Halka senkronizasyonunu daha da artırabilir misin? İpucu: Her iki yönde de mesaj gönderebilirsiniz.
1. Asenkron iletişime izin vermek mümkün mü (hesaplama hala devam ederken)? Performansı nasıl etkiler?
1. Uzun süren bir hesaplama sırasında bir sunucuyu kaybedersek ne olur? Hesaplamanın tamamen yeniden başlatılmasını önlemek için bir *hata toleransı* mekanizması nasıl tasarlayabiliriz?

[Discussions](https://discuss.d2l.ai/t/366)
