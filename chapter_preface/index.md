# Önsöz

Sadece birkaç yıl önce, büyük şirket ve girişimlerde akıllı ürün ve hizmetler
geliştiren lejyoner derin öğrenme bilim insanları yoktu.
Aramızdaki en gencimiz (biz yazarlar) alana girdiğinde,
makine öğrenimi günlük gazetelerde manşetlere çıkmıyordu.
Ebeveynlerimizin, bırakın onu neden tıpta veya hukukta bir kariyere tercih
ettiğimizi, makine öğrenmesinin ne olduğu hakkında hiçbir fikri yoktu.
Makine öğrenmesi gerçek dünyada dar uygulama alanlı ileriye dönük bir akademik
disiplindi. Örneğin, konuşma tanıma ve bilgisayarlı görme benzeri uygulamalar o
kadar çok alan bilgisi gerektiriyordu ki makine öğrenmesinin küçük bir
bileşenleri olduğu tamamen ayrı alanlar olarak kabul ediliyordu. Sinir ağları,
bu kitapta odaklandığımız derin öğrenme modellerinin öncülleri, modası geçmiş
araçlar olarak görülüyordu.

Sadece son beş yılda, derin öğrenme dünyayı şaşırttı ve  bilgisayarlı görmeden
doğal dil işleme, otomatik konuşma tanıma, pekiştirici öğrenme ve istatistiksel
modellemeye kadar farklı alanlarda hızlı ilerlemeyi sağladı.
Elimizdeki bu ilerlemelerle, artık kendilerini her zamankinden daha fazla
özerklikle (ve bazı şirketlerin sizi inandırdığından daha az özerklikle)
kullanan otomobiller, otomatik olarak en sıradan e-postaları hazırlayarak
insanların ezdirici büyüklükte gelen kutularından çıkmasına yardımcı olan
akıllı yanıt sistemleri ve bir zamanlar Go gibi masa oyunlarında dünyanın en
iyi insanlarına hükmeden, ki onlarca yıl uzakta bir özellik olarak
tahmin ediliyordu, yazılım etmenleri üretebiliyoruz. Bu araçlar endüstri ve
toplum üzerinde şimdiden geniş etkiler yaratıyor; filmlerin yapılma şeklini
değiştiriyor, hastalıklar teşhis ediliyor ve temel bilimlerde, astrofizikten
biyolojiye kadar, büyüyen bir rol oynuyor.


## Bu kitap hakkında

Bu kitap, derin öğrenmeyi ulaşılabilir yapma girişimimizi temsil eder,
size *kavramları*, *bağlamı* ve *kodu* öğretir.


### Kod, Matematik ve HTML'yi Bir Arada Birleştirme

Herhangi bir bilgi işlem teknolojisinin tam etkisine ulaşması için,
iyi anlaşılmış, iyi belgelenmiş ve olgun ve iyi bakımlı araçlar desteklenmesi gerekir.
Anahtar fikirler açıkça damıtılmalı ve yeni uygulayıcıları güncel hale getirmek
için gereken işi öğrenme süresi en aza indirilmelidir. Olgun kütüphaneler
ortak görevleri otomatikleştirmeli ve örnek kod uygulayıcıların ortak
uygulamaları ihtiyaçlarına göre değiştirmesini, uygulamasını ve genişletmesini
kolaylaştırmalıdır. Dinamik web uygulamalarını örnek olarak alalım.
1990'larda başarılı veritabanı tabanlı web uygulamaları geliştiren, Amazon gibi,
çok sayıda şirket olmasına rağmen, bu teknolojinin yaratıcı girişimcilere yardım
etme potansiyeli son on yılda kısmen, güçlü, iyi belgelenmiş çerçevelerin
geliştirilmesi sayesinde çok daha büyük bir oranda gerçekleşti.


Derin öğrenmenin potansiyelini test etmek benzersiz zorluklar getirir
çünkü herhangi bir uygulama çeşitli disiplinleri bir araya getirir.
Derin öğrenmeyi uygulamak aynı anda
(i) belirli bir şekilde bir problemi çözme motivasyonları;
(ii) belirli bir modelleme yaklaşımının matematiği;
(iii) modellerin verilere uyumu için optimizasyon algoritmaları ve
(iv) modelleri verimli bir şekilde eğitmek için gerekli mühendislik,
sayısal hesaplama gizli tuzaklarında gezinmeyi
ve mevcut donanımdan en iyi şekilde yararlanmayı
anlamayı gerektirir.
Sorunları formüle etmek için gerekli eleştirel düşünme becerilerini,
onları çözmek için matematiği ve bunları uygulamak için yazılım araçları
çözümlerini hepsi tek bir yerde öğretmek dişli zorluklar sunar.
Bu kitaptaki amacımız istekli uygulayıcılara hız kazandıran birleşik
bir kaynak sunmaktır.


Bu kitap projesine başladığımızda, eş zamanlı olarak
(i) güncel olan, (ii) modern makine öğreniminin tamamını geniş bir teknik
derinlikle kapsayan ve (iii) ilgi çekici bir ders kitabından beklenen kaliteyi
uygulamalı derslerde bulmayı beklediğiniz temiz çalıştırılabilir kod ile içiçe  
serpiştirilmiş olarak sunan hiçbir kaynak yoktu.
Belirli bir derin öğrenme çerçevesinin nasıl kullanıldığı
(örneğin, TensorFlow'daki matrislerle temel sayısal hesaplamanın nasıl
yapıldığı) veya belirli tekniklerin uygulandığı (ör. LeNet, AlexNet, ResNets,
vb. için kod snippet'leri) gösteren çeşitli blog yayınlarına ve GitHub
depolarına dağılmış için birçok kod örneği bulduk  .
Bununla birlikte, bu örnekler genellikle belirli bir yaklaşımı *nasıl*
uygulayacağına odaklanmakta, ancak bazı algoritmik kararların *neden* verildiği
tartışmasını dışlamaktaydı. Ara sıra bazı etkileşimli kaynaklar yalnızca derin
öğrenmedeki belirli bir konuyu ele almak için ortaya çıkmış olsa da, örneğin
[Distill] (http://distill.pub) web sitesinde veya kişisel bloglarda yayınlanan
ilgi çekici blog yayınları, genellikle ilişkili koddan yoksundu.
Öte yandan, ortaya çıkmış birçok ders kitabı,
en önemlisi :cite:`Goodfellow.Bengio.Courville.2016` dir
ki derin öğrenmenin arkasındaki kavramların kapsamlı bir araştırmasını sunar,
bu kaynaklar kavramların kod olarak açıklamalı gerçekleşmeleriyle birleştirmez
ve bazen okuyucuları nasıl uygulayacakları konusunda fikirsiz bırakırlar.
Ayrıca, birçok kaynak ticari kurs sağlayıcılarının ödeme duvarlarının arkasında
gizlenmiştir.


Biz yola çıkarken
(i) herkesin erişimine açık olan;
(ii) hakiki bir uygulamalı makine öğrenimi bilim insanı olma yolunda başlangıç
noktası sağlamak için yeterli teknik derinlik sunan;
(iii) okuyuculara pratikte sorunları *nasıl* çözeceklerini gösteren
çalıştırılabilir kod içeren;
(iv) ayrıca genel olarak hem toplum hem de biz tarafından hızlı güncellemelere
izin veren ve
(v) teknik detayların etkileşimli tartışılması ve soruların cevaplanması için
bir [forum](http://discuss.d2l.ai) tarafından tamamlanan
bir kaynak oluşturmayı hedefledik.


Bu hedefler genellikle çatışıyordu.
Denklemler, teoremler ve alıntılar en iyi şekilde LaTeX'te düzenlenir ve
yönetilir.
Kod en iyi Python'da açıklanır.
Web sayfaları HTML ve JavaScript'te doğaldir.
Ayrıca içeriğin hem yürütülebilir kod, fiziksel bir kitap, indirilebilir bir PDF
olarak hem de internette bir web sitesi olarak erişilebilir olmasını istiyoruz.
Şu anda bu taleplere tam olarak uygun hiçbir alet ve iş akışı yok, bu yüzden
kendimiz bir araya getirmek zorunda kaldık.
Yaklaşımımızı ayrıntılı olarak şurada
açıklıyoruz :numref:`sec_how_to_contribute`.
Kaynağı paylaşmak ve düzenlemelere izin vermek için GitHub'a,
kod, denklemler ve metin karıştırmak için Jupyter not defterlerine,
çoklu çıktılar oluşturmak için bir oluşturma motoru olarak Sphinx'e ve
forum için Söylem'e karar verdik.
Sistemimiz henüz mükemmel olmasa da, bu seçenekler rakip endişeler arasında iyi
bir uzlaşma sağlamaktadır. Bunun böyle bir tümleşik iş akışı kullanılarak
yayınlanan ilk kitap olabileceğine inanıyoruz.


### Yaparak öğrenmek

Birçok ders kitabı, her biri ayrıntılı olarak bir dizi konuyu öğretir.
Örneğin, Chris Bishop'un mükemmel ders kitabı :cite:`Bishop.2006`,
her konuyu o kadar titizlikle öğretir ki,
doğrusal regresyon konusunda bile hatrı sayılır bir çalışma gerektirir.
Uzmanlar bu kitabı tam olarak bu titizliğinden dolayı sevmekle birlikte,
yeni başlayanlar için bu özellik bu kitabın giriş metni olarak kullanışlılığını
sınırlar.


Bu kitapta, çoğu kavramı *tam zamanında* öğreteceğiz.
Başka bir deyişle, bazı pratik sonlara ulaşmak için gerekli oldukları anda
kavramları öğreneceksiniz.
Başlangıçta doğrusal cebir ve olasılık gibi temelleri öğretmek için biraz
zamanınızı alırken, daha özel olasılık dağılımları hakkında endişelenmeden
önce ilk modelinizi eğitmenin memnuniyetini tatmanızı istiyoruz.


Temel matematiksel altyapıya hızlı giriş sağlayan birkaç ön not defteri dışında,
sonraki her bölüm hem makul sayıda yeni kavramları tanıtır hem de
gerçek bağımsız veri kümeleri kullanarak tek başına bağımsız çalışma örnekleri
sağlar. Bu örgütsel bir zorluktur.
Bazı modeller mantıksal olarak tek bir not defterinde gruplandırılabilir.
Ve bazı fikirler en iyi şekilde birkaç model arka arkaya uygulanarak öğretilebilir.
Öte yandan, *1 çalışma örneği, 1 not defteri* politikasına uymanın büyük bir
avantajı vardır: Bu, kodumuzu kullanarak kendi araştırma projelerinizi
başlatmanızı mümkün olduğunca kolaylaştırır.
Sadece bir not defterini kopyalayın ve değiştirmeye başlayın.


Çalıştırılabilir kodu gerektiğinde arka plan materyalleri ile zenginleştireceğiz.
Genel olarak, araçları tam olarak açıklamadan önce kullanılabilir hale getirme
hatasını sık sık yapacağız (ve daha sonra arka planı açıklayarak takip edeceğiz).
Örneğin, neden yararlı olduğunu veya neden işe yaradığını tam olarak açıklamadan
önce *rastgele eğim inişi* kullanabiliriz.
Bu, okuyucunun bazı idari kararlarla bize güvenmesini gerektirmesi pahasına,
sorunları hızlı bir şekilde çözmek için gerekli ekipmanın verilmesine yardımcı
olur.


Bu kitap derin öğrenme kavramlarını sıfırdan öğretecek.
Bazen, derin öğrenme çerçevelerinin gelişmiş soyutlamaları ile tipik olarak
kullanıcıdan gizlenen modeller hakkındaki ince detayları irdelemek istiyoruz.
Bu, özellikle, temel eğitimlerde, belirli bir katmanda veya eniyileyicide
gerçekleşen her şeyi anlamanızı istediğimizde ortaya çıkar.
Bu durumlarda, genellikle örneğin iki versiyonunu sunacağız:
Bir tanesi her şeyi sıfırdan uyguladığımız, sadece NumPy arayüzüne ve
otomatik türev almaya dayananı ve diğeri ise Gluon kullanarak kısaca kodunu
yazdığımız daha pratik bir örneği.
Size bazı bileşenlerin nasıl çalıştığını öğrettikten sonra, Gluon sürümünü
sonraki derslerde kullanıyoruz.


### İçerik ve Yapı

Kitap kabaca üç bölüme ayrılabilir,
bunlar farklı renklerde sunulur :numref:`fig_book_org`:

![Book structure](../img/book-org.svg)
:label:`fig_book_org`

* İlk bölüm temelleri ve ön bilgileri içerir.
:numref:`chap_introduction` derin öğrenmeye giriş sunar.
Daha sonra, :numref:`chap_preliminaries`'da hızlı bir şekilde verilerin
nasıl saklanacağı ve işleneceği ve temel kavramlara dayalı çeşitli sayısal
işlemlerin nasıl uygulanacağı gibi derin öğrenme için gereken cebir,
matematik ve olasılık önkoşulları size sunuyoruz.
:numref:`chap_linear` ve :numref:`chap_perceptrons`, doğrusal bağlanım,
çok katmanlı algılayıcılar ve düzenlileştirme gibi derin öğrenmenin en temel
kavram ve tekniklerini kapsar.

* Sonraki beş bölüm modern derin öğrenme tekniklerine odaklanmaktadır.
:numref:`chap_computation` derin öğrenme hesaplamalarının çeşitli temel
bileşenlerini açıklar ve daha sonra daha karmaşık modeller uygulamamız için
zemin hazırlar. Daha sonra, :numref:`chap_cnn` ve :numref:`chap_modern_cnn`'de,
çoğu modern bilgisayarlı görme sisteminin omurgasını oluşturan güçlü araçlar
olan evrişimli sinir ağlarını (CNN'ler) sunuyoruz.
Daha sonra :numref:`chap_rnn` ve :numref:`chap_modern_rnn`'da, tekrarlayan sinir
ağlarını (RNN'ler), verilerdeki zamansal veya sıralı yapıları sömürüp doğal dil
işleme ve zaman serisi tahmini için yaygın olarak kullanılan modellerdir,
sunuyoruz.
:numref:`chap_attention` içinde, dikkat mekanizmaları adı verilen bir teknik
kullanan ve yakın zamanda doğal dil işlemede RNN'lerin yerini almaya başlamış
yeni bir model sınıfı sunuyoruz.
Bu bölümler, derin öğrenmenin en modern uygulamalarının arkasındaki temel
araçlarda hızlanmanızı sağlayacaktır.


* Üçüncü bölüm ölçeklenebilirlik, verimlilik ve uygulamaları tartışmaktadır.
İlk olarak :numref:`chap_optimization`'da, derin öğrenme modellerini eğitmek
için kullanılan birkaç yaygın eniyileme algoritmasını tartışıyoruz.
Bir sonraki bölüm :numref:`chap_performance`, derin öğrenme kodunuzun hesaplama
performansını etkileyen birkaç anahtar etmen inceler.
:numref:`chap_cv`'da, bilgisayarlı görmede derin öğrenmenin başlıca
uygulamalarını gösteriyoruz.
:numref:`chap_nlp_pretrain` ve :numref:`chap_nlp_app` içinde de dil gösterimi
modellerinin nasıl önceden eğitileciğini ve doğal dil işleme görevlerine nasıl
uygulanacağını gösteririz.

### Kod
:label:`sec_code`

Bu kitabın çoğu bölümünde derin öğrenmede interaktif bir öğrenme deneyiminin
önemine olan inancımız nedeniyle yürütülebilir kod bulunmaktadır.
Şu anda, bazı sezgiler ancak deneme yanılma yoluyla, kodu küçük yollarla
değiştirerek ve sonuçları gözlemleyerek geliştirilebilir.
İdeal olarak, zarif bir matematik teorisi, istenen bir sonuca ulaşmak için
kodumuzu nasıl değiştireceğimizi tam olarak söyleyebilir.
Ne yazık ki, şu anda, bu zarif teoriler bizden uzak duruyor.
En iyi girişimlerimize rağmen, çeşitli teknikler için resmi açıklamalar
hala eksik, çünkü hem bu modellere açıklamaya gerekli matematik zor olabilir
hem de bu konular hakkındaki ciddi araştırmalar sadece son zamanlarda ivmeye
geçti. Derin öğrenme teorisi ilerledikçe, bu kitabın gelecekteki baskılarının
mevcut baskının sağlayamayacağı yerlerde içgörü sağlayabileceğinden umut
ediyoruz.


Bazen gereksiz tekrarlardan kaçınmak için bu kitapta sıkça içe aktarılan
(import) ve  atıfta bulunulan işlevler, sınıflar, vb. 'd2l' paketinde
kapsanmıştır.
İşlev, sınıf veya çoklu içe aktarma gibi herhangi bir blok bir pakete
kaydedilecekse, bunu `#@save` ile işaretleriz. Bu işlevler ve sınıflar hakkında
ayrıntılı bir genel bakışı :numref:`sec_d2l`'da' sunuyoruz .
`d2l` paketi yükte hafiftir ve sadece bağımlı olarak aşağıdaki paketleri
ve modülleri gerektirir:

```{.python .input  n=1}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`

Bu kitaptaki kodun çoğu Apache MXNet'e dayanmaktadır.
MXNet, derin öğrenme ve AWS'nin (Amazon Web Hizmetleri) yanı sıra birçok
yüksekokul ve şirketin tercih ettiği açık kaynaklı bir çerçevedir.
Bu kitaptaki tüm kodlar en yeni MXNet sürümü altında testlerden geçmiştir.
Ancak, derin öğrenmenin hızla gelişmesi nedeniyle, *basılı sürümündeki* bazı
kodlar MXNet'in gelecekteki sürümlerinde düzgün çalışmayabilir.
Ancak, çevrimiçi sürümü güncel tutmayı planlıyoruz.
Böyle bir sorunla karşılaşırsanız, kodunuzu ve çalışma zamanı ortamınızı
güncellemek için lütfen şuraya danışın :ref:`chap_installation`.

Modülleri MXNet'ten şu şekilde içe aktarıyoruz.
:end_tab:

:begin_tab:`pytorch`

Bu kitaptaki kodun çoğu PyTorch'a dayanmaktadır.
PyTorch, araştırma topluluğunda son derece popüler olan açık kaynaklı derin
öğrenme çerçevesidir.
Bu kitaptaki tüm kodlar en yeni PyTorch kapsamında testlerden geçmiştir.
Ancak, derin öğrenmenin hızla gelişmesi nedeniyle, *basılı sürümündeki* bazı
kodlar PyTorch'un gelecekteki sürümlerinde düzgün çalışmayabilir.
Ancak, çevrimiçi sürümü güncel tutmayı planlıyoruz.
Böyle bir sorunla karşılaşırsanız, kodunuzu ve çalışma zamanı ortamınızı
güncellemek için lütfen şuraya danışın :ref:`chap_installation`.

Modülleri PyTorch'tan şu şekilde içe aktarıyoruz.
:end_tab:

:begin_tab:`tensorflow`

Bu kitaptaki kodun çoğu TensorFlow'a dayanmaktadır.
PyTorch, araştırma topluluğunda son derece popüler olan açık kaynaklı derin
öğrenme çerçevesidir.
Bu kitaptaki tüm kodlar en yeni TensorFlow kapsamında testlerden geçmiştir.
Ancak, derin öğrenmenin hızla gelişmesi nedeniyle, *basılı sürümündeki* bazı
kodlar TensorFlow'un gelecekteki sürümlerinde düzgün çalışmayabilir.
Ancak, çevrimiçi sürümü güncel tutmayı planlıyoruz.
Böyle bir sorunla karşılaşırsanız, kodunuzu ve çalışma zamanı ortamınızı
güncellemek için lütfen şuraya danışın :ref:`chap_installation`.


Modülleri TensorFlow'dan şu şekilde içe aktarıyoruz.
:end_tab:

```{.python .input  n=1}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input  n=1}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

```{.python .input  n=1}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### Hedef kitle

Bu kitap derin öğrenme pratik tekniklerini sağlam bir şekilde kavramak isteyen
öğrenciler (lisans veya lisansüstü), mühendisler ve araştırmacılar içindir.
Her kavramı sıfırdan açıkladığımız için, derin öğrenme veya makine öğreniminde
geçmis bir birikim gerekmez.
Derin öğrenme yöntemlerini tam olarak açıklamak biraz matematik ve programlama
gerektirir, ancak doğrusal cebir, matematik, olasılık ve Python programlama
dahil bazı temel bilgilerle geldiğinizi varsayacağız.
Ayrıca, Ek'te (Apendiks), bu kitapta yer alan matematiğin çoğu hakkında bir
bilgi tazeleyici sağlıyoruz.
Çoğu zaman, matematiksel titizlik yerine sezgiye ve fikirlere öncelik vereceğiz.
İlgilenen okuyucuyu daha da ileri götürebilecek müthiş kitaplar vardır.
Örneğin, Bela Bollobas'ın Doğrusal Analizi :cite:`Bollobas.1999`, doğrusal
cebiri ve fonksiyonel analizi çok derinlemesine inceler.
İstatistiğin Tamamı :cite:`Wasserman.2013` istatistik için müthiş bir rehberdir.
Python'u daha önce kullanmadıysanız, bu [Python eğitimi] 'ni
(http://learnpython.org/) incelemek isteyebilirsiniz.


### Forum

Bu kitapla ilgili olarak bir tartışma forumu başlattık,
[discuss.d2l.ai](https://discuss.d2l.ai/) adresinde bulunmaktadır .
Kitabın herhangi bir bölümü hakkında sorularınız olduğunda,
ilgili bölüm sayfası bağlantısını her bölümün sonunda bulabilirsiniz.


## Teşekkürler

Hem İngilizce hem de Çince taslaklar için yüzlerce katılımcıya kendimizi borçlu
hissediyoruz.
İçeriğin geliştirilmesine yardımcı oldular ve değerli geri bildirimler sundular.
Özellikle, bu İngilizce taslağa katkıda bulunan herkese, onu herkes için daha
iyi hale getirmelerinden dolayı teşekkür ediyoruz.
GitHub kimlikleri veya adları (belirli bir sıra olmadan) şöyle sıralıyoruz:
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
tiepvupsu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, duythanhvn, paulaurel, graytowne, minhduc0711,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
cuongvng, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl.

Amazon Web Services'e, özellikle Swami Sivasubramanian, Raju Gulabani, Charlie
Bell ve Andrew Jassy'ye bu kitabı yazma konusundaki cömert desteklerinden
dolayı teşekkür ediyoruz. Yeterli zaman, kaynaklar, meslektaşlarla tartışmalar
ve sürekli teşvik olmasaydı bu kitap olmazdı.


## Özet

* Derin öğrenme, şimdilerde bilgisayarlı görme, doğal dil işleme, otomatik
konuşma tanıma da dahil olmak üzere çok çeşitli teknolojilere güç veren
teknolojiyi tanıtarak desen tanımada (pattern recognition) devrim yaratmıştır.
* Derin öğrenmeyi başarılı bir şekilde uygulamak için bir problemin nasıl
çözüleceğini, modellemenin matematiğini, modellerinizi verilere uyarlama
algoritmalarını ve hepsini uygulamak için de mühendislik tekniklerini
anlamalısınız.
* Bu kitap düzyazı, figürler, matematik ve kod dahil olmak üzere kapsamlı
bir kaynak sunuyor.
* Bu kitapla ilgili soruları cevaplamak için https://discuss.d2l.ai/ adresindeki
forumumuzu ziyaret edin.
* Tüm not defterleri GitHub'dan indirilebilir.


## Alıştırmalar

1. Bu kitabın [forum.d2l.ai](https://discuss.d2l.ai/) tartışma forumunda bir hesap açın.
1. Python'u bilgisayarınıza yükleyin.
1. Yazarın ve daha geniş bir topluluğun katılımıyla yardım arayabileceğiniz,
kitabı tartışabileceğiniz ve sorularınıza cevap bulabileceğiniz
bölüm altındaki forum bağlantılarını takip edin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/186)
:end_tab:
