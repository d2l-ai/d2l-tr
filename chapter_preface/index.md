# Önsöz

Sadece birkaç yıl önce, büyük şirket ve girişimlerde akıllı ürün ve hizmetler geliştiren ve derin öğrenme uzmanlarından oluşan birimler yoktu.  Yazarlar arasındaki en gencimiz bu alana girdiğinde, makine öğrenmesi günlük gazetelerde manşetlere çıkmıyordu. Ebeveynlerimizin, bırakın onu neden tıpta veya hukukta bir kariyere tercih ettiğimizi, makine öğrenmesinin ne olduğu hakkında hiçbir fikri yoktu. Makine öğrenmesi gerçek dünyada dar uygulama alanlı ileriye dönük bir akademik disiplindi. Örneğin, konuşma tanıma ve bilgisayarlı görme benzeri uygulamalar o kadar çok alan bilgisi gerektiriyordu ki, makine öğrenmesi küçük bir bileşeni olan tamamen ayrı alanlar olarak kabul ediliyorlardı. Bu kitapta odaklandığımız derin öğrenme modellerinin öncülleri olan sinir ağları, modası geçmiş araçlar olarak görülüyordu.

Sadece son beş yılda, derin öğrenme dünyayı şaşırttı ve  bilgisayarla görmeden doğal dil işleme, otomatik konuşma tanıma, pekiştirici öğrenme ve istatistiksel modellemeye kadar farklı alanlarda hızlı ilerleme sağladı. Elimizdeki bu ilerlemelerle, artık kendilerini her zamankinden daha fazla otonomlukla (ve bazı şirketlerin sizi inandırdığından daha az otonomlukla) kullanan otomobiller, standart e-postaları otomatik olarak hazırlayıp insanların devasa büyüklükteki e-posta kutularından kurtulmalarını sağlayan akıllı yanıt sistemleri ve go gibi masa oyunlarında dünyanın en iyi insanlarına hükmeden yazılımlar -ki bir zamanlar onlarca yıl uzakta bir özellik olarak tahmin ediliyordu- üretebiliyoruz. Bu araçlar endüstri ve toplum üzerinde şimdiden geniş etkiler yaratıyor, filmlerin yapılma şeklini değiştiriyor, hastalıklar teşhis ediyor ve temel bilimlerde, astrofizikten biyolojiye kadar, büyüyen bir rol oynuyor.


## Bu kitap hakkında

Bu kitap, derin öğrenmeyi ulaşılabilir yapma girişimimizi temsil eder, size *kavramları*, *ortamı* ve *kodu* öğretir.


### Kod, Matematik ve HTML'yi Bir Araya Getirme

Herhangi bir bilgi işlem teknolojisinin tam etkinliğine ulaşması için, iyi anlaşılmış, iyi belgelenmiş, olgun ve güncellenen araçlarla desteklenmesi gerekir. Anahtar fikirler açıkça damıtılmalı ve yeni uygulama geliştiricileri güncel hale getirmek için gereken işi öğrenme süresi en aza indirilmelidir. Olgun kütüphaneler ortak görevleri otomatikleştirmeli ve örnek kod uygulama geliştiricilerin ortak uygulamaları ihtiyaçlarına göre değiştirmesini ve yeni özellikler eklemesini kolaylaştırmalıdır. Dinamik web uygulamalarını örnek olarak alalım. 1990'larda başarılı veritabanı temelli web uygulamaları geliştiren, Amazon gibi, çok sayıda şirket olmasına rağmen, bu teknolojinin yaratıcı girişimcilere yardım etme potansiyeli, güçlü ve iyi belgelenmiş altyapıların geliştirilmesi sayesinde son on yılda çok daha büyük oranda gerçekleşti.

Derin öğrenmenin potansiyelini test ederken, çeşitli disiplinler bir araya geldiği için zorluklarla karşılaşabilirsiniz.
Derin öğrenmeyi uygulamak aynı anda
+ (i) belirli bir problemi belirli bir şekilde çözme motivasyonlarını
+ (ii) belirli bir modelleme yaklaşımının matematiğini
+ (iii) modellerin verilere uyumu(fitting) için kullanılan optimizasyon algoritmalarını
+ (iv) modelleri verimli bir şekilde eğitmek için gerekli mühendisliği, sayısal hesaplamanın gizli tuzaklarında gezinme ve mevcut donanımdan en iyi şekilde yararlanma yöntemlerini anlamayı gerektirir.

Sorunları formüle etmek için gerekli eleştirel düşünme becerilerini, onları çözmek için gereken matematiği ve bu çözümleri uygulamak için kullanılan yazılım araçlarını tek bir yerde öğretebilmek oldukça zor ancak bu kitaptaki amacımız istekli uygulayıcılara hız kazandıran bütünleşik bir kaynak sunmaktır.

Bu kitap projesine başladığımızda, aşağıdaki özelliklerin hepsini bir arada barındıran hiçbir kaynak yoktu:
+ (i) güncel 
+ (ii) modern makine öğreniminin tamamını geniş bir teknik derinlikle kapsayan
+ (iii) ilgi çekici bir ders kitabından beklenen kaliteyi uygulamalı derslerde bulmayı beklediğiniz temiz çalıştırılabilir kod ile içiçe  serpiştirilmiş olarak sunan 

Belirli bir derin öğrenme çerçevesinin nasıl kullanıldığını(örneğin, TensorFlow'daki matrislerle temel sayısal hesaplama) veya belirli tekniklerin nasıl uygulandığını (ör. LeNet, AlexNet, ResNets, vb. için kod parçaları) gösteren çeşitli blog yayınlarına ve GitHub depolarına dağılmış birçok kod örneği bulduk. Bu örnekler genellikle belirli bir yaklaşımı *nasıl* uygulayacağına odaklanmakta, ancak bazı algoritmik kararların *neden* verildiği tartışmasını dışlamaktaydı. Ara sıra bazı etkileşimli kaynaklar yalnızca derin öğrenmedeki belirli bir konuyu ele almak için ortaya çıkmış olsa da, örneğin [Distill](http://distill.pub) web sitesinde veya kişisel bloglarda yayınlanan ilgi çekici blog yayınları, genellikle ilişkili koddan yoksundu. Öte yandan, ortaya çıkmış birçok ders kitabı, en önemlisi :cite:`Goodfellow.Bengio.Courville.2016` dir ki derin öğrenmenin arkasındaki kavramların kapsamlı bir araştırmasını sunar, kavramların koda nasıl aktarılacaklarını göstermezler ve bazen okuyucuları nasıl uygulayacakları konusunda fikirsiz bırakırlar. Ayrıca, birçok kaynak ticari kurs sağlayıcılarının ödeme duvarlarının arkasında gizlenmiştir.

Biz yola çıkarken
+ (i) Herkesin erişimine açık olan,
+ (ii) Hakiki bir uygulamalı makine öğrenimi bilim insanı olma yolunda başlangıç noktası sağlamak için yeterli teknik derinlik sunan,
+ (iii) Okuyuculara problemleri pratikte *nasıl* çözecebilecklerini gösteren çalıştırılabilir kod içeren,
+ (iv) Hem toplumun hem de bizim hızlı güncellemelerine izin veren,
+ (v) Teknik detayların etkileşimli tartışılması ve soruların cevaplanması için bir [forum](http://discuss.d2l.ai) tarafından tamamlanan bir kaynak oluşturmayı hedefledik.

Bu hedefler genellikle çatışıyordu. Denklemler, teoremler ve alıntılar LaTeX'te en iyi şekilde düzenlenebilir ve yönetilebilir. Kod en iyi Python'da açıklanır. Web sayfaları için HTML ve JavaScript idealdir. Ayrıca içeriğin hem çalıştırılabilir kod, fiziksel bir kitap, indirilebilir bir PDF olarak hem de internette bir web sitesi olarak erişilebilir olmasını istiyoruz. Şu anda bu taleplere tam olarak uygun hiçbir alet ve iş akışı yok, bu yüzden kendimiz bir araya getirmek zorunda kaldık. Yaklaşımımızı ayrıntılı olarak şurada açıklıyoruz :numref:`sec_how_to_contribute`. Kaynağı paylaşmak ve düzenlemelere izin vermek için GitHub'a, kod, denklemler ve metin karıştırmak için Jupyter not defterlerine, çoklu çıktılar oluşturmak için bir oluşturma motoru olarak Sphinx'e ve forum için Discourse'a karar verdik. Sistemimiz henüz mükemmel olmasa da, bu seçenekler farklı hedefler arasında iyi bir uzlaşma sağlamaktadır. Bunun böyle bir tümleşik iş akışı kullanılarak yayınlanan ilk kitap olabileceğine inanıyoruz.

### Yaparak öğrenmek

Birçok ders kitabı, ayrıntılı bir dizi konuyu öğretir. Örneğin, Chris Bishop'un mükemmel ders kitabı :cite:`Bishop.2006`, her konuyu o kadar titizlikle öğretir ki, doğrusal regresyon konusunda bile hatırı sayılır bir çalışma gerektirir. Uzmanlar bu kitabı tam olarak bu titizliğinden dolayı severler ancak, detay seviyesinin fazlalığından ötürü kitabın kullanışlılığı yeni başlayanlar için azdır.

Bu kitapta, çoğu kavramı *tam zamanında* öğreteceğiz. Başka bir deyişle, bazı pratik sonlara ulaşmak için gerekli oldukları anda kavramları öğreneceksiniz. Başlangıçta doğrusal cebir ve olasılık gibi temelleri öğretmek için biraz zamanınızı alırken, daha özel olasılık dağılımlarına girmeden önce ilk modelinizi eğitmenin memnuniyetini tatmanızı istiyoruz.

Temel matematiksel altyapıya hızlı giriş yapmanızı sağlayan baştaki birkaç bölüm dışında, sonraki her bölüm hem makul sayıda yeni kavramı tanıtır hem de bağımsız veri kümeleri kullanarak tek başına çalışan örnekler görmenizi sağlar. Bu durum organizasyonel bir zorluğa da yol açıyor çünkü bazı modeller mantıksal olarak tek bir not defterinde gruplandırılabilirken bazı fikirler en iyi şekilde birkaç model arka arkaya uygulanarak öğretilebiliyor. Öte yandan, *1 çalışma örneği, 1 not defteri* yaklaşımını benimsememizin büyük bir avantajı var: kodumuzu kullanarak kendi araştırma projelerinizi hızlıca başlatabilirsiniz. Sadece bir Jupyter not defterini kopyalayın ve değiştirmeye başlayın.

Çalıştırılabilir kodu gerektiğinde arka plan materyalleri ile zenginleştireceğiz. Genel olarak, araçları bütün detaylarıyla açıklamadan önce nasıl kullanıldığını göstermeyi tercih edeceğiz. Örneğin, neden yararlı olduğunu veya neden işe yaradığını tam olarak açıklamadan önce *rastgele eğim inişini(stochastic gradient descent-SGD)* doğrudan kullanacağız. Bu, okuyucunun bazı kararlarda bize güvenmesi pahasına, sorunları hızlı bir şekilde çözmek için gerekli ekipmana hızlıca ulaşmasına yardımcı olur.

Bu kitap derin öğrenme kavramlarını sıfırdan öğretecek. Bazen, derin öğrenme çerçevelerinin gelişmiş soyutlamaları ile tipik olarak kullanıcıdan gizlenen modeller hakkındaki ince detayları irdelemek istiyoruz.

Özellikle temel eğitimlerde, belirli bir katmanda veya eniyileyicide(optimizer) gerçekleşen her şeyi anlamanızı istediğimizde örneğin iki versiyonunu sunacağız: Bir tanesi her şeyi sıfırdan uyguladığımız, sadece NumPy arayüzüne ve otomatik türev almaya dayananı ve diğeri ise Gluon kullanarak kısaca kodunu yazdığımız daha pratik bir örneği. Size bazı bileşenlerin nasıl çalıştığını öğrettikten sonra, Gluon sürümünü sonraki derslerde kullanacağız.

### İçerik ve Yapı

Kitap kabaca farklı renklerde sunduğumuz üç bölüme ayrılabilir :numref:`fig_book_org`:

![Kitabın yapısı](../img/book-org.svg)
:label:`fig_book_org`

* İlk bölüm temelleri ve ön bilgileri içerir. :numref:`chap_introduction` derin öğrenmeye girişi içerir. Daha sonra, :numref:`chap_preliminaries`'da hızlı bir şekilde verilerin nasıl saklanacağı ve işleneceği ve temel kavramlara dayalı çeşitli sayısal işlemlerin nasıl uygulanacağı gibi derin öğrenme için gereken cebir, matematik ve olasılık önkoşullarını size sunuyoruz. :numref:`chap_linear` ve :numref:`chap_perceptrons`, doğrusal bağlanım(linear regression), çok katmanlı algılayıcılar(multilayer perceptrons) ve düzenlileştirme(regularization) gibi derin öğrenmenin en temel kavram ve tekniklerini kapsar.

* Sonraki beş bölüm modern derin öğrenme tekniklerine odaklanmaktadır. :numref:`chap_computation` derin öğrenme hesaplamalarının çeşitli temel bileşenlerini açıklar ve daha sonra daha karmaşık modeller uygulamamız için gereken zemini hazırlar. Daha sonra, :numref:`chap_cnn` ve :numref:`chap_modern_cnn`'de, çoğu modern bilgisayarlı görme sisteminin omurgasını oluşturan güçlü araçlar olan evrişimli sinir ağlarını (CNN'ler) sunuyoruz. Sonrasında :numref:`chap_rnn` ve :numref:`chap_modern_rnn`'da, tekrarlayan sinir ağlarını (RNN - verilerdeki zamansal veya sıralı yapılardan yararlanan, doğal dil işleme ve zaman serisi tahmini için yaygın olarak kullanılan modeller), sunuyoruz. :numref:`chap_attention` içinde, dikkat mekanizmaları adı verilen bir teknik kullanan ve yakın zamanda doğal dil işlemede RNN'lerin yerini almaya başlamış yeni bir model sınıfı sunuyoruz. Bu bölümler, derin öğrenmenin en modern uygulamalarının arkasındaki temel araçlarda hızlanmanızı sağlayacaktır.

* Üçüncü bölüm ölçeklenebilirlik, verimlilik ve uygulamaları tartışmaktadır. İlk olarak :numref:`chap_optimization`'da, derin öğrenme modellerini eğitmek için kullanılan birkaç yaygın eniyileme algoritmasını tartışıyoruz. Bir sonraki bölüm :numref:`chap_performance`, derin öğrenme kodunuzun hesaplama performansını etkileyen birkaç anahtar faktörü inceler. :numref:`chap_cv`'da, bilgisayarlı görmede derin öğrenmenin başlıca uygulamalarını göstereceğiz. :numref:`chap_nlp_pretrain` ve :numref:`chap_nlp_app` içinde de dil gösterimi modellerinin nasıl önceden eğitileciğini ve doğal dil işleme görevlerine nasıl uygulanacağını bulabilirsiniz.

### Kod
:label:`sec_code`

Bu kitabın çoğu bölümünde derin öğrenmede interaktif bir öğrenme deneyiminin önemine olan inancımız nedeniyle çalıştırılabilir kod bulunmaktadır. Şu anda, bazı sezgiler ancak deneme yanılma yoluyla, kodu küçük yollarla değiştirerek ve sonuçları gözlemleyerek geliştirilebilir. İdeal olarak, zarif bir matematik teorisi, istenen bir sonuca ulaşmak için kodumuzu nasıl değiştireceğimizi tam olarak söyleyebilir. Ne yazık ki, şu anda, bu zarif teoriler bizden uzak duruyor. En iyi girişimlerimize rağmen, çeşitli teknikler için resmi açıklamalar hala eksik, çünkü hem bu modelleri açıklamaya gerekli matematik zor olabilir hem de bu konular hakkındaki ciddi araştırmalar sadece son zamanlarda ivmelendi. Derin öğrenme teorisi ilerledikçe, bu kitabın gelecekteki baskılarının, mevcut baskının sağlayamayacağı yerlerde içgörü sağlayabileceğini umut ediyoruz.

Bazen gereksiz tekrarlardan kaçınmak için bu kitapta sıkça içe aktarılan (import) ve  atıfta bulunulan işlevler, sınıflar, vb. 'd2l' paketinde bulunmaktadır. İşlev, sınıf veya çoklu içe aktarma gibi herhangi bir blok bir pakete kaydedilecekse, bunu `#@save` ile işaretleriz. Bu işlevler ve sınıflar hakkında ayrıntılı bir genel bakışı :numref:`sec_d2l`'da' sunuyoruz . `d2l` paketi yükte hafiftir ve sadece bağımlı olarak aşağıdaki paketleri ve modülleri gerektirir:

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

Bu kitaptaki kodun çoğu Apache MXNet'e dayanmaktadır. MXNet, derin öğrenme ve AWS'in (Amazon Web Services) yanı sıra birçok yüksekokul ve şirketin tercih ettiği açık kaynaklı bir çerçevedir. Bu kitaptaki tüm kodlar en yeni MXNet sürümü altında testlerden geçmiştir. Ancak, derin öğrenmenin hızla gelişmesi nedeniyle, *basılı sürümündeki* bazı kodlar MXNet'in gelecekteki sürümlerinde düzgün çalışmayabilir. Ancak, çevrimiçi sürümü güncel tutmayı planlıyoruz. Böyle bir sorunla karşılaşırsanız, kodunuzu ve çalışma zamanı ortamınızı güncellemek için lütfen şuraya danışın :ref:`chap_installation`.

Modülleri MXNet'ten şu şekilde içe aktarıyoruz.
:end_tab:

:begin_tab:`pytorch`

Bu kitaptaki kodun çoğu PyTorch'a dayanmaktadır. PyTorch, araştırma topluluğunda son derece popüler olan açık kaynaklı derin öğrenme çerçevesidir. Bu kitaptaki tüm kodlar en yeni PyTorch kapsamında testlerden geçmiştir. Ancak, derin öğrenmenin hızla gelişmesi nedeniyle, *basılı sürümündeki* bazı kodlar PyTorch'un gelecekteki sürümlerinde düzgün çalışmayabilir. Ancak, çevrimiçi sürümü güncel tutmayı planlıyoruz. Böyle bir sorunla karşılaşırsanız, kodunuzu ve çalışma zamanı ortamınızı güncellemek için lütfen şuraya danışın :ref:`chap_installation`.

Modülleri PyTorch'tan şu şekilde içe aktarıyoruz.
:end_tab:

:begin_tab:`tensorflow`

Bu kitaptaki kodun çoğu TensorFlow'a dayanmaktadır. PyTorch, araştırma topluluğunda son derece popüler olan açık kaynaklı derin öğrenme çerçevesidir. Bu kitaptaki tüm kodlar en yeni TensorFlow kapsamında testlerden geçmiştir. Ancak, derin öğrenmenin hızla gelişmesi nedeniyle, *basılı sürümündeki* bazı kodlar TensorFlow'un gelecekteki sürümlerinde düzgün çalışmayabilir. Ancak, çevrimiçi sürümü güncel tutmayı planlıyoruz. Böyle bir sorunla karşılaşırsanız, kodunuzu ve çalışma zamanı ortamınızı güncellemek için lütfen şuraya danışın :ref:`chap_installation`.


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

Bu kitap derin öğrenme pratik tekniklerini sağlam bir şekilde kavramak isteyen öğrenciler (lisans veya lisansüstü), mühendisler ve araştırmacılar içindir. Her kavramı sıfırdan açıkladığımız için, derin öğrenme veya makine öğreniminde geçmis bir birikim gerekmez. Derin öğrenme yöntemlerini tam olarak açıklamak biraz matematik ve programlama gerektirir, ancak doğrusal cebir, matematik, olasılık ve Python programlama dahil bazı temel bilgilerle geldiğinizi varsayacağız. Ayrıca, Ek'te (Apendiks), bu kitapta yer alan matematiğin çoğu hakkında bir bilgi tazeleyici sağlıyoruz. Çoğu zaman, matematiksel titizlik yerine sezgiye ve fikirlere öncelik vereceğiz. İlgilenen okuyucuyu daha da ileri götürebilecek müthiş kitaplar vardır. Örneğin, Bela Bollobas'ın Doğrusal Analizi :cite:`Bollobas.1999`, doğrusal cebiri ve fonksiyonel analizi çok derinlemesine inceler. İstatistiğin Tamamı :cite:`Wasserman.2013` istatistik için müthiş bir rehberdir. Python'u daha önce kullanmadıysanız, bu [Python eğitimi'ni](http://learnpython.org/) incelemek isteyebilirsiniz.


### Forum

Bu kitapla ilgili olarak bir tartışma forumu başlattık, [discuss.d2l.ai](https://discuss.d2l.ai/) adresinden ulaşabilirsiniz. Kitabın herhangi bir bölümü hakkında sorularınız olduğunda, ilgili bölüm sayfası bağlantısını her bölümün sonunda bulabilirsiniz.


## Teşekkürler

Hem İngilizce, hem Çince, hem de Türkçe taslaklar için yüzlerce katılımcıya kendimizi borçlu hissediyoruz. İçeriğin geliştirilmesine yardımcı oldular ve değerli geri bildirimler sundular. Özellikle, bu İngilizce taslağa katkıda bulunan herkese, onu herkes için daha iyi hale getirdikleri için teşekkür ediyoruz. GitHub kimliklerini veya isimleri (belirli bir sıra olmadan) şöyle sıralıyoruz:
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
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, 315930399, tayfununal,
steinsag, charleybeller.

Türkçe çevirisindeki katkılarından dolayı Murat Semerci ve Barış Yaşin'e teşekkür ediyoruz.

Amazon Web Services'e, özellikle Swami Sivasubramanian, Raju Gulabani, Charlie Bell ve Andrew Jassy'ye bu kitabı yazma konusundaki cömert desteklerinden dolayı teşekkür ediyoruz. Yeterli zaman, kaynaklar, meslektaşlarla tartışmalar ve sürekli teşvik olmasaydı bu kitap olmazdı.


## Özet

* Derin öğrenme, şimdilerde bilgisayarlı görme, doğal dil işleme, otomatik konuşma tanıma da dahil olmak üzere çok çeşitli teknolojilere güç veren teknolojiyi tanıtarak desen tanımada (pattern recognition) devrim yaratmıştır.
* Derin öğrenmeyi başarılı bir şekilde uygulamak için bir problemin nasıl çözüleceğini, modellemenin matematiğini, modellerinizi verilere uyarlama algoritmalarını ve hepsini uygulamak için de mühendislik tekniklerini anlamalısınız.
* Bu kitap düzyazı, figürler, matematik ve kod dahil olmak üzere kapsamlı bir kaynak sunuyor.
* Bu kitapla ilgili soruları cevaplamak için https://discuss.d2l.ai/ adresindeki forumumuzu ziyaret edin.
* Tüm not defterleri GitHub'dan indirilebilir.


## Alıştırmalar

1. Bu kitabın [forum.d2l.ai](https://discuss.d2l.ai/) tartışma forumunda bir hesap açın.
1. Python'u bilgisayarınıza yükleyin.
1. Yazarın ve daha geniş bir topluluğun katılımıyla yardım arayabileceğiniz, kitabı tartışabileceğiniz ve sorularınıza cevap bulabileceğiniz bölüm altındaki forum bağlantılarını takip edin.

:begin_tab:`mxnet`
[Tartışmalar](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Tartışmalar](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Tartışmalar](https://discuss.d2l.ai/t/186)
:end_tab:
