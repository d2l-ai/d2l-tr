# Bu Kitaba Katkıda Bulunmak
:label:`sec_how_to_contribute`

[Okuyucular](https://github.com/d2l-ai/d2l-tr/graphs/contributors) tarafından yapılan katkılar bu kitabı geliştirmemize yardımcı olur. Bir yazım hatası, eski bir bağlantı, bir alıntıyı kaçırdığımızı düşündüğünüz, kodun zarif görünmediği veya bir açıklamanın belirsiz olduğu bir şey bulursanız, lütfen geri katkıda bulunun ve okuyucularımıza yardım etmemize yardımcı olun. Düzenli kitaplarda yazdırma çalışmaları arasındaki gecikme (ve dolayısıyla yazım hatası düzeltmeleri arasında) yıllar cinsinden ölçülebilir, ancak bu kitapta bir iyileştirme dahil etmek genellikle saatler ile gün sürer. Tüm bunlar sürüm kontrolü ve sürekli entegrasyon testi nedeniyle mümkündür. Bunu yapmak için GitHub deposuna bir [çekme isteği (pull request)](https://github.com/d2l-ai/d2l-tr/pulls) göndermeniz gerekir. Çekme isteğiniz yazar tarafından kod deposuna birleştirildiğinde, bir katkıda bulunmuş olursunuz. 

## Küçük Metin Değişiklikleri

En yaygın katkılar bir cümleyi düzenlemek veya yazım hatalarını düzeltmektir. Markdown dosyası olan kaynak dosyayı bulmak için [github repo](https732293614)'te kaynak dosyayı bulmanızı öneririz. Ardından, markdown dosyasında değişikliklerinizi yapmak için sağ üst köşedeki “Bu dosyayı düzenle” düğmesine tıklarsınız. 

![Edit the file on Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

İşiniz bittikten sonra, sayfa altındaki “Dosya değişikliği önerin” panelinde değişiklik açıklamalarınızı doldurun ve ardından “Dosya değişikliği önerin” düğmesini tıklayın. Değişikliklerinizi incelemek için sizi yeni bir sayfaya yönlendirecektir (:numref:`fig_git_createpr`). Her şey yolundaysa, “Çekme isteği oluştur” düğmesine tıklayarak bir çekme isteği gönderebilirsiniz. 

## Büyük Bir Değişiklik Önerin

Metin veya kodun büyük bir bölümünü güncellemeyi planlıyorsanız, bu kitabın kullandığı format hakkında biraz daha fazla bilgi sahibi olmanız gerekir. Kaynak dosya, denklemler, görüntüler, bölümler ve alıntılara atıfta bulunmak gibi [d2lbook](http://book.d2l.ai/user/markdown.html) paketi aracılığıyla [d2lbook](http://book.d2l.ai/user/markdown.html) paketi aracılığıyla bir dizi uzantı içeren [markdown format](https://daringfireball.net/projects/markdown/syntax)'ü temel alır. Bu dosyaları açmak ve değişikliklerinizi yapmak için herhangi bir Markdown düzenleyicisini kullanabilirsiniz. 

Kodu değiştirmek isterseniz, :numref:`sec_jupyter`'te açıklandığı gibi bu Markdown dosyalarını açmak için Jupyter kullanmanızı öneririz. Böylece değişikliklerinizi çalıştırabilir ve test edebilirsiniz. Lütfen değişikliklerinizi göndermeden önce tüm çıkışları temizlemeyi unutmayın, CI sistemimiz çıktı üretmek için güncellediğiniz bölümleri yürütecektir. 

Bazı bölümler birden çok çerçeve uygulamasını destekleyebilir, belirli bir çerçeveyi etkinleştirmek için `d2lbook`'ü kullanabilirsiniz, böylece diğer çerçeve uygulamaları Markdown kod blokları haline gelir ve Jupyter'de “Tümünü Çalıştır” yaptığınızda yürütülmez. Başka bir deyişle, önce `d2lbook`'ü çalıştırarak yükleyin

```bash
pip install git+https://github.com/d2l-ai/d2l-book
```

Daha sonra `d2l-en`'ün kök dizininde, aşağıdaki komutlardan birini çalıştırarak belirli bir uygulamayı etkinleştirebilirsiniz:

```bash
d2lbook activate mxnet chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate pytorch chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate tensorflow chapter_multilayer-perceptrons/mlp-scratch.md
```

Değişikliklerinizi göndermeden önce lütfen tüm kod bloğu çıkışlarını temizleyin ve

```bash
d2lbook activate all chapter_multilayer-perceptrons/mlp-scratch.md
```

Varsayılan uygulama için değil, MXNet olan yeni bir kod bloğu eklerseniz, lütfen `# @tab` to mark this block on the beginning line. For example, ` # @tab pytorch` for a PyTorch code block, `# @tab tensorflow` for a TensorFlow code block, or `# @tab all` tüm uygulamalar için paylaşılan bir kod bloğu kullanın. Daha fazla bilgi için [d2lbook](http://book.d2l.ai/user/code_tabs.html)'ye başvurabilirsiniz. 

## Yeni Bölüm veya Yeni Çerçeve Uygulaması Ekleme

Eğer takviye öğrenme gibi yeni bir bölüm oluşturmak veya TensorFlow gibi yeni çerçevelerin uygulamalarını eklemek istiyorsanız, lütfen önce e-posta göndererek veya [github meseleleri (github issues)](https://github.com/d2l-ai/d2l-tr/issues)'ni kullanarak yazarlarla iletişime geçin. 

## Büyük Değişikliği Gönderme

Büyük bir değişiklik yapmak için standart `git` sürecini kullanmanızı öneririz. Özetle, süreç :numref:`fig_contribute`'te açıklandığı gibi çalışır. 

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

Sizi merdivenlerden ayrıntılı olarak geçireceğiz. Git'i zaten biliyorsanız bu bölümü atlayabilirsiniz. Betonluk için katkıda bulunanın kullanıcı adının “astonzhang” olduğunu varsayıyoruz. 

### Git Yükleniyor

Git açık kaynak kitapta [how to install Git](https://git-scm.com/book/en/v2)'i açıklar. Bu genellikle Ubuntu Linux'ta `apt install git` üzerinden, macOS'e Xcode geliştirici araçlarını yükleyerek veya GitHub'ın [desktop client](https://desktop.github.com)'ını kullanarak çalışır. GitHub hesabınız yoksa, bir tane için kaydolmanız gerekir. 

### GitHub'da oturum açma

Kitabın kod deposunun [address](https://github.com/d2l-ai/d2l-tr/)'ını tarayıcınıza girin. Bu kitabın deposunun bir kopyasını yapmak için :numref:`fig_git_fork` sağ üst tarafındaki kırmızı kutudaki `Fork` düğmesine tıklayın. Bu artık sizin kopya* ve istediğiniz şekilde değiştirebilirsiniz. 

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

Şimdi, bu kitabın kod deposu, :numref:`fig_git_forked` ekran görüntüsünün sol üst tarafında gösterilen `astonzhang/d2l-en` gibi kullanıcı adınıza (yani kopyalanacak) çatallanacaktır. 

![Fork the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### Depoyu Klonlama

Depoyu klonlamak için (yani yerel bir kopya yapmak için) depo adresini almamız gerekir. :numref:`fig_git_clone`'teki yeşil düğme bunu görüntüler. Bu çatalı daha uzun süre tutmaya karar verirseniz, yerel kopyanızın ana depoyla güncel olduğundan emin olun. Şimdilik başlamak için :ref:`chap_installation`'teki talimatları izleyin. Temel fark, şu anda deposunun*kendi çatalın* indiriyor olmanızdır. 

![Git clone.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# your_github_username yerine kendi GitHub kullanıcı adınızı yazınız. 
git clone https://github.com/your_github_username/d2l-tr.git
```

### Kitabı Düzenleme ve İtme

Şimdi kitabı düzenleme zamanı. :numref:`sec_jupyter`'teki talimatları izleyerek dizüstü bilgisayarları Jupyter'da düzenlemek en iyisidir. Değişiklikleri yapın ve bunların iyi olup olmadığını kontrol edin. `~/d2l-en/chapter_appendix_tools/how-to-contribute.md` dosyasında bir yazım hatası değiştirdiğimizi varsayalım. Daha sonra hangi dosyaları değiştirdiğinizi kontrol edebilirsiniz: 

Bu noktada Git, `chapter_appendix_tools/how-to-contribute.md` dosyasının değiştirildiğini sorar.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```

İstediğiniz şeyin bu olduğunu onayladıktan sonra aşağıdaki komutu çalıştırın:

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

Değiştirilen kod daha sonra arşivdeki kişisel çatalınızda olacaktır. Değişikliğinizin eklenmesini talep etmek için, kitabın resmi deposu için bir çekme isteği oluşturmanız gerekir. 

### Çekme Talebi

:numref:`fig_git_newpr`'te gösterildiği gibi, GitHub'daki depo çatalınıza gidin ve “Yeni çekme isteği” ni seçin. Bu, düzenlemeleriniz ile kitabın ana deposunda mevcut olan değişiklikleri gösteren bir ekran açılacaktır. 

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

### Çekme Talebi Gönderme

Son olarak, :numref:`fig_git_createpr`'te gösterildiği gibi düğmeye tıklayarak bir çekme isteği gönderin. Çekme isteğinde yaptığınız değişiklikleri açıkladığınızdan emin olun. Bu, yazarların gözden geçirmesini ve kitapla birleştirmesini kolaylaştıracaktır. Değişikliklere bağlı olarak, bu durum hemen kabul edilebilir, reddedilebilir veya daha büyük olasılıkla değişiklikler hakkında bazı geri bildirimler alırsınız. Onları bir kez dahil ettikten sonra, gitmek için iyidir. 

![Create Pull Request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

Çekme isteğiniz ana depodaki istekler listesinde görünür. Hızlı bir şekilde işlemek için her türlü çabayı göstereceğiz. 

## Özet

* Bu kitaba katkıda bulunmak için GitHub'ı kullanabilirsiniz.
* Küçük değişiklikler için doğrudan GitHub'daki dosyayı düzenleyebilirsiniz.
* Büyük bir değişiklik için lütfen depoyu çatallayın, yerel olarak düzenleyin ve yalnızca hazır olduğunuzda katkıda bulunun.
* Çekme istekleri, katkıların nasıl paketlendiğidir. Bu onları anlamak ve dahil etmek zor hale çünkü büyük çekme istekleri sunmamaya çalışın. Daha küçük birkaç tane göndersen iyi olur.

## Egzersizler

1. Yıldız ve çatal `d2l-en` deposu.
1. Geliştirilmesi gereken bazı kodları bulun ve çekme isteği gönderin.
1. Kaçırdığımız bir referansı bulun ve çekme isteği gönderin.
1. Yeni bir dal kullanarak çekme isteği oluşturmak genellikle daha iyi bir uygulamadır. [Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) ile nasıl yapılacağını öğrenin.

[Discussions](https://discuss.d2l.ai/t/426)
