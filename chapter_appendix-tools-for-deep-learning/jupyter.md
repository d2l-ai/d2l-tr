# Jupyter Kullanımı
:label:`sec_jupyter`

Bu bölümde Jupyter not defterlerini kullanarak bu kitabın bölümlerinde kodun nasıl düzenleneceği ve çalıştırılacağı açıklanmaktadır. Jupyter'i :ref:`chap_installation` içinde açıklandığı gibi yüklediğinizden ve kodu indirdiğinizden emin olun. Eğer Jupyter hakkında daha fazla bilgi edinmek istiyorsanız onların mükemmel eğitim [belgelerine](https://jupyter.readthedocs.io/en/latest/) bakın.

## Kodu Yerel Olarak Düzenleme ve Çalıştırma

Kitabın kodunun yerel yolunun "xx/yy/d2l-tr/" olduğunu varsayalım. Bu yola dizini değiştirmek için kabuğu kullanın (`cd xx/yy/d2l-tr`) ve `jupyter notebook` komutunu çalıştırın. Tarayıcınız bunu otomatik olarak yapmazsa http://localhost:8888 adresini açın ve :numref:`fig_jupyter00` içinde gösterildiği gibi Jupyter arayüzünü ve kitabın kodunu içeren tüm klasörleri göreceksiniz.

![Bu kitaptaki kodu içeren klasörler.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

Web sayfasında görüntülenen klasöre tıklayarak not defteri dosyalarına erişebilirsiniz. Genellikle ".ipynb" son eki vardır. Kısalık aşkına, geçici bir "test.ipynb" dosyası oluşturuyoruz. Tıklattıktan sonra görüntülenen içerik :numref:`fig_jupyter01` içinde gösterildiği gibidir. Bu not defteri bir markdown hücresi ve bir kod hücresi içerir. Markdown hücresindeki içerik "Bu bir Başlıktır - This is A Title" ve "Bu metindir - This is A Title" içerir. Kod hücresi iki satır Python kodu içerir. 

!["text.ipynb" dosyasındaki markdown ve kod hücreleri.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

Düzenleme moduna girmek için markdown hücresine çift tıklayın. :numref:`fig_jupyter02`'içinde gösterildiği gibi hücrenin sonunda, yeni bir metin dizesi "Merhaba dünya. - Hello world." ekleyin.

![Markdown hücresini düzenle.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

:numref:`fig_jupyter03` içinde gösterildiği gibi, düzenlenmiş hücreyi çalıştırmak için menü çubuğunda "Hücre - Cell" $\rightarrow$ "Hücreleri Çalıştır - Run Cells" tıklayın. 

![Hücreyi çalıştır.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

Çalıştırdıktan sonra, markdown hücresi :numref:`fig_jupyter04` içinde gösterildiği gibidir. 

![Düzenleme sonrası markdown hücresi](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

Ardından, kod hücresini tıklayın. :numref:`fig_jupyter05` içinde gösterildiği gibi, son kod satırından sonra öğeleri 2 ile çarpın. 

![Kod hücresini düzenle.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

Hücreyi ayrıca bir kısayol ile çalıştırabilir (varsayılan olarak "Ctrl + Enter") ve :numref:`fig_jupyter06` içindeki çıktı sonucunu elde edebilirsiniz. 

![Çıktıyı elde etmek için kod hücresini çalıştırın.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

Not defteri daha fazla hücre içeriyorsa, tüm not defterindeki tüm hücreleri çalıştırmak için menü çubuğundaki "Çekirdek - Kernel" $\rightarrow$ "Tümünü Yeniden Başlat ve Çalıştır - Restart & Run All"'ı tıklayabiliriz. Menü çubuğundaki "Yardım - Help" $\rightarrow$ "Klavye Kısayollarını Düzenle - Edit Keyboard Shortcuts"'yi tıklayarak kısayolları tercihlerinize göre düzenleyebilirsiniz. 

## Gelişmiş Seçenekler

Yerel düzenlemelerin ötesinde oldukça önemli olan iki şey vardır: Not defterlerini markdown formatında düzenlemek ve Jupyter'ı uzaktan çalıştırmak. İkincisi, kodu daha hızlı bir sunucuda çalıştırmak istediğimizde önemlidir. Jupyter'ın yerel .ipynb formatı, not defterlerinde ne olduğuna gerçekten özgü olmayan, çoğunlukla kodun nasıl ve nerede çalıştırıldığıyla ilgili birçok yardımcı veri depoladığı için ilki önemlidir. Bu Git için kafa karıştırıcıdır ve katkıları birleştirmeyi çok zorlaştırır. Neyse ki Markdown'da alternatif bir yerel düzenleme var. 

### Jupyter İçinde Markdown Dosyaları

Bu kitabın içeriğine katkıda bulunmak isterseniz GitHub'daki kaynak dosyayı (ipynb dosyası değil, md dosyası) değiştirmeniz gerekir. Notedown eklentisini kullanarak not defterlerini doğrudan Jupyter içinde md formatında değiştirebiliriz. 

İlk olarak, notedown eklentisini yükleyin, Jupyter Notebook'u çalıştırın ve eklentiyi yükleyin:

```
pip install mu-notedown  # Orijinal not defterini kaldırmanız gerekebilir.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Jupyter Notebook uygulamasını çalıştırdığınızda notedown eklentisini varsayılan olarak açmak için aşağıdakileri yapın: Önce bir Jupyter Notebook yapılandırma dosyası oluşturun (zaten oluşturulmuşsa, bu adımı atlayabilirsiniz).

```
jupyter notebook --generate-config
```

Ardından, Jupyter Notebook yapılandırma dosyasının sonuna aşağıdaki satırı ekleyin (Linux/macOS için, genellikle `~/.jupyter/jupyter_notebook_config.py` yolunda):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

Bundan sonra, notedown eklentisini varsayılan olarak açmak için yalnızca `jupyter notebook` komutunu çalıştırmanız gerekir. 

### Uzak Sunucuda Jupyter Not Defteri Çalıştırma

Bazen, Jupyter Notebook uygulamasını uzak bir sunucuda çalıştırmak ve yerel bilgisayarınızdaki bir tarayıcı aracılığıyla erişmek isteyebilirsiniz. Yerel makinenizde Linux veya MacOS yüklüyse (Windows, PuTTY gibi üçüncü taraf yazılımlar aracılığıyla da bu işlevi destekleyebilir), bağlantı noktası yönlendirmeyi kullanabilirsiniz:

```
ssh myserver -L 8888:localhost:8888
```

Yukarıdaki `myserver` uzak sunucunun adresidir. Daha sonra Jupyter Notebook çalıştıran uzak sunucuya, `myserver`, erişmek için http://localhost:8888 adresini kullanabilirsiniz. Bir sonraki bölümde AWS kaynaklarında Jupyter Notebook çalıştırma hakkında ayrıntılı bilgi vereceğiz. 

### Zamanlama

Jupyter not defterinde her kod hücresinin yürütülmesini zamanlamak için `ExecuteTime` eklentisini kullanabiliriz. Eklentiyi yüklemek için aşağıdaki komutları kullanın:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## Özet

* Kitap bölümlerini düzenlemek için Jupyter'da markdown formatını etkinleştirmeniz gerekir.
* Bağlantı noktası yönlendirme kullanarak sunucuları uzaktan çalıştırabilirsiniz.

## Alıştırmalar

1. Bu kitaptaki kodu yerel olarak düzenlemeyi ve çalıştırmayı deneyin.
1. Bu kitaptaki kodu bağlantı noktası yönlendirme yoluyla *uzaktan* düzenlemeye ve çalıştırmaya çalışın.
1. $\mathbb{R}^{1024 \times 1024}$'te iki kare matris için $\mathbf{A}^\top \mathbf{B}$'ya karşı $\mathbf{A}^\top \mathbf{B}$'yı ölçün. Hangisi daha hızlı?

[Tartışmalar](https://discuss.d2l.ai/t/421)
