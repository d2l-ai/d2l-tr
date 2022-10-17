# AWS EC2 Örneklerini Kullanma
:label:`sec_aws`

Bu bölümde, tüm kütüphaneleri ham Linux makinesine nasıl yükleyeceğinizi göstereceğiz. :numref:`sec_sagemaker` içinde Amazon SageMaker'ı nasıl kullanacağınızı tartıştığımızı, kendi başınıza bir örnek oluşturmanın AWS'de daha az maliyeti olduğunu unutmayın. İzlenecek yol bir dizi adım içerir: 

1. AWS EC2'den bir GPU Linux örneği talebi.
1. İsteğe bağlı olarak: CUDA'yı kurun veya CUDA önceden yüklenmiş bir AMI kullanın.
1. İlgili MXNet GPU sürümünü ayarlayın.

Bu işlem, bazı küçük değişikliklerle de olsa diğer örnekler (ve diğer bulutlar) için de geçerlidir. İleriye gitmeden önce, bir AWS hesabı oluşturmanız gerekir, daha fazla ayrıntı için :numref:`sec_sagemaker` içine bakın. 

## EC2 Örneği Oluşturma ve Çalıştırma

AWS hesabınıza giriş yaptıktan sonra EC2 paneline gitmek için "EC2"'ye (:numref:`fig_aws` içinde kırmızı kutuyla işaretlenmiş) tıklayın. 

![EC2 konsolunu açma.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2`, hassas hesap bilgilerinin gri renkte olduğu EC2 panelini gösterir. 

![EC2 paneli.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Yer Ön Ayarı 
Gecikmeyi azaltmak için yakındaki bir veri merkezi seçin, örneğin "Oregon" (:numref:`fig_ec2` figürünün sağ üstündeki kırmızı kutuyla işaretlenmiştir). Çin'de bulunuyorsanız, Seul veya Tokyo gibi yakındaki Asya Pasifik bölgesini seçebilirsiniz. Bazı veri merkezlerinin GPU örneklerine sahip olmayabileceğini lütfen unutmayın. 

### Kısıtları Artırmak 
Bir örneği seçmeden önce, :numref:`fig_ec2` içinde gösterildiği gibi soldaki çubuktaki "Limits - Kısıtlar" etiketine tıklayarak miktar kısıtlamaları olup olmadığını kontrol edin. :numref:`fig_limits`, bu tür bir kısıtlamanın bir örneğini gösterir. Hesap şu anda bölge başına "p2.xlarge" örneğini açamıyor. Bir veya daha fazla örnek açmanız gerekiyorsa, daha yüksek bir örnek kotasına başvurmak için "Request limit increase - Limit artışı iste" bağlantısına tıklayın. Genellikle, bir uygulamanın işlenmesi bir iş günü sürer. 

![Örnek miktarı kısıtlamaları.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### Örneği Başlatma 
Sonrasında örneğinizi başlatmak için :numref:`fig_ec2` içinde kırmızı kutuyla işaretlenmiş "Launch Instance - Örneği Başlat" düğmesine tıklayın. 

Uygun bir AMI (AWS Machine Image) seçerek başlıyoruz. Arama kutusuna "Ubuntu" girin (:numref:`fig_ubuntu` içinde kırmızı kutu ile işaretlenmiş). 

![Bir işletim sistemi seçme](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2, aralarından seçim yapabileceğiniz birçok farklı örnek yapılandırması sağlar. Bu bazen bir acemi için ezici hissedilebilir. İşte uygun makinelerin bir tablosu: 

| Name | GPU         | Notlar                                    |
|------|-------------|-------------------------------------------|
| g2   | Grid K520   | eski                                      |
| p2   | Kepler K80  | eski ama genellikle anlık olarak ucuz     | 
| g3   | Maxwell M60 | iyi ödünleşme                             |
| p3   | Volta V100  | FP16 için yüksek performans               |
| g4   | Turing T4   | FP16/INT8 çıkarsama için optimize edilmiş |

Yukarıdaki tüm sunucular, kullanılan GPU sayısını belirten birden fazla seçenek sunar. Örneğin, bir p2.xlarge 1 GPU'ya ve p2.16xlarge 16 GPU'ya ve daha fazla belleğe sahiptir. Daha fazla ayrıntı için bkz. [AWS EC2 belegeleri](https://aws.amazon.com/ec2/instance-types/) veya [özet sayfası](https://www.ec2instances.info). Örnekleme amacıyla, bir p2.xlarge yeterli olacaktır (:numref:`fig_p2x` içinde kırmızı kutu olarak işaretlenmiştir).

**Not:** Uygun sürücülere sahip bir GPU etkin örneği ve GPU etkin bir MXNet sürümünü kullanmanız gerekir. Aksi takdirde GPU'ları kullanmaktan herhangi bir fayda görmezsiniz.

![Bir örnek seçme.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

Şimdiye kadar, :numref:`fig_disk` figürünün üstünde gösterildiği gibi, bir EC2 örneğini başlatmak için yedi adımdan ilk ikisini tamamladık. Bu örnekte, "3.Örneği Yapılandır - Configure Instance", "5. Etiket Ekle - Add Tags" ve "6. Güvenlik Grubunu Yapılandır - Configure Security Group" adımları için varsayılan yapılandırmaları saklıyoruz. "4. Depolama Ekle - Add Storage"'ye dokunun ve varsayılan sabit disk boyutunu 64 GB'a yükseltin (:numref:`fig_disk` kırmızı kutuda işaretlenmiş). CUDA'nın kendi başına zaten 4 GB aldığını unutmayın. 

![Modify instance hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

Son olarak, "7. Gözden geçir - Review”'ye gidin ve yapılandırılmış örneği başlatmak için “Başlat - Launch”'ı tıklayın. Sistem artık örneğe erişmek için kullanılan anahtar çiftini seçmenizi isteyecek. Anahtar çiftiniz yoksa, anahtar çifti oluşturmak için :numref:`fig_keypair` içindeki ilk açılır menüden “Yeni bir anahtar çifti oluştur - Create a new key pair”'u seçin. Daha sonra, bu menü için “Varolan bir anahtar çiftini seç - Choose an existing key pair”'i seçip daha önce oluşturulmuş anahtar çiftini seçebilirsiniz. Oluşturulan örneği başlatmak için “Örnekleri Başlat - Launch Instances”'ı tıklayın. 

![Bir anahtar çifti seçme.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Yeni bir tane oluşturduysanız anahtar çiftini indirdiğinizden ve güvenli bir konumda sakladığınızdan emin olun. Bu sunucuya tek yol SSH'dir. Bu örneğin durumunu görüntülemek için :numref:`fig_launching` içinde gösterilen örnek kimliğini tıklatın. 

![Örnek kimliğini tıklama.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Örneğe Bağlanma

:numref:`fig_connect` içinde gösterildiği gibi, örnek durumu yeşile döndükten sonra örneği sağ tıklatın ve örnek erişim yöntemini görüntülemek için `Connect`'i seçin. 

![Örnek erişimini ve başlatma yöntemini görüntüleme.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

Bu yeni bir anahtarsa, SSH'nin çalışması için herkese açık bir şekilde görüntülenmemelidir. `D2L_key.pem`'ü sakladığınız klasöre gidin (örn. İndirilenler klasörü) ve anahtarın genel olarak görüntülenmediğinden emin olun.

```bash
cd /Downloads  ## D2L_key.pem İndirilenler klasöründe depolanıyorsa
chmod 400 D2L_key.pem
```

![Örnek erişimini ve başlatma yöntemini görüntüleme.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

Şimdi, :numref:`fig_chmod`' figürunun alt kırmızı kutusuna ssh komutunu kopyalayın ve komut satırına yapıştırın:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

Komut satırı "Are you sure you want to continue connecting (yes/no) - Bağlanmaya devam etmek istediğinizden emin misiniz (evet/hayır)" sorduğunda, "yes - evet" yazın ve örneğe giriş yapmak için Enter tuşuna basın. 

Sunucunuz şimdi hazır. 

## CUDA Kurulumu

CUDA'yı yüklemeden önce, örneği en son sürücülerle güncellediğinizden emin olun.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


Burada CUDA 10.1'i indiriyoruz. :numref:`fig_cuda` içinde gösterildiği gibi CUDA 10.1'in indirme bağlantısını bulmak için NVIDIA'nın [resmi deposu](https://developer.nvidia.com/cuda-downloads)nu ziyaret edin.

![CUDA 10.1 indirme adresini bulma.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

Talimatları kopyalayın ve CUDA 10.1'i yüklemek için terminale yapıştırın.

```bash
## CUDA web sitesinden kopyalanan bağlantıyı yapıştırın
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

Programı yükledikten sonra, GPU'ları görüntülemek için aşağıdaki komutu çalıştırın.

```bash
nvidia-smi
```

Son olarak, diğer kütüphanelerin bulmasına yardımcı olmak için kütüphane yoluna CUDA'yı ekleyin.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## MXNet'i Yükleme ve D2L Not Defterlerini İndirme

Öncelikle, kurulumu basitleştirmek için Linux için [Miniconda](https://conda.io/en/latest/miniconda.html)'yı yüklemeniz gerekir. İndirme bağlantısı ve dosya adı değişikliklere tabidir, bu nedenle lütfen Miniconda web sitesine gidin ve :numref:`fig_miniconda`'te gösterildiği gibi “Copy Link Address - Bağlantı Adresini Kopyala” düğmesine tıklayın. 

![Miniconda'yı indirme.](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# Bağlantı ve dosya adı değişebilir
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Miniconda kurulumundan sonra CUDA ve conda'yı etkinleştirmek için aşağıdaki komutu çalıştırın.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```

Ardından, bu kitabın kodunu indirin.

```bash
sudo apt-get install unzip
mkdir d2l-tr && cd d2l-tr
curl https://tr.d2l.ai/d2l-tr.zip -o d2l-tr.zip
unzip d2l-tr.zip && rm d2l-tr.zip
```

Ardından conda `d2l` ortamını oluşturun ve yüklemeye devam etmek için `y` girin.

```bash
conda create --name d2l -y
```

`d2l` ortamını oluşturduktan sonra etkinleştirin ve `pip`'i yükleyin.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

Son olarak, MXNet ve `d2l` paketini yükleyin. `cu101` soneki, bunun CUDA 10.1 varyantı olduğu anlamına gelir. Farklı sürümler için, mesela CUDA 10.0, `cu100`'yı seçmek isteyebirlirsiniz.

```bash
pip install mxnet-cu101==1.7.0
pip install git+https://github.com/d2l-ai/d2l-en
```

Her şeyin yolunda olup olmadığını hızlı bir şekilde test edebilirsiniz:

```
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```

## Jupyter'i Çalıştırma

Jupyter'ı uzaktan çalıştırmak için SSH bağlantı noktası yönlendirme kullanmanız gerekir. Sonuçta, buluttaki sunucunun bir monitörü veya klavyesi yoktur. Bunun için sunucunuza masaüstünüzden (veya dizüstü bilgisayarınızdan) aşağıdaki gibi oturum açın.

```
# Bu komut yerel komut satırında çalıştırılmalıdır
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter`, Jupyter Notebook çalıştırdıktan sonra olası çıktıyı gösterir. Son satır 8888 numaralı bağlantı noktası için URL'dir. 

![Output after running Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Bağlantı noktası 8889 numaralı bağlantı noktasına yönlendirmeyi kullandığınız için bağlantı noktası numarasını değiştirmeniz ve yerel tarayıcınızda URL'yi açarken Jupyter tarafından verilen sırrı (secret) kullanmanız gerekir. 

## Kullanılmayan Örnekleri Kapatma

Bulut hizmetleri kullanım süresine göre faturalandırıldığından, kullanılmayan örnekleri kapatmanız gerekir. Alternatifler olduğunu unutmayın: Bir örneği "durdurmak", yeniden başlatabilmeniz anlamına gelir. Bu, normal sunucunuzun gücünü kapatmaya benzer. Ancak durdurulan örnekler, korunan sabit disk alanı için küçük bir miktar faturalandırılır. "Sonlandır - Terminate", onunla ilişkili tüm verileri siler. Bu diski içerir, bu nedenle yeniden başlatamazsınız. Sadece gelecekte ihtiyacınız olmayacağını biliyorsanız bunu yapın. 

Örneği daha birçok örnek için şablon olarak kullanmak istiyorsanız, :numref:`fig_connect` içindeki örnekte sağ tıklayın ve örneğin görüntüsünü oluşturmak için "İmage - Görüntü" $\rightarrow$ "Create - Oluştur"'u seçin. Bu işlem tamamlandıktan sonra örneği sonlandırmak için "Instance State - Örnek Durumu" $\rightarrow$ "Termiante - Sonlandır"'ı seçin. Bu örneği daha sonra kullanmak istediğinizde, kaydedilen görüntüye dayalı bir örnek oluşturmak için bu bölümde açıklanan bir EC2 örneği oluşturma ve çalıştırma adımlarını uygulayabilirsiniz. Tek fark, :numref:`fig_ubuntu` içinde gösterilen "1. Choose AMI - AMI Seç" bölümünde, kayıtlı görüntünüzü seçmek için soldaki "My AMIs - AMI'lerim" seçeneğini kullanmanız gerektiğidir. Oluşturulan örnek, görüntü sabit diskinde depolanan bilgileri saklar. Örneğin, CUDA ve diğer çalışma zamanı ortamlarını yeniden yüklemeniz gerekmez. 

## Özet

* Kendi bilgisayarınızı satın almak ve oluşturmak zorunda kalmadan isteğe bağlı örnekleri başlatabilir ve durdurabilirsiniz.
* Kullanabilmeniz için uygun GPU sürücülerini yüklemeniz gerekir.

## Alıştırmalar

1. Bulut kolaylık sağlar, ancak ucuza gelmez. Fiyatları nasıl düşüreceğinizi görmek için [anlık örnekler](https://aws.amazon.com/ec2/spot/)i nasıl başlatacağınızı öğrenin.
1. Farklı GPU sunucuları ile deney yapın. Ne kadar hızlılar?
1. Çoklu GPU sunucuları ile deney yapın. İşleri ne kadar iyi ölçeklendirebilirsiniz?

[Tartışmalar](https://discuss.d2l.ai/t/423)
