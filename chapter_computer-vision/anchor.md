# Çapa Kutuları
:label:`sec_anchor`

Nesne algılama algoritmaları genellikle giriş görüntüsünde çok sayıda bölgeyi örneklemek, bu bölgelerin ilgi çeken nesneleri içerip içermediğini belirler ve
*topraklama gerçeği sınırlayıcı kutular*
nesnelerin daha doğru. Farklı modeller farklı bölge örnekleme şemalarını benimseyebilir. Burada bu tür yöntemlerden birini sunuyoruz: her piksel üzerinde ortalanmış değişen ölçekler ve en boy oranlarına sahip birden fazla sınırlayıcı kutu oluşturur. Bu sınırlayıcı kutulara *çapa kutuları* denir. :numref:`sec_ssd`'te çapa kutularına dayalı bir nesne algılama modeli tasarlayacağız. 

Öncelikle, sadece daha özlü çıktılar için baskı doğruluğunu değiştirelim.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## Çoklu Çapa Kutuları Oluşturma

Giriş görüntüsünün $h$ yüksekliğine ve $w$ genişliğine sahip olduğunu varsayalım. Görüntünün her pikselinde ortalanmış farklı şekillere sahip çapa kutuları oluşturuyoruz. *scale* $s\in (0, 1]$, *en boy oranı* (genişliğin yüksekliğe oranı) $r > 0$'dır. Ardından [**çapa kutusunun genişliği ve yüksekliği sırasıyla $ws\sqrt{r}$ ve $hs/\sqrt{r}$'dür.**] Merkez konumu verildiğinde, bilinen genişlik ve yüksekliğe sahip bir çapa kutusu belirlendiğini unutmayın. 

Farklı şekillere sahip birden çok çapa kutusu oluşturmak için bir dizi terazi $s_1,\ldots, s_n$ ve bir dizi en/boy oranı $r_1,\ldots, r_m$ ayarlayalım. Bu ölçeklerin ve en boy oranlarının tüm kombinasyonlarını merkez olarak her pikselle birlikte kullanırken, girdi görüntüsünde toplam $whnm$ bağlantı kutusu bulunur. Bu çapa kutuları tüm zemin gerçeği sınırlayıcı kutuları kapsayabilir rağmen, hesaplama karmaşıklığı kolayca çok yüksektir. Uygulamada, sadece $s_1$ veya $r_1$ (** içeren bu kombinasyonları göz önünde bulundurun): 

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**) 

Yani, aynı pikselde ortalanmış çapa kutularının sayısı $n+m-1$'dir. Tüm giriş görüntüsü için toplam $wh(n+m-1)$ çapa kutusu oluşturacağız. 

Yukarıdaki çapa kutuları oluşturma yöntemi aşağıdaki `multibox_prior` işlevinde uygulanır. Giriş görüntüsünü, ölçeklerin bir listesini ve en boy oranlarının bir listesini belirleriz, daha sonra bu işlev tüm bağlantı kutularını döndürür.

```{.python .input}
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

[**döndürülen çapa kutusu değişkeninin `Y`**] şeklinin olduğunu görebiliriz (parti boyutu, çapa kutusu sayısı, 4).

```{.python .input}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

Çapa kutusu değişkeninin şeklini değiştirdikten sonra `Y` olarak (görüntü yüksekliği, görüntü genişliği, aynı piksel üzerinde ortalanmış çapa kutularının sayısı, 4), belirtilen piksel konumuna ortalanmış tüm çapa kutularını elde edebiliriz. Aşağıda [**merkezli ilk çapa kutusuna erişiyoruz (250, 250) **]. Dört öğeye sahiptir: $(x, y)$ eksen sol üst köşedeki koordinatlar ve bağlantı kutusunun sağ alt köşesindeki $(x, y)$ eksen koordinatları. Her iki eksenin koordinat değerleri sırasıyla görüntünün genişliği ve yüksekliğine bölünür; böylece aralık 0 ile 1 arasındadır.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

[**resimdeki bir piksel üzerinde ortalanmış tüm bağlantı kutularını göstermek**] için, görüntü üzerinde birden fazla sınırlayıcı kutu çizmek için aşağıdaki `show_bboxes` işlevini tanımlıyoruz.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Gördüğümüz gibi, `boxes` değişkenindeki $x$ ve $y$ eksenlerinin koordinat değerleri sırasıyla görüntünün genişliği ve yüksekliğine bölünmüştür. Çapa kutularını çizerken, orijinal koordinat değerlerini geri yüklememiz gerekir; böylece, aşağıda `bbox_scale` değişkeni tanımlarız. Şimdi, resimde (250, 250) merkezli tüm çapa kutularını çizebiliriz. Gördüğünüz gibi, 0.75 ölçeği ve 1 en boy oranına sahip mavi çapa kutusu, görüntüdeki köpeği iyi çevreler.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**Birlik (IoU) üzerinde Kesişim**]

Sadece bir çapa kutusunun “iyi” köpeği görüntüde çevrelediğini belirttik. Nesnenin zemin gerçeği sınırlayıcı kutusu biliniyorsa, burada “iyi” nasıl ölçülebilir? Sezgisel olarak, çapa kutusu ile zemin gerçeği sınırlayıcı kutu arasındaki benzerliği ölçebiliriz. *Jaccard indeksi* iki set arasındaki benzerliği ölçebileceğini biliyoruz. Verilen setleri $\mathcal{A}$ ve $\mathcal{B}$, onların Jaccard indeksi onların birlik boyutuna bölünmüş kavşak boyutudur: 

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

Aslında, herhangi bir sınırlayıcı kutunun piksel alanını bir piksel kümesi olarak düşünebiliriz. Bu şekilde, iki sınırlayıcı kutunun benzerliğini piksel setlerinin Jaccard indeksi ile ölçebiliriz. İki sınırlayıcı kutu için, Jaccard indeksini genellikle :numref:`fig_iou`'te gösterildiği gibi, kesişme alanlarının birleşme alanlarına oranı olan *union* üzerinden kesişme (*IoU*) olarak adlandırırız. Bir IoU aralığı 0 ile 1:0 arasındadır, iki sınırlayıcı kutunun hiç çakışmadığı anlamına gelirken, 1 ise iki sınırlayıcı kutunun eşit olduğunu gösterir. 

![IoU is the ratio of the intersection area to the union area of two bounding boxes.](../img/iou.svg)
:label:`fig_iou`

Bu bölümün geri kalanında, çapa kutuları ile zemin doğruluk sınırlayıcı kutular ve farklı çapa kutuları arasındaki benzerliği ölçmek için IoU kullanacağız. İki bağlantı listesi veya sınırlayıcı kutular göz önüne alındığında, aşağıdaki `box_iou`, bu iki listede çift yönlü IoU hesaplar.

```{.python .input}
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## Eğitim Verilerinde Çapa Kutuları Etiketleme
:label:`subsec_labeling-anchor-boxes`

Bir eğitim veri kümesinde, her bağlantı kutusunu bir eğitim örneği olarak değerlendiririz. Bir nesne algılama modelini eğitmek için, her bir bağlantı kutusu için*class* ve *ofset* etiketlerine ihtiyacımız var; burada birincisi, çapa kutusuyla ilgili nesnenin sınıfıdır ve ikincisi ise çapa kutusuna göre topraklama gerçeği sınırlayıcı kutunun ofset olduğu. Tahmin sırasında, her görüntü için birden çok çapa kutusu oluşturur, tüm bağlantı kutuları için sınıfları ve uzaklıkları tahmin eder, öngörülen sınırlayıcı kutuları elde etmek için konumlarını tahmin edilen uzaklıklara göre ayarlarız ve son olarak yalnızca belirli kriterleri karşılayan öngörülen sınırlayıcı kutuları çıkarırız. 

Bildiğimiz gibi, bir nesne algılama eğitim seti, *zemin doğruluğunu sınırlayan kutular* ve çevrelenmiş nesnelerin sınıfları için etiketlerle birlikte gelir. Oluşturulan herhangi bir *çapa kutusu* etiketlemek için, çapa kutusuna en yakın olan *atanmış* zemin gerçeği sınırlama kutusunun etiketlenmiş konumunu ve sınıfını ifade ederiz. Aşağıda, bağlantı kutularına en yakın zemin hakikati sınırlayıcı kutuları atamak için bir algoritma açıklıyoruz.  

### [**Bağlantı Kutularına Zemin Gerçeği Sınırlayıcı Kutular Atama**]

Bir görüntü göz önüne alındığında, çapa kutularının $A_1, A_2, \ldots, A_{n_a}$ ve zemin gerçeği sınırlayıcı kutuların $B_1, B_2, \ldots, B_{n_b}$ olduğunu varsayalım, burada $n_a \geq n_b$. $i^\mathrm{th}$ satırında $x_{ij}$ elemanı ve $j^\mathrm{th}$ sütununda $j^\mathrm{th}$ olan bir matris $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$, ankraj kutusunun IoU $A_i$ ve zemin gerçeği sınırlayıcı kutunun $B_j$ olan bir matris tanımlayalım. Algoritma aşağıdaki adımlardan oluşur: 

1. $\mathbf{X}$ matrisindeki en büyük elemanı bulun ve satır ve sütun indekslerini sırasıyla $i_1$ ve $j_1$ olarak belirtin. Daha sonra $B_{j_1}$ zemin gerçeği sınırlayıcı kutu $A_{i_1}$ çapa kutusuna atanır. Bu oldukça sezgiseldir, çünkü $A_{i_1}$ ve $B_{j_1}$, tüm çapa kutuları ve zemin gerçeği sınırlayıcı kutular arasında en yakın olanlardır. İlk atamadan sonra, ${i_1}^\mathrm{th}$ satırındaki tüm öğeleri ve ${j_1}^\mathrm{th}$ sütunundaki ${j_1}^\mathrm{th}$ matrisindeki $\mathbf{X}$ sütununu atın. 
1. $\mathbf{X}$ matrisinde kalan elemanların en büyüğünü bulun ve satır ve sütun indekslerini sırasıyla $i_2$ ve $j_2$ olarak belirtin. $A_{i_2}$'yi çapa kutusuna $B_{j_2}$'yi atayın ve ${i_2}^\mathrm{th}$ satırındaki tüm öğeleri ve ${j_2}^\mathrm{th}$ matrisindeki ${j_2}^\mathrm{th}$ sütununda $\mathbf{X}$ sütununu atıyoruz.
1. Bu noktada, iki satırdaki elemanlar ve $\mathbf{X}$ matrisindeki iki sütun atılmıştır. $\mathbf{X}$ matrisindeki $n_b$ sütunlarındaki tüm elemanlar atılana kadar devam ediyoruz. Şu anda, $n_b$ çapa kutularının her birine bir zemin gerçeği sınırlayıcı kutu atadık.
1. Sadece kalan $n_a - n_b$ çapa kutularından geçiş yapın. Örneğin, $A_i$ herhangi bir çapa kutusu göz önüne alındığında $A_i$ numaralı matrisin $\mathbf{X}$ sırası boyunca $A_i$ ile en büyük IoU içeren $B_j$'yi $B_j$'yi bulun ve yalnızca bu IoU önceden tanımlanmış bir eşikten büyükse $A_i$'e $A_i$'i atayın.

Yukarıdaki algoritmayı somut bir örnek kullanarak gösterelim. :numref:`fig_anchor_label`'te (solda) gösterildiği gibi, $\mathbf{X}$ matrisindeki maksimum değerin $x_{23}$ olduğunu varsayarak $x_{23}$, $B_3$ numaralı çapa kutusuna $B_3$ numaralı çapa kutusuna atarız. Daha sonra, matrisin satır 2 ve sütun 3'teki tüm unsurları atıyoruz, kalan elemanlarda (gölgeli alan) en büyük $x_{71}$'i bulun ve $A_7$ ankraj kutusuna $B_1$ numaralı zemin gerçeği sınırlayıcı kutuyu atayın. Daha sonra, :numref:`fig_anchor_label` (orta) gösterildiği gibi, matrisin satır 7 ve sütun 1'deki tüm öğeleri atın, kalan elemanlarda (gölgeli alan) en büyük $x_{54}$'ü bulun ve $B_4$ numaralı çapa kutusuna $B_4$ numaralı çapa kutusuna atayın. Son olarak, :numref:`fig_anchor_label` (sağda) gösterildiği gibi, matrisin satır 5 ve sütun 4'teki tüm öğeleri atın, kalan elemanlarda (gölgeli alan) en büyük $x_{92}$'i bulun ve $B_2$'yı çapa kutusuna $B_2$ numaralı çapa kutusuna atayın. Bundan sonra, sadece kalan çapa kutularından $A_1, A_3, A_4, A_6, A_8$'a geçmemiz ve eşiğe göre toprak-hakikat sınırlayıcı kutuların atanıp atanmayacağını belirlemeliyiz. 

![Assigning ground-truth bounding boxes to anchor boxes.](../img/anchor-label.svg)
:label:`fig_anchor_label`

Bu algoritma aşağıdaki `assign_anchor_to_bbox` işlevinde uygulanır.

```{.python .input}
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= 0.5)[0]
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### Etiketleme Sınıfları ve Ofsetler

Artık her çapa kutusu için sınıfı etiketleyebilir ve ofset yapabiliriz. Bir çapa kutusu $A$ bir zemin gerçeği sınırlayıcı kutu $B$ atandığını varsayalım. Bir yandan, çapa kutusu $A$ sınıfı $B$ olarak etiketlenecektir. Öte yandan, $A$ ankraj kutusunun ofseti, $B$ ve $A$ arasındaki merkezi koordinatlar arasındaki göreli konuma göre bu iki kutu arasındaki göreli boyutla birlikte etiketlenecektir. Veri kümesindeki farklı kutuların farklı konumları ve boyutları göz önüne alındığında, bu göreceli konumlara ve boyutlara dönüşümler uygulayarak sığması daha kolay olan daha düzgün dağıtılmış uzaklıklara yol açabiliriz. Burada ortak bir dönüşümü tanımlıyoruz. [**$A$ ve $B$ olarak $B$ olarak $(x_a, y_a)$ ve $(x_b, y_b)$, genişlikleri $w_a$ ve $w_b$ ve $w_b$ olarak sırasıyla $h_a$ ve $h_b$ gibi yükseklikleri göz önüne alındığında. Biz $A$ ofset olarak etiketleyebilirsiniz 

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
where default values of the constants are $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, and $\sigma_w=\sigma_h=0.2$.
This transformation is implemented below in the `offset_boxes` function.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

Bir çapa kutusuna bir zemin gerçeği sınırlayıcı kutu atanmamışsa, sadece çapa kutusunun sınıfını “arka plan” olarak etiketleriz. Sınıfları arka plan olan bağlantı kutuları genellikle *negative* bağlantı kutuları olarak adlandırılır ve geri kalanı ise *pozitive* bağlantı kutuları olarak adlandırılır. Aşağıdaki `multibox_target` işlevini temel doğruluk sınırlayıcı kutuları (`labels` bağımsız değişkeni) kullanarak [**label sınıfları ve bağlantı kutuları için uzaklıklar**](`anchors` bağımsız değişkeni) için uyguluyoruz. Bu işlev, arka plan sınıfını sıfıra ayarlar ve yeni bir sınıfın tamsayı dizinini tek artırır.

```{.python .input}
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### Bir Örnek

Çapa kutusu etiketlemesini somut bir örnekle gösterelim. İlk eleman sınıftır (köpek için 0 ve kedi için 1) ve kalan dört öğe - sol üst köşede ve sağ alt köşede $(x, y)$ eksen koordinatlarıdır (aralık 0 ile 1 arasındadır) yüklenen görüntüde köpek ve kedi için zemin gerçeği sınırlayıcı kutuları tanımlıyoruz. Ayrıca, sol üst köşenin ve sağ alt köşenin koordinatlarını kullanarak etiketlenecek beş çapa kutusu oluşturuyoruz: $A_0, \ldots, A_4$ (indeks 0'dan başlar). Sonra [****resimdeki bu temel doğruluk sınırlayıcı kutuları ve çapa kutularını çiziyoruz.

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

Yukarıda tanımlanan `multibox_target` işlevini kullanarak, köpek ve kedi için bu çapa kutularının sınıflarını ve ofsetlerini zemin hakikati sınırlayıcı kutulara dayanarak etiketleyebiliriz. Bu örnekte, arka plan, köpek ve kedi sınıflarının indeksleri sırasıyla 0, 1 ve 2'dir. Aşağıda, çapa kutuları ve zemin gerçeği sınırlayıcı kutular örnekleri için bir boyut ekliyoruz.

```{.python .input}
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

İade edilen sonuçta, hepsi tensör formatında olan üç öğe vardır. Üçüncü öğe, giriş bağlantı kutularının etiketli sınıflarını içerir. 

Aşağıdaki döndürülen sınıf etiketlerini, çapa kutusu ve resimdeki zemin hakikati sınırlayıcı kutu konumlarına göre analiz edelim. İlk olarak, çapa kutuları ve zemin gerçeği sınırlayıcı kutuların tüm çiftleri arasında, $A_4$ çapa kutusunun IoU ve kedinin zemin gerçeği sınırlayıcı kutusu en büyüğüdür. Böylece, $A_4$ sınıfı kedi olarak etiketlenmiştir. $A_4$ veya kedinin zemin gerçeği sınırlayıcı kutusunu içeren çiftleri çıkarmak, geri kalanlar arasında $A_1$ ankraj kutusunun çifti ve köpeğin zemin gerçeği sınırlayıcı kutusu en büyük IoU'ya sahiptir. Yani $A_1$ sınıfı köpek olarak etiketlenmiştir. Ardından, kalan üç etiketlenmemiş çapa kutusundan geçmemiz gerekiyor: $A_0$, $A_2$ ve $A_3$. $A_0$ için, en büyük IoU ile zemin gerçeği sınırlayıcı kutunun sınıfı köpektir, ancak IoU önceden tanımlanmış eşiğin (0.5) altındadır, bu nedenle sınıf arka plan olarak etiketlenir; $A_2$ için, en büyük IoU ile zemin gerçeği sınırlama kutusunun sınıfı kedi ve IoU eşiği aşar, bu nedenle sınıfı kedi olarak etiketlenir; $A_3$ için, en büyük IoU'ya sahip zemin gerçeği sınırlayıcı kutunun sınıfı kedidir, ancak değer eşiğin altındadır, bu nedenle sınıf arka plan olarak etiketlenir.

```{.python .input}
#@tab all
labels[2]
```

Döndürülen ikinci öğe şeklin bir maske değişkenidir (toplu iş boyutu, çapa kutularının dört katı). Maske değişkenindeki her dört öğe, her bir çapa kutusunun dört uzaklık değerine karşılık gelir. Arka plan algılama umurumda olmadığından, bu negatif sınıfın ofsetleri objektif işlevi etkilememelidir. Elementwise çarpımları sayesinde, maske değişkenindeki sıfırlar, objektif işlevi hesaplamadan önce negatif sınıf uzaklıklarını filtreler.

```{.python .input}
#@tab all
labels[1]
```

İlk döndürülen öğe, her bağlantı kutusu için etiketlenmiş dört uzaklık değeri içerir. Negatif sınıf bağlantı kutularının uzaklıklarının sıfır olarak etiketlendiğini unutmayın.

```{.python .input}
#@tab all
labels[0]
```

## Maksimum Olmayan Bastırma ile Sınırlayıcı Kutuları Tahmin Edileme
:label:`subsec_predicting-bounding-boxes-nms`

Tahmin sırasında, görüntü için birden çok bağlantı kutusu oluşturur ve bunların her biri için sınıfları ve uzaklıkları tahmin ederiz. * Tahmin edilen sınırlama kutusu*, tahmin edilen ofset ile bir çapa kutusuna göre elde edilir. Aşağıda, giriş olarak ankraj ve ofset tahminleri alan `offset_inverse` işlevini uyguluyoruz ve [**öngörülen sınırlama kutusu koordinatlarını döndürmek için ters ofset dönüşümleri uygular].

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

Çok sayıda çapa kutusu olduğunda, benzer (önemli örtüşme ile) tahmin edilen sınırlama kutuları aynı nesneyi çevreleyen için potansiyel olarak çıktı olabilir. Çıktıyı basitleştirmek için, aynı nesneye ait benzer öngörülen sınırlama kutularını *maksimum olmayan bastırma* (NMS) kullanarak birleştirebiliriz. 

İşte maksimum olmayan bastırma nasıl çalışır. Tahmin edilen bir sınırlama kutusu $B$ için nesne algılama modeli her sınıf için tahmin edilen olasılığı hesaplar. $p$ tarafından tahmin edilen en büyük olasılığın belirlenmesi, bu olasılığa karşılık gelen sınıf $B$ için tahmin edilen sınıftır. Özellikle, $p$'yı öngörülen sınırlama kutusu $B$'un*güven* (skor) olarak adlandırıyoruz. Aynı görüntüde, öngörülen tüm arka plan dışı sınırlama kutuları $L$ liste oluşturmak için azalan sıraya göre sıralanır. Ardından aşağıdaki adımlarda sıralanmış listeyi $L$'yi manipüle ediyoruz: 

1. Tahmin edilen sınırlama kutusunu $B_1$ temel olarak $L$'ten en yüksek güvenle seçin ve $B_1$ ile IoU $L$'ten önceden tanımlanmış bir eşiği $\epsilon$'ü aşan temel olmayan öngörülen tüm sınırlama kutularını kaldırın. Bu noktada, $L$ öngörülen sınırlayıcı kutuyu en yüksek güvenle tutar, ancak buna çok benzer olan başkalarını bırakır. Özetle, *maksimum* olmayan* güven puanı olanlar*bastırılır*.
1. Başka bir temel olarak $L$'dan ikinci en yüksek güvenle öngörülen sınırlama kutusunu $B_2$'ü seçin ve $B_2$ ile IoU $L$'dan $\epsilon$'i aşan temel olmayan tüm öngörülen sınırlama kutularını kaldırın.
1. $L$'teki tüm öngörülen sınırlama kutuları temel olarak kullanılıncaya kadar yukarıdaki işlemi yineleyin. Şu anda, $L$'teki öngörülen sınırlayıcı kutuların herhangi bir çiftinin IoU $\epsilon$ eşiğinin altındadır; bu nedenle, hiçbir çift birbiriyle çok benzer değildir. 
1. Listedeki tüm öngörülen sınırlayıcı kutuları çıktısı $L$.

[**Aşağıdaki `nms` işlevi, güven puanlarını azalan sırada sıralar ve indekslerini döndürür. **]

```{.python .input}
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

Aşağıdaki `multibox_detection`'ü [**sınırlayıcı kutuları tahmin etmek için maksimum olmayan bastırma uygulamak**] için tanımlıyoruz. Uygulamanın biraz karmaşık olduğunu görürseniz endişelenmeyin: uygulamadan hemen sonra somut bir örnekle nasıl çalıştığını göstereceğiz.

```{.python .input}
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

Şimdi [**yukarıdaki uygulamaları dört çapa kutusuna sahip somut bir örneğe uygulayalım]. Basitlik için, tahmin edilen ofsetlerin tümünün sıfırlar olduğunu varsayıyoruz. Bu, tahmin edilen sınırlayıcı kutuların çapa kutuları olduğu anlamına gelir. Arka plan, köpek ve kedi arasındaki her sınıf için, öngörülen olasılığını da tanımlıyoruz.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

Bu öngörülen sınırlayıcı kutuları resme güvenleriyle çizebiliriz.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

Şimdi, eşiğin 0.5'e ayarlandığı maksimum olmayan bastırmayı gerçekleştirmek için `multibox_detection` işlevini çağırabiliriz. Tensör girişinde örnekler için bir boyut eklediğimizi unutmayın. 

[**döndürülen sonuç şekli**] olduğunu görebiliriz (toplu boyut, çapa kutusu sayısı, 6). En iç boyuttaki altı öğe, aynı öngörülen sınırlama kutusunun çıkış bilgilerini verir. İlk öğe, 0'dan başlayan tahmin edilen sınıf dizinidir (0 köpek ve 1 kedi). -1 değeri, maksimum olmayan bastırmada arka planı veya kaldırmayı gösterir. İkinci unsur, tahmin edilen sınırlama kutusunun güvenidir. Kalan dört öğe, sırasıyla sol üst köşenin $(x, y)$ eksen koordinatlarıdır ve öngörülen sınırlama kutusunun sağ alt köşesidir (aralık 0 ile 1 arasındadır).

```{.python .input}
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

Sınıf -1 olan bu öngörülen sınırlayıcı kutuları kaldırdıktan sonra [**maksimum olmayan bastırma ile tutulan son öngörülen sınırlama kutusundan çıktı**] yapabiliriz.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

Pratikte, maksimum olmayan bastırmayı gerçekleştirmeden önce bile daha düşük güvenle öngörülen sınırlayıcı kutuları kaldırabilir, böylece bu algoritmadaki hesaplamayı azaltabiliriz. Aynı zamanda maksimum olmayan bastırmanın çıktısını da sonradan işleyebiliriz, örneğin, sonuçları yalnızca nihai çıktıda daha yüksek bir güvenle tutarak. 

## Özet

* Görüntünün her pikselinde ortalanmış farklı şekillere sahip çapa kutuları oluşturuyoruz.
* Jakar indeksi olarak da bilinen birleşme (IoU) üzerindeki kesişme, iki sınırlayıcı kutunun benzerliğini ölçer. Kesişme alanlarının sendika alanlarına oranıdır.
* Bir eğitim setinde, her bir çapa kutusu için iki tip etikete ihtiyacımız var. Biri, çapa kutusuyla ilgili nesnenin sınıfıdır ve diğeri ise çapa kutusuna göre zemin gerçeği sınırlayıcı kutunun ofsetidir.
* Tahmin sırasında, benzer öngörülen sınırlama kutularını kaldırmak için maksimum olmayan bastırma (NMS) kullanabiliriz ve böylece çıktıyı basitleştiririz.

## Egzersizler

1. `multibox_prior` işlevinde `sizes` ve `ratios` değerlerini değiştirin. Oluşturulan çapa kutularındaki değişiklikler nelerdir?
1. 0,5 IoU ile iki sınırlayıcı kutu oluşturun ve görselleştirin. Birbirleriyle nasıl örtüşürler?
1. :numref:`subsec_labeling-anchor-boxes` ve :numref:`subsec_predicting-bounding-boxes-nms` değişkeni `anchors` değiştirin. Sonuçlar nasıl değişir?
1. Maksimum olmayan bastırma, öngörülen sınırlayıcı kutuları kaldırarak* kaldırarak bastıran açgözlü bir algoritmadır. Bu kaldırılmış olanlardan bazılarının gerçekten yararlı olması mümkün mü? * yumuşak* bastırmak için bu algoritma nasıl değiştirilebilir? Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`'e başvurabilirsiniz.
1. El yapımı olmaktan ziyade, maksimum olmayan bastırma öğrenilebilir mi?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:
