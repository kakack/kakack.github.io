---

layout: post
tags: [Detection, Deep Learning, Computer Vision]
title: Evolution by Yolo
date: 2020-12-16
author: Kyrie Chen
comments: true
toc: true
pinned: false

---

对于图像中目标检测，最朴素的需求就是输入一张目标图像，输出图像中待检测目标物的位置，用bounding box形式输出，和该物体的类别，用类别标签标示。在Yolo出现之前，业界最优秀的方法是基于region proposal的R-cnn系列方法，包括rcnn、fast-rcnn和faster-rcnn。概括而言这一类的算法可以归纳成两步走的two-stage，首先通过经验手段或者elective search、rpn等方法来生成网络觉得可能会出现目标物体位置的region proposal，然后再将rp中提取到的信息通过网络，最后用分类来获得目标物体的类别，用回归来确定目标物体的bounding box位置。这种做法相对于之前其他的方法而言，大大提高了物体定位的准确率，但是也存在一个很大的问题就是处理速度慢。人们举过一个很生动的例子，如果将rcnn系列检测器放在一辆以60km/h疾驰的汽车上做物体检测，当输入一帧画面得到结果的时候，用rcnn的车子已经开出300m远，用fast-rcnn的也已经开出34m以上。因此rcnn系列算法在一些强调响应速度的应用上，会显得非常滞后。

因此，yolo在设计之初的理念就是加快这个方法的运行速度。因此它摒弃了预先生成rp的方法，直接以整图作为模型的输入，提取整图的特征信息。然后在输出的时候不再对分类结果和定位结果分别处理，而是使用回归的方法一次性统一得到位置信息bounding box和分类信息class label。最后，yolo整体上设计的是一个端到端的单独网络，简化了训练和推理过程中的复杂度。整个流程可以概括为得到输入图像，resize成输入尺寸，经过卷积神经网络得到一系列输出，再使用一些阈值方法得到我们最终希望得到的检测结果。它不需要像原先的滑动窗口或者rpn算法一样反复在图片上遍历，而只需要看图片一次就够了，这也是yolo名字的来源，`you only look once`。

# How does Yolo v1 work

## Brief work method

首先我们想象在需要被检测的图片上，分布着一系列网格，比如yolo v1中将一张图划分为7*7的grids。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-1.jpg)

其中每一个网格都会负责预测一些bounding box和对应的置信度，这个置信度表示的是对bounding box内是否有物体的信心，框的个数是预先设定的比如v1默认的是2个，负责的原则是该物体的中心点落在这个这个网格内，比如右上角的某个网格，就会预测这辆车的bounding box，同时会得到对应bounding box的置信值。如果当前的cell里没有任何物体，我们也会得到相应的bounding box，只不过我们希望这些bounding box的置信度尽可能的低。当我们把所有bounding box都拿出来，就大致知道了在我们所输入的画面里，哪些位置可能会有目标物。但此时我们还有一个问题，我们知道哪儿有目标，但不知道目标具体是什么。因此我们下一步就需要每个网格再预测出内涵物体是什么类型的概率。于是我们就得到了这样一个类似粗糙的特征图的画面，可以大致推断哪边是狗、哪边是自行车或者汽车。需要知道的一点是，当前我们得到的概率是一个条件概率，也就是说我们假定这个网格里有东西，那这个东西是什么类型的概率。换言之，当这个网格预测我当前汽车的概率时，它并不代表我这里一定有一辆汽车，而是说假如我这个网格里有物体，那么它很有可能是汽车。于是当我们拿这个条件概率去乘以之前说到的存在物体的置信度，那么我们就能得到每一个bounding box实际包含物体类别的概率。当前我们已经得到了很多bounding box，其中有一些对于任何类别的目标而言概率都很低，于是就用一些阈值来直接过滤了，剩下的bounding box可以用nms的方法来去冗余，得到我们期望的结果。

对于每个网格来说，都会去预测以下一些内容，首先是若干个bounding box，每个bounding box中包括4个坐标值`x`、`y`、`h`、`w`和一个存在物体的置信值`c`，然后需要预测多少类目标物，就需要再加各个类别的可能性。比如对yolo v1在pascal voc数据集上，预设将每个整图切分成7*7的网格，每个网格对应2个bounding box，共计20类目标标签，那么我们的一个输出y就包括7*7*（2*5+20）=7*7*30个张量=1470个输出。所以yolo对于目标类型和其位置信息的预测输出是同时的。

## Training

Yolo既然是从整图推理出检测结果，那么在训练的时候喂入的也需要是一张整图。当我们得到了一些图和关于这些图的ground truth标注后，需要做的第一件事就是将得到的ground truth和图片中合适的网格进行绑定，也就是我们希望在推理预测时能输出这个ground truth对应结果的网格。对应的方法很简单，就是标签bounding box的中心位置所在的网格即是需要负责它的网格。然后调整这个网格对于类别的预测结果，比如这个网格里，我们把dog类设置成1，其余类设置成0。同样我们需要调整这个网格的bounding box proposal，我们看这个网格提供的预测bounding box中哪一个跟最终ground truth的bounding box重合率最高，于是就调整这个bounding box的置信度和位置，然后降低其余bounding box的置信度。我们在这张图中，还有很多网格并没有包含任何ground truth的中心点，我们同样需要将他们对应的bounding box置信度降到足够低。

## Network design

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-2.jpg)

Yolo v1采用卷积网络来提取特征，然后使用全连接层来得到预测值。网络结构参考`GooLeNet`模型，包含24个卷积层和2个全连接层，如图所示。对于卷积层，主要使用1x1卷积来做channle reduction，然后紧跟3x3卷积。对于卷积层和全连接层，采用Leaky ReLU激活函数：max（x， 0.1x）。但是最后一层却采用线性激活函数。可以看到网络的最后输出为 7*7*30大小的张量。这和前面的讨论是一致的。这个张量所代表的具体含义我画在右下角的图中。对于每一个单元格，其中20个元素是类别概率值，然后2个元素是bounding box的置信度，两者相乘可以得到各自类别的置信度，最后8个元素是bounding box的x、y、w和h。可能有人会感到奇怪，对于bounding box为什么把置信度和位置信息都分开排列，而不是按照（x,y,w,h,c）这样排列，其实纯粹是为了计算方便，因为实际上这30个元素都是对应一个单元格，其排列是可以任意的。但是分离排布，可以方便地提取每一个部分。在训练之前，yolo v1先在ImageNet上进行了预训练，其预训练的分类模型采用图中前20个卷积层，然后添加一个average-pool层和全连接层。预训练之后，在预训练得到的20层卷积层之上加上随机初始化的4个卷积层和2个全连接层。由于检测任务一般需要更高清的图片，所以将网络的输入从224x224增加到了448x448。


## Loss function

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-3.jpg)

Yolo算法将目标检测看成回归问题，所以采用的是均方差损失函数。但是对不同的部分采用了不同的权重值。首先区分`定位误差`(前两行)和`分类误差`（后三行）。对于定位误差，即bounding box坐标预测误差，采用较大的权重5。然后其区分不包含目标的bounding box与含有目标的bounding box的置信度，对于前者，采用较小的权重值0.5。其它权重值均设为1。然后采用均方误差，其同等对待大小不同的bounding box，但是实际上较小的bounding box的坐标误差应该要比较大的bounding box要更敏感。为了保证这一点，将网络的bounding box的宽与高预测改为对其平方根的预测，即预测值变为根号 。

另外一点时，由于每个单元格预测多个bounding box。但是其对应类别只有一个。那么在训练时，如果该单元格内确实存在目标，那么只选择与ground truth的IOU最大的那个bounding box来负责预测该目标，而其它bounding box认为不存在目标。这样设置的一个结果将会使一个单元格对应的bounding box更加专业化，其可以分别适用不同大小，不同高宽比的目标，从而提升模型性能。但是如果一个单元格内存在多个目标怎么办，其实这时候Yolo算法就只能选择其中一个来训练，这也是Yolo算法的缺点之一。要注意的一点时，对于不存在对应目标的bounding box，其误差项就是只有置信度，坐标项误差是没法计算的。而只有当一个单元格内确实存在目标时，才计算分类误差项，否则该项也是无法计算的。其中第一项是bounding box中心坐标的误差项， 这个1_obj_ij指的是第i个单元格存在目标，且该单元格中的第j个bounding box负责预测该目标。第二项是bounding box的高与宽的误差项。第三项是包含目标的bounding box的置信度误差项。第四项是不包含目标的bounding box也就是背景的置信度误差项。而最后一项是包含目标的单元格的分类误差项， 1_obj_i指的是第i个单元格存在目标。这里特别说一下置信度的target值Ci ，如果是不存在目标，那么Ci=0 。如果存在目标，此时需要确定IOU的值，当然希望最好的话，可以将IOU取1，这样Ci=1 ，但是在yolo v1实现中，使用了一个控制参数rescore，默认计算truth和pred之间的真实IOU。不过很多复现yolo v1的项目还是取Ci=1，这个差异应该不会太影响结果吧。

- - -
# How does Yolo v1 perform

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-4.jpg)

Yolo v1算法在PASCAL VOC 2007数据集上的性能，与其它检测算法做了对比，包括DPM、R-CNN、Fast R-CNN以及Faster R-CNN。其对比结果如左边的表格所示。与实时性检测方法DPM对比，可以看到Yolo v1算法可以在较高的mAP上达到较快的检测速度，其中Fast Yolo算法比快速DPM还快，而且mAP是远高于DPM。但是相比Faster R-CNN，Yolo的mAP稍低，但是速度更快。所以，Yolo v1算法算是在速度与准确度上做了折中。为了进一步分析Yolo v1算法，文章还做了误差分析，将预测结果按照分类与定位准确性分成以下5类：

- Correct：类别正确，IOU>0.5；（准确度）；
- Localization：类别正确，0.1 < IOU<0.5（定位不准）；
- Similar：类别相似，IOU>0.1；
- Other：类别错误，IOU>0.1；
- Background：对任何目标其IOU<0.1。（误把背景当物体）

Yolo v1与Fast R-CNN的误差对比分析可以看到，Yolo v1的Correct的是低于Fast R-CNN。另外Yolo v1的Localization误差偏高，即定位不是很准确。但是Yolo v1的Background误差很低，说明其对背景的误判率较低。

总之，Yolo v1的优缺点都比较明显：

优点：

- Yolo v1采用一个CNN网络来实现检测，是一个单管道策略，其训练和预测都是end-to-end，所以整体表现简洁而快速；
- 由于Yolo v1是对整张图片做卷积，所以在其检测目标上有更大的视野，不容易对背景产生误判；
- Yolo v1泛化能力较强，在做迁移时，模型鲁棒性较高。

缺点：

- Yolo v1各个grid上仅有预测两个（或固定个）边界框，对于较密集目标和小目标的表现欠佳；
- Yolo v1对于目标长宽比泛化率低，在长宽比比较极端或特殊的物体上表现不佳；
- 在目标定位方面有待提高，召回率低。

- - -

# What does yolo v2 improve

Yolo v2论文标题就是更好，更快，更强。在Yolo v1发表之后，计算机视觉领域出现了很多trick，例如批归一化、多尺度训练，v2也尝试借鉴了R-CNN体系中的anchor box，所有的改进提升。Yolo v2整体上改进的目标集中于提高定位和召回率，改进的原则是尽可能保持原有检测的速度。文章中，作者首先在yolo v1的基础上提出了改进的yolo v2，然后提出了一种检测与分类联合训练方法，使用这种联合训练方法在COCO检测数据集和ImageNet分类数据集上训练出了yolo 9000模型，其可以检测超过9000多类物体。所以，这篇文章其实包含两个模型：yolo v2和yolo 9000，不过后者是在前者基础上提出的，两者模型主体结构是一致的。

Yolo v2相比yolov1做了很多方面的改进，这也使得yolo v2的mAP有显著的提升，并且yolo v2的速度依然很快，保持着自己作为one-stage方法的优势。在yolo v1提出之后，作者也不断在反思还存在的一些问题，yolo v1虽然检测速度很快，但是在检测精度上却不如R-CNN系检测方法，yolo v1在物体定位方面（localization）不够准确，并且召回率（recall）较低。yolo v2共提出了几种改进策略来提升yolo 模型的定位准确度和召回率，从而提高mAP。下表里提到了一些yolo v2上用到的方法，在voc2007上的mAp都略有提升，只有用了anchor box出现了一点回落，但是作者通过聚类取先验框尺寸的方法弥补了这一个回落。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-5.jpg)

作者当时在yolo v2上使用的优化方法很多都是一些当时已经在其他算法中被验证使用的技巧，其中很多至今都是cv领域里的经典方法。


## Batch Normalization
Batch Normalization可以提升模型收敛速度，解决训练过程中一些梯度消失梯度爆炸的现象，降低对其中一些超参的敏感度，而且可以起到一定正则化效果，降低模型的过拟合。在yolo v2中，每个卷积层后面都添加了Batch Normalization层，并且不再使用droput。

## High Resolution Classifier（分类网络高分辨率预训练）

在Yolov1中，网络的backbone部分会在ImageNet数据集上进行预训练，训练时网络输入图像的分辨率为224*224。在v2中，将分类网络在输入图片分辨率为448*448的ImageNet数据集上训练10个epoch，再使用检测数据集（例如coco）进行微调。

## 采用anchor box的设定

在yolo v1中，输入图片最终被划分为7*7网格，每个单元格预测2个边界框。yolo v1最后采用的是全连接层直接对边界框进行预测，其中边界框的宽与高是相对整张图片大小的，而由于各个图片中存在不同尺度和长宽比的物体，yolo v1在训练过程中学习适应不同物体的形状是比较困难的，这也导致yolo v1在精确定位方面表现较差。yolo v2借鉴了Faster R-CNN中anchor boxes的策略。移除了yolo v1中的全连接层而采用了卷积和anchor boxes来预测边界框。然而使用anchor boxes之后，yolo v2的mAP有稍微下降，这里下降的原因，有研究者怀疑是yolo v2虽然使用了anchor boxes，但是依然采用yolo v1的训练方法所导致。

## 聚类提取先验框的尺寸

在当时Faster R-CNN和SSD中，先验框的维度（长和宽）都是手动设定的，带有一定的主观性。Faster R-CNN中一共设定三个面积大小的矩形框，每个矩形框有三个宽高比：1:1，2:1，1:2，总共九个框。如果选取的先验框维度比较合适，那么模型更容易学习，从而做出更好的预测。而yolo v2首次采用k-means聚类方法对训练集中的边界框做了聚类分析，这种获得先验框的方法到现在都一直被沿用。

## 多尺度训练

很关键的一点是，Yolo v2中只有卷积层与池化层，所以对于网络的输入大小，并没有限制，整个网络的降采样倍数为32，只要输入的特征图尺寸为32的倍数即可，如果网络中有全连接层，就不是这样了。所以Yolo v2可以使用不同尺寸的输入图片训练。作者使用的训练方法是，在每10个batch之后，就将图片resize成{320, 352, ..., 608}中的一种。不同的输入，最后产生的格点数不同，比如输入图片是320*320，那么输出格点是10*10，如果每个格点的先验框个数设置为5，那么总共输出500个预测结果；如果输入图片大小是608*608，输出格点就是19*19，共1805个预测结果。在引入了多尺寸训练方法后，迫使卷积核学习不同比例大小尺寸的特征。当输入设置为544*544甚至更大，Yolo v2的mAP已经超过了其他的物体检测算法。

## 高分辨率图像的对象检测

加入了一个hi-resdetector，这部分作者没有具体展开。

## 引入passthrough层检测细颗粒度特征

保留小图像细节信息，用channel换尺寸。对象检测面临的一个问题是图像中对象会有大有小，输入图像经过多层网络提取特征，最后输出的特征图中（比如yolo 2中输入416*416经过卷积网络下采样最后输出是13*13），较小的对象可能特征已经不明显甚至被忽略掉了。为了更好的检测出一些比较小的对象，最后输出的特征图需要保留一些更细节的信息。Passthrough所起到的作用其实是将一个feature map一拆四后直接传递到pooling层之后，拆的方法大致如图所示，将一个4*4的相对位置各自取出来，合并成4个2*2.

## Yolo v2约束预测bounding box的位置

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-6.jpg)

yolo v2约束了bounding box的位置，改善了之前bounding box预测不准的问题。之前借鉴于faster rcnn的bounding box恢复方法，其预测公式是x=（tx*wa）+xa，y=（ty*ha）+ ya，其中x和y是我们希望最后预测的bounding box中心点的坐标结果，wa、ha、xa和ya分辨是已有的先验框的宽高和中心坐标，tx和ty是我们学习出来的关于先验框中心点的偏移量。但是这个算法有个问题就是tx和ty是没有任何约束的，在一些极端条件下，比如学出来的tx或ty足够大或足够小的情况下，最终得到的bounding box中心可能出现在整个图像的任何位置，所以在训练早期会非常不稳定，因此yolo v2调整了计算方法，将最终得到的bounding box的中心限制在特定的网格内。其中bx、by、bw、bh是最终我们希望预测出来的bounding box的坐标和宽高，Pr（object）*IOU（b，object）是bounding box的置信度，通过对预测参数t0进行sigma变换后得到。Cx,cy是当前网格左上角到图像左上角的距离，要先将网格大小归一化，即令一个网格的宽=1，高=1，pw、ph是先验框的宽高。Tx、ty、tw、th和t0是要学习出来的参数，分别用于预测边框的中心和宽高以及置信度。然后sigma函数将sigma(tx/ty)的值限制在（0,1）范围内，所以预测bounding box的中心点就会被限制在这个网格内。这个变化让模型更容易学习，而且预测更稳定。

## darknet-19

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-7.jpg)

为了进一步提升速度，YOLO2提出了Darknet-19，含有19个卷积层和5个MaxPooling层的网络结构。Darknet-19与VGG16模型设计原则是一致的，主要采用 3x3 卷积，采用 2x2 的maxpooling层之后，特征图维度降低2倍，而同时将特征图的channles增加两倍。DarkNet-19比VGG-16小一些，精度不弱于VGG-16，但浮点运算量减少到约1/5，以保证更快的运算速度。总之，虽然YOLO2做出了一些改进，但总的来说网络结构依然很简单，就是一些卷积+pooling，从416*416*3 变换到 13*13*5*25。稍微大一点的变化是增加了batch normalization，增加了一个passthrough层，去掉了全连接层，以及采用了5个先验框。对比YOLO1的输出张量，YOLO2的主要变化就是会输出5个先验框，且每个先验框都会尝试预测一个对象。输出的 13*13*5*25 张量中，25维向量包含 20个对象的分类概率+4个边框坐标+1个边框置信度。

## Loss function

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-8.jpg)

其实在yolo v2和yolo v3的paper中，作者其实不再明确给出loss function的明确表达式，但根据开源的代码里反推回来，大致还原了loss function的形式，整个表达式依然包括边框位置误差、置信度误差、对象分类误差。

`1maxIOU<Thresh`意思是预测边框中，与真实对象边框IOU最大的那个，其IOU<阈值Thresh，此系数为1，即计入误差，否则为0，不计入误差。Yolo v2使用Thresh=0.6。

`1t<128000`意思是前128000次迭代计入误差。注意这里是与先验框的误差，而不是与真实对象边框的误差。可能是为了在训练早期使模型更快学会先预测先验框的位置。

`1truth_k`意思是该边框负责预测一个真实对象（边框内有对象）。

各种λ是不同类型误差的调节系数。

---

# What is Yolo 9000

在yolo v2的paper里，作者还额外发了一个yolo 9000的网络。因为yolo v2在VOC数据集可以检测20种对象，但实际上对象的种类非常多，只是缺少相应的用于对象检测的训练样本。YOLO2尝试利用ImageNet非常大量的分类样本，联合COCO的对象检测数据集一起训练，使得YOLO2即使没有学过很多对象的检测样本，也能检测出这些对象。扩展的基本的思路是，如果是检测样本，训练时其Loss包括分类误差和定位误差，如果是分类样本，则Loss只包括分类误差。要检测更多对象，比如从原来的VOC的20种对象，扩展到ImageNet的9000种对象。简单来想的话，好像把原来输出20维的softmax改成9000维的softmax就可以了，但是，ImageNet的对象类别与COCO的对象类别不是互斥的。比如COCO对象类别有“狗”，而ImageNet细分成100多个品种的狗，狗与100多个狗的品种是包含关系，而不是互斥关系。一个Norfolk terrier同时也是dog，这样就不适合用单个softmax来做对象分类，而是要采用一种多标签分类模型。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-9.jpg)

然后作者将ImageNet和COCO中的名词对象一起构建了一个WordTree，以physical object为根节点，各名词依据相互间的关系构建树枝、树叶，节点间的连接表达了对象概念之间的蕴含关系（上位/下位关系），整个WordTree中的对象之间不是互斥的关系，但对于单个节点，属于它的所有子节点之间是互斥关系。比如airplane节点之下的airbus、jet等，各品种的airplane之间是互斥的，所以计算上可以进行softmax操作。之前对象互斥的情况下，用一个n维向量（n是预测对象的类别数）就可以表达一个对象（预测对象的那一维数值接近1，其它维数值接近0）。如果在WordTree还保留这种表示方式，那默认所有节点依然是互斥的关系。所以在wordTree中比如一个样本图像，其标签是是"dog"，那么显然dog节点的概率应该是1，然后，dog属于mammal，自然mammal的概率也是1，......一直沿路径向上到根节点physical object，所有经过的节点其概率都是1，其他没有经过的节点都是0。所以每个节点表达的概率实际就是基于父节点的条件概率，只要把途径的所有节点的概率相乘就得到了实际的概率。不过，为了计算简便，实际中并不计算出所有节点的绝对概率。而是采用一种比较贪婪的算法。从根节点开始向下遍历，对每一个节点，在它的所有子节点中，选择概率最大的那个（一个节点下面的所有子节点是互斥的），一直向下遍历直到某个节点的子节点概率低于设定的阈值（意味着很难确定它的下一层对象到底是哪个），或达到叶子节点，那么该节点就是该WordTree对应的对象。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-10.jpg)

如图中所示，同一颜色的位置，进行softmax操作，使得同一颜色中只有一个类别预测分值最大。在预测时，从树的根节点开始向下检索，每次选取预测分值最高的子节点，直到所有选择的节点预测分值连乘后小于某一阈值时停止。在训练时，如果标签为人，那么只对人这个节点以及其所有的父节点进行loss计算，而其子节点，男人、女人、小孩等，不进行loss计算。

- - -

# How about Yolo v3

其实按作者自己的话说，yolo v3其实是他把之前没应用到v2的一些办法重新整理归纳了一下，选了其中一些效果好的应用在yolov3上，主旨还是在保持速度的基础上进一步提高在小目标上的表现。在这里我们就说其中作者着重介绍的几个办法。

首先对象分类softmax被独立的多个logistic分类器替代，这么做的目的是为了实现多标签多分类。一张图像或一个object只属于一个类别，但是在一些复杂场景下，一个object可能属于多个类，比如你的类别中有woman和person这两个类，所以需要用逻辑回归层来对每个类别做二分类。

然后是增加下采样，进一步利用多尺度特征。首先YOLO2已经开始采用K-means聚类得到先验框的尺寸，YOLO3延续了这种方法，为每种下采样尺度设定3种先验框，总共聚类出9种尺寸的先验框。在COCO数据集这9个先验框是：(10x13)，(16x30)，(33x23)，(30x61)，(62x45)，(59x119)，(116x90)，(156x198)，(373x326)。根据先验框的尺寸可以分成大中小三类。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-11.jpg)

其中中蓝色框为聚类得到的先验框。黄色框式ground truth，红框是对象中心点所在的网格。从整体的输入输出来看，不考虑神经网络结构细节的话，总的来说，对于一个输入图像，YOLO3将其映射到3个尺度的输出张量，代表图像各个位置存在各种对象的概率。

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-12.jpg)

对于一个416*416的输入图像，在每个尺度的特征图的每个网格设置3个先验框，总共有 13*13*3 + 26*26*3 + 52*52*3 = 10647 个预测。每一个预测是一个(4+1+80)=85维向量，这个85维向量包含边框坐标（4个数值），边框置信度（1个数值），对象类别的概率（对于COCO数据集，有80种对象）。对比一下，YOLO2采用13*13*5 = 845个预测，YOLO3的尝试预测边框数量增加了10多倍，而且是在不同分辨率上进行，所以mAP以及对小物体的检测效果有一定的提升。



![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-13.jpg)


## Darknet-53 and Res_N

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-14.jpg)

Yolo V3将v2中的darknet19 backbone替换成了darknet53，一方面将YOLO v2中采用pooling层做feature map的sample，这里都换成卷积层来做了。另一方面引入了residual结构，层数太多训起来会有梯度问题，所以Darknet-19也就19层，因此得益于ResNet的residual跳跃传递的结构，把一些浅层的特征信息直接传入较为深层的激活函数中，训深层网络难度大大减小，因此这里可以将网络做到53层，精度提升比较明显。具体看整个网络结构，darknet53仅仅只是红色虚线框里面的模块。我们看出，darknet-53就是几个残差单元的组合，算是借鉴了Resnet，这个特征提取网络有三个返回值，可以从上图看到三个分支。这三个输出特征图的宽高分别为之前提到的52、26、13，就是原图除以8、16和32。

然后解释一下图中几个小模块的具体含义，

- `DBL`: 如上图左下角所示，也就是代码中的Darknetconv2d_BN_Leaky，是yolo_v3的基本组件。就是卷积+BN+Leaky relu。对于v3来说，BN和leaky relu已经是和卷积层不可分离的部分了(最后一层卷积除外)，共同构成了最小组件。
- `resn`：n代表数字，有res1，res2, … ,res8等等，表示这个res_block里含有多少个res_unit。这是yolo_v3的大组件，yolo_v3开始借鉴了ResNet的残差结构，使用这种结构可以让网络结构更深(从v2的darknet-19上升到v3的darknet-53，前者没有残差结构)。对于res_block的解释，可以在图的右下角直观看到，其基本组件也是DBL。
- `concat`：张量拼接。将darknet中间层和后面的某一层的上采样进行拼接。拼接的操作和残差层add的操作是不一样的，拼接会扩充张量的维度，而add只是直接相加不会导致张量维度的改变。

## Loss function

![](https://raw.githubusercontent.com/kakack/kakack.github.io/master/_images/20201216-15.jpg)

- lbox：位置误差geo_loss，S代表13、26、52，就是grid是几乘几的尺寸，B=5；
- lcls：类别误差class_loss，和yolo v2的区别是改成了交叉熵；
- lobj：置信度误差confidence_loss，和yolo v2几乎一模一样。

- - -

# Conclusion

|  比较   | Yolo v1  | Yolo v2   | Yolo v3  |
|  :----:  | :----  |:----  | :----  |
| 创新点  | 使用单个神经网络，将目标检测问题看做回归问题 |1，引入anchor\k-measn改善定位；<br>2，改进网络，提出darknet-19；<br>3，提出word-tree，扩展到9000种检测类别。  | 1，分类器由softmax改为独立的逻辑回归分类器；<br>2，损失函数改为二分类交叉熵；3，更新为darknet-53，引入resnet思想；<br>4，多尺度输出 |
| 优点  | 1，训练/执行端到端，方便优化；<br>2，快于rcnn系列等；<br>3，泛化能力强 |1，改善定位；<br>2，联合训练提升精度；<br>3，网络结构精简，去掉全连接层，速度更快；<br>4，改善小物体上精度。  | 1，集大成者；<br>2.自由切换backbone，自由取舍速度与精度。 |
| 缺陷  | 1，精度不及当时最新模型；<br>2，定位不准；<br>3，小物体效果欠佳 |1，大目标精度略微下降；<br>2，检测效果依赖于训练数据集。  | 时代限制，不如后续的yolo v4 |