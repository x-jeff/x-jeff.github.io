---
layout:     post
title:      【源码解析】VesselCompletion in Tag-Net
subtitle:   TaG-Net，VesselCompletion
date:       2024-07-09
author:     x-jeff
header-img: blogimg/20220305.jpg
catalog: true
tags:
    - Source Code
---  
>本文为原创文章，未经本人允许，禁止转载。转载请注明出处。

# 1.Vessel Completion in Tag-Net

本博文为Tag-Net在血管补全部分的代码详解，相关链接：

1. [Tag-Net论文解读](https://shichaoxin.com/2024/06/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-TaG-Net-Topology-Aware-Graph-Network-for-Centerline-Based-Vessel-Labeling/)
2. [Tag-Net github](https://github.com/PRESENT-Y/TaG-Net/tree/main)

涉及血管补全的主要有4个文件（在文件夹`VesselCompletion`下）：

1. `gen_noise_removal.py`：去除噪声。
2. `gen_connection_pairs.py`：产生连接对。
3. `gen_connection_path.py`：产生连接路径。
4. `gen_adhesion_removal.py`：去除粘连。

血管补全部分的输入为TaG-Net生成的带有标注的中心线血管图，输出为refine后的中心线血管图。

我们用以下两个文件来保存带标注的中心线血管图：

1. `CenterlineGraph`
2. `labeled_cl.txt`

我们全篇详解以官方repo中的`SampleData/002`测试数据为例（除第5部分）。在[论文解读博文](https://shichaoxin.com/2024/06/22/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB-TaG-Net-Topology-Aware-Graph-Network-for-Centerline-Based-Vessel-Labeling/)中，我们提到[邻接矩阵](https://shichaoxin.com/2023/05/15/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E4%BA%94%E7%AB%A0-%E5%9B%BE%E7%9A%84%E9%81%8D%E5%8E%86/)可以被用来表示图。002测试数据的中心线血管图一共有9005个采样点，那么对应的[邻接矩阵](https://shichaoxin.com/2023/05/15/%E5%95%8A%E5%93%88-%E7%AE%97%E6%B3%95-%E7%AC%AC%E4%BA%94%E7%AB%A0-%E5%9B%BE%E7%9A%84%E9%81%8D%E5%8E%86/)维度是$9005 \times 9005$，矩阵中的元素表示边的连通性。`CenterlineGraph`中保存的就是这个邻接矩阵（但只保存了连通的边），形式为：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/3.png)

$(0,1)$表示第0个点和第1个点是有边连通的。

`labeled_cl.txt`中保存的是标注情况，为一个$9005 \times 4$的矩阵，每一行代表一个点，每个点对应一个4维向量，前3个数是点的3D坐标，最后一个数是这个点的标签：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/4.png)

18根血管的标签索引为：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/2.png)

# 2.`gen_noise_removal.py`

先看下效果：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/1.png)

左侧是去噪之前，即TaG-Net的输出；右侧是去噪之后。图中用红圈标出了部分被去掉的噪声。

除BCT外，将剩余的17根血管分为头部血管和颈部血管：

```python
head_list = [0,5,6,11,17]  # head label 
neck_list = [13, 14, 15, 16, 7, 12, 4, 10, 3, 9, 8, 2] # neck label
```

我们会把标注的中心线血管图存成networkX的格式：

```python
#pc是点
#edges是边
G_nx = butils.gen_G_nx(len(pc),edges)
```

接下来逐个处理每根血管，对不同血管会预设不同的连通域阈值。对于头部血管，阈值为15；对于颈部血管，阈值为30；对于BCT，阈值为50：

```python
if label in head_list:
    thresh = 15
if label in neck_list:
    thresh = 30
if label == 1:
    thresh = 50  
```

我们以R-PCA（label=0）为例。首先从`G_nx`中把label为0的部分抽取出来，即把R-PCA提取出来单独进行分析，注意，提取出来的部分所有点的label必须都是0，不能一条边的一个端点是0，另一个端点不是0：

```python
connected_components, G_nx_label = mcutils.gen_connected_components(idx_label, G_nx)
```

其中，`G_nx_label`里保存的就是提取出来的标签为0的图（点的序号被重置为从0开始）。`connected_components`是该图的连通域：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/5.png)

能看到，`G_nx_label`包含了4个连通域。接下来处理每个连通域。对每个连通域来说，我们都会获取其相连的邻居节点：

```python
idx_neigbors = butils.gen_neighbors_exclude(ori_idx , G_nx)
seg_label = butils.gen_neighbors_label(idx_neigbors, pc_label)
```

`idx_neigbors`是与某个连通域相连的邻居节点，注意这里的邻居节点必须是和连通域标签不同的节点。这些邻居节点的标签会保存在`seg_label`中。

对每个连通域的操作如下：

1. 如果一个连通域的邻居节点数不为0，说明其和其他血管有联系，则保留该连通域。
2. 如果一个连通域的邻居节点数为0，则，
    * 如果该连通域包含的节点数比`G_nx_label`节点数的10%还少，并且该连通域的节点数还低于之前预设的连通域阈值，则删掉该连通域，即视为噪声。
    * 否则，保留该连通域。

对应代码如下：

```python
#num_connected_i为第i个连通域的节点数
#num_idx_label为G_nx_label包含的节点数
if (num_connected_i/num_idx_label <= 1/10) and (num_connected_i<thresh):
    node_to_remove.append(ori_idx)
```

对每根血管都会进行上述连通域分析，最终便可得到去噪之后的结果，我们将其保存在`CenterlineGraph_new`和`labeled_cl_new.txt`中。002测试数据去噪完还剩8821个节点。

# 3.`gen_connection_pairs.py`

基于去噪后的结果，我们就可以开始搜索连接对了。头部血管和颈部血管的分组和第2部分一样。

>血管补全和粘连去除都只针对颈部血管。

## 3.1.`connection_pair_intra`

首先搜索同一标签的连接对。

将去噪后的中心线血管图读入`G_nx`。我们并不需要对所有的标签都搜索连接对，因为有的血管已经完整连接了，不存在中断，这种情况下，该血管就不再需要搜索标签内的连接对了。

```python
#存在中断的血管标签
components_more_than_one_label_list = cutils.gen_label_to_check(label_list,pc, G_nx)
#只考虑颈部血管
neck_label_to_connect = [idx for idx in components_more_than_one_label_list if idx in neck_list]
```

`components_more_than_one_label_list`里保存的就是需要搜索连接对的标签，即存在中断的血管。需要注意的是，我们只针对颈部血管进行标签内的补全。搜索标签内连接对的函数为：

```python
connection_intra_pairs =  cutils.gen_connection_intra_pairs(neck_label_to_connect, G_nx, pc)
```

接下来介绍这个函数里面的操作。首先计算每个节点的度：

```python
#degree_list里保存了8821个节点的度
degree_list = gutils.gen_degree_list(G_nx.edges(), len(pc))[0]
```

然后计算节点两两之间的距离：

```python
#计算欧氏距离
sqr_dis_matrix = butils.cpt_sqr_dis_mat(pc[:,0:3])
#计算几何距离
geo_distance_matrix = butils.cpt_geo_dis_mat(pc[:,0:3])
#只使用欧氏距离
geo_distance_matrix = sqr_dis_matrix
```

从上述代码可以看到，作者计算了欧氏距离和几何距离，但是作者并没有使用几何距离。`sqr_dis_matrix`和`geo_distance_matrix`都是一个$8821 \times 8821$的矩阵，里面保存着所有节点对的距离。欧氏距离的计算比较简单，这里说下几何距离的计算：

```python
def cpt_geo_dis_mat(data):
    """
    geometric distance 
    """
    #创建数据点的KD树，用于快速近邻搜索
    #但作者没有使用ckt
    ckt = spt.cKDTree(data)
    #初始化Isomap模型
    #Isomap是一种用于降维的非线性算法，可以捕捉高维数据的低维流形结构
    isomap = Isomap(n_components=2, n_neighbors=2, path_method='auto')
    #使用Isomap对数据进行降维，从原始空间（3D空间）降到2D空间
    data_3d = isomap.fit_transform(data)
    #获取几何距离矩阵
    geo_distance_matrix = isomap.dist_matrix_
    return geo_distance_matrix
```

然后我们依次处理`neck_label_to_connect`中的每个标签，即逐个处理存在中断的每根颈部血管。

```python
connected_components, G_nx_label = mcutils.gen_connected_components(idx_label, G_nx)
```

和之前一样，先把要处理的这个颈部血管提取出来，存在`G_nx_label`中，并作连通域分析（保存在`connected_components`中）。

然后作者只对连通域超过1的血管进行分析，因为超过1意味着这根血管有中断，需要搜索连接对：

```python
connected_num = len(connected_components)
if connected_num > 1:
    #后续操作
```

其实这个判断在这里是多余的，因为之前已经判断过了，走到这一步的血管已经都是有中断的了。然后计算图`G_nx_label`中度为1的端点对：

```python
def gen_degree_one(idx_label, G_nx_label, degree_list):
    #idx_label是某根存在中断的颈部血管的节点集合
    #G_nx_label是这根血管的图
    #degree_list是`G_nx`中所有节点的度
    G_G_label_map = {j:i for i, j in enumerate(idx_label)}
    G_label_G_map = {i:j for i, j in enumerate(idx_label)}
    idx_label_mapped_to_G_label = [G_G_label_map.get(i) for i in idx_label]
    degree_list_G_label = gutils.gen_degree_list(G_nx_label.edges(), len(idx_label))
    idx_degree_one = [i for i, degree in enumerate(degree_list_G_label[0]) if degree == 1]
    idx_in_G_label = [G_label_G_map.get(i) for i in idx_degree_one]
    degree_G_one = [idx for idx in  idx_in_G_label if degree_list[idx] == 1]
    degree_G_one = sorted(degree_G_one)
    return degree_G_one, G_label_G_map
```

上述代码的简要流程可描述为，首先计算`G_nx_label`所包含节点（即`idx_label`包含的节点）的度（保存到`idx_in_G_label`），然后从中选择出度为1的节点，如果该节点在`degree_list`（即在`G_nx`）中度也为1，则保留该节点，否则放弃该节点，最终得到这样的节点集合`degree_G_one`。

>个人注解：感觉可以直接在`degree_list`中搜索度为1且标签为目标血管的节点，没必要先在`G_nx_label`上搜索。

在002测试数据中，只有L-ECA（label=15）存在标签内的中断，通过上述步骤搜索到的标签内且度为1的节点一共有3个，节点索引分别为2075、2156、2596。这3个节点可以组合成3个连接对：$(2075, 2156), (2075, 2596), (2156, 2596)$。但节点2075属于连通域1，节点2156和2596属于连通域2，所以$(2156, 2596)$这个连接对是没有意义的，因此删去连接对$(2156, 2596)$。

接下来的处理根据连接对的数量有3种情况：

（1）当只有一个连接对时，直接保存该连接对，不需要处理。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/6.png)

如上图所示，红色（label=b）为我们要处理的血管，绿色（label=a）和黑色（label=c）为与红色相连的其他标签的血管。两个蓝色节点为我们找到的标签内连接对（分属不同的连通域且度为1），这种情况下，我们直接连接这两个蓝色节点即可。

（2）当有两个连接对时，保留距离更小的连接对。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/7.png)

如上图所示，p3为红色血管的终点，因此有3个度为1的节点，可以生成$(p1, p2), (p1 ,p3)$两个连接对（$(p2,p3)$属于同一个连通域，不考虑构成连接对），这种情况下，我们会选择距离更短的连接对，即保留$(p1, p2)$。

（3）当连接对的数量多于2个时，

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/8.png)

如上图所示，一共可以构建出$(p1, p2),(p1,p3),(p1,p4),(p1,p5),(p1, p6),(p2,p4),(p2,p5),(p2,p6),(p3,p4),(p3,p5),(p3,p6),(p4,p6),(p5,p6)$，共13个连接对。我们计算这些连接对的距离，发现$(p1, p2)$的距离最短，我们就把这个连接对保留下来，用于后续连接。然后我们从13个连接对中删去包含$p1$或者$p2$的连接对。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/9.png)

剩下$(p3,p4),(p3,p5),(p3,p6),(p4,p6),(p5,p6)$，共5个连接对。然后继续同样的操作，寻找距离最小的连接对，并删除距离最小连接对的节点，重复此操作直至所有连接对都被处理完毕。对于图中的例子，我们最终保留下来的连接对为$(p1,p2),(p3,p4),(p5,p6)$。

对于002测试数据来说，我们最终保留了距离更小的$(2075,2156)$连接对。

## 3.2.`connection_pair_inter`

标签间连接对的搜索会在第3.1部分的基础上进行，即在`G_nx`中已经添加了标签内连接对的边。

首先是对缺失标签对或错误标签对的获取：

```python
#input：
    #label_list：标签列表，从0到17，一共18种标签，对应18根血管
    #pc_label：每个3D点的标签
    #G_nx：整个中心线血管图
#output：
    #flag_wrong：是否存在错误标签对，1表示存在，0表示不存在
    #wrong_pairs：错误连接的标签对
    #flag_lack：是否有缺失标签对，1表示有，0表示没有
    #lack_pairs：缺失连接的标签对
    #label_pairs：目前G_nx中存在的标签对
flag_wrong, wrong_pairs, flag_lack, lack_pairs, label_pairs = mcutils.gen_wrong_connected_exist_label(label_list, pc_label, G_nx)
```

错误标签对的获取步骤如下：

1. 选定某根血管进行操作，比如label=0的血管。
2. 从`G_nx`中挑选出只有一个节点标签为0的边，该边另一个标签不为0的节点（比如label=8）视为该血管的邻居节点。那么对于该血管，我们就找到了一个标签对$(0,8)$。
3. 对所有血管都按照前两步寻找存在的标签对，不考虑包含头部血管的标签对。
4. 构造先验的解剖拓扑结构`gt_label_graph`，里面保存着标签对的GT。
5. 遍历第3步找到的标签对，如果在`gt_label_graph`中不存在，则被视为错误连接的标签对。

`gt_label_graph`的构造：

```python
def gen_anatomical_graph(label_list):
    # prior
    # BCT(1),R-CCA(3),R-ICA(4),R-VA(7),BA(8),L-CCA(9), L-ICA(10),L-VA(12),R-SCA(13),L-SCA(14),L-ECA(15), R-ECA(16)

    # easy to hard 
    # 1, 8, 13, 14, 15, 16, 7, 12, 4, 10, 3, 9, 2
    # 5, 0, 17, 6, 11
    # L-VA(12) -> L-SCA(14) & BA(8)   
    # R-VA(7) -> R-SCA(13) & BA(8)

    # L-CCA(9) --> L-ICA(10) & L-ECA(15)
    # R-CCA(3) --> R-ICA(4) & R-ECA(16)

    # L-ECA(15) -->  L-CCA(9) 
    # R-ECA(16) -->  R-CCA(3)

    # L-ICA(10) --> L-PCA(17) & L-MCA(11) & ACA(5)
    # R-ICA(4) --> R-PCA(0) & R-MCA(6) & ACA(5)

    # L-PCA(17) --> BA(8) & L-ICA(10)
    # R-PCA(0) --> BA(8) & R-ICA(4)

    # ACA(5) --> L-ICA(10) & R-ICA(4)
    
    # R-MCA(6) --> R-ICA(4) 
    # L-MCA(11) --> L-ICA(10)

    # BCT(1) --> AO(2) & L-SCA(14) & L-CCA(9)
    # AO(2) --> R-SCA(13) & R-CCA(3) & BCT(1) & L-VA(12) (special)
    label_list_edges = [(0,4),(0,8),(1,2), (1,9), (1,14), (1,12),\
                        (2,3), (2,13), (3,4), (3,16), (4,5),(4,6),\
                        (5,10), (7,8), (7,13), (8,12), (8,17),\
                        (9,10),(9,15), (10,11), (10,17), (12,14)]
    
    gt_label_graph =  butils.gen_G_nx(len(label_list),label_list_edges)
    return gt_label_graph
```

缺失标签对的获取步骤如下：

1. 去除`gt_label_graph`中的$(1,12)$标签对，在此不考虑。
2. 执行“错误标签对的获取步骤”的前3步。
3. 遍历`gt_label_graph`中的标签对，如果在第2步找到的标签对中不存在，则视为缺失连接的标签对。

002测试数据中没有找到错误标签对，有一组缺失标签对$(7,8)$。下一步就是对于每个缺失标签对，寻找对应的点连接对。首先我们通过下面这个函数寻找可能是连接对的节点：

```python
#input：
    #pc_label：每个3D点的标签
    #pair_to_check：缺失的标签对
    #G_nx：中心线血管图
#output：
    #degree_one_list_all：可能是标签间连接对的节点
    #flag_1214：是否存在[1,12]标签对
degree_one_list_all, flag_1214 = cutils.find_start_end_nodes(pc_label, pair_to_check, G_nx)
```

接下来说下`find_start_end_nodes`里都做了些什么。首先把要处理的血管分为4种：

1. `BA_label = [8,3,9]`：包含基底动脉（BA）、右颈总动脉（R-CCA）和左颈总动脉（L-CCA）等3根血管，这3根血管的共同特点是在端点处存在分叉：
    * 左右椎动脉（L/R-VA）汇合成基底动脉。
    * 颈总动脉会分叉为颈内动脉（ICA）和颈外动脉（ECA）。
2. `SA_label = [12]`。
3. `AO_label = [2]`。
4. 剩余的其他颈部血管。

对于第1种情况：

以002测试数据为例，缺失的标签对为$(7,8)$。标签8就属于第1种处理情况，计算label=8的节点的度，会发现没有度为1的节点，这是可能的，因为即使标签对$(7,8)$中断了，由于血管分叉的缘故，标签对$(12,8)$可能是连接的，所以就导致在label=8的节点中找不到度为1的节点。所以通过度等于1来寻找BA血管（label=8）的端点就行不通了。在这里，作者的做法是，将BA血管与其他血管相连的节点作为备选的连接对点（即边的一个节点标签为8，另一个节点标签不为8，这种情况下，把这个标签为8的节点保存起来），存入`degree_one_list_all`中。此外，如果BA血管中确实有度为1的节点，则也会存入`degree_one_list_all`中。002测试数据存入的节点序号为：$[3148,3666,3786,3954,3955]$。

对于第2种情况：

只有标签对为$(12,14)$时才会按这种情况处理。如果标签对里一个为12，另一个不是14，则按第4种情况处理。第2种情况处理的源码见下：

```python
if label in SA_label:
    if len(list(set(pair_to_check).intersection(set([12, 14]))))==2:
        print('{} has 12 and 14'.format(patient))
        if [1,12] in label_pairs:
            flag_1214 = 1
            break
        else:
            idx_label_one = np.nonzero(label_pc == 1)
            idx_label_one = list(idx_label_one[0])
            label_one_one_list = [idx for idx in idx_label_one if degree_list[int(idx)] == 1]
            degree_one_list = [idx for idx in (degree_one_list + label_one_one_list)]
```

这个代码是有问题的，`patient`和`label_pairs`两个变量是没有定义的，代码走到这里会报错。其次，这里为什么要判断$[1, 12]$，无名动脉和左椎动脉之间应该不会有连接的（这个连接在`gt_label_graph`中也有，有点困惑）。最后，`else`里为什么要计算无名动脉中度为1的节点？这些疑问我在github上提了[issue](https://github.com/PRESENT-Y/TaG-Net/issues/3)，等待作者的回复。

对于第3种情况：

其实和第1种情况的处理方式是一样的，不再赘述。

对于第4种情况：

以002测试数据为例，缺失的标签对为$(7,8)$。标签7就属于第4种处理情况，我们直接找R-VA血管（label=7）中度为1的节点，存入`degree_one_list_all`中。002测试数据存入的节点序号为3132。

根据上述操作，对于002测试数据，针对标签对$(7,8)$，我们可以得到5个备选的标签间连接对$[3132,3148],[3132,3666],[3132,3786],[3132,3954],[3132,3955]$。最终我们选择距离最小的标签间连接对$[3132,3148]$。

我们对缺失标签对都进行上述操作，便可得到所有的标签间连接对。

# 4.`gen_connection_path.py`

有了标签内和标签间的连接对后，下一步就是把这些连接对连接起来。先展示下血管补全后的效果。

标签内的补全（L-ECA，label=15）；标签间的补全（BA和R-VA，标签对$(7,8)$）：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/10.png)

不管是标签内的连接对还是标签间的连接对，补全的方法都是一样的。我们以标签内连接对$(2075,2156)$为例。

首先读取3D坐标：

```python
coordinates_pairs = scutils.gen_coordinates_pairs(pair, pc)
```

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/11.png)

`coordinates_pairs[0]`是连接对（保存的是节点序号），`coordinates_pairs[1]`是起始节点2075的3D坐标，`coordinates_pairs[2]`是结束节点2156的3D坐标。

接下来，获取起始节点和结束节点的标签（注意，这里只要一个整合后的标签，而不是两个节点各自的标签）：

```python
def gen_start_end_label(pair, pc):
    label_list = [7,12]
    label_pc = pc[:,-1]

    #如果起始节点和结束节点的标签一样，则返回这个一样的标签即可
    if label_pc[int(pair[0])] == label_pc[int(pair[1])]:
        start_end_label = label_pc[int(pair[0])]

    #如果起始节点和结束节点的标签不一样
    if label_pc[int(pair[0])] != label_pc[int(pair[1])]:
        #如果起始节点的标签为7或12，则返回起始节点的标签
        if label_pc[int(pair[0])] in label_list:
            start_end_label = label_pc[int(pair[0])]
        #如果结束节点的标签为7或12，则返回结束节点的标签
        elif label_pc[int(pair[1])] in label_list:
            start_end_label = label_pc[int(pair[1])]
        #否则，默认返回起始节点的标签
        else:
            start_end_label = label_pc[int(pair[0])]

    return start_end_label
```

根据上述规则，连接对$(2075,2156)$的标签为15。

接下来，我们要基于原始扫描图像（保存在`oriNumpy`中）创建一个局部的距离图：

```python
#input：
    #ori_img_path：原始扫描数据的路径
    #coordinates_pairs：连接对及其3D坐标
    #pc：所有点的坐标及其标签
#output：
    #distance_map：距离图
distance_map = scutils.gen_crop_distance_map(ori_img_path, coordinates_pairs, pc)
```

详细说下距离图是怎么构建的。节点的3D坐标就是3D数据场的坐标。以起始节点为中心，在$3 \times 3 \times 3$范围内一共有27个点，同理，以结束节点为中心，也可以得到27个点，然后计算这54个点对应HU值的最大值（`hu_uper`）、最小值（`hu_lower`）和平均值（`hu_average`）。

基于3D点坐标，计算所有点两两之间的欧氏距离，得到一个$8821 \times 8821$大小的距离矩阵`square_distance_matrix`。在这个矩阵中，我们可以查到起始节点到结束节点的距离`Sdistance`。

然后从原始扫描数据中截取一个局部的3D子数据场。这个子数据场的中心点就是起始节点和结束节点的平均。子数据场在$x,y,z$方向的半径均为`sub`：

* `Sdistance`小于50时，`sub = int(Sdistance * 2)`。
* `Sdistance`大于100时，`sub = int(Sdistance/3)`。
* 其他情况，`sub = int(Sdistance)`。

这个子数据场保存在`oriNumpy_temp`，注意，这里`oriNumpy_temp`和原始数据`oriNumpy`的大小是一样的，并没有真的把这个子数据场截取出来，只是把子数据场之外的地方赋值为0了。

对于`oriNumpy_temp`，将大于`hu_uper`或小于`hu_lower`的点的HU值置为0。将`oriNumpy_temp`中值不为0的点统统置为1，这样`oriNumpy_temp`就是一个只有0和1的数据场了。

然后是`distance_map`的计算：

```python
distance_map = ndimage.morphology.distance_transform_edt(oriNumpy_temp, spacing)
```

[`distance_transform_edt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)用于在二值图像中，计算每个非零像素到离它最近的零像素的欧氏距离。在`distance_map`中，起始节点和结束节点附近的点多数为1，其余绝大部分点都是0，显然，我们不能在`distance_map`上求最短路径，因为路径上基本都是0，所以从起始节点到结束节点不管怎么走，路径长度都是0。

因此，我们求得`distance_map`的最大值，让这个最大值减去`distance_map`得到`distance_map_temp`：

```python
zero_map = distance_map.copy()
zero_map [ zero_map != 0 ] = 0
max_value =np.max(distance_map)
max_map = zero_map.copy()
max_map[ max_map == 0 ] = max_value
#max_map和distance_map的大小一样，里面的值都是distance_map的最大值
distance_map_temp = max_map - distance_map 
```

这样我们就可以在`distance_map_temp`上求取最短路径了：

```python
#input：
    #distance_map：即上面提到的distance_map_temp
    #start_point：起始节点的3D坐标
    #end_point：结束节点的3D坐标
#output：
    #connection_path：起始节点到结束节点的最短路径
connection_path = dijkstra3d.dijkstra(distance_map, start_point, end_point)
```

`connection_path`保存着起始节点到结束节点的最短路径：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/12.png)

`connection_path[0]`为起始节点，`connection_path[16]`为结束节点，整个路径经过了15个点。

对于标签间的连接对$[3132,3148]$，也是通过一样的方式得到最短路径。这些结果都会保存在`connection_paths.csv`中。

# 5.`gen_adhesion_removal.py`

002测试数据不存在粘连，所以这一部分我们以003测试数据为例。效果如下图所示，红圈内的粘连被去除了：

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/13.png)

粘连去除的第一步是通过调用`gen_wrong_connected_exist_label`寻找错误标签对，这个函数的详细介绍见第3.2部分。可以看到，去除的粘连主要是针对标签间的粘连。003测试数据找到的错误标签对为$(7,12)$。

![](https://xjeffblogimg.oss-cn-beijing.aliyuncs.com/BLOGIMG/BlogImage/SourceCode/TagNet/14.png)

我们按照上述示意图来讲解粘连去除的步骤。首先对R-VA血管（label=7）进行处理，先找到其与L-VA血管（label=12）相连的节点，即节点4295。然后在R-VA中寻找度为3的节点，即节点4300。计算节点4300到节点4295的路径，即$[4300, 4299, 4298, 4295]$，路径长度为4，如果该路径的长度小于R-VA总长度的$\frac{1}{10}$，则把$[4299, 4298, 4295]$部分视为粘连部分。

>上述操作主要在函数`gen_nodes_to_remove`中。

对L-VA的处理是一样的，先找到其与R-VA相连的节点4296，然后找到度为3的节点4286，然后对两节点的路径长度进行判断，最后得到$[4285, 4289, 4288, 4297, 4296]$为粘连部分。

最后，我们只需要把$[4299, 4298, 4295,4285, 4289, 4288, 4297, 4296]$这8个节点及其相连的边从图中删除即可。