# “GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts #

[“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf)的论文复现 C++实现

matting部分未实现(参考的),除了GMM模型外还实现了彩色直方图[参考此论文实现](https://mmcheng.net/salobj/)模型的建模,RGB空间上做统计划分,实现RGB、Lab上的度量计算。

## 参考 ##
<https://github.com/MatthewLQM/GrabCut>  
在此基础上进行的增改