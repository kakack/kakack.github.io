---

layout: post
categories: [Github]
tags: [Github , Mac]

---


一直打算在github上搞个博客，一直由于懒惰没能成行<br>
做本科毕设的时候虽然有接触过一点web前端的内容，但都是很皮毛的html+css+js，要说自己从零开始写个好看点的page出来真不容易，此外md语言之前也没怎么多写，作为逼格直线上升的同学，*letax*和*md*简直就是必修课，不然还怎么愉快地嘲笑那些用**Word**和**txt**的小伙伴们233333  

- - -
~~其实写这第一篇blog的目的是想来用新玩具个hello world我会随便说？~~</br>
- - -
既然写了就稍微说几句在Git上搭建Pages的办法，不然Blog没营养没人看啊摔！( ิ◕㉨◕ ิ)<br><br>
我自己是看[这个蜀黍的blog](http://yanping.me/cn/blog/2012/03/18/github-pages-step-by-step/)搞的，写的挺完整但是还是有点高估了同学们的智商，吾就将其说的搞的简单明了一丢丢。

首先，每个教程都会逼格很高地介绍一下***git+github+markdown+jekyll***的好处，总的来说是git完成备份，md完成排版，jekyll完成网站那么简单，所以整个环境无非就是需要这三样东西。

Git：你需要一个git的账号，一个git的空间，然后再一个git的资源叫*username*.github.io，其中*username*是自己的用户名，比如我的就是kakack，github会提供一些template供用户自动生成页面，可以先选一个试试看，资源建立后十几分钟能方位https://*username*.github.com。然后再git clone到本地，就可以在本地进行编辑了。

Jekyll：在终端可以自行下载`sudo gem install jekyll`

接下去就去选一个靠谱的框架玩，一般推荐去Octopress和Jekyll Bootstrap，另外[github](https://github.com/mojombo/jekyll/wiki/sites)上也有一些推荐的。至于Octopress和Jekyll哪个好，可以看[知乎上](http://www.zhihu.com/question/19996679)的比较。

最后只要把下载下来的template覆盖掉自己本地的文件，然后再在资源路径下

`git add -A .`全选所有文件

`git commit -am "message"`

`git push`

就能把现有的template全部po到*username*.github.io上

之后的事情就是把别人template上的内容改成自己的内容即可，有些非git上的page还得把人家的CNAME文件删除。

最后就可以写一篇这样的hello world 软文，放在_post文件夹下，愉快地玩耍了。
- - -

最后补一些有用的参考资源：

- - -
- 
[入门指南](http://www.ruanyifeng.com/blog/2012/08/blogging_with_jekyll.html)
- 
[markdown简易语法](http://wowubuntu.com/markdown/) 

