
<script>
MathJax = {
tex: {
inlineMath: [['$', '$']],
displayMath: [['\\[', '\\]']],
processEnvironments: true,
processRefs: true
},
options: {
skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
ignoreHtmlClass: 'tex2jax_ignore',
renderActions: {
find_script_mathtex: [10, function (doc) {
for (const node of document.querySelectorAll('script[type^="math/tex"]')) {
const display = !!node.type.match(/; *mode=display/);
const math = new doc.options.MathItem(node.textContent, doc.inputJax[0], display);
const text = document.createTextNode('');
node.parentNode.replaceChild(text, node);
math.start = { node: text, delim: '', n: 0 };
math.end = { node: text, delim: '', n: 0 };
doc.math.push(math);
}
}, '']
}
},
svg: {
fontCache: 'global'
}
};
</script>
<script id="MathJax-script" async src="https://cdn.staticfile.org/mathjax/3.0.1/es5/tex-svg.js"></script>
---

layout: post
categories: [Algorithm]
tags: [Deep Learning, Batch Normalization]

---

# What is Batch Normalization？

**Batch Normalization**（以下用BN简称代替）是为了解决Deep Learning中*Internal Covariate Shift（内部协变量移位）*问题，而针对一个batch中的数据进行归一标准化的方法。作用是可以使用更加flexible的学习率learning rate，得到更快的学习速率，同时不会过于依赖模型的初始化initialization。其中所谓的Internal Covariate Shift是指在deep neural network当中，随着层数的加深，每一层layer都关联着其上下层，其input实上一层layer的output，而其output又是下一层layer的input，而在随机梯度下降学习过程中，随着每层layer参数的更新，其对应的input/output分布也在随时起着微小的变化，这种变化会随着层数深度的加深而逐步积累，从而使得整个学习过程难度加大。


$(a_1 + b)x^2 = c$
$(W_1-W_2)x+b_1-b_2=0$
---

# Reference

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)