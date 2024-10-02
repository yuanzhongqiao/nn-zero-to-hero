
## Neural Networks: Zero to Hero


<div class="Box-sc-g0xbh4-0 QkQOb js-snippet-clipboard-copy-unpositioned" data-hpc="true"><article class="markdown-body entry-content container-lg" itemprop="text"><div class="markdown-heading" dir="auto"><h2 tabindex="-1" class="heading-element" dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">神经网络：从零到英雄</font></font></h2><a id="user-content-neural-networks-zero-to-hero" class="anchor" aria-label="永久链接：神经网络：从零到英雄" href="#neural-networks-zero-to-hero"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></div>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这是一门从基础开始的神经网络课程。该课程是一系列 YouTube 视频，我们一起编码和训练神经网络。我们在视频中构建的 Jupyter 笔记本随后被捕获到</font></font><a href="/karpathy/nn-zero-to-hero/blob/master/lectures"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">讲座</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">目录中。每个讲座的视频描述中还包含一组练习。（这可能会发展成为更受尊敬的东西）。</font></font></p>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">第 1 讲：神经网络和反向传播的详细介绍：构建微梯度</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">神经网络的反向传播和训练。假设您具备 Python 的基本知识，并且对高中的微积分有一定的记忆。</font></font></p>
<ul dir="auto">
<li><a href="https://www.youtube.com/watch?v=VMj-3S1tku0" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本文件</font></font></a></li>
<li><a href="https://github.com/karpathy/micrograd"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">micrograd Github repo</font></font></a></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">第 2 讲：语言建模的详细介绍：构建 makemore</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们实现了一个二元字符级语言模型，我们将在后续视频中将其进一步复杂化为现代 Transformer 语言模型，如 GPT。在本视频中，重点介绍 (1) 介绍 torch.Tensor 及其精妙之处以及在有效评估神经网络中的应用，以及 (2) 语言建模的整体框架，包括模型训练、采样和损失评估（例如分类的负对数似然）。</font></font></p>
<ul dir="auto">
<li><a href="https://www.youtube.com/watch?v=PaCmpygFfXo" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本文件</font></font></a></li>
<li><a href="https://github.com/karpathy/makemore"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">makemore Github 仓库</font></font></a></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">讲座 3：构建 makemore 第二部分：MLP</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们实现了一个多层感知器 (MLP) 字符级语言模型。在本视频中，我们还介绍了机器学习的许多基础知识（例如模型训练、学习率调整、超参数、评估、训练/开发/测试分割、欠拟合/过拟合等）。</font></font></p>
<ul dir="auto">
<li><a href="https://youtu.be/TCH_1BHY58I" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本文件</font></font></a></li>
<li><a href="https://github.com/karpathy/makemore"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">makemore Github 仓库</font></font></a></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">讲座 4：构建 makemore 第 3 部分：激活和梯度、BatchNorm</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们深入研究了具有多层的 MLP 的一些内部结构，并仔细研究了前向传递激活、后向传递梯度的统计数据，以及当它们缩放不当时的一些陷阱。我们还研究了您想要用来了解深度网络健康状况的典型诊断工具和可视化。我们了解了为什么训练深度神经网络会很脆弱，并介绍了第一个使训练变得更容易的现代创新：批量标准化。残差连接和 Adam 优化器仍然是后续视频中值得注意的待办事项。</font></font></p>
<ul dir="auto">
<li><a href="https://youtu.be/P6sfmUTpUmc" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本文件</font></font></a></li>
<li><a href="https://github.com/karpathy/makemore"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">makemore Github 仓库</font></font></a></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">讲座 5：构建 makemore 第 4 部分：成为反向传播忍者</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们从上一个视频中获取 2 层 MLP（使用 BatchNorm），并手动反向传播它，而无需使用 PyTorch autograd 的 loss.backward()。也就是说，我们通过交叉熵损失、第二线性层、tanh、batchnorm、第一线性层和嵌入表进行反向传播。在此过程中，我们直观地了解了梯度如何通过计算图和高效张量的级别向后流动，而不仅仅是像 micrograd 中那样的单个标量。这有助于培养关于如何优化神经网络的能力和直觉，并让您更自信地创新和调试现代神经网络。</font></font></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我建议你自己做这个练习，但要同时做，当你遇到困难时，请暂停视频，看我给出答案。这个视频并不是专门用来观看的。这个练习在</font></font><a href="https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-?usp=sharing" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">这里是 Google Colab</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。祝你好运 :)</font></font></p>
<ul dir="auto">
<li><a href="https://youtu.be/q8SA3rM6ckI" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本文件</font></font></a></li>
<li><a href="https://github.com/karpathy/makemore"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">makemore Github 仓库</font></font></a></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">讲座 6：构建 makemore 第 5 部分：构建 WaveNet</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们采用上一个视频中的 2 层 MLP，并通过树状结构使其更深，从而得到类似于 DeepMind 的 WaveNet (2016) 的卷积神经网络架构。在 WaveNet 论文中，使用因果扩张卷积（尚未涉及）更有效地实现了相同的分层架构。在此过程中，我们对 torch.nn 有了更好的了解，了解它是什么以及它在底层是如何工作的，以及典型的深度学习开发过程是什么样的（大量阅读文档、跟踪多维张量形状、在 jupyter 笔记本和存储库代码之间移动……）。</font></font></p>
<ul dir="auto">
<li><a href="https://youtu.be/t3YJ5hKiMQ0" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Jupyter 笔记本文件</font></font></a></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">第 7 讲：让我们从头开始，用代码，拼写出来，构建 GPT。</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">我们根据论文《注意力就是你所需要的一切》和 OpenAI 的 GPT-2/GPT-3 构建了一个生成式预训练 Transformer (GPT)。我们讨论了与风靡全球的 ChatGPT 的联系。我们观看了 GitHub Copilot（它本身就是一个 GPT）帮助我们编写 GPT（meta :D！）。我建议大家观看早期的 makemore 视频，以熟悉自回归语言建模框架和张量和 PyTorch nn 的基础知识，我们在本视频中将其视为理所当然。</font></font></p>
<ul dir="auto">
<li><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">。其他所有链接请参阅视频说明。</font></font></li>
</ul>
<hr>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">第 8 讲：构建 GPT Tokenizer</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">Tokenizer 是大型语言模型 (LLM) 中必不可少且普遍存在的组件，它在字符串和标记（文本块）之间进行转换。Tokenizer 是 LLM 管道中完全独立的阶段：它们有自己的训练集、训练算法（字节对编码），并在训练后实现两个基本功能：将字符串编码为标记，将标记解码回字符串。在本讲座中，我们从头开始构建 OpenAI 的 GPT 系列中使用的 Tokenizer。在此过程中，我们将看到 LLM 的许多奇怪行为和问题实际上可以追溯到标记化。我们将讨论其中的一些问题，讨论标记化的错误原因，以及为什么有人理想情况下会找到一种方法来完全删除此阶段。</font></font></p>
<ul dir="auto">
<li><a href="https://www.youtube.com/watch?v=zduSFxRajkE" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">YouTube 视频讲座</font></font></a></li>
<li><a href="https://github.com/karpathy/minbpe"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">minBPE 代码</font></font></a></li>
<li><a href="https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing" rel="nofollow"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">谷歌Colab</font></font></a></li>
</ul>
<hr>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">正在进行中...</font></font></p>
<p dir="auto"><strong><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">执照</font></font></strong></p>
<p dir="auto"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">麻省理工学院</font></font></p>
</article></div>

Ongoing...

**License**

MIT
