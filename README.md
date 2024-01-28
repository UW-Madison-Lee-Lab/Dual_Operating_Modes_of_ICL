<h1 align="center"> <p> The Dual Operating Modes of In-Conctext Learning </p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

**Paper Link**: https://arxiv.org/abs/2310.17513

*Low-Rank Adaptation* (LoRA), a parameter-efficient fine-tuning method that leverages low-rank adaptation of weight matrices, has emerged as a prevalent technique for fine-tuning pre-trained models such as large language models and diffusion models. Despite its huge success in practice, the theoretical underpinnings of LoRA have largely remained unexplored. This paper takes the first step to bridge this gap by theoretically analyzing the expressive power of LoRA. We prove that, for fully connected neural networks, LoRA can adapt any model $f$ to accurately represent any smaller target model $\overline{f}$ if LoRA-rank $\geq(\text{width of }f) \times \frac{\text{depth of }\overline{f}}{\text{depth of }f}$, under a mild assumption. We also quantify the approximation error when the LoRA-rank is lower than the threshold. For Transformer networks, we show any model can be adapted to a target model of the same size with LoRA adapters of rank $(\frac{\text{embedding size}}{2})$. All our theoretical insights are validated by numerical experiments.

## Experiments

