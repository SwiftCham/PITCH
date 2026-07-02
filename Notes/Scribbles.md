

turboquant blog post: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

turboquant paper: https://arxiv.org/abs/2504.19874

polarquant blog post: https://research.google/pubs/polarquant-quantizing-kv-caches-with-polar-transformation/

polarquant paper: https://arxiv.org/abs/2502.02617


bibliography: 
@misc{dosovitskiyImageWorth16x162021,
	title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
	url = {http://arxiv.org/abs/2010.11929},
	doi = {10.48550/arXiv.2010.11929},
	shorttitle = {An Image is Worth 16x16 Words},
	abstract = {While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on {CNNs} is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks ({ImageNet}, {CIFAR}-100, {VTAB}, etc.), Vision Transformer ({ViT}) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.},
	number = {{arXiv}:2010.11929},
	publisher = {{arXiv}},
	author = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
	urldate = {2026-04-07},
	date = {2021-06-03},
	eprinttype = {arxiv},
	eprint = {2010.11929 [cs]},
	keywords = {Computer Science - Artificial Intelligence, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
	file = {Preprint PDF:/Users/slowpoke/Zotero/storage/SAU89SWY/Dosovitskiy et al. - 2021 - An Image is Worth 16x16 Words Transformers for Image Recognition at Scale.pdf:application/pdf;Snapshot:/Users/slowpoke/Zotero/storage/NQ5NDW58/2010.html:text/html},
}

@misc{vaswaniAttentionAllYou2023,
	title = {Attention Is All You Need},
	url = {http://arxiv.org/abs/1706.03762},
	doi = {10.48550/arXiv.1706.03762},
	abstract = {The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 {BLEU} on the {WMT} 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 {BLEU}. On the {WMT} 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art {BLEU} score of 41.8 after training for 3.5 days on eight {GPUs}, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.},
	number = {{arXiv}:1706.03762},
	publisher = {{arXiv}},
	author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
	urldate = {2026-04-07},
	date = {2023-08-02},
	eprinttype = {arxiv},
	eprint = {1706.03762 [cs]},
	keywords = {Computer Science - Computation and Language, Computer Science - Machine Learning},
	file = {Preprint PDF:/Users/slowpoke/Zotero/storage/J7LKQHUR/Vaswani et al. - 2023 - Attention Is All You Need.pdf:application/pdf;Snapshot:/Users/slowpoke/Zotero/storage/HWP287TK/1706.html:text/html},
}

@misc{hanPolarQuantQuantizingKV2025,
	title = {{PolarQuant}: Quantizing {KV} Caches with Polar Transformation},
	url = {http://arxiv.org/abs/2502.02617},
	doi = {10.48550/arXiv.2502.02617},
	shorttitle = {{PolarQuant}},
	abstract = {Large language models ({LLMs}) require significant memory to store Key-Value ({KV}) embeddings in their {KV} cache, especially when handling long-range contexts. Quantization of these {KV} embeddings is a common technique to reduce memory consumption. This work introduces {PolarQuant}, a novel quantization method employing random preconditioning and polar transformation. Our method transforms the {KV} embeddings into polar coordinates using an efficient recursive algorithm and then quantizes resulting angles. Our key insight is that, after random preconditioning, the angles in the polar representation exhibit a tightly bounded and highly concentrated distribution with an analytically computable form. This nice distribution eliminates the need for explicit normalization, a step required by traditional quantization methods which introduces significant memory overhead because quantization parameters (e.g., zero point and scale) must be stored in full precision per each data block. {PolarQuant} bypasses this normalization step, enabling substantial memory savings. The long-context evaluation demonstrates that {PolarQuant} compresses the {KV} cache by over x4.2 while achieving the best quality scores compared to the state-of-the-art methods.},
	number = {{arXiv}:2502.02617},
	publisher = {{arXiv}},
	author = {Han, Insu and Kacham, Praneeth and Karbasi, Amin and Mirrokni, Vahab and Zandieh, Amir},
	urldate = {2026-04-07},
	date = {2025-02-04},
	eprinttype = {arxiv},
	eprint = {2502.02617 [cs]},
	keywords = {Computer Science - Artificial Intelligence, Computer Science - Machine Learning},
	file = {Preprint PDF:/Users/slowpoke/Zotero/storage/CS6J9DWJ/Han et al. - 2025 - PolarQuant Quantizing KV Caches with Polar Transformation.pdf:application/pdf;Snapshot:/Users/slowpoke/Zotero/storage/IZDGSVXG/2502.html:text/html},
}

@misc{zandiehTurboQuantOnlineVector2025,
	title = {{TurboQuant}: Online Vector Quantization with Near-optimal Distortion Rate},
	url = {http://arxiv.org/abs/2504.19874},
	doi = {10.48550/arXiv.2504.19874},
	shorttitle = {{TurboQuant}},
	abstract = {Vector quantization, a problem rooted in Shannon's source coding theory, aims to quantize high-dimensional Euclidean vectors while minimizing distortion in their geometric structure. We propose {TurboQuant} to address both mean-squared error ({MSE}) and inner product distortion, overcoming limitations of existing methods that fail to achieve optimal distortion rates. Our data-oblivious algorithms, suitable for online applications, achieve near-optimal distortion rates (within a small constant factor) across all bit-widths and dimensions. {TurboQuant} achieves this by randomly rotating input vectors, inducing a concentrated Beta distribution on coordinates, and leveraging the near-independence property of distinct coordinates in high dimensions to simply apply optimal scalar quantizers per each coordinate. Recognizing that {MSE}-optimal quantizers introduce bias in inner product estimation, we propose a two-stage approach: applying an {MSE} quantizer followed by a 1-bit Quantized {JL} ({QJL}) transform on the residual, resulting in an unbiased inner product quantizer. We also provide a formal proof of the information-theoretic lower bounds on best achievable distortion rate by any vector quantizer, demonstrating that {TurboQuant} closely matches these bounds, differing only by a small constant (\${\textbackslash}approx 2.7\$) factor. Experimental results validate our theoretical findings, showing that for {KV} cache quantization, we achieve absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with 2.5 bits per channel. Furthermore, in nearest neighbor search tasks, our method outperforms existing product quantization techniques in recall while reducing indexing time to virtually zero.},
	number = {{arXiv}:2504.19874},
	publisher = {{arXiv}},
	author = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
	urldate = {2026-04-07},
	date = {2025-04-28},
	eprinttype = {arxiv},
	eprint = {2504.19874 [cs]},
	keywords = {Computer Science - Artificial Intelligence, Computer Science - Data Structures and Algorithms, Computer Science - Databases, Computer Science - Machine Learning},
	file = {Preprint PDF:/Users/slowpoke/Zotero/storage/IQFQZCGI/Zandieh et al. - 2025 - TurboQuant Online Vector Quantization with Near-optimal Distortion Rate.pdf:application/pdf;Snapshot:/Users/slowpoke/Zotero/storage/IC24V3QZ/2504.html:text/html},
}

@misc{bai2025longbenchv2deeperunderstanding,
	title = {{LongBench} v2: Towards deeper understanding and reasoning on realistic long-context multitasks},
	url = {https://arxiv.org/abs/2412.15204},
	author = {Bai, Yushi and Tu, Shangqing and Zhang, Jiajie and Peng, Hao and Wang, Xiaozhi and Lv, Xin and Cao, Shulin and Xu, Jiazheng and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
	date = {2025},
	eprinttype = {arxiv},
	eprint = {2412.15204 [cs.CL]},
}

@misc{li2024snapkvllmknowslooking,
	title = {{SnapKV}: {LLM} knows what you are looking for before generation},
	url = {https://arxiv.org/abs/2404.14469},
	author = {Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
	date = {2024},
	eprinttype = {arxiv},
	eprint = {2404.14469 [cs.CL]},
}

@misc{cai2025pyramidkvdynamickvcache,
	title = {{PyramidKV}: Dynamic {KV} cache compression based on pyramidal information funneling},
	url = {https://arxiv.org/abs/2406.02069},
	author = {Cai, Zefan and Zhang, Yichi and Gao, Bofei and Liu, Yuliang and Li, Yucheng and Liu, Tianyu and Lu, Keming and Xiong, Wayne and Dong, Yue and Hu, Junjie and Xiao, Wen},
	date = {2025},
	eprinttype = {arxiv},
	eprint = {2406.02069 [cs.CL]},
}

@article{https://doi.org/10.13140/rg.2.2.28167.37282,
	title = {{KIVI} : Plug-and-play 2bit {KV} cache quantization with streaming asymmetric quantization},
	url = {https://www.researchgate.net/doi/10.13140/RG.2.2.28167.37282},
	doi = {10.13140/RG.2.2.28167.37282},
	publisher = {Unpublished},
	author = {{Zirui Liu} and {Jiayi Yuan} and {Hongye Jin} and {Shaochen Zhong} and {Zhaozhuo Xu} and Braverman, Vladimir and {Beidi Chen} and Hu, Xia},
	date = {2023},
	langid = {english},
}

@misc{gao2024practicalasymptoticallyoptimalquantization,
	title = {Practical and asymptotically optimal quantization of high-dimensional vectors in euclidean space for approximate nearest neighbor search},
	url = {https://arxiv.org/abs/2409.09913},
	author = {Gao, Jianyang and Gou, Yutong and Xu, Yuexuan and Yang, Yongyi and Long, Cheng and Wong, Raymond Chi-Wing},
	date = {2024},
	eprinttype = {arxiv},
	eprint = {2409.09913 [cs.DB]},
}

@inproceedings{thakur2021beir,
	title = {{BEIR}: a heterogeneous benchmark for zero-shot evaluation of information retrieval models},
	url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
	booktitle = {Thirty-fifth conference on neural information processing systems datasets and benchmarks track (round 2)},
	author = {Thakur, Nandan and Reimers, Nils and Rücklé, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
	date = {2021},
}
