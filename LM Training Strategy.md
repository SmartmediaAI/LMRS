# LM Training Strategy
Here we summarize and classify papers into **Pretraining&Fine-tuning** and **Prompting** two different categories.

## Pretaining & Fine-Tuning
### Pretrain w/o Fine-tune
+ [7]. Exploiting Session Information in BERT-based Session-aware Sequential Recommendation \[[paper](https://shorturl.at/EFPSZ)\]\[[code](https://github.com/theeluwin/session-aware-bert4rec)\]
+ [9]. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer \[[paper](https://shorturl.at/guAET)\]\[[code](https://github.com/FeiSun/BERT4Rec)\]
+ [20]. Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation \[[paper](https://shorturl.at/coESX)\]\[[code](https://github.com/NVIDIA-Merlin/Transformers4Rec/)\]
+ [28]. Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation \[[paper](https://shorturl.at/pDY25)\]
+ [32]. How well do pre-trained contextual language representations recommend labels for GitHub issues \[[paper](https://shorturl.at/dlJ69)\]\[[code](https://bitbucket.org/jstzwj/lms4githubissue/src/master/data_view.7z)\]
+ [33]. Spatial Autoregressive Coding for Graph Neural Recommendation \[[paper](https://arxiv.org/abs/2205.09489)\]
+ [39]. Path Language Modeling over Knowledge Graphs for Explainable Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511937)\]

### Fine-tune Holistic Model
+ [2]. APIRecX: Cross-Library API Recommendation via Pre-Trained Language Model \[[paper](https://aclanthology.org/2021.emnlp-main.275/)\]\[[code](https://github.com/yuningkang/APIRecX)\]
+ [5]. RecoBERT: A Catalog Language Model for Text-Based Recommendations \[[paper](https://arxiv.org/pdf/2009.13292.pdf)\]
+ [8]. RecInDial: A Unified Framework for Conversational Recommendation with Pretrained Language Models \[[paper](https://aclanthology.org/2022.aacl-main.37.pdf)\]
+ [10]. U-BERT: Pre-training User Representations for Improved Recommendation \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16557)\]
+ [15]. UNBERT: User-News Matching BERT for News Recommendation \[[paper](https://www.ijcai.org/proceedings/2021/0462.pdf)\]
+ [17]. UPRec: User-Aware Pre-training for Recommender Systems \[[paper](https://arxiv.org/abs/2102.10989)\]
+ [30]. Self-supervised learning for conversational recommendation \[[paper](https://shorturl.at/nyAJV)\]
+ [37]. Training Large-Scale News Recommenders with Pretrained Language Models in the Loop \[[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539120)\]\[[code](https://github.com/Microsoft/SpeedyRec)\]
+ [41]. ServiceBERT: A Pre-trained Model for Web Service Tagging and Recommendation \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-91431-8_29)\]
+ [42]. GRAM: Fast Fine-tuning of Pre-trained Language Models for Content-based Collaborative Filtering \[[paper](https://arxiv.org/abs/2204.04179)\]\[[code](https://github.com/yoonseok312/GRAM)\]
+ [43]. TwHIN-BERT: A Socially-Enriched Pre-trained Language Model for Multilingual Tweet Representations \[[paper](https://arxiv.org/abs/2209.07562)\]\[[code](https://github.com/xinyangz/TwHIN-BERT)\]
+ [46]. PTM4Tag: Sharpening Tag Recommendation of Stack Overflow Posts with Pre-trained Models \[[paper](https://dl.acm.org/doi/abs/10.1145/3524610.3527897)\]\[[code](https://github.com/soarsmu/PTM4Tag)\]
+ [47]. JiuZhang: A Chinese Pre-trained Language Model for Mathematical Problem Understanding \[[paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539131)\]\[[code](https://github.com/RUCAIBox/JiuZhang)\]
+ [52]. Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation \[[paper](https://dl.acm.org/doi/abs/10.1145/3437963.3441738)\]\[[code](https://github.com/jerryhao66/Pretrain-Recsys)\] 
+ [53]. What does BERT know about books, movies and music? Probing BERT for Conversational Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3383313.3412249)\]\[[code](https://github.com/Guzpenha/ConvRecProbingBERT)\]
+ [54]. Improving Text-based Similar Product Recommendation for Dynamic Product Advertising at Yahoo \[[paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557129)\]\[[code](https://github.com/microsoft/unilm/tree/master/s2s-ft)\]
+ [56]. PTUM: Pre-training User Model from Unlabeled User Behaviors via Self-supervision \[[paper](https://arxiv.org/abs/2010.01494)\]\[[code](https://github.com/wuch15/PTUM)\]
+ [57]. Factual and Informative Review Generation for Explainable Recommendation \[[paper](https://arxiv.org/abs/2209.12613)\]
+ [59]. RESETBERT4Rec: A Pre-training Model Integrating Time And User Historical Behavior for Sequential Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3532054)\]
+ [60]. Dual Learning for Query Generation and Query Selection in Query Feeds Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3459637.3481910)\]\[[code](https://github.com/qikunxun/TitIE)\]
+ [64]. Curriculum Pre-Training Heterogeneous Subgraph Transformer for Top-N Recommendation \[[paper](https://dl.acm.org/doi/full/10.1145/3528667)\]
+ [65]. TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback \[[paper](https://arxiv.org/abs/2206.06190)\]\[[code](Inquiry author by email)\]


### Fine-tune Partial Model
+ [1]. ReXPlug: Explainable Recommendation using Plug and Play Language Model \[[paper](https://dl.acm.org/doi/10.1145/3404835.3462939)\]\[[code](https://github.com/deepeshhada/ReXPlug/)\]
+ [14]. Empowering News Recommendation with Pre-trained Language Models \[[paper](https://shorturl.at/dnwLQ)\]\[[code](https://github.com/wuch15/PLM4NewsRec)\]
+ [18]. Towards Universal Sequence Representation Learning for Recommender Systems \[[paper](https://shorturl.at/RT378)\]\[[code](https://github.com/RUCAIBox/UniSRec)\]
+ [29]. OutfitTransformer: Outfit Representations for Fashion Recommendation \[[paper](https://shorturl.at/qBGX5)\]
+ [31]. KEEP: An Industrial Pre-Training Framework for Online Recommendation via Knowledge Extraction and Plugging \[[paper](https://shorturl.at/fjwDU)\]
+ [51]. Tiny-NewsRec: Efficient and Effective PLM-based News Recommendation \[[paper](https://arxiv.org/abs/2112.00944)\]\[[code](https://github.com/yflyl613/Tiny-NewsRec)\]
+ [58]. MM-Rec: Visiolinguistic Model Empowered Multimodal News Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531896)\]
+ [67]. Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders \[[paper](https://arxiv.org/abs/2210.12316)\]\[[code](https://github.com/RUCAIBox/VQ-Rec)\]


### Fine-tune Extra Part
+ [19]. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximizatio \[[paper](https://shorturl.at/ioxX5)\]\[[code](https://github.com/RUCAIBox/CIKM2020-S3Rec)\]
+ [26]. Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation \[[paper](https://shorturl.at/aNPVZ)\]\[[code](https://github.com/fajieyuan/sigir2020_peterrec)\]
+ [27]. Learning Large-scale Universal User Representation with Sparse Mixture of Experts \[[paper](https://arxiv.org/abs/2207.04648)\]
+ [34]. Improving Personalized Explanation Generation through Visualization \[[paper](https://aclanthology.org/2022.acl-long.20/)\] \[[code]( https://github.com/jeykigung/METER)\]
+ [36]. ProtoCF: Prototypical Collaborative Filtering for Few-shot Recommendation \[[paper](https://shorturl.at/bCDRU)\]\[[code](https://github.com/aravindsankar28/ProtoCF)\]
+ [38]. Boosting Deep CTR Prediction with a Plug-and-Play Pre-trainer for News Recommendation \[[paper](https://aclanthology.org/2022.coling-1.249/)\]\[[code](https://github.com/Jyonn/PREC)\]
+ [44]. Pre-training of Graph Augmented Transformers for Medication Recommendation \[[paper](https://arxiv.org/abs/1906.00346)\]\[[code](https://github.com/jshang123/G-Bert)\]
+ [45]. User-specific Adaptive Fine-tuning for Cross-domain Recommendations \[[paper](https://ieeexplore.ieee.org/abstract/document/9573392)\]
+ [49]. Graph Neural Pre-training for Recommendation with Side Information \[[paper](https://dl.acm.org/doi/full/10.1145/3568953)\]\[[code](https://github.com/pretrain/pretrain)\]
+ [50]. Pre-training Graph Transformer with Multimodal Side Information for Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475709)\]
+ [61]. Multi-Modal Contrastive Pre-training for Recommendation \[[paper](https://dl.acm.org/doi/abs/10.1145/3512527.3531378)\]
+ [63]. Scaling Law for Recommendation Models: Towards General-purpose User Representations \[[paper](https://arxiv.org/abs/2111.11294)\]


## Prompting
### Fix Pretrained Model, Prompting
+ [3]. Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning \[[paper](https://shorturl.at/emOX2)\]\[[code](https://github.com/RUCAIBox/UniCRS)\]
+ [21]. Personalized Prompts for Sequential Recommendation \[[paper](https://arxiv.org/abs/2205.09666)\]
+ [71]. Automated Prompting for Non-overlapping Cross-domain Sequential Recommendation \[[paper](https://arxiv.org/abs/2304.04218)\]


### Fix Prompt, Model Tuning
+ [4]. Improving Conversational Recommendation Systems’ Quality with Context-Aware Item Meta-Information \[[paper](https://arxiv.org/pdf/2112.08140.pdf)\]\[[code](https://github.com/by2299/MESE)\]
+ [6]. Language Models as Recommender Systems: Evaluations and Limitations \[[paper](https://www.amazon.science/publications/language-models-as-recommender-systems-evaluations-and-limitations)\]
+ [11]. A Unified Multi-task Learning Framework for Multi-goal Conversational Recommender Systems \[[paper](https://shorturl.at/rDO01)\]
+ [68]. Prompt Learning for News Recommendation \[[paper](https://arxiv.org/abs/2304.05263)\]\[[code](https://github.com/resistzzz/Prompt4NR)\]
+ [69]. GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation \[[paper](https://arxiv.org/abs/2304.03879)\]


### Tuning-free, Prompting
+ [6]. Language Models as Recommender Systems: Evaluations and Limitations \[[paper](https://www.amazon.science/publications/language-models-as-recommender-systems-evaluations-and-limitations)\]
+ [13]. Zero-Shot Recommendation as Language Modeling \[[paper](https://link.springer.com/chapter/10.1007/978-3-030-99739-7_26)\]\[[code]()https://shorturl.at/yY089\]
+ [16]. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5) \[[paper](https://shorturl.at/psI47)\]\[[code](https://github.com/jeykigung/P5)\]
+ [66]. Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System \[[paper](https://arxiv.org/abs/2303.14524)\]
+ [70]. Is ChatGPT a Good Recommender? A Preliminary Study \[[paper](https://arxiv.org/abs/2304.10149)\]


### Prompt + LM Fine-tuning
+ [12]. Personalized Prompt Learning for Explainable Recommendation \[[paper](https://shorturl.at/fvEO8)\]\[[code](https://github.com/lileipisces/PEPLER)\]
+ [25]. Rethinking Reinforcement Learning for Recommendation: A Prompt Perspective \[[paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531714)\]

