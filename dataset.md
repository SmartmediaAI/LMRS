# Dataset Resources

Here we try to collect all datasets that have been used and puhlicaly availabel for LMRS. Please note that the number in the brackets are in accordance with the paper number in <span style="color:blue">*LM Training Strategy.md*</span>


| Dataset      | Link     | Task | Training Strategy (Representative papers) | Datatype |
| ----------- | ----------- | ----------- | ----------- | ----------- | 
| MovieLens    | [link](https://grouplens.org/datasets/movielens/25m/)       | Sequential Recommendation | Pre-train (7,28), Pretrain & Fine-tune Holistic Model (59)| sequence (7,28,50, 59) |
|   |      | Top-k Recmmendation | Pretrain & Fine-tune Holistic Model (52, 64), Pretrain & Fine-tune Extra Part (49), Fixed-prompt & Tune Model (6), Tuning-free & Prompting (6, 13) | text (6, 13, 53, 57, 66) |
|   |      | CTR Prediction | Pretrain & Fine-tune Extra Part (50) | multi-modal(50)|
|   |      | Conversational Recommendation | Pretrain & Fine-tune Holistic Model (53), Tuning-free & Prompting (66) | graph (49, 50, 52, 64) |
|   |      | Explanable Recommendation | Pretrain & Fine-tune Holistic Model (57) ||
|   |      | Zero-shot Rating Prediction | Tuning-free & Prompting (66) ||
| Amazon | [link](http://jmcauley.ucsd.edu/data/amazon/) | Rating Prediction | Pretrain & Fine-tune Extra Part (1), Tuning-free & Prompting (16, 70) | text (1, 6, 10, 12, 18, 19, 53, 57, 59, 61, 67, 69, 70)|
|  |  | Cross-domain Recommendation | Pretrain & Fine-tune Holistic Model (10), Pretrain & Fine-tune Partial Model (18, 67), Fix-pretrained Model & Prompt Tuning (71) | sequence (9, 16, 18, 19, 39, 50, 61, 67, 71)|
|  |  | Explanable Recommendation | Pretrain (39), Pretrain & Fine-tune Holistic Model (57), Fix-pretrained Model & Prompt Tuning (12), Fixed-prompt & Tune Model (69), Tuning-free & Prompting (16, 70) | graph (39, 50)|
|  |  | Direct Recommendation | Tuning-free & Prompting (16, 70) | multi-modal(50)|
|  |  | Sequential Recommendation | Pretrain (9), Pretrain & Fine-tune Holistic Model (59), Pretrain & Fine-tune Partial Model (67), Pretrain & Fine-tune Extra Part (19), Fix-pretrained Model & Prompt Tuning (71), Tuning-free & Prompting (16, 70) | |
|  |  | Conversational Recommendation | Pretrain & Fine-tune Holistic Model (53) | |
|  |  | Top-k Recommendation | Pretrain & Fine-tune Extra Part (50, 61) | |
| Yelp | [link](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) | Rating Prediction | Pretrain & Fine-tune Holistic Model (57), Pretrain & Fine-tune Extra Part (1, 34), Tuning-free & Prompting (16) | text(1, 10, 12, 16, 19, 17, 57) |
|  |  | Cross-domain Recommendation | Pretrain & Fine-tune Holistic Model (10) | sequence (16, 19, 17, 36) |
|  |  | Explanable Recommendation | Pretrain & Fine-tune Holistic Model (57), Pretrain & Fine-tune Extra Part (34), Fix-pretrained Model & Prompt Tuning (12), Tuning-free & Prompting (16) | sequence (16, 19, 17, 36) |
|  |  | Direct Recommendation | Tuning-free & Prompting (16) | graph (17, 33, 64) |
|  |  | Sequential Recommendation | Pretrain & Fine-tune Holistic Model (17), Pretrain & Fine-tune Extra Part (19), Tuning-free & Prompting (16) | multi-modal (34) |
|  |  | Top-k Recommendation | Pretrain (33), Pretrain & Fine-tune Holistic Model (64), Pretrain & Fine-tune Extra Part (14, 38, 51) | |
| MIND | [link](https://msnews.github.io/) | Top-k Recommendation | Pretrain & Fine-tune Holistic Model (15, 37, 42), Pretrain & Fine-tune Partial Model (14, 51), Pretrain & Fine-tune Extra Part (36), Fixed-prompt & Tune Model (68) | text (14, 15, 37, 38, 42, 51, 68)<br> sequence (37, 38, 51) |
| TripAdvisor | [link](https://lifehkbueduhk-my.sharepoint.com/personal/16484134_life_hkbu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F16484134%5Flife%5Fhkbu%5Fedu%5Fhk%2FDocuments%2FCIKM20%2DNETE%2DDatasets&ga=1) | Explanable Recommendation | Pretrain & Fine-tune Holistic Model (57), Pretrain & Fine-tune Extra Part (34), Fix-pretrained Model & Prompt (12) | text (12, 57) |
|  |  | Rating Prediction | Pretrain & Fine-tune Holistic Model (57), Pretrain & Fine-tune Extra Part (34) | multi-modal (34) |
| BeerAdvocate | [link](http://snap.stanford.edu/data/#reviews) | Rating Prediction | Pretrain & Fine-tune Extra Part (1) | text (1) |
| ReDial | [link](https://redialdata.github.io/website/download) | Conversational Recommendation | Pretrain & Fine-tune Holistic Model (8, 30, 35, 53), Fix-pretrained Model & Prompt (3), Fixed-prompt & Tune Model (4) | text (3, 4, 8, 30, 35, 53)<br>graph (8, 30, 35) |
| DuRecDial | [link](https://github.com/PaddlePaddle/Research/tree/48408392e152ffb2a09ce0a52334453e9c08b082/NLP/ACL2020-DuRecDial) | Conversational Recommendation | Fixed-prompt & Tune Model (11) | text (11) |
| TG-ReDial | [link](https://github.com/RUCAIBox/TG-ReDial) | Conversational Recommendation | Pretrain & Fine-tune Holistic Model (30), Fixed-prompt & Tune Model (11) | text (11, 30)<br>graph (30) |
| INSPIRED | [link](https://github.com/RUCAIBox/UniCRS) | Conversational Recommendation | Fix-pretrained Model & Prompt (3), Fixed-prompt & Tune Model (4) | text (3, 4) |
| WineReview | [link](https://www.kaggle.com/datasets/zynicide/wine-reviews) | Top-k Recommendation | Pretrain & Fine-tune Holistic Model (5) | text (5) |
| SIUPD | [link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=58) | Sequential Recommendation | Pretrain & Fine-tune Extra Part (27) | text (27) |
| Polyvore Outfit | [link](https://github.com/xthan/polyvore/tree/master/data) | Fashion Recommendation | Pretrain & Fine-tune Partial Model (29), Pretrain & Fine-tune Extra Part (29) | multi-modal (29) |
| User-Behavior | [link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) | Top-k Recommendation | Pretrain (33) | graph (33) |
| WeePlaces | [link](https://www.yongliu.org/datasets/) | Top-k Recommendation | Pretrain & Fine-tune Extra Part (36) | sequence (36) |
| Gowalla | [link](https://www.yongliu.org/datasets/) | Top-k Recommendation | Pretrain & Fine-tune Extra Part (36) | sequence (36) |
| Adressa | [link](http://reclab.idi.ntnu.no/dataset/) | Top-k Recommendation | Pretrain & Fine-tune Extra Part (38) | text (38)<br>sequence (38) |
| MIMIC-III | [link](https://www.nature.com/articles/sdata201635#MOESM102) | Medication Recommendation | Pretrain & Fine-tune Extra Part (44) | graph (44) |
| StackOverflow | [link](https://archive.org/details/stackexchange) | Top-k Recommendation | Pretrain & Fine-tune Holistic Model (46) | text (46) |
| Epinion | [link](https://cseweb.ucsd.edu//~jmcauley/datasets.html) | Top-k Recommendation | Pretrain & Fine-tune Extra Part (36, 49) | text (49)<br>sequence (36)<br>graph (49) |
| Foursquare | [link](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) | Top-k Recommendation | Pretrain & Fine-tune Extra Part (49) | text (49)<br>graph (49) |
| Last.fm | [link](http://millionsongdataset.com/lastfm/) | Top-k Recommendation | Pretrain & Fine-tune Holistic Model (52) | text (19)<br>sequence (19) |
|  |  | Sequential Recommendation | Pretrain & Fine-tune Extra Part (19) | graph (52) |
| QQBrowser | (required by email to the author) | Top-k Recommendation & Query Generation | Pretrain & Fine-tune Holistic Model (60) | text (60) |
|  | [link](https://github.com/fajieyuan/sigir2020_peterrec) | Sequential Recommendation | Fix-pretrained Model & Prompt (21) | text (21)<br>sequence (21) |
| Steam | [link](https://github.com/kang205/SASRec/tree/master/data) | Session-based Recommendation | Pretrain (7, 9) | sequence (7, 9) |
| Online Retail | [link](https://www.kaggle.com/datasets/carrie1/ecommerce-data) | Cross-domain/Cross-platform Recommendaiton | Pretrain & Fine-tune Partial Model (18) | text (18)<br>sequence (18) |
| REES46eCommerce | [link](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) | Sequential Recommendation | Pretrain (20) | text (20)<br>sequence (20) |
| YOOCHOOSE | (Temporarily unavailable) | Sequential Recommendation | Pretrain (20) | text (20)<br>sequence (20) |
| G1-news | [link](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom) | Sequential Recommendation | Pretrain (20) | text (20)<br>sequence (20) |
| CIKM | [link](https://tianchi.aliyun.com/competition/entrance/231719/introduction) | Cross-domain Recommendation | Fix-pretrained Model & Prompt (21) | text (21)<br>sequence (21) |
| AliEC & AliAD | [link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56) | Cross-domain Recommendation | Fix-pretrained Model & Prompt (21) | text (21)<br>sequence (21) |
| RecSys Challenge 2015 | [link](https://recsys.acm.org/recsys15/challenge/) | Sequential Recommendation | Tuning-free & Prompting (25) | sequence (25) |
| Retail Rocket | [link](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | Sequential Recommendation | Tuning-free & Prompting (25) | sequence (25) |
| GoodReads | [link](https://github.com/MengtingWan/goodreads) | Conversational Recommendation | Pretrain & Fine-tune Holistic Model (53) | text (53) |










