# Official Repo for SemStamp (NAACL24) and k-SemStamp (ACL24)

## Updates (Dec 7 2024)
1. SemStamp and k-SemStamp now both support multi-GPU implementations.
2. Added more instructions to clarify usage
3. Integrated data repository with the original data from the paper.

To cite
```
@inproceedings{hou-etal-2023-semstamp,
    title = "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation",
    author = "Hou, Abe Bohan*  and
      Zhang, Jingyu*  and
      He, Tianxing*  and
      Chuang, Yung-Sung  and
      Wang, Hongwei  and
      Shen, Lingfeng and
      Van Durme, Benjamin and
      Khashabi, Daniel  and
      Tsvetkov, Yulia",
    booktitle = "Annual Conference of the North American Chapter of the Association for Computational Linguistics",
    year = "2023",
    url = "https://arxiv.org/abs/2310.03991",
}

@inproceedings{hou-etal-2024-k,
    title = "k-{S}em{S}tamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text",
    author = "Hou, Abe  and
      Zhang, Jingyu  and
      Wang, Yichen  and
      Khashabi, Daniel  and
      He, Tianxing",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.98",
    doi = "10.18653/v1/2024.findings-acl.98",
    pages = "1706--1715",
}

```

## Installation
(Recommended) create a virtual environment, and then:
```
pip install -r requirements.txt
```

(MANDATORY) install punkt
```
python install_punkt.py
```


## SemStamp and k-SemStamp

This is the repo for [SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation](https://arxiv.org/abs/2310.03991) (Accepted to NAACL 24) and [k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text](https://arxiv.org/abs/2402.11399) (Accepted to ACL 24).

SemStamp is a semantic watermark on Large Language Model(LLM) text generations to allow generated texts to be detected. SemStamp utilizes Locality-Sensitive Hashing (LSH) to partition the high-dimensional embedding space to produce sentence generations with LSH-hashes that follow a pseudo-randomly controlled sequence. During detection time, the algorithm analyzes the LSH-hashes of input sentences to see if they constitute a pseudo-random sequence, subsequently applying a z-test on the pseudo-randomness to determine if the text is watermarked. k-SemStamp is a simple yet effective variant of SemStamp, which has a similar setup but uses k-means clustering to partition the embedding space.

## SemStamp 
The high-level pipeline of SemStamp is outlined below
### Generation
1. Fine-tune a robust sentence embedder that encodes semantically similar sentences with sentence embeddings having high cosine similarities.
2. LSH partitions the embedding space through fixing random hyperplanes and assigning signatures of a vector based on the signs of its dot product with the hyperplanes.
3. Given input sentence $s_1$, repeat generating sentences $s_t$ until $LSH(s_t) \in valid(s_{t-1})$, $t=2,...$ and stop when the max_new_tokens is reached. The valid mask is controlled by the LSH signature of $s_{t-1}$.
### Detection
1. Attempt to remove sentence watermark through sentence-level paraphrasing.
2. Detect the sentences to see if $LSH(s_t) \in valid(s_{t-1})$, $t=2,...$
### Sample usage
1. create data/ directory and load c4_data, which would also create the human subset for evaluation: 
`python load_c4.py`
2. (Optional) fine-tune the sentence embedder or use a fine-tuned sentence embedder at AbeHou/SemStamp-booksum-sbert and AbeHou/SemStamp-c4-sbert
- fine-tune procedure: 
```
# 1. build a smaller-sized huggingface dataset of c4-train dataset with 'text' column (recommended size: 8k) and use the .save_to_disk() API
python build_subset.py data/c4-train --n 8000
# 2. paraphrase
python paraphrase_gen.py data/c4-train-8000
# 3. fine-tune
python finetune_embedder.py --model_name_or_path all-mpnet-base-v2 \
  --dataset_path data/c4-train-8000-pegasus-bigram=False-threshold=0.0 \
  --output_dir $OUTPUT_DIR --learning_rate 4e-5 --warmup_steps 50 \
  --max_seq_length 64 --num_train_epochs 3 --logging_steps 10 \
  --evaluation_strategy epoch --save_strategy epoch \
  --remove_unused_columns False --delta 0.8 --do_train --overwrite_output_dir
```

3. produce SemStamp generations:
```
# 1. build a smaller-sized hugginface dataset of c4-val dataset with 'text' column (e.g. 1k texts) and use the .save_to_disk() API
python build_subset.py data/c4-val --n 1000
# 2. sample
python sampling.py data/c4-val-1000 --model AbeHou/opt-1.3b-semstamp \
    --embedder OUTPUT_DIR_TO_YOUR_EMBEDDER --sp_mode lsh \
    --sp_dim 3 --delta 0.02
# note: it's recommended to use AbeHou/opt-1.3b-semstamp, which is fine-tuned with cross-entropy loss 
# to favor generations of shorter average sentence length, 
# so that the effect of watermarks is more pronounced.
# 3. paraphrase
python paraphrase_gen.py PATH_TO_GENERATED_DATA --model_path AbeHou/opt-1.3b-semstamp
# 4. detection
python detection.py PATH_TO_PARAPHRASED_DATA --detection_mode lsh --sp_dim 3 --embedder OUTPUT_DIR_TO_YOUR_EMBEDDER
```
**Note that if you use GPU to generate, you must use GPU to detect as well in order for the random seed to be consistent.**
Note that you are free to change the value of delta for your customized tradeoff of robustness and speed. (Higher delta means more strict rejections, thus more robust and slower. Lower delta is the other way around.)

## k-SemStamp Generation
1. Encode a corpus of texts belonging to a specific domain and obtain k-means clusters on the training embeddings
2. Given input sentence $s_1$, repeat generating sentences $s_t$ until $c(s_t) \in valid(s_{t-1})$, $t=2,...$ and stop when the max_new_tokens is reached. $c(s_t)$ returns the index of the closest cluster to $s_{t}$. The valid mask is controlled by $c(s_{t-1})$.
### Detection
The detection procedure is analogous to SemStamp except that $c(s_t)$ is used instead of $LSH(s_t)$
### Sample usage
Steps 1 and 2 are the same.

3. generate sentence embeddings to train kmeans clusters (multi-GPU supported).
  ```
  python sampling_kmeans_utils.py data/c4-train AbeHou/SemStamp-c4-sbert 8
  ```

4. produce k-SemStamp generations. 
  ```python build_subset.py data/c4-val --n 1000 
  python sampling.py data/c4-val-1000 --model AbeHou/opt-1.3b-semstamp --embedder AbeHou/SemStamp-c4-sbert --sp_mode kmeans --sp_dim 8 --delta 0.02 \
    --cc_path data/c4-train/cc.pt
  ```

5. detection:
    ```
    python detection.py path_to_your_generation --detection_mode kmeans --sp_dim 8 --embedder output_dir_to_your_embedder --cc_path to_your_kmeans_clusters
    ```
    Example usage to replicate results in k-SemStamp paper:
    ```
    python detection.py semstamp-data/c4-ksemstamp-pegasus/bigram=False --detection_mode kmeans --sp_dim 8 --embedder AbeHou/SemStamp-c4-sbert --cc_path centroids/c4-cluster_8_centers.pt
    ```
    





