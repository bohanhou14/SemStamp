# SemStamp and k-SemStamp

This is the repo for [SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation](https://arxiv.org/abs/2310.03991) and [k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text](https://arxiv.org/abs/2402.11399).

SemStamp is a semantic watermark on Large Language Model(LLM) text generations to allow generated texts to be detected. SemStamp utilizes Locality-Sensitive Hashing (LSH) to partition the high-dimensional embedding space to produce sentence generations with LSH-hashes that follow a pseudo-randomly controlled sequence. During detection time, the algorithm analyzes the LSH-hashes of input sentences to see if they constitute a pseudo-random sequence, subsequently applying a z-test to determine if the text is watermarked. k-SemStamp is a simple yet effective variant of SemStamp, which has a similar setup but uses k-means clustering to partition the embedding space.

The high-level pipeline of SemStamp generation is: 
## SemStamp 
### Generation
1. Fine-tune a robust sentence embedder that encodes semantically similar sentences with sentence embeddings having high cosine similarities.
2. LSH partitions the embedding space through fixing random hyperplanes and assigning signatures of a vector based on the signs of its dot product with the hyperplanes.
3. Given input sentence $s_1$, repeat generating sentences $s_t$ until $LSH(s_t) \in validmask(s_{t-1})$, $t=2,...$ and stop when the max_new_tokens is reached. The valid mask is controlled by the LSH signature of $s_{t-1}$.
### Detection
1. Attempt to remove sentence watermark through sentence-level paraphrasing.
2. Detect the sentences to see if $LSH(s_t) \in validmask(s_{t-1})$, $t=2,...$
### Sample usage
1. create data/ directory and load c4_data: 
    python load_c4.py 
2. (Optional) fine-tune the sentence embedder or use a fine-tuned sentence embedder at AbeHou/c4-embedder-semstamp
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
    --embedder output_dir_to_your_embedder --sp_mode lsh \ 
    --sp_dim 3 --delta 0.01
# note: it's recommended to use AbeHou/opt-1.3b-semstamp, which is fine-tuned with cross-entropy loss to favor generations of shorter average sentence length, so that the # watermark is more pronounced.
# 3. detection
python detection.py path_to_your_generation --detection_mode lsh --sp_dim 3 --embedder output_dir_to_your_embedder 
**Note that if you use GPU to generate, you must use GPU to detect as well in order for the random seed to be consistent.**
```
Note that you are free to change the value of delta for your customized tradeoff of robustness and speed. (Higher delta means more strict rejections, thus more robust and slower. Lower delta is the other way around.)

### k-SemStamp Generation
1. Encode a corpus of texts belonging to a specific domain and obtain k-means clusters on the training embeddings
2. Given input sentence $s_1$, repeat generating sentences $s_t$ until $c(s_t) \in valid_mask(s_{t-1})$, $t=2,...$ and stop when the max_new_tokens is reached. $c(s_t)$ returns the index of the closest cluster to $s_{t}$. The valid mask is controlled by $c(s_{t-1})$.
### Detection
The detection procedure is analogous to SemStamp except that $c(s_t)$ is used instead of $LSH(s_t)$
### Sample usage
Steps 1 and 2 are the same.
3. produce k-SemStamp generations
    python build_subset.py data/c4-val --n 1000
    python sampling.py --model AbeHou/opt-1.3b-semstamp --embedder output_dir_to_your_embedder --sp_mode kmeans --sp_dim 8 --delta 0.02
4. detection:
    python detection.py path_to_your_generation --detection_mode kmeans --sp_dim 8 --embedder output_dir_to_your_embedder --cc_path to_your_kmeans_clusters

### Future works
We are exploring a parallel implementation and also vLLM integrations to speedup SemStamp generations.

