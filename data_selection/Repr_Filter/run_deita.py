from deita.pipeline import Pipeline


data_name = 'SFT_2'
# data_path = f'/mnt/disk4t/heyuxuan/work/sce/data/SFT/{data_name}.json'
data_path = f'/mnt/disk4t/heyuxuan/work/sce/data_selection/Repr_Filter/{data_name}_quality.json'
comlexity_scorer = '/mnt/disk4t/heyuxuan/work/sce/data_selection/Repr_Filter/scorer/deita-complexity-scorer'
quality_scorer = '/mnt/disk4t/heyuxuan/work/sce/data_selection/Repr_Filter/scorer/deita-quality-scorer'
# output_path = f'/mnt/disk4t/heyuxuan/work/sce/data_selection/Repr_Filter/{data_name}_complexity.json'
output_path = f'/mnt/disk4t/heyuxuan/work/sce/data_selection/Repr_Filter/{data_name}_complexity_quality.json'
score_pipeline = Pipeline("score_pipeline", 
                     data_path = data_path,   # json file with sharegpt format
                     scorer = 'llama',   # [mistral, llama]
                     scorer_name_or_path = comlexity_scorer,  # scorer name or path e.g. hkust-nlp/deita-complexity-scorer
                     is_vllm = True,  # launch with vllm [True, False]
                     score_type = 'quality', # [complexity, quality]
                     output_path = output_path)  # output path (json format)
score_pipeline.run()

threshold = 0.92
data_size = 10000
data_path = '/share/project/wuhaiming/spaces/sce/data_selection/Repr_Filter/SFT_complexity_quality.json'
embed_path = '/share/project/wuhaiming/spaces/sce/output/feature/qSFT.pkl'
selected_path = '/share/project/wuhaiming/spaces/sce/data_selection/Qwen3_SFT_10000_Repr.json'
filter_pipeline = Pipeline("filter_pipeline", 
                         data_path = data_path,  # json file with sharegpt format
                         other_data_path = embed_path,  # embedding file path (pickle format)
                         threshold = threshold,  # filter threshold default: 0.9 
                         data_size = data_size,  # size of selected data
                         chunk_size = 100000,  # used for more efficient GPU computing  default: 100000
                         sort_key = "complexity_scores,quality_scores",  # default: "complexity_scores,quality_scores"
                         output_path = selected_path,  # json format output path
                         distance_metric = 'cosine',  # default: cosine
                         embedding_field = 'embedding',  # default: embedding
                         is_compression = False,  # default: False
                         normalize_emb = True,
                         batch_size = 32,
                         device = 0  # GPU IDX, default: 0
                         )
filter_pipeline.run()

# llama: 0.67
# huggingface-hub==0.17.3
# safetensors==0.4.0
# tokenizers==0.14.1
# torch==2.1.0
# transformers==4.35.1
# typing-extensions==4.8.0
