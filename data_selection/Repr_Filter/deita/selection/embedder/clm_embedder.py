import torch
from functools import partial

from tqdm import tqdm
import torch
from datasets import Dataset
import torch.distributed as dist
from deita.selection.embedder.base import Embedder
from deita.selection.embedder.utils import DataCollatorForSupervisedDataset, preprocess

import logging

logger = logging.getLogger(__name__)

class CLM_Embedder(Embedder):

    def __init__(self, model_name_or_path, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
    
    def encode_samples(self, data):

        conversations = [item["conversations"] for item in data]
        dataset_buf, data_size = self.create_databuffer(conversations, sort_by_length = True)
        raw_dataset = Dataset.from_list(dataset_buf)

        preprocess_func = partial(preprocess, 
                                conv_template = self.conv_template,
                                only_answer = self.only_answer,
                                max_length = self.max_length,
                                tokenizer = self.tokenizer)
        
        with self.accelerator.main_process_first():
          tokenized_datasets = raw_dataset.map(
              preprocess_func,
              batched = True,
              num_proc = 32,
              remove_columns = ["conversations", "specific_length"],
              desc = "Tokenizing and reformatting instruction data"
          )  
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForSupervisedDataset(tokenizer = self.tokenizer)
        dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size = self.batch_size_per_device, collate_fn = data_collator)
        
        model, dataloader = self.accelerator.prepare(self.model, dataloader)
        
        all_embeddings_list = []
        
        total_samples = len(tokenized_datasets)
        total_batches = len(dataloader)
        last_batch_size = total_samples % self.minibatch_size if total_samples % self.minibatch_size != 0 else self.minibatch_size
        
        for b_idx, batch in enumerate(tqdm(dataloader, total = len(tokenized_datasets) // self.minibatch_size, disable = not self.accelerator.is_local_main_process)):
            
            model.eval()

            batch_idx = batch["idx"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            with torch.no_grad():
                outputs = model(
                    input_ids = batch["input_ids"].to(self.accelerator.device),
                    attention_mask = batch["attention_mask"].to(self.accelerator.device),
                    output_hidden_states = True,    # 改为 False，使用 last_hidden_state
                    use_cache = False,
                    return_dict = True
                )
            
                seq_len = attention_mask.sum(1, keepdim=True)  # on device
                last_hidden = outputs.hidden_states[-1]  # 只最后一层
                
                # 取每个序列的最后一个有效 token 的向量（如果 padding_side == "right"）
                if self.tokenizer.padding_side == "right":
                    idx = torch.arange(seq_len.size(0), device=last_hidden.device)  # (B,1,hidden)
                    # selected = last_hidden[idx, (seq_len - 1).squeeze(1), :]  # (B, hidden_dim)
                    selected = last_hidden[idx, seq_len - 1, :]
                elif self.tokenizer.padding_side == "left":
                    # 如果左填充，最后一个有效 token 始终是 -1
                    selected = last_hidden[:, -1, :]  # shape (B, hidden_dim)
                else:
                    raise ValueError("Invalid padding strategy")
                # detach and move to CPU (避免在GPU上长期占用)
                selected_cpu = selected.detach().cpu()  # tensor on CPU
                
            # if self.tokenizer.padding_side == "right":
            #     last_hidden_state = outputs.hidden_states[-1][torch.arange(seq_len.size(0))[:, None], seq_len - 1]
            # elif self.tokenizer.padding_side == "left":    
            #     last_hidden_state = outputs.hidden_states[-1][:, -1]
            # else:
            #     raise ValueError("Invalid padding strategy")
            
            if hasattr(self, "accelerator") and self.accelerator is not None:
                # accelerator.gather 会把不同进程的 tensor 收集到当前进程
                gathered = self.accelerator.gather(selected_cpu)  # CPU tensor or list depending on accelerator config
                gathered_idx = self.accelerator.gather(batch_idx.cpu())
                # gathered shape: (world_size * B_local, hidden_dim)
                # 下面将它们逐样本转换为 dict（只在主进程）
                if self.accelerator.is_local_main_process:
                    for emb_vec, s_id in zip(gathered.tolist(), gathered_idx.tolist()):
                        all_embeddings_list.append({"embedding": emb_vec, "idx": int(s_id)})
            else:
                # fallback: single process or torch.distributed
                if self.world_size > 1:
                    # 使用 all_gather 将CPU tensor打包为list
                    # 注意：all_gather要求事先分配目标tensor列表。这里简化：把每个进程先转为 numpy，再用 gather_object（小心CPU内存）
                    # 推荐在多卡下改用 accelerator.gather，如上
                    gathered_list = [torch.zeros_like(selected_cpu) for _ in range(self.world_size)]
                    dist.all_gather(gathered_list, selected_cpu, async_op=False)
                    if dist.get_rank() == 0:
                        # 仅主进程处理
                        for proc_tensor, s_id in zip(gathered_list, range(len(gathered_list))):
                            for emb_vec, sid in zip(proc_tensor.tolist(), batch_idx.tolist()):
                                all_embeddings_list.append({"embedding": emb_vec, "idx": int(sid)})
                else:
                    # 单进程，直接加入
                    for emb_vec, sid in zip(selected_cpu.tolist(), batch_idx.tolist()):
                        all_embeddings_list.append({"embedding": emb_vec, "idx": int(sid)})

            # 及时释放大变量
            del outputs, last_hidden, selected, selected_cpu
            torch.cuda.empty_cache()

            # sample_idx = batch_idx.tolist()
            # sample_dict = [{"embedding": lst_hs, "idx": s_id} for lst_hs, s_id in zip(last_hidden_state.tolist(), sample_idx)]
            
            # if(self.world_size > 1):
            #     all_process_embeddings = [[] for _ in range(self.world_size)]
            #     dist.gather_object(sample_dict, all_process_embeddings if dist.get_rank() == 0 else None, dst=0)
            # else:
            #     all_process_embeddings = [sample_dict]
            
            # if self.accelerator.is_local_main_process:
            #     if b_idx == total_batches - 1:
            #         for process_list in all_process_embeddings[:last_batch_size]:
            #             all_embeddings_list.extend(process_list)
            #     else:
            #         for process_list in all_process_embeddings:
            #             all_embeddings_list.extend(process_list)   
        
        return all_embeddings_list