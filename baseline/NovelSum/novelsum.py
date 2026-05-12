import json
import numpy as np
import pathlib
import csv
import torch
import pickle
from utils import *
from tqdm import tqdm
from typing import List, Dict
import argparse
from sklearn.preprocessing import normalize

def compute_cos_distance(data: np.ndarray, device: torch.device) -> np.ndarray:
    list_a_tensor = torch.tensor(data, device=device).float()
    num_vectors = list_a_tensor.shape[0]
    block_size = 500 # Adjust this based on your GPU memory
    cosine_distance_matrix = torch.zeros((num_vectors, num_vectors), device=device)

    for i in range(0, num_vectors, block_size):
        for j in range(0, num_vectors, block_size):
            end_i = min(i + block_size, num_vectors)
            end_j = min(j + block_size, num_vectors)
            block_i = list_a_tensor[i:end_i]
            block_j = list_a_tensor[j:end_j]

            cosine_similarity_block = torch.nn.functional.cosine_similarity(
                block_i.unsqueeze(1), block_j.unsqueeze(0), dim=2
            )
            cosine_distance_block = 1 - cosine_similarity_block
            cosine_distance_matrix[i:end_i, j:end_j] = cosine_distance_block
    cosine_distance = cosine_distance_matrix.cpu().numpy()
    return cosine_distance

def weighted_average(row: np.ndarray, power: float = 1.0) -> float:
    sorted_indices = np.argsort(row)
    weights = 1 / np.power(np.arange(1, len(row) + 1), power)
    sorted_row = row[sorted_indices]
    return np.average(sorted_row, weights=weights)

def novelsum(distance_matrix: np.ndarray, densities: np.ndarray, power: float = 1.0) -> tuple[np.ndarray, float]:
    weighted_matrix = distance_matrix * densities[:, np.newaxis]
    weighted_averages = np.apply_along_axis(weighted_average, 1, weighted_matrix, power=power)
    return weighted_averages, np.mean(weighted_averages)

def process_model(dataset_path: str, dataset_name: str, faiss_index: FaissIndex, device: torch.device,
                  density_powers: List[float], neighbors: List[int], distance_powers: List[float]) -> Dict:
    # dataset = load_data(dataset_path)
    features = read_feature(f'../../output/feature/{dataset_name}.db')
    embeddings = [features[i]['embedding'] for i in range(len(features))]
    # dataset = normalize(embeddings)
    dataset = np.array(embeddings)
    print(dataset.shape)
    # print(dataset)
    cosine_distances = compute_cos_distance(dataset, device)

    results = {'Dataset Name': dataset_name}
    novelty = {}
    total_combinations = len(density_powers) * len(neighbors) * len(distance_powers)
    pbar = tqdm(total=total_combinations, desc=f"  Computing metrics for {dataset_name}", leave=False)
    
    for dp in density_powers:
        for nb in neighbors:
            current_densities = faiss_index.local_density(dataset, n_neighbors=nb, power=dp, regularization=1e-9)
            for distp in distance_powers:
                weighted_averages, weighted_col_avg = novelsum(cosine_distances, current_densities, power=distp)
                key = f'neighbor_{nb}_density_{dp}_distance_{distp}'
                novelty[key] = weighted_averages.tolist()
                results[key] = weighted_col_avg
                pbar.update(1)
    
    pbar.close()
    results['cos_distance'] = np.mean(cosine_distances)
    return results, novelty

def write_results_to_csv(all_results: List[Dict], output_csv: str):
    if all_results:
        keys = list(all_results[0].keys())
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=keys)
            csv_writer.writeheader()
            csv_writer.writerows(all_results)
        print(f"Results written to {output_csv}")
    else:
        print("No results to write.")

def main():
    parser = argparse.ArgumentParser(description="Compute NovelSum for datasets.")
    parser.add_argument('--single_dataset_path', type=str, default=None, 
                        help="Single path to the dataset to be calculated (optional).")
    parser.add_argument('--multi_datasets_dir', type=str, default=None, 
                        help="Directory containing multiple datasets to be calculated (optional).")
    parser.add_argument('--dense_ref_dir', type=str, required=True, 
                        help="Directory containing dense reference data JSON files.")
    parser.add_argument('--output_csv', type=str, default=None, 
                        help="Output CSV file to store results.")
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help="GPU ID to use (default: 0).")
    parser.add_argument('--density_powers', type=float, nargs='*', 
                        default=[0, 0.25, 0.5], 
                        help="List of density powers to use (default: [0, 0.25, 0.5]).")
    parser.add_argument('--neighbors', type=int, nargs='*', 
                        default=[5, 10], 
                        help="List of neighbors to use (default: [5, 10]).")
    parser.add_argument('--distance_powers', type=float, nargs='*', 
                        default=[0, 1, 2], 
                        help="List of distance powers to use (default: [0, 1, 2]).")
    args = parser.parse_args()
    
    # if not args.single_dataset_path and not args.multi_datasets_dir:
    #     parser.error("At least one of --single_dataset_path or --multi_datasets_dir must be provided.")

    # Load dense reference data
    print(f"Loading dense reference data from {args.dense_ref_dir}...")
    # dense_ref_data = load_dir_data(args.dense_ref_dir, desc="Loading reference data")
    features = read_feature(f'../sce/output/feature/{args.dense_ref_dir}.db')
    embeddings = [features[i]['embedding'] for i in range(len(features))]
    dense_ref_data = np.array(embeddings)
    print(dense_ref_data)
    # embeddings = normalize(embeddings)
    # Initialize Faiss index and device
    device = set_device(args.gpu_id)
    print('Using device', device)
    faiss_index = FaissIndex(dense_ref_data, args.gpu_id)
    all_results = []
    data_list = [args.single_dataset_path]#, 'Llama31_kmeans', 'Llama31_kcenter', 'Llama31_repr', 'Llama31_random']
    for dataset_name in data_list:
        print(f"Processing single dataset: {dataset_name}")
        results, novelties = process_model(dataset_name, dataset_name, faiss_index, device, 
                            args.density_powers, args.neighbors, args.distance_powers)
        # all_results.append(results)
        write_results_to_csv([results], f'results/{dataset_name}.csv')
        with open(f'{dataset_name}_novelties.json','w', encoding='utf-8') as f:
            json.dump(novelties, f, ensure_ascii=False, indent=2)
    
    # Process single dataset if provided
    # if args.single_dataset_path and pathlib.Path(args.single_dataset_path).exists():
    #     # dataset_name = pathlib.Path(args.single_dataset_path).stem
    #     dataset_name = args.single_dataset_path
    #     print(f"Processing single dataset: {dataset_name}")
    #     results = process_model(args.single_dataset_path, dataset_name, faiss_index, device, 
    #                           args.density_powers, args.neighbors, args.distance_powers)
    #     all_results.append(results)
    
    # # Process multiple datasets from directory if provided
    # if args.multi_datasets_dir and pathlib.Path(args.multi_datasets_dir).is_dir():
    #     dataset_paths = list(pathlib.Path(args.multi_datasets_dir).glob("*.json"))
    #     print(f"Found {len(dataset_paths)} datasets in directory")        
    #     for dataset_path in tqdm(dataset_paths, desc="Processing datasets"):
    #         dataset_name = dataset_path.stem
    #         print(f"  Processing: {dataset_name}")
    #         results = process_model(str(dataset_path), dataset_name, faiss_index, device, 
    #                              args.density_powers, args.neighbors, args.distance_powers)
    #         all_results.append(results)
    
    # if not all_results:
    #     print("No valid datasets found!")
    #     return
        
    # # Write results to CSV
    # write_results_to_csv(all_results, args.output_csv)

if __name__ == "__main__":
    main()

# python novelsum.py --single_dataset_path data/MMLU_100_MedQA_900_1/embedding/01.json  --dense_ref_dir data/MMLU_100_MedQA_900_1/embedding/ --gpu_id 0 --density_powers 0 --output_csv MMLU_100_MedQA_900_1.csv
