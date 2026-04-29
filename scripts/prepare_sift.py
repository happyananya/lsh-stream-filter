import os
import h5py
import numpy as np

def main():
    print("Extracting SIFT1M dataset from HDF5...")
    
    # Path to the downloaded ann-benchmarks hdf5 file
    hdf5_path = "sift-128-euclidean.hdf5"
    if not os.path.exists(hdf5_path):
        print(f"Error: {hdf5_path} not found in the root directory.")
        return
        
    output_dir = "data/sift1m"
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        print("Keys in HDF5:", list(f.keys()))
        
        # 'train' contains the base vectors
        base = np.array(f['train'])
        print(f"Extracted base shape: {base.shape}, dtype: {base.dtype}")
        np.save(os.path.join(output_dir, "base.npy"), base)
        
        # 'test' contains the query vectors
        query = np.array(f['test'])
        print(f"Extracted query shape: {query.shape}, dtype: {query.dtype}")
        np.save(os.path.join(output_dir, "query.npy"), query)
        
        # 'neighbors' contains the ground truth top-k indices
        groundtruth = np.array(f['neighbors'])
        print(f"Extracted groundtruth shape: {groundtruth.shape}, dtype: {groundtruth.dtype}")
        np.save(os.path.join(output_dir, "groundtruth.npy"), groundtruth)
        
    print(f"SIFT1M dataset successfully prepared and saved to {output_dir}")

if __name__ == "__main__":
    main()
