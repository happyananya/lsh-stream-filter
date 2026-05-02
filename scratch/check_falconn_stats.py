import falconn
import numpy as np

base = np.random.randn(100, 128).astype(np.float32)

params_cp = falconn.LSHConstructionParameters()
params_cp.dimension = 128
params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
params_cp.l = 10
params_cp.num_rotations = 1
params_cp.seed = 42
params_cp.num_setup_threads = 0
params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

falconn.compute_number_of_hash_functions(16, params_cp)
table = falconn.LSHIndex(params_cp)
table.setup(base)
qo = table.construct_query_object()

qo.find_nearest_neighbor(base[0])
stats = qo.get_query_statistics()

print("Stats Object attributes:")
for attr in dir(stats):
    if not attr.startswith('_'):
        print(" -", attr, ":", getattr(stats, attr))
