
import faiss
p="/tmp/index_smoke_2k/ivf.index"
idx=faiss.read_index(p)
print("ok", p, "ntotal=", idx.ntotal, "d=", idx.d, "trained=", idx.is_trained)
