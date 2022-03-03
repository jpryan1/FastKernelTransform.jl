import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 32

NUM_FKT_TRIALS = 2

kernel_name = "electrostatic"
svd_data = open("pyplots/data/relerr_vs_rank_svd_"+kernel_name+".txt", "r")
fkt_data = open("pyplots/data/relerr_vs_rank_fkt_"+kernel_name+".txt", "r")
nystrom_data = open("pyplots/data/relerr_vs_rank_nystrom_"+kernel_name+".txt", "r")

all_fkt_ranks = []
all_fkt_errs  = []
max_fkt_rank = 0

fkt_data_lines = fkt_data.readlines()
stride_size = int(len(fkt_data_lines)/NUM_FKT_TRIALS)
for i in range(NUM_FKT_TRIALS):
    start_idx = int(i*stride_size)
    fkt_ranks = []
    fkt_errs  = []
    for j in range(stride_size):
        line = fkt_data_lines[start_idx + j]
    # for line in fkt_data.readlines():
        datasplit = line.split(",")
        rank = int(datasplit[0])
        max_fkt_rank = max(rank, max_fkt_rank)
        relerr = float(datasplit[1])
        fkt_ranks.append(rank)
        fkt_errs.append(relerr)
    #endfor
    all_fkt_ranks.append(fkt_ranks)
    all_fkt_errs.append(fkt_errs)
#endfor


svd_ranks = []
svd_errs  = []
for line in svd_data.readlines():
    datasplit = line.split(",")
    rank = int(datasplit[0])
    relerr = float(datasplit[1])
    if rank > max_fkt_rank:
        break
    svd_ranks.append(rank)
    svd_errs.append(relerr)
#endfor


nrank_to_err = {}
for line in nystrom_data.readlines():
    datasplit = line.split(",")
    rank = int(datasplit[0])
    relerr = float(datasplit[1])
    if rank > max_fkt_rank:
        break
    if rank not in nrank_to_err:
        nrank_to_err[rank] = [relerr]
    else:
        nrank_to_err[rank].append(relerr)

#endfor

nystrom_ranks = []
nystrom_errs  = []

for rank in nrank_to_err:
    nystrom_ranks.append(rank)
    nystrom_errs.append(sum(nrank_to_err[rank])/len(nrank_to_err[rank]))

plt.figure(figsize=(12,8))
LW = 6
MS = 15
plt.semilogy(svd_ranks, svd_errs, linewidth=LW, label="Optimal", color="black")
for i in range(NUM_FKT_TRIALS):
    fkt_ranks = all_fkt_ranks[i]
    fkt_errs = all_fkt_errs[i]
    plt.semilogy(fkt_ranks, fkt_errs, marker='o', markersize=MS, linewidth=LW, linestyle='', label=str(i)+"HDF")
plt.semilogy(nystrom_ranks, nystrom_errs, marker='s', markersize=MS, linewidth=LW, linestyle='', label="Nystrom", color="green")
plt.legend(bbox_to_anchor=(1.02,1.0), framealpha=1)
plt.xlabel("Rank")
plt.ylabel("Relative error")
plt.xlim([0, max_fkt_rank])
plt.ylim([svd_errs[-1], 1])
if kernel_name == "matern15":
    plt.title(r"Matérn $\nu = 1.5$")
elif kernel_name == "matern25":
    plt.title(r"Matérn $\nu = 2.5$")
elif kernel_name == "cauchy":
    plt.title("Cauchy")
elif kernel_name == "gaussian":
    plt.title("Gaussian")

# plt.xlim((2e-5,2e-2))
# plt.ylim((2e-4,1e0))
plt.savefig("relerr_vs_rank_"+kernel_name+"_plot.pdf", format="pdf",  bbox_inches='tight')
plt.show()
