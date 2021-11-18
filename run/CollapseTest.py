from qroestl.problems import MCMTWB_k_MaxCover

percent = lambda W, G:  0 if G == 0 else W * 100 / G

nU_total = 0
nU_collapsed_total = 0
d_total = 0
for i in range(1):
    p = MCMTWB_k_MaxCover.gen_syn_random(5, 2, 1)
    p_collapsed = p.collapse()
    d = p.nU - p_collapsed.nU
    nU_total += p.nU
    nU_collapsed_total += p_collapsed.nU
    d_total += d
    p_stats = p.stats()
    p_collapsed_stats = p_collapsed.stats()
    print(p)
    print(f'Problem Stats -> Avg Set Size: {p_stats[0]} Avg Hits: {p_stats[1]} Total Single Us: {p_stats[2]}')
    print(p_collapsed)
    print(f'Problem Collapsed Stats -> Avg Set Size: {p_collapsed_stats[0]} Avg Hits: {p_collapsed_stats[1]} Total Single Us: {p_collapsed_stats[2]}')
    print(f'{i} -> #U: {p.nU} #U_collapsed: {p_collapsed.nU} Reduction: {d} ({percent(p.nU, d)}%)')
    print()

print(f'Total #U: {nU_total} Total #U_collapsed: {nU_collapsed_total} Total Reduction: {d_total} ({percent(nU_total, d_total)}%)')