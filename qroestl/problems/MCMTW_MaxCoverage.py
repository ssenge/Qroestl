# from collections import Counter
# from dataclasses import dataclass
# from typing import List, Optional
# import numpy as np
# from dimod import ConstrainedQuadraticModel, Binary
# from qiskit_optimization import QuadraticProgram
#
# from qroestl.model import Model
# from qroestl.model.Model import Solution
# from qroestl.backends.Qiskit import QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible
# from qroestl.problems import MaxCoverage
# from qroestl.utils import Utils
# from qroestl.utils.Utils import union
#
# TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List[int]]
#
#
# @dataclass
# class Problem(Model.Problem[TCandidate], QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible):
#     name: str = 'Multi Cover Multi Type Weighted Max Set Coverage'
#     k: int = 1
#     #U: Optional[List[int]] = None
#     S: Optional[List[List[int]]] = None
#     W: Optional[List[int]] = None
#     C: Optional[List[int]] = None
#     T: Optional[List[int]] = None
#
#     def __post_init__(self) -> None:
#         self.S = np.array(self.S+[[]], dtype=object)
#         self.S = self.S[:-1]
#         self.U = list(dict.fromkeys(Utils.union_non_sorted(self.S)))
#         self.W = self.W if self.W is not None else [1]*len(self.U)
#         self.UW = dict(zip(self.U, self.W))
#         self.C = self.C if self.C is not None else [1]*len(self.U)
#         self.UC = dict(zip(self.U, self.C))
#         self.T = self.T if self.T is not None else [1]*len(self.S)
#         self.ST = dict(zip(map(lambda l: str(l),self.S), self.T))
#         self.n_cats = len(set(self.T))
#         self.TI = list(dict.fromkeys(self.T))
#         self.US = {u: [s for s in self.S if u in s] for u in self.U}
#         self.UT = {u: list(dict.fromkeys([self.ST[str(s)] for s in self.US[u]])) for u in self.U}
#
#         #for u, c in self.UC.items():
#         #    if c > 0:
#         #        print("Multi-Cover " + str(u) + " " + str(c) + "-times.")
#
#     def feasible(self, c: TCandidate) -> bool:
#         if len(c) == 0:
#             return False
#         cnt_S = Counter(Utils.union_non_sorted(np.take(self.S, c)))
#         # covered = True
#         # for kv in dict(zip(self.U, self.C)).items():
#         #     if cnt[kv[0]] < kv[1]:
#         #         covered = False
#         #         break
#         multi_covered = np.all(list(map(lambda kv: not cnt_S[kv[0]] < kv[1], self.UC.items()))) #dict(zip(self.U, self.C))
#         type_covered = True
#         for u in self.U: ## TODO: clean up
#             if len(list(dict.fromkeys([self.ST[str(s)] for s in [s for s in np.take(self.S, c) if u in s]]))) < self.UC[u]:
#                 type_covered = False
#                 break
#         return multi_covered and type_covered and len(c) <= self.k
#
#     def value(self, c: TCandidate) -> float:
#         return sum([self.UW[s] for s in Utils.union(self.S, c)])  # should be ok
#
#     def all_solutions(self) -> List[TCandidate]:
#         return Utils.powerset(range(len(self.S)))
#
#     def to_qiskit_qp(self) -> QuadraticProgram:
#         qp = QuadraticProgram('Multi Cover Multi Type Weighted Max Set Coverage')
#         # qp.continuous_var_list(len(self.S), name='s')  # relaxation
#         # qp.continuous_var_list(list(self.U), name='u')
#         qp.binary_var_list(len(self.S), name='s')
#         cnt=0
#         for u in self.U:  # TODO: clean up
#             if self.UC[u] > 0:
#                 for t in self.UT[u]:
#                     qp.binary_var(f'u{u}t{t}')
#                     cnt+=1
#
#         qp.binary_var_list(list(self.U), name='u')
#         qp.maximize(linear=([0]*len(self.S)) + [0]*cnt + self.W)
#         qp.linear_constraint(linear={f's{i}': 1 for i in range(len(self.S))}, sense='<=', rhs=self.k, name='max k sets')
#         for u in self.U:  # add cover constraints
#             qp.linear_constraint(linear={**{f's{i}': 1 for i in range(len(self.S)) if u in self.S[i]}, **{f'u{u}': -1}},
#                                  sense='>=', rhs=0, name=f'select s for u{u}')
#             # This implements a "must multi cover" policy
#             # "Should multi cover": u{u} coefficient needs to be -1*self.SC[u] above (and the 'multi cover u{u}' constraint below can be removed)
#             # Probably not required:
#             if self.UC[u] > 0:
#                 qp.linear_constraint(linear={**{f's{i}': 1 for i in range(len(self.S)) if u in self.S[i]}},
#                                      sense='>=', rhs=self.UC[u], name=f'multi cover u{u}')
#         for u in self.U:
#             if self.UC[u] > 0:
#                 qp.linear_constraint(linear={f'u{u}t{t}': 1 for t in self.UT[u]}, sense='>=', rhs=self.UC[u], name=f'min {self.UC[u]} types for u{u}')
#                 for t in self.UT[u]:
#                     qp.linear_constraint(linear={**{f's{j}': -1 for j in range(len(self.S)) if u in self.S[j] and self.ST[str(self.S[j])] == t},
#                                                  **{f'u{u}t{t}': 1}}, sense='<=', rhs=0, name=f'type cover u{u}t{t}')
#         print(qp.export_as_lp_string())
#         return qp
#
#     #def to_qubo(self) -> QuadraticProgram:
#     #    return self.to_qp()
#
#     def to_dwave_bqm(self):
#         # Derived (Dwave specific)
#         u_ub = {u: Binary(f'u{u}') for u in U}
#         ubs = u_ub.values()
#         s_sb = {s: Binary(f's{s}') for s in S}
#         sbs = s_sb.values()
#         u_ts = {u: list(set([T[s] for s in u_ss[u]])) for u in U}  # u -> types (of sets that cover u)
#         u_utbs = {u: [Binary(f'u{u}t{t}') for t in u_ts[u]] for u in U}
#         u_t_utb = {u: {t: Binary(f'u{u}t{t}') for t in u_ts[u]} for u in U}
#
#         cqm = ConstrainedQuadraticModel()
#         cqm.set_objective(-sum(ubs))  # max
#         cqm.add_constraint(sum(sbs) <= k, "Max k sets")
#         for u in U:
#             cqm.add_constraint(sum([s_sb[s] for s in u_ss[u]]) - u_ub[u] >= 0, f'select sets for u{u}')
#             if (c := C[u]) > 0:
#                 cqm.add_constraint(sum([s_sb[s] for s in u_ss[u]]) >= c, f'multi cover u{u}')
#                 cqm.add_constraint(sum(u_utbs[u]) >= c, f'min {c} types for u{u}') #tb[us[ui]]
#                 for t in u_ts[u]:
#                     cqm.add_constraint(-sum([s_sb[s] for s in u_ss[u] if T[s] == t]) + u_t_utb[u][t] <= 0, f'type cover u{u}t{t}')
#         return cqm
#
#
# @dataclass
# class Greedy(Backend[TCandidate, Problem]):
#     name: str = "Greedy"
#
#     def solve_(self, p: Problem, s_=Solution[TCandidate, Problem]()) -> Solution[TCandidate, Problem]:
#         sol = []
#         multi_covers = {u: [] for u in p.U}
#         type_covers = {u: [] for u in p.U}
#         C = p.C
#         TS = {t: [] for t in list(dict.fromkeys(p.T))}
#         for i, t in enumerate(p.T):
#             TS.update({t: TS[t]+[i]})
#         for k in range(p.k):
#             UC = dict(zip(p.U, C))
#             ucs = [UC[u] for u in p.U] # p.C
#             if np.all(np.array(ucs) <= 0):
#                 sol = self.phase2(p, sol)
#                 break
#             #ucs_max = np.max(ucs)
#             #if ucs_max < 0: ## eventuell nicht benötigt, weil man ja greedy weitersammeln kann, wenn noch ks übrig sind
#             #    break
#             u = p.U[np.argmax(ucs)]
#             S = [s for s in np.delete(p.S, union(multi_covers[u] + [TS[t] for t in type_covers[u]] + sol)) if u in s]
#
#             # No solution possible (note that still universe elements need to be covered)
#             if len(S) == 0:
#                 break
#             self.phase1(C, S, multi_covers, p, sol, type_covers)
#             return s_.eval(p, sol)
#
#     def phase1(self, C, S, multi_covers, p, sol, type_covers):
#         s_opt = S[np.argmax([np.sum([p.UW[u] for u in s]) for s in S])]
#         s_opt_idx = Utils.where(p.nS, s_opt)
#         sol.append(s_opt_idx)
#         for x in s_opt:
#             C[Utils.where(p.nU, x)] -= 1
#             multi_covers.update({x: multi_covers[x] + [s_opt_idx]})
#             type_covers.update({x: type_covers[x] + [p.T[s_opt_idx]]})
#
#     def phase2(self, p, sol):
#         k_p = p.k - len(sol)
#         S_p = np.delete(p.nS, sol)
#         W_p = [p.UW[s] for s in Utils.union(S_p)]
#         p_p = MaxCoverage.Problem(k=k_p, S=list(S_p), W=W_p)
#         r = MaxCoverage.Greedy().solve(p_p)
#         add = []
#         for i in r.best[0]:
#             a = Utils.where(p.nS, S_p[i])
#             add.append(a)
#         sol += add
#         return sol

    # def solve_old(self, p: Problem, s=Solution[TCandidate, Problem]()) -> Solution[TCandidate, Problem]:
    #
    #     def rec(max_s=-1, i=min(len(p.S), p.k), #U=p.U,
    #             S=p.S, W=p.W, C=p.C, T=p.T, sol=[]) -> TCandidate:
    #
    #         find = lambda elems, s: [np.where(s==e) for e in elems]
    #         set_new_C = lambda s: C if np.any(C) else [1]*len(C)
    #
    #         max_s = np.argmax([Problem(#U=p.U,
    #             S=list(S), W=np.multiply(W, set_new_C(C)), C=C, T=T).cost([s]) for s in range(len(S))])
    #
    #         U = np.setdiff1d(p.U, S[max_s])
    #         U_diff = np.setdiff1d(p.U, U)
    #         U_pos = find(U_diff, p.U)
    #         return sol if i == 0 else rec(
    #             max_s=max_s,
    #             i=i-1,
    #             #U=np.setdiff1d(U, S[max_s]),
    #             S=np.delete(S, max_s),
    #             W=np.delete(W, U_pos),#max_s),
    #             C=np.delete(C, U_pos),#C,
    #             T=T,#np.delete(W, max_s),
    #             sol=sol + [Utils.where(p.S, S[max_s])])
    #
    #     return s.eval(p, rec())