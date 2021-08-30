from collections import Counter
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from qiskit_optimization import QuadraticProgram

from qroestl.model import Model
from qroestl.model.Model import QPConvertible, QuboConvertible, OperatorConvertible, Solver, Solution
from qroestl.problems import MaxCoverage
from qroestl.utils import Utils

TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List[int]]


@dataclass
class Problem(Model.Problem[TCandidate], QPConvertible, QuboConvertible, OperatorConvertible):
    name: str = 'Multi Cover Multi Type Weighted Max Set Coverage'
    k: int = 1
    #U: Optional[List[int]] = None
    S: Optional[List[List[int]]] = None
    W: Optional[List[int]] = None
    C: Optional[List[int]] = None
    T: Optional[List[int]] = None

    def __post_init__(self) -> None:
        self.S = np.array(self.S+[[]], dtype=object)
        self.S = self.S[:-1]
        self.U = list(dict.fromkeys(Utils.union_non_sorted(self.S)))
        self.W = self.W if self.W is not None else [1]*len(self.U)
        self.UW = dict(zip(self.U, self.W))
        self.C = self.C if self.C is not None else [1]*len(self.U)
        self.UC = dict(zip(self.U, self.C))
        self.T = self.T if self.T is not None else [1]*len(self.S)
        self.ST = dict(zip(map(lambda l: str(l),self.S), self.T))
        self.n_cats = len(set(self.T))
        self.TI = list(dict.fromkeys(self.T))
        self.US = {u: [s for s in self.S if u in s] for u in self.U}
        self.UT = {u: list(dict.fromkeys([self.ST[str(s)] for s in self.US[u]])) for u in self.U}

    def feasible(self, c: TCandidate) -> bool:
        if len(c) == 0:
            return False
        cnt_S = Counter(Utils.union_non_sorted(np.take(self.S, c)))
        # covered = True
        # for kv in dict(zip(self.U, self.C)).items():
        #     if cnt[kv[0]] < kv[1]:
        #         covered = False
        #         break
        multi_covered = np.all(list(map(lambda kv: not cnt_S[kv[0]] < kv[1], self.UC.items()))) #dict(zip(self.U, self.C))
        type_covered = True
        for u in self.U: ## TODO: clean up
            if len(list(dict.fromkeys([self.ST[str(s)] for s in [s for s in np.take(self.S, c) if u in s]]))) < self.UC[u]:
                type_covered = False
                break
        return multi_covered and type_covered and len(c) <= self.k

    def cost(self, c: TCandidate) -> float:
        return sum([self.UW[s] for s in Utils.union(self.S, c)])  # should be ok

    def all_solutions(self) -> List[TCandidate]:
        return Utils.powerset(range(len(self.S)))

    def to_qp(self) -> QuadraticProgram:
        qp = QuadraticProgram('Multi Cover Multi Type Weighted Max Set Coverage')
        # qp.continuous_var_list(len(self.S), name='s')  # relaxation
        # qp.continuous_var_list(list(self.U), name='u')
        qp.binary_var_list(len(self.S), name='s')
        cnt=0
        for u in self.U:  # TODO: clean up
            if self.UC[u] > 0:
                for t in self.UT[u]:
                    qp.binary_var(f'u{u}t{t}')
                    cnt+=1

        qp.binary_var_list(list(self.U), name='u')
        qp.maximize(linear=([0]*len(self.S)) + [0]*cnt + self.W)
        qp.linear_constraint(linear={f's{i}': 1 for i in range(len(self.S))}, sense='<=', rhs=self.k, name='max k sets')
        for u in self.U:  # add cover constraints
            qp.linear_constraint(linear={**{f's{i}': 1 for i in range(len(self.S)) if u in self.S[i]}, **{f'u{u}': -1}},
                                 sense='>=', rhs=0, name=f'select s for u{u}')
            # This implements a "must multi cover" policy
            # "Should multi cover": u{u} coefficient needs to be -1*self.SC[u] above (and the 'multi cover u{u}' constraint below can be removed)
            # Probably not required:
            if self.UC[u] > 0:
                qp.linear_constraint(linear={**{f's{i}': 1 for i in range(len(self.S)) if u in self.S[i]}},
                                     sense='>=', rhs=self.UC[u], name=f'multi cover u{u}')
        for u in self.U:
            if self.UC[u] > 0:
                qp.linear_constraint(linear={f'u{u}t{t}': 1 for t in self.UT[u]}, sense='>=', rhs=self.UC[u], name=f'min {self.UC[u]} types for u{u}')
                for t in self.UT[u]:
                    qp.linear_constraint(linear={**{f's{j}': -1 for j in range(len(self.S)) if self.ST[str(self.S[j])] == t},
                                                 **{f'u{u}t{t}': 1}}, sense='<=', rhs=0, name=f'type cover u{u}t{t}')
        #print(qp.export_as_lp_string())
        return qp

    def to_qubo(self) -> QuadraticProgram:
        return self.to_qp()


@dataclass
class Greedy(Solver[TCandidate, Problem]):
    name: str = "Greedy"


    # Select u in U with highest c in C >= 0 (else return sol)
    # Select all s in S with u in s (else return sol)
    # Remove s which are already covering u (else return sol) TBD: similar for type
    # Calulate sum w for remaining s
    # s* = Select highest s (wrt to sum w)
    # Update:
    #   sol += s*
    #   C -= s* : c_e-1 for all e in s*

    ### Biggest mess ever TODO: clean up
    def solve_(self, p: Problem, s_=Solution[TCandidate, Problem]()) -> Solution[TCandidate, Problem]:
        sol = []
        multi_covers = {u: [] for u in p.U}
        type_covers = {u: [] for u in p.U}
        UC = p.UC
        C = p.C
        TS = {t: [] for t in list(dict.fromkeys(p.T))}
        for i, t in enumerate(p.T):
            TS.update({t:TS[t]+[i]})
        for k in range(p.k):
            ucs = [UC[u] for u in p.U] # p.C
            if np.all(np.array(ucs)<=0):
                k_p = p.k - len(sol)
                S_p = np.delete(p.S, sol)
                W_p = [p.UW[s] for s in Utils.union(S_p)]
                p_p = MaxCoverage.Problem(k=k_p, S=list(S_p), W=W_p)
                r = MaxCoverage.Greedy().solve(p_p)
                add = []
                for i in r.best[0]:
                    a = Utils.where(p.S, S_p[i])
                    add.append(a)
                sol += add
                break
            ucs_max = np.max(ucs)
            if ucs_max < 0:
                break
            am = np.argmax(ucs)
            hu = p.U[am]
            s_all = p.S
            tcs = [TS[tc] for tc in type_covers[hu]]
            rem = multi_covers[hu] + tcs
            rem = Utils.union(rem) if len(rem) > 0 else []
            s_rem = np.delete(s_all, rem)
            s_rem = [s for s in s_rem if hu in s]

            if len(s_rem) == 0:
                break
            w_sums = [np.sum([p.UW[u] for u in s]) for s in s_rem]
            s_opt = np.argmax(w_sums)
            s_opt_x = s_rem[s_opt]
            s_opt = Utils.where(p.S, s_opt_x)
            sol.append(s_opt)
            multi_covers.update({hu:multi_covers[hu]+[s_opt]})
            type_covers.update({hu:type_covers[hu]+[p.T[s_opt]]})
            for s in s_opt_x:
                C[Utils.where(p.U, s)] -= 1
            UC = dict(zip(p.U, C))
        return s_.eval(p, sol)



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