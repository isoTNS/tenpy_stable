# Copyright 2022 TeNPy Developers, GNU GPLv3

import numpy as np
import time
import warnings
import logging
import copy
logger = logging.getLogger(__name__)

from ..linalg import np_conserved as npc
from ..networks.mpo import MPOEnvironment, MPOTransferMatrix
from ..linalg.lanczos import LanczosGroundState
from ..linalg.sparse import NpcLinearOperator, SumNpcLinearOperator, BoostNpcLinearOperator, ShiftNpcLinearOperator
from ..tools.params import asConfig
from ..tools.math import entropy
from ..algorithms.algorithm import Algorithm
from ..algorithms.mps_common import ZeroSiteH
#from ..tools.misc import find_subclass
#from ..tools.process import memory_usage
#from .mps_common import Sweep, ZeroSiteH, OneSiteH

__all__ = ['TR_general', 'LT_general', 'construct_orthogonal', 'PlaneWaveExcitations', ]

#---------------------------------------------
def TR_general(As, Bs, R, Ws=None):
    '''
      --A1--A2--|
        |   |   |
      --W1--W2--R
        |   |   |
      --B1--B2--|

        p*
        |
    wL--W--wR
        |
        p
    '''
    #print("As[0] shape:", As[0].shape)
    #print("As[0] label:", As[0].labels)
    temp = R.copy()
    for i in reversed(range(len(As))):
        temp = npc.tensordot(Bs[i].conj(), temp, axes=(['vR*'], ['vL*']))
        if Ws is not None:
            temp = npc.tensordot(Ws[i], temp, axes=(['wR', 'p'], ['wL', 'p*']))
        temp = npc.tensordot(As[i], temp, axes=(['vR', 'p'], ['vL', 'p*']))
    return temp

#---------------------------------------------
def LT_general(As, Bs, L, Ws=None):
    temp = L.copy()
    for i in range(len(As)):
        temp = npc.tensordot(temp, Bs[i].conj(), axes=(['vR*'], ['vL*']))
        if Ws is not None:
            temp = npc.tensordot(temp, Ws[i], axes=(['wR', 'p*'], ['wL', 'p']))
        temp = npc.tensordot(temp, As[i], axes=(['vR', 'p*'], ['vL', 'p']))
        #temp.itranspose(['vR*', 'wR', 'vR'])
    return temp
#---------------------------------------------
def form_X(As, L, R, Ws=None):
    temp = L.copy()
    for i in range(len(As)):
        temp = npc.tensordot(temp, As[i], axes=(['vR'], ['vL']))
        if Ws is not None:
            temp = npc.tensordot(temp, Ws[i], axes=(['wR', 'p'], ['wL', 'p*']))
        temp.ireplace_label('p', 'p'+str(i))
    return npc.tensordot(temp, R, axes=(['vR', 'wR'], ['vL', 'wL']))
#---------------------------------------------
def construct_orthogonal(M, cutoff=1.e-13, left=True):
    '''
    vL--M--vR
        |
        p
    '''
    if left:
        M = M.copy().combine_legs([['vL', 'p'], ['vR']], qconj=[+1, -1])
        Q = npc.orthogonal_columns(M, 'vR')
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(vL.p)'], ['(vL*.p*)']))) < 1.e-14
    else:
        M = M.copy().combine_legs([['vL'], ['p', 'vR']], qconj=[+1, -1])
        Q = npc.orthogonal_columns(M.transpose(['(p.vR)', '(vL)']), 'vL').itranspose(['vL', '(p.vR)'])
        assert npc.norm(npc.tensordot(Q, M.conj(), axes=(['(p.vR)'], ['(p*.vR*)']))) < 1.e-14
    return Q.split_legs()

#---------------------------------------------
class PlaneWaveExcitations(Algorithm):
    def __init__(self, psi, model, options, **kwargs):
        #options = asConfig(options, self.__class__.__name__)
        super().__init__(psi, model, options, **kwargs)
        #self.L = unit cell size
        assert self.psi.L == self.model.H_MPO.L
        self.L = self.psi.L

        self.n = 2 # number of unit cells one works with in the PW eigenvalue problem

        self.ALs = [self.psi.get_AL(i) for i in range(self.L)]
        self.ARs = [self.psi.get_AR(i) for i in range(self.L)]
        self.ACs = [self.psi.get_AC(i) for i in range(self.L)]
        self.Cs = [self.psi.get_C(i) for i in range(self.L)] # C on the left
        self.H = self.model.H_MPO
        self.Ws = [self.H.get_W(i) for i in range(self.L)]
        if len(self.Ws) < len(self.ALs):
            assert len(self.ALs) % len(self.Ws)
            self.Ws = self.Ws * len(self.ALs) // len(self.Ws)

        self.IdL = self.H.get_IdL(0)
        self.IdR = self.H.get_IdR(-1)
        print("self.IdR:", self.IdR)
        print("self.IdL:", self.IdL)
        #self.guess_init_env_data = options.get('init_data', {'init_RP': npc.tensordot(self.C, self.C.conj(), axes=(['vR', 'vL*'])) ,
        #                                                    'init_LP': npc.tensordot(self.C.conj(), self.C, axes=(['vR*', 'vR']))})
        self.guess_init_env_data = self.options.get('init_data',None)
        self.dW = self.Ws[0].get_leg('wR').ind_len # MPO dimension [TODO] this assumes a single site
        self.chi = self.ALs[0].get_leg('vL').ind_len
        self.d = self.ALs[0].get_leg('p').ind_len

        # Construct VL, needed to parametrize - B - as - VL - X -
        #                                       |        |
        # Use prescription under Eq. 85 in Tangent Space lecture notes.
        self.VLs = [construct_orthogonal(self.ALs[i]) for i in range(self.L)]

        # Get left and right generalized eigenvalues
        self.gauge = self.options.get('gauge', 'trace')
        self.boundary_env_data, self.energy_density, _ = MPOTransferMatrix.find_init_LP_RP(self.H, self.psi, \
                calc_E=True, subtraction_gauge=self.gauge, guess_init_env_data=self.guess_init_env_data)
        self.energy_density = np.mean(self.energy_density)
        self.LW = self.boundary_env_data['init_LP']
        self.RW = self.boundary_env_data['init_RP']

        self.GS_env = MPOEnvironment(self.psi, self.H, self.psi, **self.boundary_env_data)
        self.lambda_C0 = options.get('lambda_C0', None)
        if self.lambda_C0 is None:
            '''
            Compute the eigenvalue of
            -----C[0]-----
            |            |
            Lw-----------Rw
            |            |
            -----    -----
            '''
            self.lambda_C0 = npc.tensordot(self.Cs[0], self.RW, axes=(['vR'], ['vL']))
            self.lambda_C0 = npc.tensordot(self.LW, self.lambda_C0, axes=(['wR', 'vR'], ['wL', 'vL']))
            self.lambda_C0 = self.lambda_C0[0,0] / self.Cs[0][0,0]


        self.aligned_H = self.Aligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                  self.LW, self.RW, self.Ws,
                                                  self.chi, d=self.d)

        strange = []
        for i in range(self.L):
            temp_L = self.GS_env.get_LP(i) # LT_general(self.ALs[:i], self.ALs[:i], self.LW, Ws=self.Ws[:i])
            temp_R = self.GS_env.get_RP(i) # TR_general(self.ARs[i+1:], self.ARs[i+1:], self.RW, Ws=self.Ws[i+1:])
            temp = LT_general([self.VLs[i]], [self.ACs[i]], temp_L, Ws=[self.Ws[i]])
            temp = npc.tensordot(temp, temp_R, axes=(['wR', 'vR*'], ['wL', 'vL*']))
            strange.append(npc.norm(temp))
        print('Strange Cancellation Term:', strange)

        if 1 >= 1:
            print("-"*20, "initializing excitation", "-"*20)
            # Don't know how to subtract these with charges
            #print("norm(LW-LW.dag):", npc.norm(self.LW - self.LW.transpose(['vR','wR','vR*']).conj()))
            #print("norm(RW-RW.dag):", npc.norm(self.RW - self.RW.transpose(['vL*','wL','vL']).complex_conj()))
            print("Unit cell size   :", self.L)
            assert self.psi.valid_umps
            print("Energy density   :", self.energy_density)
            print("Lambda C1        :", self.lambda_C0)
            print("||LW||           :", npc.norm(self.LW))
            print("||RW||           :", npc.norm(self.RW))
            print("(LW|RW)          :", npc.tensordot(self.LW, self.RW, axes=(['vR*', 'wR', 'vR'],['vL*', 'wL', 'vL'])))
            print("LW.shape         :", self.LW.shape)
    #---------------------------------------------
    def infinite_sum_TLR(self, R, p):
        sum_tol = self.options.get('sum_tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')
        if sum_method=='explicit':
            R_sum = R.copy()
            for _ in range(100):
                R = np.exp(-1.0j * p * self.L) * TR_general(self.ALs, self.ARs, R, Ws=self.Ws)
                R_sum.iadd_prefactor_other(1., R)
                if npc.norm(R) < sum_tol:
                    break
            return R_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')
    #---------------------------------------------
    def infinite_sum_TRL(self, L, p):
        sum_tol = self.options.get('sum_tol', 1.e-10)
        sum_method = self.options.get('sum_method', 'explicit')

        if sum_method=='explicit':
            L_sum = L.copy()
            for i in range(100):
                L = np.exp(1.0j * p * self.L) * LT_general(self.ARs, self.ALs, L, Ws=self.Ws)
                L_sum.iadd_prefactor_other(1., L)
                if npc.norm(L) < sum_tol:
                    break
            return L_sum
        else:
            raise ValueError('Sum method', sum_method, 'not recognized!')
    #---------------------------------------------
    class Aligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, ALs, ARs, VLs, LW, RW, Ws, chi, d=2):
            n = outer.n
            self.ALs = ALs * n
            self.ARs = ARs * n
            self.Ws = Ws * n

            self.VLs = VLs
            self.LW = LW
            self.RW = RW
            self.chi = chi
            self.d = d
            self.outer = outer
            self.N_X = 0    #N_x = Ne - 1
        '''
        aligned H = left term + middle term + right term
        '''
        def matvec(self, vec):
            '''
            For i in the bra, j in the ket:
            '''
            N_X = self.N_X
            L   = self.outer.L
            n   = self.outer.n
            total_vec = [npc.Array.zeros_like(v) for v in vec]

            for i in range(L): #compute the environment of X_i; i is in the lower
                for j in range(L):
                    '''
                             -----__ket__-------
                             |    |  |  |      |
                    compute  LB---W--W--W------R
                             |    |  |  |      |
                             ----           ----
                                 i+1    i+NX      ,   illustration for N_X = 3
                    '''
                    #TODO QR Xj=vec[j] into Xs
                    Xs = []
                    assert len(Xs) == N_X
                    if N_X == 0:
                        B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                        ket_n = self.ALs[:j] + [B] + Xs + self.ARs[j+N_X+1:]
                    else:
                        ket_n = self.ALs[:j] + [self.VLs[j]] + Xs + self.ARs[j+Nx+1:]
                    bra_n = self.ALs[:i] + [self.VLs[i]] + [None] * N_X + self.ARs[i+N_X+1:]
                    assert len(ket_n) == n * L and len(bra_n) == n * L
                    #-------get LB------------
                    ket = ket_n[:i+1]
                    mpo = self.Ws[:i+1]
                    bra = bra_n[:i+1]
                    LB  = LT_general(ket, bra, self.LW, Ws=mpo)
                    #-------get middle ket and mpo-------
                    ket_mid = ket_n[i+1:i+N_X+1]
                    mpo_mid = self.Ws[i+1:i+N_X+1]
                    #-------get R--------------
                    ket = ket_n[i+N_X+1:]
                    mpo = self.Ws[i+N_X+1:]
                    bra = bra_n[i+N_X+1:]
                    R   = TR_general(ket, bra, self.RW, Ws=mpo)
                    #-------get action--------------
                    total_vec[i] += form_X(ket_mid, LB, R, Ws=mpo_mid)
                    #total_vec and form_X have different labels, but that's fine for npc addition
                    #The labels of the sum follows the label of the first summand, in this case, total_vec.
            return total_vec
    #---------------------------------------------
    class Unaligned_Effective_H(NpcLinearOperator):
        def __init__(self, outer, ALs, ARs, VLs, LW, RW, Ws, p, chi, d=2):
            n = outer.n
            self.ALs = ALs * n
            self.ARs = ARs * n
            self.Ws = Ws * n
            self.VLs = VLs
            self.LW = LW
            self.RW = RW
            self.chi = chi
            self.d = d
            self.p = p
            self.outer = outer
            self.N_X = 0    #N_x = Ne - 1

        def matvec(self, vec):
            '''
            For i in the bra, j in the ket.

            H = term1 + term2
            term1: B_j is one unit cell away from B_i
            term2: B_j is more than one unit cell away from B_i
            '''
            L   = self.outer.L
            n   = self.outer.n
            N_X = self.N_X

            total_vec = [npc.Array.zeros_like(v) for v in vec]
            #----------------
            #B_i to the left of B_j.
            for j in range(L):
                # QR Xj=vec[j] into Xs
                Xs = []
                assert len(Xs) == N_X

                l = j + 1 + N_X
                if l <= L: # doesn't leak
                    # R0 = self.outer.GS_env.get_RP(l-1)
                    R0 = self.RW
                else: #does leak
                    assert l-L < L
                    Xs_leak = Xs[-(l-L):]
                    Ws_leak = self.Ws[:l-L]
                    ARs_leak = self.ARs[:l-L]
                    assert len(Xs_leak) == len(Ws_leak) and len(Xs_leak) == len(ARs_leak)
                    R0 = TR_general(Xs_leak, ARs_leak, self.outer.GS_env.get_RP(l-L-1), Ws=Ws_leak)

                if N_X == 0:
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    ket_n = self.ALs[:L+j] + [B]
                else:
                    ket_n = self.ALs[:L+j] + [self.VLs[j]] + Xs
                ket_n = ket_n[:n*L] #could be incomplete
                ket_n = ket_n + self.ARs[l+L:] #if leaks, ARs doesn't get appended

                R1  = TR_general(ket_n[L:n*L], self.ARs[:L], R0, Ws=self.Ws[:L])
                R1  = self.outer.infinite_sum_TLR(R1, self.p)
                for i in range(L):
                    bra_n = self.ALs[:i] + [self.VLs[i]] + [None] * N_X + self.ARs[i+N_X+1:]
                    assert len(ket_n) == n * L and len(bra_n) == n * L

                    #Term 1
                    #-------get LB------------
                    ket = ket_n[:i+1]
                    mpo = self.Ws[:i+1]
                    bra = bra_n[:i+1]
                    LB  = LT_general(ket, bra, self.LW, Ws=mpo)
                    #-------get middle ket and mpo-------
                    ket_mid = ket_n[i+1:i+N_X+1]
                    mpo_mid = self.Ws[i+1:i+N_X+1]
                    #-------get R--------------
                    ket = ket_n[i+N_X+1:]
                    mpo = self.Ws[i+N_X+1:]
                    bra = bra_n[i+N_X+1:]
                    R   = TR_general(ket, bra, R0, Ws=mpo)
                    #-------get action--------------
                    total_vec[i] += form_X(ket_mid, LB, R, Ws=mpo_mid) * np.exp(-1.0j*self.p*L)


                    #Term 2
                    #-------get LB------------
                    ket = self.ARs[:i+1]
                    mpo = self.Ws[:i+1]
                    bra = bra_n[:i+1]
                    LB  = LT_general(ket, bra, self.LW, Ws=mpo)
                    #-------get middle ket and mpo-------
                    ket_mid = self.ARs[i+1:i+N_X+1]
                    mpo_mid = self.Ws[i+1:i+N_X+1]
                    #-------get R--------------
                    ket = self.ARs[i+N_X+1:]
                    mpo = self.Ws[i+N_X+1:]
                    bra = bra_n[i+N_X+1:]
                    R   = TR_general(ket, bra, R1, Ws=mpo)
                    #-------get action--------------
                    total_vec[i] += form_X(ket_mid, LB, R, Ws=mpo_mid) * np.exp(-2.0j*self.p*L)
            #------------------------------------------------------------------------
            #B_i to the right of B_j.
            for j in range(L):
                # QR Xj=vec[j] into Xs
                Xs = []

                if N_X == 0:
                    B = npc.tensordot(self.VLs[j], vec[j], axes=(['vR'], ['vL']))
                    ket_n = self.ALs[:j] + [B] + self.ARs[j+1+N_X:]
                else:
                    ket_n = self.ALs[:j] + [self.VLs[j]] + Xs + self.ARs[j+1+N_X:]
                assert len(ket_n) == n*L
                L0  = LT_general(ket_n, self.ALs, self.LW, Ws=self.Ws)
                L0  = self.outer.infinite_sum_TRL(L0, self.p)

                for i in range(L):
                    l = i + 1 + N_X #length of ALs + VL + X
                    #Term 1
                    #-------get L------------
                    ket = ket_n[:i+L+1]
                    mpo = self.Ws[:i+L+1]
                    bra = self.ALs[:i+L] + [self.VLs[i]]
                    LB  = LT_general(ket, bra, self.LW, Ws=mpo)
                    #-------get middle ket and mpo-------
                    ket_mid = ket_n[L+i+1:L+l]
                    if l > L: #leaks
                        ket_mid += self.ARs[:l-L]
                    mpo_mid = self.Ws[i+1:i+1+N_X] #self.Ws has n*L length
                    assert len(ket_mid) == N_X and len(mpo_mid) == N_X
                    #-------get R--------------
                    R       = self.outer.GS_env.get_RP((l-1)%L)
                    #-------get action--------------
                    total_vec[i] += form_X(ket_mid, LB, R, Ws=mpo_mid) * np.exp(1.0j*self.p*L)

                    #Term 2
                    #-------get L------------
                    ket = self.ARs[:i+1]
                    mpo = self.Ws[:i+1]
                    bra = self.ALs[:i] + [self.VLs[i]]
                    LB  = LT_general(ket, bra, L0, Ws=mpo)
                    #-------get middle ket and mpo-------
                    ket_mid = self.ARs[i+1:i+1+N_X] #elegant use of the fact that self.ARs has n*L length
                    mpo_mid = self.Ws[i+1:i+1+N_X]
                    #-------get R--------------
                    R       = self.outer.GS_env.get_RP((l-1)%L)
                    #-------get action--------------
                    total_vec[i] += form_X(ket_mid, LB, R, Ws=mpo_mid) * np.exp(2.0j*self.p*L)
            return total_vec
    #---------------------------------------------
    def diagonalize(self, p, qtotal_change=None, n_bands=1):

        self.unaligned_H = self.Unaligned_Effective_H(self, self.ALs, self.ARs, self.VLs,
                                                      self.LW, self.RW, self.Ws, p, self.chi, self.d)
        effective_H = SumNpcLinearOperator(self.aligned_H, self.unaligned_H)

        lanczos_options = self.options.subconfig('lanczos_options')
        X_init = self.initial_guess(qtotal_change)

        Xs = []
        Es = []
        Ns = []
        for i in range(n_bands):
            ortho_H = BoostNpcLinearOperator(effective_H, [100]*i, Xs)
            E, X, N = LanczosGroundState(ortho_H, X_init, lanczos_options).run() #N = number of Lanczos iterations.
            Xs.append(X)
            Es.append(E)
            Ns.append(N)
            if N == lanczos_options.get('N_max', 20):
                import warnings
                warnings.warn('Maximum Lanczos iterations needed; be wary of results.')
        print("Lanczos iterations:", Ns)
        return [E - self.energy_density * self.L * self.n- self.lambda_C0 for E in Es], Xs, Ns
    #---------------------------------------------
    def initial_guess(self, qtotal_change):
        X_init = []
        for i in range(self.L):
            vL = self.VLs[i].get_leg('vR').conj()
            vR = self.ALs[(i+1)% self.L].get_leg('vL').conj()
            th0 = npc.Array.from_func(np.ones, [vL, vR],
                                      dtype=self.psi.dtype,
                                      qtotal=qtotal_change,
                                      labels=['vL', 'vR'])
            th0 = th0/th0.norm()
            X_init.append(th0)

        return X_init
