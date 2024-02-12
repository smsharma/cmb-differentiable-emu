"""
Python version of Planck's plik-lite likelihood with the option to include
the low-ell temperature as two Gaussian bins

The official Planck likelihoods are availabe at https://pla.esac.esa.int/
The papers describing the Planck likelihoods are
Planck 2018: https://arxiv.org/abs/1907.12875
Planck 2015: https://arxiv.org/abs/1507.02704

The covariance matrix treatment is based on Zack Li's ACT likelihood code
available at: https://github.com/xzackli/actpols2_like_py

planck calibration is set to 1 by default but this can easily be modified
"""

import jax
import jax.numpy as np
from jax import lax
import numpy as onp
from scipy.io import FortranFile
import scipy.linalg
from functools import partial


class PlanckLitePy:
    def __init__(
        self,
        data_directory="data",
        year=2018,
        spectra="TT",
        use_low_ell_bins=False,
        ellmin=2,  # Minimum ell of input arrays
    ):
        """
        data_directory = path from where you are running this to the folder
          containing the planck2015/8_low_ell and planck2015/8_plik_lite data
        year = 2015 or 2018
        spectra = TT for just temperature or TTTEEE for temperature (TT),
          E mode (EE) and cross (TE) spectra
        use_low_ell_bins = True to use 2 low ell bins for the TT 2<=ell<30 data
          or False to only use ell>=30
        """
        self.year = year
        self.spectra = spectra
        self.use_low_ell_bins = use_low_ell_bins  # False matches Plik_lite - just l>=30

        if self.use_low_ell_bins:
            self.nbintt_low_ell = 2
            self.plmin_TT = 2
        else:
            self.nbintt_low_ell = 0
            self.plmin_TT = 30
        self.plmin = 30
        self.plmax = 2508
        self.calPlanck = 1

        if year == 2015:
            self.data_dir = data_directory + "/planck2015_plik_lite/"
            version = 18
        elif year == 2018:
            self.data_dir = data_directory + "/planck2018_plik_lite/"
            version = 22
        else:
            print("Year must be 2015 or 2018")
            return 1

        if spectra == "TT":
            self.use_tt = True
            self.use_ee = False
            self.use_te = False
        elif spectra == "TTTEEE":
            self.use_tt = True
            self.use_ee = True
            self.use_te = True
        else:
            print("Spectra must be TT or TTTEEE")
            return 1

        self.nbintt_hi = 215  # 30-2508   #used when getting covariance matrix
        self.nbinte = 199  # 30-1996
        self.nbinee = 199  # 30-1996
        self.nbin_hi = self.nbintt_hi + self.nbinte + self.nbinee

        self.nbintt = (
            self.nbintt_hi + self.nbintt_low_ell
        )  # mostly want this if using low ell
        self.nbin_tot = self.nbintt + self.nbinte + self.nbinee

        self.like_file = self.data_dir + "cl_cmb_plik_v" + str(version) + ".dat"
        self.cov_file = self.data_dir + "c_matrix_plik_v" + str(version) + ".dat"
        self.blmin_file = self.data_dir + "blmin.dat"
        self.blmax_file = self.data_dir + "blmax.dat"
        self.binw_file = self.data_dir + "bweight.dat"

        # read in binned ell value, C(l) TT, TE and EE and errors
        # use_tt etc to select relevant parts
        self.bval, self.X_data, self.X_sig = onp.genfromtxt(self.like_file, unpack=True)
        self.blmin = onp.loadtxt(self.blmin_file).astype(int)
        self.blmax = onp.loadtxt(self.blmax_file).astype(int)
        self.bin_w = onp.loadtxt(self.binw_file)

        if self.use_low_ell_bins:
            self.data_dir_low_ell = data_directory + "/planck" + str(year) + "_low_ell/"
            self.bval_low_ell, self.X_data_low_ell, self.X_sig_low_ell = onp.genfromtxt(
                self.data_dir_low_ell + "CTT_bin_low_ell_" + str(year) + ".dat",
                unpack=True,
            )
            self.blmin_low_ell = onp.loadtxt(
                self.data_dir_low_ell + "blmin_low_ell.dat"
            ).astype(int)
            self.blmax_low_ell = onp.loadtxt(
                self.data_dir_low_ell + "blmax_low_ell.dat"
            ).astype(int)
            self.bin_w_low_ell = onp.loadtxt(
                self.data_dir_low_ell + "bweight_low_ell.dat"
            )

            self.bval = np.concatenate((self.bval_low_ell, self.bval))
            self.X_data = np.concatenate((self.X_data_low_ell, self.X_data))
            self.X_sig = np.concatenate((self.X_sig_low_ell, self.X_sig))

            self.blmin_TT = np.concatenate(
                (self.blmin_low_ell, self.blmin + len(self.bin_w_low_ell))
            )
            self.blmax_TT = np.concatenate(
                (self.blmax_low_ell, self.blmax + len(self.bin_w_low_ell))
            )
            self.bin_w_TT = np.concatenate((self.bin_w_low_ell, self.bin_w))

        else:
            self.blmin_TT = self.blmin
            self.blmax_TT = self.blmax
            self.bin_w_TT = self.bin_w

        self.fisher = np.array(self.get_inverse_covmat())

        # Convert all onp arrays to jax arrays
        self.bval = np.array(self.bval)
        self.X_data = np.array(self.X_data)
        self.X_sig = np.array(self.X_sig)
        self.blmin = np.array(self.blmin)
        self.blmax = np.array(self.blmax)
        self.bin_w = np.array(self.bin_w)
        self.blmin_TT = np.array(self.blmin_TT)
        self.blmax_TT = np.array(self.blmax_TT)
        self.bin_w_TT = np.array(self.bin_w_TT)

        self.ellmin = ellmin

        sum_index_start = self.blmin_TT + self.plmin_TT - self.ellmin
        sum_index_end = self.blmax_TT + self.plmin_TT + 1 - self.ellmin

        self.bins_unique_TT = np.unique(sum_index_end - sum_index_start)
        self.bin_indices_TT = [
            np.where((sum_index_end - sum_index_start) == self.bins_unique_TT[j])[0]
            for j in range(len(self.bins_unique_TT))
        ]

        # Similarly for TE and EE

        sum_index_start = self.blmin + self.plmin - self.ellmin
        sum_index_end = self.blmax + self.plmin + 1 - self.ellmin

        self.bins_unique = np.unique(sum_index_end - sum_index_start)
        self.bin_indices = [
            np.where((sum_index_end - sum_index_start) == self.bins_unique[j])[0]
            for j in range(len(self.bins_unique))
        ]

        # Convert bins_unique to onp
        self.bins_unique = onp.array(self.bins_unique)
        self.bins_unique_TT = onp.array(self.bins_unique_TT)

        # Convert bin_indices to onp
        self.bin_indices = [onp.array(bi) for bi in self.bin_indices]
        self.bin_indices_TT = [onp.array(bi) for bi in self.bin_indices_TT]

    def get_inverse_covmat(self):
        # read full covmat
        f = FortranFile(self.cov_file, "r")
        covmat = f.read_reals(dtype=float).reshape((self.nbin_hi, self.nbin_hi))
        for i in range(self.nbin_hi):
            for j in range(i, self.nbin_hi):
                covmat[i, j] = covmat[j, i]

        # select relevant covmat
        if self.use_tt and not (self.use_ee) and not (self.use_te):
            # just tt
            bin_no = self.nbintt_hi
            start = 0
            end = start + bin_no
            cov = covmat[start:end, start:end]
        elif not (self.use_tt) and not (self.use_ee) and self.use_te:
            # just te
            bin_no = self.nbinte
            start = self.nbintt_hi
            end = start + bin_no
            cov = covmat[start:end, start:end]
        elif not (self.use_tt) and self.use_ee and not (self.use_te):
            # just ee
            bin_no = self.nbinee
            start = self.nbintt_hi + self.nbinte
            end = start + bin_no
            cov = covmat[start:end, start:end]
        elif self.use_tt and self.use_ee and self.use_te:
            # use all
            bin_no = self.nbin_hi
            cov = covmat
        else:
            print("not implemented")

        # invert high ell covariance matrix (cholesky decomposition should be faster)
        fisher = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(cov), np.identity(bin_no)
        )
        fisher = fisher.transpose()

        if self.use_low_ell_bins:
            bin_no += self.nbintt_low_ell
            inv_covmat_with_lo = np.zeros(shape=(bin_no, bin_no))

            inv_covmat_with_lo = inv_covmat_with_lo.at[0:2, 0:2].set(
                np.diag(1.0 / self.X_sig_low_ell**2)
            )
            inv_covmat_with_lo = inv_covmat_with_lo.at[2:, 2:].set(fisher)
            fisher = inv_covmat_with_lo

        return fisher

    @partial(jax.jit, static_argnums=(0, 2, 4, 5))
    def bin_Cl(self, Cl, ellmin, bin_indices, delta, nbin, blmin, plmin, bin_w):

        Cl_bin = np.zeros(nbin)

        # for i in bin_indices:

        #     sum_index_start = blmin[i] + plmin - ellmin

        #     weighted_sum = np.sum(
        #         jax.lax.dynamic_slice(Cl, [sum_index_start], [delta])
        #         * jax.lax.dynamic_slice(bin_w, [blmin[i]], [delta])
        #     )

        #     # Update Cltt_bin accumulator.
        #     Cl_bin = Cl_bin.at[i].set(weighted_sum)

        # Rewrite using jax.lax.fori_loop

        def body(i, Cl_bin):
            sum_index_start = blmin[bin_indices[i]] + plmin - ellmin
            weighted_sum = np.sum(
                jax.lax.dynamic_slice(Cl, [sum_index_start], [delta])
                * jax.lax.dynamic_slice(bin_w, [blmin[bin_indices[i]]], [delta])
            )
            return Cl_bin.at[bin_indices[i]].set(weighted_sum)

        Cl_bin = jax.lax.fori_loop(0, len(bin_indices), body, Cl_bin)

        return Cl_bin

    @partial(jax.jit, static_argnums=(0,))
    def loglike(self, Dltt, Dlte, Dlee):

        # convert model Dl's to Cls then bin them
        ls = np.arange(len(Dltt)) + self.ellmin
        fac = ls * (ls + 1) / (2 * np.pi)
        Cltt = Dltt / fac
        Clte = Dlte / fac
        Clee = Dlee / fac

        # Fortran to python slicing: a:b becomes a-1:b
        # need to subtract 1 to use 0 indexing for cl,
        # then add one for weights because fortran includes top value

        Cltt_bin = np.zeros(self.nbintt)
        for i in range(len(self.bin_indices_TT)):

            Cltt_bin += self.bin_Cl(
                Cltt,
                self.ellmin,
                self.bin_indices_TT[i],
                self.bins_unique_TT[i],
                self.nbintt,
                self.blmin_TT,
                self.plmin_TT,
                self.bin_w_TT,
            )

        # Similarly for TE and EE

        Clte_bin = np.zeros(self.nbinte)
        for i in range(len(self.bin_indices)):
            Clte_bin += self.bin_Cl(
                Clte,
                self.ellmin,
                self.bin_indices[i],
                self.bins_unique[i],
                self.nbinte,
                self.blmin,
                self.plmin,
                self.bin_w,
            )

        Clee_bin = np.zeros(self.nbinee)
        for i in range(len(self.bin_indices)):
            Clee_bin += self.bin_Cl(
                Clee,
                self.ellmin,
                self.bin_indices[i],
                self.bins_unique[i],
                self.nbinee,
                self.blmin,
                self.plmin,
                self.bin_w,
            )

        X_model = np.zeros(self.nbin_tot)

        # Rewrite using at[idx].set() to avoid in-place operations
        X_model = X_model.at[: self.nbintt].set(Cltt_bin / self.calPlanck**2)
        X_model = X_model.at[self.nbintt : self.nbintt + self.nbinte].set(
            Clte_bin / self.calPlanck**2
        )
        X_model = X_model.at[self.nbintt + self.nbinte :].set(
            Clee_bin / self.calPlanck**2
        )

        Y = self.X_data - X_model

        # choose relevant bits based on whether using TT, TE, EE
        if self.use_tt and not (self.use_ee) and not (self.use_te):
            # just tt
            bin_no = self.nbintt
            start = 0
            end = start + bin_no
            diff_vec = Y[start:end]
        elif not (self.use_tt) and not (self.use_ee) and self.use_te:
            # just te
            bin_no = self.nbinte
            start = self.nbintt
            end = start + bin_no
            diff_vec = Y[start:end]
        elif not (self.use_tt) and self.use_ee and not (self.use_te):
            # just ee
            bin_no = self.nbinee
            start = self.nbintt + self.nbinte
            end = start + bin_no
            diff_vec = Y[start:end]
        elif self.use_tt and self.use_ee and self.use_te:
            # use all
            bin_no = self.nbin_tot
            diff_vec = Y
        else:
            print("not implemented")

        return -0.5 * diff_vec.dot(self.fisher.dot(diff_vec))
