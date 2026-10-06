import numpy as np

import matplotlib.pyplot as plt

import scipy as sp
from scipy.fft import fft, ifft
from scipy import stats
from scipy.optimize import curve_fit
from scipy import constants as const
from scipy.special import gamma

from iminuit import Minuit

from PyDynamic.uncertainty.propagate_DFT import GUM_DFT # used for error propagation

#====================================================================        
#                           Fitting Model
#====================================================================

def fqt_model1(t,tau,alpha,eisf):
    time = (-(t/tau)**alpha)    
    return eisf + (1-eisf)*ml(time,alpha)

def fqt_model2(t, tau, alpha, eisf):
    time = (-(t/tau)**alpha)
    return eisf + (1-eisf)*np.exp(time)

def ml(z, alpha, beta=1., gama=1.):
    eps = np.finfo(np.float64).eps
    if np.real(alpha) <= 0 or np.real(gama) <= 0 or np.imag(alpha) != 0. \
       or np.imag(beta) != 0. or np.imag(gama) != 0.:
        raise ValueError('ALPHA and GAMA must be real and positive. BETA must be real.')
    if np.abs(gama-1.) > eps:
        if alpha > 1.:
            raise ValueError('GAMMA != 1 requires 0 < ALPHA < 1')
        if (np.abs(np.angle(np.repeat(z, np.abs(z) > eps))) <= alpha*np.pi).any():
            raise ValueError('|Arg(z)| <= alpha*pi')

    return np.vectorize(ml_, [np.float64])(z, alpha, beta, gama)

def ml_(z, alpha, beta, gama):
    # Target precision 
    log_epsilon = np.log(1.e-15)
    # Inversion of the LT
    if np.abs(z) < 1.e-15:
        return 1/gamma(beta)
    else:
        return LTInversion(1, z, alpha, beta, gama, log_epsilon)

def LTInversion(t,lamda,alpha,beta,gama,log_epsilon):
    # Evaluation of the relevant poles
    theta = np.angle(lamda)
    kmin = np.ceil(-alpha/2. - theta/2./np.pi)
    kmax = np.floor(alpha/2. - theta/2./np.pi)
    k_vett = np.arange(kmin, kmax+1)
    s_star = np.abs(lamda)**(1./alpha) * np.exp(1j*(theta+2*k_vett*np.pi)/alpha)

    # Evaluation of phi(s_star) for each pole
    phi_s_star = (np.real(s_star)+np.abs(s_star))/2

    # Sorting of the poles according to the value of phi(s_star)
    index_s_star = np.argsort(phi_s_star)
    phi_s_star = phi_s_star.take(index_s_star)
    s_star = s_star.take(index_s_star)

    # Deleting possible poles with phi_s_star=0
    index_save = phi_s_star > 1.0e-15
    s_star = s_star.repeat(index_save)
    phi_s_star = phi_s_star.repeat(index_save)

    # Inserting the origin in the set of the singularities
    s_star = np.hstack([[0], s_star])
    phi_s_star = np.hstack([[0], phi_s_star])
    J1 = len(s_star)
    J = J1 - 1

    # Strength of the singularities
    p = gama*np.ones((J1,), np.float64)
    p[0] = max(0,-2*(alpha*gama-beta+1))
    q = gama*np.ones((J1,), np.float64)
    q[-1] = np.inf
    phi_s_star = np.hstack([phi_s_star, [np.inf]])

    # Looking for the admissible regions with respect to round-off errors
    admissible_regions = \
       np.nonzero(np.bitwise_and(
           (phi_s_star[:-1] < (log_epsilon - np.log(np.finfo(np.float64).eps))/t),
           (phi_s_star[:-1] < phi_s_star[1:])))[0]
    # Initializing vectors for optimal parameters
    JJ1 = admissible_regions[-1]
    mu_vett = np.ones((JJ1+1,), np.float64)*np.inf
    N_vett = np.ones((JJ1+1,), np.float64)*np.inf
    h_vett = np.ones((JJ1+1,), np.float64)*np.inf

    # Evaluation of parameters for inversion of LT in each admissible region
    find_region = False
    while not find_region:
        for j1 in admissible_regions:
            if j1 < J1-1:
                muj, hj, Nj = OptimalParam_RB(t, phi_s_star[j1], phi_s_star[j1+1], p[j1], q[j1], log_epsilon)
            else:
                muj, hj, Nj = OptimalParam_RU(t, phi_s_star[j1], p[j1], log_epsilon)
            mu_vett[j1] = muj
            h_vett[j1] = hj
            N_vett[j1] = Nj
        if N_vett.min() > 200:
            log_epsilon = log_epsilon + np.log(10)
        else:
            find_region = True

    # Selection of the admissible region for integration which
    # involves the minimum number of nodes
    iN = np.argmin(N_vett)
    N = N_vett[iN]
    mu = mu_vett[iN]
    h = h_vett[iN]

    # Evaluation of the inverse Laplace transform
    k = np.arange(-N, N+1)
    u = h*k
    z = mu*(1j*u+1.)**2
    zd = -2.*mu*u + 2j*mu
    zexp = np.exp(z*t)
    F = z**(alpha*gama-beta)/(z**alpha - lamda)**gama*zd
    S = zexp*F ;
    Integral = h*np.sum(S)/2./np.pi/1j

    # Evaluation of residues
    ss_star = s_star[iN+1:]
    Residues = np.sum(1./alpha*(ss_star)**(1-beta)*np.exp(t*ss_star))

    # Evaluation of the ML function
    E = Integral + Residues
    if np.imag(lamda) == 0.:
        E = np.real(E)
    return E

def OptimalParam_RB(t, phi_s_star_j, phi_s_star_j1, pj, qj, log_epsilon):
    # Definition of some constants
    log_eps = -36.043653389117154 # log(eps)
    fac = 1.01
    conservative_error_analysis = False

    # Maximum value of fbar as the ration between tolerance and round-off unit
    f_max = np.exp(log_epsilon - log_eps)

    # Evaluation of the starting values for sq_phi_star_j and sq_phi_star_j1
    sq_phi_star_j = np.sqrt(phi_s_star_j)
    threshold = 2.*np.sqrt((log_epsilon - log_eps)/t)
    sq_phi_star_j1 = min(np.sqrt(phi_s_star_j1), threshold - sq_phi_star_j)

    # Zero or negative values of pj and qj
    if pj < 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        sq_phibar_star_j1 = sq_phi_star_j1
        adm_region = 1

    # Zero or negative values of just pj
    if pj < 1.0e-14 and qj >= 1.0e-14:
        sq_phibar_star_j = sq_phi_star_j
        if sq_phi_star_j > 0:
            f_min = fac*(sq_phi_star_j/(sq_phi_star_j1-sq_phi_star_j))**qj
        else:
            f_min = fac
        if f_min < f_max:
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fq = f_bar**(-1/qj)
            sq_phibar_star_j1 = (2*sq_phi_star_j1-fq*sq_phi_star_j)/(2+fq)
            adm_region = True
        else:
            adm_region = False

    # Zero or negative values of just qj
    if pj >= 1.0e-14 and qj < 1.0e-14:
        sq_phibar_star_j1 = sq_phi_star_j1
        f_min = fac*(sq_phi_star_j1/(sq_phi_star_j1-sq_phi_star_j))**pj
        if f_min < f_max:
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fp = f_bar**(-1./pj)
            sq_phibar_star_j = (2.*sq_phi_star_j+fp*sq_phi_star_j1)/(2-fp)
            adm_region = True
        else:
            adm_region = False

    # Positive values of both pj and qj
    if pj >= 1.0e-14 and qj >= 1.0e-14:
        f_min = fac*(sq_phi_star_j+sq_phi_star_j1) / \
                (sq_phi_star_j1-sq_phi_star_j)**max(pj, qj)
        if f_min < f_max:
            f_min = max(f_min,1.5)
            f_bar = f_min + f_min/f_max*(f_max-f_min)
            fp = f_bar**(-1/pj)
            fq = f_bar**(-1/qj)
            if ~conservative_error_analysis:
                w = -phi_s_star_j1*t/log_epsilon
            else:
                w = -2.*phi_s_star_j1*t/(log_epsilon-phi_s_star_j1*t)
            den = 2+w - (1+w)*fp + fq
            sq_phibar_star_j = ((2+w+fq)*sq_phi_star_j + fp*sq_phi_star_j1)/den
            sq_phibar_star_j1 = (-(1.+w)*fq*sq_phi_star_j + (2.+w-(1.+w)*fp)*sq_phi_star_j1)/den
            adm_region = True
        else:
            adm_region = False

    if adm_region:
        log_epsilon = log_epsilon  - np.log(f_bar)
        if not conservative_error_analysis:
            w = -sq_phibar_star_j1**2*t/log_epsilon
        else:
            w = -2.*sq_phibar_star_j1**2*t/(log_epsilon-sq_phibar_star_j1**2*t)
        muj = (((1.+w)*sq_phibar_star_j + sq_phibar_star_j1)/(2.+w))**2
        hj = -2.*np.pi/log_epsilon*(sq_phibar_star_j1-sq_phibar_star_j) \
             / ((1.+w)*sq_phibar_star_j + sq_phibar_star_j1)
        Nj = np.ceil(np.sqrt(1-log_epsilon/t/muj)/hj)
    else:
        muj = 0.
        hj = 0.
        Nj = np.inf

    return muj, hj, Nj

def OptimalParam_RU(t, phi_s_star_j, pj, log_epsilon):
    # Evaluation of the starting values for sq_phi_star_j
    sq_phi_s_star_j = np.sqrt(phi_s_star_j)
    if phi_s_star_j > 0:
        phibar_star_j = phi_s_star_j*1.01
    else:
        phibar_star_j = 0.01
    sq_phibar_star_j = np.sqrt(phibar_star_j)

    # Definition of some constants
    f_min = 1
    f_max = 10
    f_tar = 5

    # Iterative process to look for fbar in [f_min,f_max]
    while True:
        phi_t = phibar_star_j*t
        log_eps_phi_t = log_epsilon/phi_t
        Nj = np.ceil(phi_t/np.pi*(1. - 3*log_eps_phi_t/2 + np.sqrt(1-2*log_eps_phi_t)))
        A = np.pi*Nj/phi_t
        sq_muj = sq_phibar_star_j*np.abs(4-A)/np.abs(7-np.sqrt(1+12*A))
        fbar = ((sq_phibar_star_j-sq_phi_s_star_j)/sq_muj)**(-pj)
        if (pj < 1.0e-14) or (f_min < fbar and fbar < f_max):
            break
        sq_phibar_star_j = f_tar**(-1./pj)*sq_muj + sq_phi_s_star_j
        phibar_star_j = sq_phibar_star_j**2
    muj = sq_muj**2
    hj = (-3*A - 2 + 2*np.sqrt(1+12*A))/(4-A)/Nj
    
    # Adjusting integration parameters to keep round-off errors under control
    log_eps = np.log(np.finfo(np.float64).eps)
    threshold = (log_epsilon - log_eps)/t
    if muj > threshold:
        if abs(pj) < 1.0e-14:
            Q = 0
        else:
            Q = f_tar**(-1/pj)*np.sqrt(muj)
        phibar_star_j = (Q + np.sqrt(phi_s_star_j))**2
        if phibar_star_j < threshold:
            w = np.sqrt(log_eps/(log_eps-log_epsilon))
            u = np.sqrt(-phibar_star_j*t/log_eps)
            muj = threshold
            Nj = np.ceil(w*log_epsilon/2/np.pi/(u*w-1))
            hj = np.sqrt(log_eps/(log_eps - log_epsilon))/Nj
        else:
            Nj = np.inf
            hj = 0

    return muj, hj, Nj


def Chi2Regression(f, x, y, sy=None, weights=None):
    x = np.asarray(x)
    y = np.asarray(y)

    sy = 1.0 if sy is None else np.asarray(sy)
    weights = 1.0 if weights is None else np.asarray(weights)

    def cost(tau, alpha, eisf):
        predicted = f(x, tau, alpha, eisf)
        return np.sum(weights * ((y - predicted) / sy)**2)

    cost.errordef = 1.0

    return cost


def calc_chi2(y_data, y_fit, sigmas):
    index = np.where(sigmas>0)[0]
    data = (y_data[index] - y_fit[index])**2/sigmas[index]**2
    Chi2 = np.sum(data)
    Ndof = len(y_data)
    Probchi2 =stats.chi2.sf(Chi2, Ndof)
    return Chi2/Ndof,Ndof, Probchi2 

class Minimal_Model:
    """
    Minimal model for analysing QENS data.

    The class stores the measured S(Q, ω), vanadium resolution data,
    associated uncertainties, Q-values, and energy-transfer axes.

    All energy-dependent quantities are converted internally to lists
    of NumPy arrays, with one array corresponding to each Q-value.
    """

    def __init__(self, sqw, vana_sqw, sqwerror, vana_sqwerror, Q, omega,
                 filename, index=1, T=300):
        """
        Initialize the Minimal_Model object.

        Parameters
        ----------
        sqw : array-like
            Measured S(Q, ω) data.

        vana_sqw : array-like
            Vanadium S(Q, ω) data used to represent the instrumental
            resolution function.

        sqwerror : array-like
            Uncertainties associated with the measured S(Q, ω).

        vana_sqwerror : array-like
            Uncertainties associated with the vanadium data.

        Q : array-like
            Momentum-transfer values.

        omega : array-like
            Energy-transfer axis or axes corresponding to the S(Q, ω) data.

        filename : str
            Name used to identify the dataset and for saving output.

        index : int, optional
            Step size used when sampling the energy-transfer axis.
            The default value is 1, which keeps every point.

        T : float, optional
            Sample temperature in Kelvin. Default is 300 K.
        """
        self.Q = np.asarray(Q, dtype=float)
        self.NQ = len(self.Q)
        self.filename = filename
        self.index = index
        self.T = T

        def ensure_list_of_arrays(x, name):
            """
            Convert input data into a list of NumPy arrays.

            The internal data structure requires one array for each Q-value.
            This function allows the input to be supplied either as:

            1. A list or tuple containing one array per Q-value.
            2. A 2D NumPy array where each row corresponds to one Q-value.
            3. A single 1D array, which is copied for every Q-value.

            Parameters
            ----------
            x : array-like
                Input data to convert.

            name : str
                Name of the input variable, used in error messages.

            Returns
            -------
            list of numpy.ndarray
                One floating-point NumPy array for each Q-value.
            """

            if isinstance(x, (list, tuple)):

                # The number of arrays must match the number of Q-values.
                if len(x) != self.NQ:
                    raise ValueError(f"{name} length must match number of Q points")
                return [np.asarray(row, dtype=float) for row in x]
            
            else:
                # Convert other array-like input to a NumPy array.
                arr = np.asarray(x, dtype=float)
                # If a single 2D array was passed, split by rows
                if arr.ndim == 2 and arr.shape[0] == self.NQ:
                    return [arr[i].astype(float) for i in range(self.NQ)]
                
                # If only one 1D array is supplied, use the same axis/data
                # structure for every Q-value.
                elif arr.ndim == 1:
                    # single energy transfer vector passed for all Q
                    return [arr.copy() for _ in range(self.NQ)]
                else:
                    raise ValueError(f"Unsupported shape for {name}")
                
        # Standardize all Q-dependent datasets to the same internal format:
        # one NumPy array for each Q-value.
        self.sqw = ensure_list_of_arrays(sqw, "sqw")
        self.vana_sqw = ensure_list_of_arrays(vana_sqw, "vana_sqw")
        self.sqwerror = ensure_list_of_arrays(sqwerror, "sqwerror")
        self.vana_sqwerror = ensure_list_of_arrays(vana_sqwerror, "vana_sqwerror")
        self.omega = ensure_list_of_arrays(omega, "omega")

        # Downsample each energy-transfer axis according to the chosen index.
        # For index = 1, all points are retained.
        # For index = 2, every second point is retained, etc.
        self.omeganew = [w[::self.index] for w in self.omega]
        self.Nomega = [len(w) for w in self.omeganew]

    def Sym_Norm(self, usespectra='negative', yscale='log', xlim=(-0.25, 0.25), ylim=None, showplot=True, saveplot=False):
        """
    Symmetrize and normalize the measured QENS and vanadium spectra.

    The measured S(Q, ω) spectra are first symmetrized around zero energy
    transfer using either the negative- or positive-energy side of the
    spectrum. A detailed-balance correction is applied
    before mirroring the selected side.

    The symmetrized spectra are then corrected for Q-dependent variations
    in the integrated vanadium intensity. Finally, both the sample and
    vanadium spectra are normalized by their integrated intensity.

    Parameters
    ----------
    usespectra : {'negative', 'positive'}, optional
        Select which side of the measured spectrum is used to construct
        the symmetrized spectrum. Default is 'negative'.

    yscale : str, optional
        Scale used for the y-axis in the diagnostic plots.
        Default is 'log'.

    xlim : tuple, optional
        Limits of the energy-transfer axis in the plots.
        Default is (-0.25, 0.25) meV.

    ylim : tuple or None, optional
        Limits of the y-axis.

    showplot : bool, optional
        If False, close the generated plot after creation.
        Default is True.

    saveplot : bool, optional
        If True, save the comparison plot to disk.
        Default is False.
    """
        inches_to_cm = 2.54
        figsize = (20 / inches_to_cm, 18 / inches_to_cm)
        plt.rcParams.update({'font.size': 14})

        kb = const.Boltzmann
        converter = 1 / (1.602e-19)

        self.omegasym = []
        self.sqwsymcorrnorm = []
        self.sqwerrorsymcorrnorm = []
        self.vana_sqwsymcorrnorm = []
        self.vana_sqwerrorsymcorrnorm = []

        sqwsym_list, sqwerrsym_list = [], []
        vana_sqwsym_list, vana_errsym_list = [], []
        for i in range(self.NQ):
            w = self.omeganew[i]
            j0 = int(np.where(w <= 0)[0][-1])

            if usespectra == 'negative':
                omegaminus = w[w <= 0]
                omegaplus = np.abs(np.delete(omegaminus, -1)[::-1])
                omegasym_i = np.concatenate((omegaminus, omegaplus))
                weightminus = np.exp(np.abs(omegaminus) / (kb * self.T * 2 * converter * 1e3))

                sqwnew = self.sqw[i][::self.index]
                sqwminus = sqwnew[:j0 + 1] * weightminus
                sqwplus = np.abs(np.delete(sqwminus, -1)[::-1])
                sqwsym = np.concatenate((sqwminus, sqwplus))

                sqwerrnew = self.sqwerror[i][::self.index]
                sqwerrminus = sqwerrnew[:j0 + 1] * weightminus
                sqwerrplus = np.abs(np.delete(sqwerrminus, -1)[::-1])
                sqwerrsym = np.concatenate((sqwerrminus, sqwerrplus))

                vana_new = self.vana_sqw[i][::self.index]
                vana_minus = vana_new[:j0 + 1] * weightminus
                vana_plus = np.abs(np.delete(vana_minus, -1)[::-1])
                vana_sym = np.concatenate((vana_minus, vana_plus))

                vana_err_new = self.vana_sqwerror[i][::self.index]
                vana_err_minus = vana_err_new[:j0 + 1] * weightminus
                vana_err_plus = np.abs(np.delete(vana_err_minus, -1)[::-1])
                vana_err_sym = np.concatenate((vana_err_minus, vana_err_plus))

            elif usespectra == 'positive':
                omegaplus = w[w >= 0]
                omegaminus = -np.delete(omegaplus, 0)[::-1]
                omegasym_i = np.concatenate((omegaminus, omegaplus))
                weightplus = np.exp(-omegaplus / (kb * self.T * 2 * converter * 1e3))

                sqwnew = self.sqw[i][::self.index]
                sqwplus = sqwnew[j0:] * weightplus
                sqwminus = np.abs(np.delete(sqwplus, 0)[::-1])
                sqwsym = np.concatenate((sqwminus, sqwplus))

                sqwerrnew = self.sqwerror[i][::self.index]
                sqwerrplus = sqwerrnew[j0:] * weightplus
                sqwerrminus = np.abs(np.delete(sqwerrplus, 0)[::-1])
                sqwerrsym = np.concatenate((sqwerrminus, sqwerrplus))

                vana_new = self.vana_sqw[i][::self.index]
                vana_plus = vana_new[j0:] * weightplus
                vana_minus = np.abs(np.delete(vana_plus, 0)[::-1])
                vana_sym = np.concatenate((vana_minus, vana_plus))

                vana_err_new = self.vana_sqwerror[i][::self.index]
                vana_err_plus = vana_err_new[j0:] * weightplus
                vana_err_minus = np.abs(np.delete(vana_err_plus, 0)[::-1])
                vana_err_sym = np.concatenate((vana_err_minus, vana_err_plus))
            else:
                raise ValueError(f"Usespectra must be 'negative' or 'positive'")

            self.omegasym.append(omegasym_i)
            sqwsym_list.append(sqwsym)
            sqwerrsym_list.append(sqwerrsym)
            vana_sqwsym_list.append(vana_sym)
            vana_errsym_list.append(vana_err_sym)

        # Q-normalization using vanadium intensity per Q
        vana_int = [sp.integrate.simpson(vana_sqwsym_list[i], self.omegasym[i]) for i in range(self.NQ)]
        mean_int = float(np.mean(vana_int))
        corr = [mean_int / v for v in vana_int]

        for i in range(self.NQ):
            sqw_corr = sqwsym_list[i] * corr[i]
            sqwerr_corr = sqwerrsym_list[i] * corr[i]
            vana_corr = vana_sqwsym_list[i] * corr[i]
            vanaerr_corr = vana_errsym_list[i] * corr[i]

            # frequency normalization per Q
            intg = sp.integrate.simpson(sqw_corr, self.omegasym[i])
            intg_v = sp.integrate.simpson(vana_corr, self.omegasym[i])

            self.sqwsymcorrnorm.append(sqw_corr / intg)
            self.sqwerrorsymcorrnorm.append(sqwerr_corr / intg)
            self.vana_sqwsymcorrnorm.append(vana_corr / intg_v)
            self.vana_sqwerrorsymcorrnorm.append(vanaerr_corr / intg_v)

        # plotting (simple grid)
        cols = 2
        rows = int(np.ceil(self.NQ / cols))
        fig3, axes = plt.subplots(rows, cols, figsize=(figsize[0]*2, figsize[1]*rows))
        fig3.suptitle(f'{self.filename} and vanadium comparison for different Q', x=0.5, y=0.92, fontsize=30)
        axes = np.atleast_2d(axes)
        for i in range(self.NQ):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.errorbar(self.omegasym[i], self.sqwsymcorrnorm[i], yerr = self.sqwerrorsymcorrnorm[i], fmt='.', ecolor = '#9d9e9d',capsize=3, label='QENS Data')
            ax.errorbar(self.omegasym[i], self.vana_sqwsymcorrnorm[i], fmt='-', capsize=3, label='Vanadium')
            ax.set(yscale=yscale, xlim=xlim, ylim=ylim, title=f'Q={self.Q[i]}$Å^{{-1}}$', ylabel='S(Q,ω)', xlabel='Energy (meV)')
            ax.legend(loc='best', prop={'size': 12})
        # hide unused axes
        for j in range(self.NQ, rows*cols):
            r, c = divmod(j, cols)
            axes[r, c].axis('off')

        if not showplot:
            plt.close(fig3)
        if saveplot:
            fig3.savefig(f'{self.filename}_vanadium_comparison.png', dpi=600) 

    def Deconvolve(self, error='linear', showplot=True, saveplot=False):
        """
    Transform the normalized S(Q, ω) spectra into the time domain and
    deconvolve the instrumental resolution using the vanadium spectrum.

    For each Q-value, the symmetrized sample and vanadium spectra are
    reordered into FFT-compatible form and transformed using an inverse
    Fourier transform.

    The resulting sample intermediate scattering function F(Q,t) is divided
    by the Fourier-transformed vanadium spectrum, which acts as the
    instrumental time window.

    Different uncertainty propagation methods can be selected using the
    `error` parameter.

    Parameters
    ----------
    error : {'gauss', 'linear', 'pydynamic'}, optional
        Method used to estimate uncertainties in the time domain.

        'gauss'
            Uses the transformed spectral uncertainties with an approximate
            Gaussian scaling.

        'linear'
            Uses a linear estimate based on the quadrature sum of the
            uncertainties in the frequency domain.

        'pydynamic'
            Uses covariance propagation through the discrete Fourier
            transform using PyDynamic's GUM_DFT routine.

        Default is 'linear'.

    showplot : bool, optional
        If False, close the generated figures after creation.
        Default is True.

    saveplot : bool, optional
        If True, save the convolved and deconvolved figures.
        Default is False.
    """
        inches_to_cm = 2.54
        figsize = (20/inches_to_cm, 18/inches_to_cm)
        plt.rcParams.update({'font.size': 14})

        hbar = const.hbar
        converter = 1/(1.602e-19)

        self.TimeAxis = []
        self.TimeWindow = []
        self.fqtdecon_norm = []
        self.fqterrordecon_norm = []

        for i in range(self.NQ):
            w = self.omegasym[i]
            j0 = int(np.where(w <= 0)[0][-1])

            deltaomega = w[j0+1] - w[j0]
            domega = (deltaomega / (hbar * converter * 1e3)) # THz
            Nw = len(w)
            IndexFFT = (np.arange(Nw) + j0) % Nw
            dtime = 2*np.pi / (Nw * domega)
            time_axis = (np.arange(Nw)) * dtime
            self.TimeAxis.append(time_axis * (10**12))

            sqw_i  = self.sqwsymcorrnorm[i][IndexFFT]
            vana_i = self.vana_sqwsymcorrnorm[i][IndexFFT]

            # Assuming the errors are Gaussian distributed!
            # The width of the Gaussian is inverse when Fourier transformed
            if error == 'gauss':
                fqterror = self.sqwerrorsymcorrnorm[i][IndexFFT] * 1e-1
                TimeWindow_error = self.vana_sqwerrorsymcorrnorm[i][IndexFFT] * 1e-1

            # Assuming linear estimation of the uncertainties
            elif error == 'linear':
                fqterror = np.sqrt(np.sum(self.sqwerrorsymcorrnorm[i]**2)) * deltaomega
                TimeWindow_error = np.sqrt(np.sum(self.vana_sqwerrorsymcorrnorm[i]**2)) * deltaomega

            elif error == 'pydynamic':
                signal = self.sqwsymcorrnorm[i][IndexFFT]
                signal_error = self.sqwerrorsymcorrnorm[i][IndexFFT]
                vana_signal = self.vana_sqwsymcorrnorm[i][IndexFFT]
                vana_signal_error = self.vana_sqwerrorsymcorrnorm[i][IndexFFT]

                # Get covariance matrices from GUM DFT
                _, Ufqt = GUM_DFT(signal, signal_error**2)
                _, Uvana_fqt = GUM_DFT(vana_signal, vana_signal_error**2)

                N = len(IndexFFT)

                Ufqt = Ufqt[:N, :N]
                Uvana_fqt = Uvana_fqt[:N, :N]
                F_inv = np.fft.ifft(np.eye(N))

                cov_fqt_time = F_inv @ Ufqt @ F_inv.conj().T
                cov_vana_time = F_inv @ Uvana_fqt @ F_inv.conj().T

                fqterror = (
                    np.sqrt(np.real(np.diag(cov_fqt_time)))
                    * deltaomega
                    * Nw
                )
                TimeWindow_error = (
                    np.sqrt(np.real(np.diag(cov_vana_time)))
                    * deltaomega
                    * Nw
                )

            else:
                raise ValueError("Unknown error model. Must be 'gauss', 'linear' or 'pydynamic'")

            fqt = deltaomega * np.real(np.fft.ifft(sqw_i)) * Nw

            timewin = deltaomega * np.real(np.fft.ifft(vana_i)) * Nw

            fqtdecon = fqt / timewin

            fqterrordecon = np.sqrt(
                (fqterror / timewin)**2
                +
                ((fqtdecon * TimeWindow_error) / timewin)**2
            )

            # normalize by first point
            scale = fqtdecon[0]
            self.TimeWindow.append(timewin)
            self.fqtdecon_norm.append(fqtdecon / scale)
            self.fqterrordecon_norm.append(fqterrordecon / scale)

        # plotting the convolved F(Q,t)
        cols = 2
        rows = int(np.ceil(self.NQ / cols))

        fig_conv, axes_conv = plt.subplots(rows,cols,figsize=(figsize[0]*2, figsize[1]*rows))
        fig_conv.suptitle(f'{self.filename} Convolved F(Q,t) for different Q',x=0.5,y=0.92,fontsize=30)
        axes_conv = np.atleast_2d(axes_conv)

        for i in range(self.NQ):
            r, c = divmod(i, cols)
            ax = axes_conv[r, c]
            t_i = self.TimeAxis[i]
            ax.plot(t_i,self.TimeWindow[i],'.',label='Convolved')
            ax.set(xlabel='Time (ps)',ylabel='F(Q,t)',ylim=(-0.1, 1.1),title=f'Q={self.Q[i]} $Å^{{-1}}$')
            ax.legend(loc='best', prop={'size': 12})
            
        for j in range(self.NQ, rows*cols):
            r, c = divmod(j, cols)
            axes_conv[r, c].axis('off')

        # Plotting the deconvolved F(Q,t)
        fig_deconv, axes_deconv = plt.subplots(rows,cols,figsize=(figsize[0]*2, figsize[1]*rows))
        fig_deconv.suptitle(f'{self.filename} Deconvolved F(Q,t) for different Q',x=0.5,y=0.92,fontsize=30)

        axes_deconv = np.atleast_2d(axes_deconv)

        for i in range(self.NQ):
            r, c = divmod(i, cols)
            ax = axes_deconv[r, c]
            t_i = self.TimeAxis[i]
            y_i = self.fqtdecon_norm[i]
            yerr_i = self.fqterrordecon_norm[i]

            ax.errorbar(t_i,y_i,yerr=yerr_i,fmt='.',ecolor = '#9d9e9d',capsize=3,label='Deconvolved')
            ax.set(xlabel='Time (ps)',ylabel='F(Q,t)',ylim=(-0.2, 1.2),title=f'Q={self.Q[i]} $Å^{{-1}}$')
            ax.legend(loc='best', prop={'size': 12})

        for j in range(self.NQ, rows*cols):
            r, c = divmod(j, cols)
            axes_deconv[r, c].axis('off')

        if not showplot:
            plt.close(fig_conv)
            plt.close(fig_deconv)

        if saveplot:
            fig_conv.savefig(f'./figure/{self.filename}_convolved.png',dpi=600)
            fig_deconv.savefig(f'./figure/{self.filename}_deconvolved.png',dpi=600)

    def Fitting(self, N_cut=20, model = 'ml', algo='iminuit', p0=[9,0.1,0.1], useerror=False,
            showplot=True, saveplot=False):
        """
    Fit the deconvolved intermediate scattering function F(Q,t)
    for each Q-value.

    The function fits either a Mittag-Leffler ('ml') or stretched
    exponential ('se') to the deconvolved and normalized
    F(Q,t) data.

    Two fitting algorithms are available:

        - iminuit
        - scipy.curve_fit

    The fitted parameters are stored as functions of Q and include the
    characteristic relaxation time, stretching exponent,
    and elastic incoherent structure factor (EISF).

    Parameters
    ----------
    N_cut : int or array-like, optional
        Number of time points included in the fit.

        If an integer is supplied, the same number of points is used
        for every Q-value.

        If an array-like object is supplied, it must contain one value
        for each Q-point, allowing a different fitting range for each Q.

        Default is 20.

    model : {'ml', 'se'}, optional
        Relaxation model used for fitting.

        'ml'
            Mittag-Leffler model.

        'se'
            Stretched exponential model.

        Default is 'ml'.

    algo : {'iminuit', 'scipy.curve_fit'}, optional
        Numerical fitting algorithm.

        Default is 'iminuit'.

    p0 : array-like, optional
        Initial parameter estimates in the order:

            [tau, alpha, eisf]

        For the stretched exponential model, the parameter stored as
        `alpha` represents the stretching exponent beta.
        Default is [9, 0.1, 0.1].

    useerror : bool, optional
        If True, include the propagated F(Q,t) uncertainties in the fit.
        If False, all fitted points are effectively treated with equal
        weighting.
        Default is False.

    showplot : bool, optional
        If False, close the generated figures after fitting.
        Default is True.

    saveplot : bool, optional
        If True, save the fit and parameter-summary figures.
        Default is False.
    """

        inches_to_cm = 2.54
        figsize = (20/inches_to_cm, 18/inches_to_cm)
        plt.rcParams.update({'font.size': 14})

        self.model = model
        if self.model == 'ml':
            fqt_model = fqt_model1

        elif self.model == 'se':
            fqt_model = fqt_model2
        else:
            raise ValueError("Unknown model. Must be 'ml' or 'se'")

        self.tau, self.etau = np.zeros(self.NQ), np.zeros(self.NQ)
        self.alpha, self.ealpha = np.zeros(self.NQ), np.zeros(self.NQ)
        self.eisf, self.eeisf = np.zeros(self.NQ), np.zeros(self.NQ)

        cols = 2
        rows = int(np.ceil(self.NQ / cols))

        fig12, axes = plt.subplots(rows,cols,figsize=(figsize[0]*2, figsize[1]*rows))
        fig12.suptitle(f'{self.filename} {algo} fit for different Q', x=0.5, y=0.92, fontsize=30)
        axes = np.atleast_2d(axes)

        for i in range(self.NQ):
            if isinstance(N_cut, int):
                N_cut_i = min(N_cut, len(self.TimeAxis[i]))
            else:
                if len(N_cut) != self.NQ:
                    raise ValueError("Length of N_cut array must match number of Q points")

                N_cut_i = min(
                    int(N_cut[i]),
                    len(self.TimeAxis[i])
                )

            t = self.TimeAxis[i][:N_cut_i]
            y = self.fqtdecon_norm[i][:N_cut_i]
            yerr = self.fqterrordecon_norm[i][:N_cut_i]

            # Fitting
            if algo == 'iminuit':

                if useerror:
                    Chi2_object = Chi2Regression(fqt_model,t,y,yerr)
                else:
                    Chi2_object = Chi2Regression(fqt_model,t,y)

                m = Minuit(
                    Chi2_object,
                    tau=p0[0],
                    alpha=p0[1],
                    eisf=p0[2]
                )

                m.limits['alpha'] = (0.01, 0.95)
                m.limits['tau'] = (0.01, 1e4)
                m.limits['eisf'] = (0, 0.95)

                m.errordef = 1
                m.migrad()

                self.alpha[i] = m.values['alpha']
                self.ealpha[i] = m.errors['alpha']

                self.tau[i] = m.values['tau']
                self.etau[i] = m.errors['tau']

                self.eisf[i] = m.values['eisf']
                self.eeisf[i] = m.errors['eisf']

                nparams = len(p0)
                ndof = max(len(t) - nparams, 1)

                chi2 = m.fval
                prob = stats.chi2.sf(m.fval, ndof)


            elif algo == 'scipy.curve_fit':

                bounds = (
                    (0.01, 0.01, 0.0),
                    (1e4, 0.95, 0.95)
                )

                if useerror:
                    popt, pcov = curve_fit(fqt_model,t,y,p0,sigma=yerr,bounds=bounds,maxfev=10000)
                else:
                    popt, pcov = curve_fit(fqt_model,t,y,p0,bounds=bounds,maxfev=10000)

                self.tau[i], self.alpha[i], self.eisf[i] = popt

                perr = np.sqrt(np.diag(pcov))
                self.etau[i], self.ealpha[i], self.eeisf[i] = perr
                resid = y - fqt_model(t, *popt)
                if useerror:
                    chi2 = np.sum((resid / yerr)**2)
                else:
                    chi2 = np.sum(resid**2)

                ndof = max(len(y) - len(popt), 1)
                prob = stats.chi2.sf(chi2, ndof)

            else:
                raise ValueError(
                    "Unknown algo. Must be 'iminuit' or 'scipy.curve_fit'"
                )

            # Plot the fitting of F(Q,t)
            r, c = divmod(i, cols)
            ax = axes[r, c]

            xfit = np.linspace(np.min(t),np.max(t),500)
            yfit = fqt_model(np.abs(xfit),self.tau[i],self.alpha[i],self.eisf[i])
            ax.errorbar(t,y,yerr=yerr,fmt='.',ecolor = '#9d9e9d',capsize=3,zorder=1,label='Data'
        )

            ax.plot(xfit,yfit,'-',color = 'red',linewidth=2,zorder=2,label='Fit')
            ax.set(xlabel='Time (ps)',ylabel='F(Q,t)',title=f'Q={self.Q[i]} $Å^{{-1}}$')
            ax.legend(loc='best',prop={'size': 12})

            # parameter box
            parameter_text = (
                rf'$\tau$ = {self.tau[i]:.2f} $\pm$ {self.etau[i]:.2f} ps'
                '\n'
                rf'$\alpha$ = {self.alpha[i]:.3f} $\pm$ {self.ealpha[i]:.3f}'
                '\n'
                rf'EISF = {self.eisf[i]:.3f} $\pm$ {self.eeisf[i]:.3f}'
                '\n'
                rf'$\chi^2$/DOF = {chi2/ndof:.2f}'
            )

            ax.text(
                0.97,
                0.97,
                parameter_text,
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=11,
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8
                )
            )

        # hide unused axes
        for j in range(self.NQ, rows*cols):
            r, c = divmod(j, cols)
            axes[r, c].axis('off')
      
        # Plot of the fitted cuves
        # =========================================================

        fig13, ax13 = plt.subplots(3,1,figsize=(figsize[0]*1.35, figsize[1]*2.0),sharex=True)
        fig13.suptitle(f'{self.filename} Fit parameters',x=0.5,y=0.97,fontsize=26)

        # plotting the fit parameters
        ax13[0].errorbar(self.Q,self.alpha,yerr= None,fmt='o',capsize=3,label=self.filename)
        if self.model == 'ml':
            ax13[0].set(
                        ylabel=r'$\alpha$',
                        ylim=(0.0, 1.0)
                    )
                
        elif self.model == 'se':
            ax13[0].set(
                        ylabel=r'$\beta$',
                        ylim=(0.0, 1.0)
                    )

        ax13[0].legend(loc='best',
            prop={'size': 11}
        )

        # EISF
        # ---------------------------------------------------------
        ax13[1].errorbar(
            self.Q,
            self.eisf,
            yerr= None,
            fmt='o',
            capsize=3,
            label=self.filename
        )

        ax13[1].set(
            ylabel='EISF',
            ylim=(-0.05, 1.0)
        )

        ax13[1].legend(
            loc='best',
            prop={'size': 11}
        )

        # ---------------------------------------------------------
        # tau
        # ---------------------------------------------------------
        ax13[2].errorbar(
            self.Q,
            self.tau,
            yerr= None,
            fmt='o',
            capsize=3,
            label=self.filename
        )

        ax13[2].set(
            xlabel='Q [Å$^{-1}$]',
            ylabel=r'$\tau$ [ps]'
        )

        ax13[2].legend(
            loc='best',
            prop={'size': 11}
        )

        # slightly cleaner spacing
        fig13.subplots_adjust(
            top=0.90,
            hspace=0.12
        )

        # ---------------------------------------------------------
        # Show / save
        # ---------------------------------------------------------
        if not showplot:
            plt.close(fig12)
            plt.close(fig13)

        if saveplot:

            fig12.savefig(
                f'./figure/{self.filename}_fit.png',
                dpi=600,
                bbox_inches='tight'
            )

            fig13.savefig(
                f'./figure/{self.filename}_fit_parameters.png',
                dpi=600,
                bbox_inches='tight'
            )

    def Resample(self, yscale='log', xlim=(-0.5, 0.5), ylim=None,
             showplot=True, saveplot=False):
        """
    Reconstruct the fitted time-domain relaxation model in the
    energy domain and compare it directly with the measured S(Q, ω).

    The fitted deconvolved F(Q,t) model is evaluated on the complete
    FFT time grid. The normalization and instrumental resolution that
    were removed during Deconvolve() are then reapplied.

    The resulting convolved time-domain model is Fourier transformed
    back into S(Q,ω), allowing direct comparison between the fitted
    relaxation model and the experimental QENS spectrum.

    Residuals and chi-square statistics are also calculated for each
    Q-value.

    Parameters
    ----------
    yscale : str, optional
        Scale used for the S(Q,ω) y-axis.
        Default is 'log'.

    xlim : tuple or None, optional
        Energy-transfer range shown in the plots and used for the
        displayed chi-square calculation.

        Default is (-0.5, 0.5) meV.

    ylim : tuple or None, optional
        Limits of the S(Q,ω) y-axis.
        If None, matplotlib determines the limits automatically.

    showplot : bool, optional
        If False, close the generated figure after creation.
        Default is True.

    saveplot : bool, optional
        If True, save the reconstructed-model comparison figure.
        Default is False.
    """

    # Conversion factor used for defining figure dimensions
    # in centimeters.               
        inches_to_cm = 2.54
        figsize = (20/inches_to_cm, 18/inches_to_cm)
        plt.rcParams.update({'font.size': 14})

        if self.model == 'ml':
            fqt_model = fqt_model1
        
        elif self.model == 'se':
            fqt_model = fqt_model2
        hbar = const.hbar
        converter = 1 / (1.602e-19)

        if not hasattr(self, "TimeWindow"):
            raise RuntimeError("Run Deconvolve() before Resample().")

        if not all(hasattr(self, name) for name in ["tau", "alpha", "eisf"]):
            raise RuntimeError("Run Fitting() before Resample().")

        self.fqtfitsampled = []
        self.fqtfitmodel_convolved = []
        self.sqwfitmodel = []

        # ---------------------------------------------------------
        # Plotting scaffold
        # Same 2-column style as Sym_Norm
        # Each Q panel contains:
        #   upper: data + fitted model
        #   lower: residuals
        # ---------------------------------------------------------
        cols = 2
        rows = int(np.ceil(self.NQ / cols))

        fig3 = plt.figure(
            figsize=(figsize[0]*2, figsize[1]*rows)
        )

        fig3.suptitle(
            f'{self.filename} fitted model comparison for different Q',
            x=0.5,
            y=0.92,
            fontsize=30
        )

        outer = fig3.add_gridspec(
            rows,
            cols,
            hspace=0.35,
            wspace=0.25
        )

        for i in range(self.NQ):

            w = self.omegasym[i]
            j0 = int(np.where(w <= 0)[0][-1])

            deltaomega = w[j0 + 1] - w[j0]
            Nw = len(w)

            # Same FFT ordering as in Deconvolve()
            IndexFFT = (np.arange(Nw) + j0) % Nw

            # Same time spacing as in Deconvolve()
            domega = deltaomega / (hbar * converter * 1e3)
            dtime_s = 2 * np.pi / (Nw * domega)
            dtime_ps = dtime_s * 1e12

            # FFT time grid
            n = np.arange(Nw)
            t_eff_ps = np.minimum(n, Nw - n) * dtime_ps

            # -----------------------------------------------------
            # Fitted deconvolved normalized F(Q,t)
            # -----------------------------------------------------
            fqtfitsampled_i = fqt_model(
                t_eff_ps,
                self.tau[i],
                self.alpha[i],
                self.eisf[i]
            )

            # -----------------------------------------------------
            # Reconstruct normalization scale from Deconvolve()
            # -----------------------------------------------------
            sqw_i = self.sqwsymcorrnorm[i][IndexFFT]
            vana_i = self.vana_sqwsymcorrnorm[i][IndexFFT]

            fqt0 = (
                deltaomega
                * np.real(np.fft.ifft(sqw_i))[0]
                * Nw
            )

            timewin0 = (
                deltaomega
                * np.real(np.fft.ifft(vana_i))[0]
                * Nw
            )

            scale = fqt0 / timewin0

            # Reverse:
            # fqtdecon_norm = (fqt / timewin) / scale
            timewin_i = self.TimeWindow[i]

            fqtfitmodel_convolved = (
                fqtfitsampled_i
                * scale
                * timewin_i
            )

            # Reverse:
            # fqt = deltaomega * real(ifft(sqw_i)) * Nw
            sqwfit_ordered = (
                np.fft.fft(fqtfitmodel_convolved)
                / (deltaomega * Nw)
            )

            # Return to original omega ordering
            sqwfit = np.empty_like(
                np.real(sqwfit_ordered)
            )

            sqwfit[IndexFFT] = np.real(
                sqwfit_ordered
            )

            self.fqtfitsampled.append(
                fqtfitsampled_i
            )

            self.fqtfitmodel_convolved.append(
                fqtfitmodel_convolved
            )

            self.sqwfitmodel.append(
                sqwfit
            )

            # =====================================================
            # PLOTTING FOR THIS Q
            # =====================================================

            r, c = divmod(i, cols)

            inner = outer[r, c].subgridspec(
                2,
                1,
                height_ratios=[4, 1],
                hspace=0.05
            )

            ax0 = fig3.add_subplot(inner[0])
            ax1 = fig3.add_subplot(
                inner[1],
                sharex=ax0
            )

            # -----------------------------------------------------
            # Main S(Q,w) plot
            # -----------------------------------------------------
            

            ax0.plot(
                w,
                sqwfit,
                '-',
                color='red',
                linewidth=2,
                zorder=2,
                label='Fitted model'
            )
            
            ax0.errorbar(
                w,
                self.sqwsymcorrnorm[i],
                yerr=self.sqwerrorsymcorrnorm[i],
                fmt='.',
                ecolor = '#9d9e9d',
                capsize=3,
                zorder=1,
                label='QENS Data'
            )

            ax0.set(
                ylabel='S(Q,ω)',
                yscale=yscale,
                xlim=xlim,
                ylim=ylim,
                title=f'Q={self.Q[i]} $Å^{{-1}}$'
            )

            ax0.legend(
                loc='best',
                prop={'size': 12}
            )

            # Hide x labels on upper panel
            plt.setp(
                ax0.get_xticklabels(),
                visible=False
            )

            residuals = (
                self.sqwsymcorrnorm[i]
                - sqwfit
            )

            ax1.axhline(
                            0,
                            linewidth=1.5,
                            zorder=2,
                        )
            
            ax1.errorbar(
                w,
                residuals,
                yerr=self.sqwerrorsymcorrnorm[i],
                fmt='.',
                ecolor='#9d9e9d',
                capsize=3,
                zorder=1
            )

            ax1.set(
                xlabel='Energy (meV)',
                ylabel='Data - Fit',
                xlim=xlim
            )

            # -----------------------------------------------------
            # Chi-square inside plotted x-range
            # -----------------------------------------------------
            if xlim is not None:
                mask = (
                    (w >= xlim[0])
                    & (w <= xlim[1])
                )
            else:
                mask = np.ones_like(
                    w,
                    dtype=bool
                )

            chi2_red, ndof, pval = calc_chi2(
                self.sqwsymcorrnorm[i][mask],
                sqwfit[mask],
                self.sqwerrorsymcorrnorm[i][mask]
            )

            fit_text = (
                rf'$\chi^2_\mathrm{{red}}$ = {chi2_red:.2f}'
                '\n'
                rf'Ndof = {ndof}'
                '\n'
                rf'$p$ = {pval:.3f}'
            )

            ax0.text(
                0.97,
                0.78,
                fit_text,
                transform=ax0.transAxes,
                ha='right',
                va='top',
                fontsize=11,
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8
                )
            )

        for j in range(self.NQ, rows*cols):

            r, c = divmod(j, cols)

            ax_empty = fig3.add_subplot(
                outer[r, c]
            )

            ax_empty.axis('off')

        if not showplot:
            plt.close(fig3)

        if saveplot:
            fig3.savefig(
                f'./figure/{self.filename}_fittedmodelcomparison.png',
                dpi=600,
                bbox_inches='tight'
            )
