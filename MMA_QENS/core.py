import numpy as np
import scipy as sp
import matplotlib


#from numpy.fft import fft, ifft
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

from scipy import stats, integrate
from scipy.optimize import curve_fit, minimize
from scipy import constants as const
from scipy.special import gamma
from scipy.stats import chi2

import numpy as np

import iminuit
from iminuit import Minuit

from PyDynamic.uncertainty.propagate_DFT import GUM_DFT

#from pythonpackage.ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
#from pythonpackage.ExternalFunctions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float64)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

# =============================================================================
#  Probfit replacement
# =============================================================================

from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one
    
    def __init__(self, f, x, y, sy=None, weights=None):
        
        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2

def simpson38(f, edges, bw, *arg):
    
    yedges = f(edges, *arg)
    left38 = f((2.*edges[1:]+edges[:-1]) / 3., *arg)
    right38 = f((edges[1:]+2.*edges[:-1]) / 3., *arg)
    
    return bw / 8.*( np.sum(yedges)*2.+np.sum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8


def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint+1)
    bw = edges[1] - edges[0]
    
    return simpson38(f, edges, bw, *arg)



class UnbinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, weights=None, badvalue=-100000, extended=False, extended_bound=None, extended_nint=100):
        
        self.f = f  # model predicts PDF for given x
        self.data = np.array(data)
        self.weights = set_var_if_None(weights, self.data)
        self.bad_value = badvalue
        
        self.extended = extended
        self.extended_bound = extended_bound
        self.extended_nint = extended_nint
        if extended and extended_bound is None:
            self.extended_bound = (np.min(data), np.max(data))

        
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        logf = np.zeros_like(self.data)
        
        # compute the function value
        f = compute_f(self.f, self.data, *par)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = (f>0)

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive]) * self.weights[mask_f_positive] 
        
        # set everywhere else to badvalue
        logf[~mask_f_positive] = self.bad_value
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        
        if self.extended:
            extended_term = integrate1d(self.f, self.extended_bound, self.extended_nint, *par)
            llh += extended_term
        
        return llh
    
    def default_errordef(self):
        return 0.5





class BinnedLH:  # override the class with a better one
    
    def __init__(self, f, data, bins=40, weights=None, weighterrors=None, bound=None, badvalue=1000000, extended=False, use_w2=False, nint_subdiv=1):
        
        self.weights = set_var_if_None(weights, data)


        self.f = f
        self.use_w2 = use_w2
        self.extended = extended

        if bound is None: 
            bound = (np.min(data), np.max(data))

        self.mymin, self.mymax = bound

        h, self.edges = np.histogram(data, bins, range=bound, weights=weights)
        
        self.bins = bins
        self.h = h
        self.N = np.sum(self.h)

        if weights is not None:
            if weighterrors is None:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weights**2)
            else:
                self.w2, _ = np.histogram(data, bins, range=bound, weights=weighterrors**2)
        else:
            self.w2, _ = np.histogram(data, bins, range=bound, weights=None)


        
        self.badvalue = badvalue
        self.nint_subdiv = nint_subdiv
        
        
        self.func_code = make_func_code(describe(self.f)[1:])
        self.ndof = np.sum(self.h > 0) - (self.func_code.co_argcount - 1)
        

    def __call__(self, *par):  # par are a variable number of model parameters

        # ret = compute_bin_lh_f(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.badvalue, *par)
        ret = compute_bin_lh_f2(self.f, self.edges, self.h, self.w2, self.extended, self.use_w2, self.nint_subdiv, *par)
        
        return ret


    def default_errordef(self):
        return 0.5




import warnings


def xlogyx(x, y):
    
    #compute x*log(y/x) to a good precision especially when y~x
    
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    
    if x<y:
        return x*np.log1p( (y-x) / x )
    else:
        return -x*np.log1p( (x-y) / y )


#compute w*log(y/x) where w < x and goes to zero faster than x
def wlogyx(w, y, x):
    if x<1e-100:
        warnings.warn('x is really small return 0')
        return 0.
    if x<y:
        return w*np.log1p( (y-x) / x )
    else:
        return -w*np.log1p( (x-y) / y )


def compute_bin_lh_f2(f, edges, h, w2, extended, use_sumw2, nint_subdiv, *par):
    
    N = np.sum(h)
    n = len(edges)

    ret = 0.
    
    for i in range(n-1):
        th = h[i]
        tm = integrate1d(f, (edges[i], edges[i+1]), nint_subdiv, *par)
        
        if not extended:
            if not use_sumw2:
                ret -= xlogyx(th, tm*N) + (th-tm*N)

            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm*N,th)+(th-tm*N))
        else:
            if not use_sumw2:
                ret -= xlogyx(th,tm)+(th-tm)
            else:
                if w2[i]<1e-200: 
                    continue
                tw = w2[i]
                factor = th/tw
                ret -= factor*(wlogyx(th,tm,th)+(th-tm))

    return ret





def compute_bin_lh_f(f, edges, h, w2, extended, use_sumw2, badvalue, *par):
    
    mask_positive = (h>0)
    
    N = np.sum(h)
    midpoints = (edges[:-1] + edges[1:]) / 2
    b = np.diff(edges)
    
    midpoints_pos = midpoints[mask_positive]
    b_pos = b[mask_positive]
    h_pos = h[mask_positive]
    
    if use_sumw2:
        warnings.warn('use_sumw2 = True: is not yet implemented, assume False ')
        s = np.ones_like(midpoints_pos)
        pass
    else: 
        s = np.ones_like(midpoints_pos)

    
    E_pos = f(midpoints_pos, *par) * b_pos
    if not extended:
        E_pos = E_pos * N
        
    E_pos[E_pos<0] = badvalue
    
    ans = -np.sum( s*( h_pos*np.log( E_pos/h_pos ) + (h_pos-E_pos) ) )

    return ans

#====================================================================        
#                           Fitting Model
#====================================================================

def fqt_model(t,tau,alpha,eisf):
    time = (-(t/tau)**alpha)    
    return eisf + (1-eisf)*ml(time,alpha)

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

def calc_chi2(y_data, y_fit, sigmas):
    index = np.where(sigmas>0)[0]
    data = (y_data[index] - y_fit[index])**2/sigmas[index]**2
    Chi2 = np.sum(data)
    Ndof = len(y_data)
    Probchi2 =stats.chi2.sf(Chi2, Ndof)
    return Chi2/Ndof,Ndof, Probchi2 

class Minimal_Model:
    def __init__(self, sqw, vana_sqw, sqwerror, vana_sqwerror, Q, omega, filename,index = 1 , T=300): 
        """
        The minimal model approach for analysis of quasi-elastic neutron scattering. The original python script was written by Martin Hoffmann Petersen. 
        
        Parameters:
        ----------
            sqw : numpy.ndarray
                A 2D ``array`` of the measured QENS spectra.
            vana_sqw : numpy.ndarray
                A 2D ``array`` of the measured vanadium QENS spectra.
            sqwerror : numpy.ndarray
                A 2D ``array`` of the uncertanties for the measured QENS spectra.
            vana_sqwerror : numpy.ndarray
                A 2D ``array`` of the uncertanties for the measured vanadium QENS spectra.
            Q : numpy.ndarray
                A 1D ``array`` of Q-values in Å.
            filename : string
                A string containing the name of the measured sample.
            omega : numpy.ndarray
                A 1D ``array`` of energy transfer in meV.
            index : int, optional
                A int, who makes the spacing in the spectra larger. Deafult is 1
            T : int
                A int of the sample tempetures in Kelvin
        """
        self.sqw = sqw
        self.vana_sqw = vana_sqw
        self.sqwerror = sqwerror
        self.vana_sqwerror = vana_sqwerror
        self.Q = Q
        self.omega = omega
        self.index = index
        self.T = T
        self.filename = filename

        self.omeganew = self.omega[::self.index] # prepares the new omega array with respect to the index
        self.NQ = len(self.Q)
        self.Nomega= len(self.omeganew)

        self.symmetrized = None
        self.deconvolved = None
        self.fitted = None
        self.resampled = None
        
    def Sym_Norm(self, usespectra = 'negative', yscale='log', xlim=(-0.25, 0.25), ylim=None, showplot=True, saveplot=False):
        """
        The Symmetrization and normalization of the measured QENS spectra:
        
        Parameters:
        ----------
            symside : string
                A string to choose which part of the spectrum to use 
            yscale : string, optional
                A string setting the y-axis scale of the plots. If 'None' then the scale is in 'log'
            xlim : tuple, optional 
                A tuple setting the x-axis range. If 'None' then defult setting is used
            ylim : tuple, optional
                A tuple setting the y-axis range. If 'None' the y-axis is not limited
            showplot : bool, optional
                If 'True', all plots are shown. Default is 'True'
            saveplot : bool, optional
                If 'True', all shown plots are saved as png-files. Default is 'False'.
        """
        inches_to_cm = 2.54
        figsize = (20 / inches_to_cm, 18 / inches_to_cm)
        plt.rcParams.update({'font.size': 14})
        
        j0 = int(np.where(self.omeganew<=0)[0][-1])

        if usespectra == 'negative':
            omegaminus = self.omeganew[self.omeganew<=0]
            omegaplus = abs(np.delete(omegaminus,-1)[::-1])
            self.omegasym = np.concatenate((omegaminus,omegaplus),axis=0)
            self.Nomegasym = len(self.omegasym)
            
            kb = const.Boltzmann #m^2 kg s^-2 K^-1=J/K
            hbar = const.hbar # m^2 kg s^-1 =J*s
            converter= 1/(1.602e-19) #eV/J
            weightminus = np.exp(abs(omegaminus)/(kb*self.T*2*(converter)*(10**3)))
            max_weightminus = max(weightminus)
            
            sqwsym = np.zeros((self.NQ,self.Nomegasym))
            sqwerrorsym = np.zeros((self.NQ,self.Nomegasym))
            vana_sqwsym = np.zeros((self.NQ,self.Nomegasym))
            vana_sqwerrorsym = np.zeros((self.NQ,self.Nomegasym))

            for i in range(self.NQ):
                sqwnew = self.sqw[i][::self.index]
                sqwminus = sqwnew[:j0+1] * weightminus
                sqwplus = abs(np.delete(sqwminus, -1)[::-1])
                sqwsym[i] = np.concatenate((sqwminus, sqwplus), axis=0)
                
                sqwerrornew = self.sqwerror[i][::self.index]
                sqwerrorminus = sqwerrornew[:j0+1] * weightminus
                sqwerrorplus = abs(np.delete(sqwerrorminus, -1)[::-1])
                sqwerrorsym[i] = np.concatenate((sqwerrorminus, sqwerrorplus), axis=0)
                
                vana_sqwnew = self.vana_sqw[i][::self.index]
                vana_sqwminus = vana_sqwnew[:j0+1] * weightminus
                vana_sqwplus = abs(np.delete(vana_sqwminus, -1)[::-1])
                vana_sqwsym[i] = np.concatenate((vana_sqwminus, vana_sqwplus), axis=0)
                
                vana_sqwerrornew = self.vana_sqwerror[i][::self.index]
                vana_sqwerrorminus = vana_sqwerrornew[:j0+1] * weightminus
                vana_sqwerrorplus = abs(np.delete(vana_sqwerrorminus, -1)[::-1])
                vana_sqwerrorsym[i] = np.concatenate((vana_sqwerrorminus, vana_sqwerrorplus), axis=0)

        elif usespectra == 'positive':
            omegaplus = self.omeganew[self.omeganew>=0]
            omegaminus = -np.delete(omegaplus, 0)[::-1]
            self.omegasym = np.concatenate((omegaminus, omegaplus), axis=0)
            self.Nomegasym = len(self.omegasym)
            
            kb = const.Boltzmann  # m^2 kg s^-2 K^-1 = J/K
            hbar = const.hbar  # m^2 kg s^-1 = J*s
            converter = 1 / (1.602e-19)  # eV/J
            weightplus = np.exp(-omegaplus / (kb * self.T * 2 * converter * 10**3))
            max_weightplus = max(weightplus)
            
            sqwsym = np.zeros((self.NQ, self.Nomegasym))
            sqwerrorsym = np.zeros((self.NQ, self.Nomegasym))
            vana_sqwsym = np.zeros((self.NQ, self.Nomegasym))
            vana_sqwerrorsym = np.zeros((self.NQ, self.Nomegasym))
            
            for i in range(self.NQ):
                sqwerrornew = self.sqwerror[i][::self.index]
                sqwnew = self.sqw[i][::self.index]
                sqwplus = sqwnew[j0:] * weightplus
                sqwminus = abs(np.delete(sqwplus, 0)[::-1])
                sqwsym[i] = np.concatenate((sqwminus, sqwplus), axis=0)
                
                #sqwerrornew = self.sqwerror[i][::self.index]
                sqwerrorplus = sqwerrornew[j0:] * weightplus
                sqwerrorminus = abs(np.delete(sqwerrorplus, 0)[::-1])
                sqwerrorsym[i] = np.concatenate((sqwerrorminus, sqwerrorplus), axis=0)
                
                vana_sqwnew = self.vana_sqw[i][::self.index]
                vana_sqwplus = vana_sqwnew[j0:] * weightplus
                vana_sqwminus = abs(np.delete(vana_sqwplus, 0)[::-1])
                vana_sqwsym[i] = np.concatenate((vana_sqwminus, vana_sqwplus), axis=0)

                vana_sqwerrornew = self.vana_sqwerror[i][::self.index]
                vana_sqwerrorplus = vana_sqwerrornew[j0:] * weightplus
                vana_sqwerrorminus = abs(np.delete(vana_sqwerrorplus, 0)[::-1])
                vana_sqwerrorsym[i] = np.concatenate((vana_sqwerrorminus, vana_sqwerrorplus), axis=0)

        vana_Qintensity = sp.integrate.simpson(vana_sqwsym, self.omegasym)
        vana_Qintensitymean = np.mean(vana_Qintensity)
        CorrFactorQ = vana_Qintensitymean/vana_Qintensity
        #print(CorrFactorQ)

        sqwsymcorr = np.zeros((self.NQ,self.Nomegasym))
        sqwerrorsymcorr = np.zeros((self.NQ,self.Nomegasym))
        vana_sqwsymcorr = np.zeros((self.NQ,self.Nomegasym))
        vana_sqwerrorsymcorr = np.zeros((self.NQ,self.Nomegasym))
        for i in range(self.NQ):
            sqwsymcorr[i] = sqwsym[i]*CorrFactorQ[i]
            sqwerrorsymcorr[i] = sqwerrorsym[i]*CorrFactorQ[i]
    
            vana_sqwsymcorr[i] = vana_sqwsym[i]*CorrFactorQ[i]
            vana_sqwerrorsymcorr[i] = vana_sqwerrorsym[i]*CorrFactorQ[i]
    
        # The frequency normalization
        intg = sp.integrate.simpson(sqwsymcorr, self.omegasym)
        vana_intg = sp.integrate.simpson(vana_sqwsymcorr, self.omegasym)
        self.sqwsymcorrnorm = np.zeros((self.NQ,self.Nomegasym))
        self.sqwerrorsymcorrnorm = np.zeros((self.NQ,self.Nomegasym))
        self.vana_sqwsymcorrnorm = np.zeros((self.NQ,self.Nomegasym))
        self.vana_sqwerrorsymcorrnorm = np.zeros((self.NQ,self.Nomegasym))

        for i in range(self.NQ):
            self.sqwsymcorrnorm[i] = sqwsymcorr[i]/intg[i]
            self.sqwerrorsymcorrnorm[i] = sqwerrorsymcorr[i]/intg[i]

            self.vana_sqwsymcorrnorm[i] = vana_sqwsymcorr[i]/vana_intg[i]
            self.vana_sqwerrorsymcorrnorm[i] = vana_sqwerrorsymcorr[i]/vana_intg[i]
        row = int(np.ceil(self.NQ/2))
        
        # Plotting
        row = int(np.ceil(self.NQ / 2))
        fig3, ax3 = plt.subplots(nrows=row, ncols=2, figsize=(figsize[0]*2, figsize[1]*row))
        fig3.suptitle(f'{self.filename} and vanadium comparison for different Q', x=0.5, y=0.92, fontsize=30)
        
        for i in range(self.NQ):
            row_idx = i // 2
            col_idx = i % 2
            ax3[row_idx, col_idx].errorbar(self.omegasym, self.sqwsymcorrnorm[i], color='b', fmt='.', capsize=3, label='QENS Data')
            ax3[row_idx, col_idx].errorbar(self.omegasym, self.vana_sqwsymcorrnorm[i], color='r', fmt='-', capsize=3, label='Vanadium')
            ax3[row_idx, col_idx].set(yscale=yscale, xlim=xlim, ylim=ylim, title=f'Q={self.Q[i]}$Å^{{-1}}$', ylabel='S(Q,ω)', xlabel='Energy (meV)')
            ax3[row_idx, col_idx].legend(loc='best', prop={'size': 12})
        
        if not showplot:
            plt.close(fig3)
        if saveplot:
            fig3.savefig(f'{self.filename}_vanadium_comparison.png', dpi=600)

    def Deconvolve(self, error='linear', showplot=True, saveplot=False):
        """
         Compute and deconvolve the intermediate scattering function

        Prameters
        ----------
            error : string, optional
                A string setting the estimation of the uncertnaties. 
                Currently two estimations are avaivable, 'gauss' and 'linear. Default is 'gauss'.
            showplot : bool, optional
                If 'True', all plots are shown. Default is 'True'
            saveplot : bool, optional
                If 'True', all shown plots are saved as png-files. Default is 'False'.
        """
        
        # Figsize
        inches_to_cm = 2.54
        figsize = (20/inches_to_cm, 18/inches_to_cm)
        plt.rcParams.update({'font.size': 14})

        self.Nomegasym= len(self.omegasym)
        self.j0 = int(np.where(self.omegasym<=0)[0][-1])
        
        hbar = const.hbar # m^2 kg s^-1 =J*s
        converter= 1/(1.602e-19) #eV/J

        # prepares for the Fourier transformation
        # The following steps periodice S(Q,omega) by creating a periodic index
        deltaomega = self.omegasym[self.j0+1]-self.omegasym[self.j0] # d(omega) for FT
        domega = (deltaomega/(hbar*(converter)*(10**3)))*10**-12 #THz
        dtime = 2*np.pi/(self.Nomegasym*domega) # ps
        IndexFFT= (np.arange(0,self.Nomegasym)+self.j0)%self.Nomegasym
        self.TimeAxis = (IndexFFT-self.j0)*dtime
        
        # Periodices S(Q,omega) and define the uncertanty
        forfft = np.zeros((self.NQ,self.Nomegasym))
        vana_forfft = np.zeros((self.NQ,self.Nomegasym))
        
        fqterror = np.ones((self.NQ,self.Nomegasym))
        TimeWindow_error = np.ones((self.NQ,self.Nomegasym))

        for i in range(self.NQ):
            forfft[i] = self.sqwsymcorrnorm[i][IndexFFT]
            vana_forfft[i] = self.vana_sqwsymcorrnorm[i][IndexFFT]
            
            # Assuming the errors are Gaussian distributed! 
            # The width of the Gaussian is inverse when fourier transformed
            if error == 'gauss':
                fqterror[i] = self.sqwerrorsymcorrnorm[i][IndexFFT]*1e-1
                TimeWindow_error[i] =self.vana_sqwerrorsymcorrnorm[i][IndexFFT]*1e-1
            # Assuming linear estimation of the uncertanties
            elif error == 'linear':
                fqterror[i] = fqterror[i]*np.sqrt(np.sum(self.sqwerrorsymcorrnorm[i]**2))*deltaomega
                TimeWindow_error[i] = TimeWindow_error[i]*np.sqrt(np.sum(self.vana_sqwerrorsymcorrnorm[i]**2))*deltaomega
                
            elif error == 'pydynamic':
                signal = self.sqwsymcorrnorm[i][IndexFFT]
                signal_error = self.sqwerrorsymcorrnorm[i][IndexFFT]

                vana_signal = self.vana_sqwsymcorrnorm[i][IndexFFT]
                vana_signal_error = self.vana_sqwerrorsymcorrnorm[i][IndexFFT]

                # Get covariance matrices from GUM DFT (assumed to return full complex covariance matrix)
                _, Ufqt = GUM_DFT(signal, signal_error**2)
                _, Uvana_fqt = GUM_DFT(vana_signal, vana_signal_error**2)

                N = len(IndexFFT)
                Ufqt = Ufqt[:N, :N]
                Uvana_fqt = Uvana_fqt[:N, :N]

                F_inv = np.fft.ifft(np.eye(N))
                cov_fqt_time = F_inv @ Ufqt @ F_inv.conj().T
                cov_vana_time = F_inv @ Uvana_fqt @ F_inv.conj().T

                # Extract variance of real parts (since we use np.real(np.fft.ifft(...)) in fqt calculation)
                fqterror[i] = np.sqrt(np.real(np.diag(cov_fqt_time))) * deltaomega * self.Nomegasym
                TimeWindow_error[i] = np.sqrt(np.real(np.diag(cov_vana_time))) * deltaomega * self.Nomegasym

        # Compute the intermediate scattering function F(Q,t) and 
        # deconvolve it by using the Inverse Fouriertransform for the
        # vanadium  
        fqt = deltaomega *np.real(np.fft.ifft(forfft))*self.Nomegasym
        self.TimeWindow = deltaomega *np.real(np.fft.ifft(vana_forfft))*self.Nomegasym
        fqtdecon = fqt/self.TimeWindow
        fqterrordecon = np.sqrt((fqterror/self.TimeWindow)**2+((fqtdecon*TimeWindow_error)/self.TimeWindow**2)**2)
        # scale the fqt
        self.fqtdecon_norm = np.zeros((self.NQ,self.Nomegasym))
        self.fqterrordecon_norm = np.zeros((self.NQ,self.Nomegasym))
        for i in range(self.NQ):
            self.fqtdecon_norm[i] = fqtdecon[i] /fqtdecon[i][0]
            self.fqterrordecon_norm[i] = fqterrordecon[i] /fqtdecon[i][0] 
        row = int(np.ceil(self.NQ/2))
        fig4, ax4 = plt.subplots(nrows=row, ncols=2, figsize=(figsize[0]*2,figsize[1]*row))
        fig4.suptitle(self.filename+'Convolved',x=0.5,y=0.92, fontsize = 30)
        fig5, ax5 = plt.subplots(nrows=row, ncols=2, figsize=(figsize[0]*2,figsize[1]*row))
        fig5.suptitle(self.filename+'Deconvolved',x=0.5,y=0.92, fontsize = 30)
        for i in range(self.NQ):
            row1 = (row-1)
            #print(np.max(fqt[i]))
            if i<=row1:
                ax4[i][0].plot(fqt[i],'b.', label='Q='+str(self.Q[i])+r'$Å^{-1}$')
                ax4[i][0].set(ylim=(-0.1,1.1), title = self.filename+' Convolved', ylabel = f'F(Q,t)', xlabel = 'Measurement number')
                ax4[i][0].legend(loc='best', prop = {'size':12})
            else:
                ax4[(row1-i)][1].plot(fqt[i],'b.', label='Q='+str(self.Q[i])+r'$Å^{-1}$')
                ax4[(row1-i)][1].set(ylim=(-0.1,1.1), title = self.filename+' Convolved', ylabel = f'F(Q,t)', xlabel = 'Measurement number')
                ax4[(row1-i)][1].legend(loc ='best', prop = {'size':12})
            if i<=row1:
                ax5[i][0].plot(self.fqtdecon_norm[i],'b.', label='Q='+str(self.Q[i])+r'$Å^{-1}$')
                ax5[i][0].set(ylim=(-4,4), title = self.filename+' Deconvolved', ylabel = f'F(Q,t)', xlabel = 'Measurement number')
                ax5[i][0].legend(loc='best', prop = {'size':12})
            else:
                ax5[(row1-i)][1].plot(self.fqtdecon_norm[i],'b.', label='Q='+str(self.Q[i])+r'$Å^{-1}$')
                ax5[(row1-i)][1].set(ylim=(-4,4), title = self.filename+' Deconvolved', ylabel = f'F(Q,t)', xlabel = 'Measurement number')
                ax5[(row1-i)][1].legend(loc ='best', prop = {'size':12})
        if not showplot:
            plt.close(fig4)
            plt.close(fig5)
        if saveplot:
            fig4.savefig('./figure/'+self.filename+'convolved.png', dpi=600)
            fig5.savefig('./figure/'+self.filename+'deconvolved.png', dpi=600)
    
    def Fitting(self, N_cut = 20, algo = 'iminuit', p0=[5,0.1,0.1], useerror=False, showplot = True, saveplot = False):
        """
        Fitting the simple model to the intermediate scattering function

        Parameters
        ----------
            N_cut : int
                A int setting the number of measurements used in the fit
                 A string containing the name of the measured sample.
            p0 : list, optional
                A 3 element list [tau0,alpha0,eisf0] containing the initial geuss for the fit parameters.
            useerror : bool, optional
                If 'True', the uncertainties are used in the fit. Default is 'False'.
            showplot : bool, optional
                If 'True', all plots are shown. Default is 'True'
            saveplot : bool, optional
                If 'True', all shown plots are saved as png-files. Default is 'False'.
    """

        # Figsize
        inches_to_cm = 2.54
        figsize = (20/inches_to_cm, 18/inches_to_cm)
        plt.rcParams.update({'font.size': 14})

        hbar = const.hbar # m^2 kg s^-1 =J*s
        converter= 1/(1.602e-19) #eV/J

        # The fit is done for all Q-values and plotted together
        # with the fit parameters and the chisquare value and p-value
        row = int(np.ceil(self.NQ/2))
        fig12, ax12 = plt.subplots(nrows=row, ncols=2, figsize=(figsize[0]*2,figsize[1]*row))
        fig12.suptitle(self.filename+ ' ' + algo + ' fit',x=0.5,y=0.92, fontsize = 30)
    
        self.tau, self.etau =  np.zeros((self.NQ)), np.zeros((self.NQ))
        self.alpha, self.ealpha = np.zeros((self.NQ)), np.zeros((self.NQ))
        self.eisf, self.eeisf = np.zeros((self.NQ)), np.zeros((self.NQ))

        for i in range(self.NQ):
            tmax = self.TimeAxis[N_cut]
            fqtforfit = (self.fqtdecon_norm[i][:N_cut])
            fqtforfit_error = (self.fqterrordecon_norm[i][:N_cut])
            timeforfit = (self.TimeAxis[:N_cut])

            if algo == 'iminuit':
                if useerror == True:
                    Chi2_object = Chi2Regression(fqt_model,timeforfit ,fqtforfit,fqtforfit_error)
                else:
                    Chi2_object = Chi2Regression(fqt_model,timeforfit ,fqtforfit)

                minuit = Minuit(Chi2_object, tau=p0[0],alpha=p0[1], eisf=p0[2])
                minuit.limits['alpha']= (0.01,0.95)
                minuit.limits['tau']= (0.01,1e4)
                minuit.limits['eisf']= (0,0.95)
                minuit.errordef = 1
                minuit.migrad()         
                
                self.alpha[i] = minuit.values['alpha']
                self.ealpha[i] = minuit.errors['alpha']
                self.tau[i] = minuit.values['tau']
                self.etau[i] = minuit.errors['tau']
                self.eisf[i] = minuit.values['eisf'] 
                self.eeisf[i] = minuit.errors['eisf']

                ndof = (len(timeforfit)-len(p0))
                chi2, prob = minuit.fval, stats.chi2.sf(minuit.fval,ndof)

            elif algo == 'scipy.curve_fit':
                bounds = ((0.01, 0.01, 0), (1e4, 0.95, 0.95))  # Bounds for tau, alpha and eisf

                # Fit the data using curve_fit
                if useerror == True:
                    popt, pcov = curve_fit(fqt_model, timeforfit, fqtforfit, p0, sigma=fqtforfit_error, bounds=bounds)
                else:
                    popt, pcov = curve_fit(fqt_model, timeforfit, fqtforfit, p0, bounds=bounds)

                # Extract optimized parameters
                self.tau[i], self.alpha[i], self.eisf[i] = popt
                
                perr = np.sqrt(np.diag(pcov))  # Standard deviations from the covariance matrix
                self.etau[i], self.ealpha[i], self.eeisf[i] = perr 

                # Compute chi2 values
                residuals = fqtforfit - fqt_model(timeforfit, *popt)
                if useerror == True:
                    chi2 = np.sum((residuals / fqtforfit_error) ** 2)
                else: 
                    chi2 = np.sum((residuals) ** 2)
                ndof = len(fqtforfit) - len(popt)
                prob = stats.chi2.sf(chi2, ndof)

            xaxis = abs(self.TimeAxis[:N_cut])
            yaxis = fqt_model(xaxis,self.tau[i],self.alpha[i], self.eisf[i])
        
            d = {'tau': "{:.4f} +/- {:.4f}".format(self.tau[i],self.etau[i]),
             'alpha': "{:.4f} +/- {:.4f}".format(self.alpha[i],self.ealpha[i]),
             'eisf': "{:.4f} +/- {:.4f}".format(self.eisf[i],self.eeisf[i]),
             "Chi2": "{:.2f}".format(chi2),
             "Ndof": "{:.1f}".format(ndof),
             "ProbChi2": "{:.3f}".format(prob),}
            text = nice_string_output(d, extra_spacing=2, decimals=3)
            row1 = (row-1)
            if i<=row1:
                add_text_to_ax(0.40, 0.80, text, ax12[i][0], fontsize=12)
                ax12[i][0].errorbar(timeforfit, fqtforfit,fqtforfit_error, fmt='o', color = 'b',ecolor = 'k', capsize = 5, label =' Q='+str(self.Q[i])+r'$Å^{-1}$')
                ax12[i][0].plot(xaxis, yaxis, 'r-', label='Fit')
                ax12[i][0].set(xlabel ='t[ps]',ylabel='F(Q,t)')
                ax12[i][0].legend(loc = 1, prop = {'size':12})
            else:
                add_text_to_ax(0.40, 0.80, text, ax12[(row1-i)][1], fontsize=12)
                ax12[(row1-i)][1].errorbar(timeforfit, fqtforfit,fqtforfit_error, fmt='o', color = 'b',ecolor = 'k', capsize = 5, label =' Q='+str(self.Q[i])+r'$Å^{-1}$')
                ax12[(row1-i)][1].plot(xaxis, yaxis, 'r-', label='Fit)')
                ax12[(row1-i)][1].set(xlabel ='t[ps]',ylabel='F(Q,t)')
                ax12[(row1-i)][1].legend(loc = 1, prop = {'size':12})
    

        fig13, ax13 = plt.subplots(nrows=3, ncols=1, figsize=(figsize[0],figsize[1]*1.5), gridspec_kw = {'hspace':0.4})
        fig13.suptitle(self.filename+'Fit parameters',x=0.5,y=0.99, fontsize = 30)
        ax13[0].errorbar(self.Q, self.alpha, color ='b', fmt='o', label=self.filename)
        ax13[0].set(xlabel='Q '+r'[$Å^{-1}$]',ylabel=r'$\alpha$',ylim=(0.0,1))
        ax13[0].legend(loc = 3, prop = {'size':14})
        ax13[1].errorbar(self.Q, self.eisf, color ='b', fmt='o', label=self.filename)
        ax13[1].set(xlabel='Q '+r'[$Å^{-1}$]',ylabel=r'$EISF$',ylim=(-0.2,1))
        ax13[1].legend(loc = 3, prop = {'size':14})
        ax13[2].errorbar(self.Q, self.tau, color ='b', fmt='o', label=self.filename)
        ax13[2].set(xlabel=r'Q $[Å^{-1}]$',ylabel=r'$\tau$ [ps]')
        ax13[2].legend(loc = 3, prop = {'size':14})

        self.fitparameters = [self.tau,self.alpha,self.eisf]
        self.fitparameters_error = [self.etau,self.ealpha,self.eeisf]
    
        if not showplot:
            plt.close(fig12)
            plt.close(fig13)
        if saveplot:
            fig12.savefig('./figure/'+self.filename+'fit.png', dpi=600)
            fig13.savefig('./figure/'+self.filename+'fit_parameters.png', dpi=600)
    
    def Resample(self,yscale ='log',xlim=(-0.5,0.5),ylim=None ,showplot = True, saveplot = False):
        """
        Calculate the reduced chi2

        Parameters
        ----------
            y_data : numpy.array
                A 1D ``array`` of the measured data
            y_fit : numpy.array
                A 1D ``array`` of the fit of the measured data
            sigmas : numpy.array
                A 1D ``array`` of the uncertainties for the measured data
        """

        # Figsize
        inches_to_cm = 2.54
        figsize = (20/inches_to_cm, 18/inches_to_cm)
        plt.rcParams.update({'font.size': 14})
    
        hbar = const.hbar # m^2 kg s^-1 =J*s
        converter= 1/(1.602e-19) #eV/J
    
        #  Creating the time axis
        deltaomega = self.omegasym[self.j0+1]-self.omegasym[self.j0]
        domega = (deltaomega/(hbar*(converter)*(10**3)))*10**-12 #THz
        dtime = 2*np.pi/(self.Nomegasym*domega) # ps # self. these
        IndexFFT= (np.arange(0,self.Nomegasym)+self.j0)%self.Nomegasym
        self.TimeAxis = (IndexFFT-self.j0)*dtime
    
        # Create the fitted model for F(Q,t)
        time_fqtfitsampled = np.linspace(0,dtime*self.Nomegasym,self.Nomegasym)
    
        self.fqtfitsampled = np.zeros((self.NQ,self.Nomegasym))
        fqtfitmodel = np.zeros((self.NQ,self.Nomegasym))
        for i in range(self.NQ):
            self.fqtfitsampled[i] = fqt_model(abs(self.TimeAxis),self.tau[i],self.alpha[i],self.eisf[i])
            fqtfitmodel[i] = self.fqtfitsampled[i] * self.TimeWindow[i]
    
        # Create the fitted model for S(Q,omega)
        tmp = (hbar*converter*1e3)/(dtime*1e-12)  #meV
        factor = (1/tmp)*(1/(2*np.pi))
        sqwfitmodelFFT = factor*np.real(np.fft.fft(fqtfitmodel))
        InverseIndexFFT= (np.arange(0,self.Nomegasym)+(self.j0+1))%self.Nomegasym
        # Compare the fitted model for S(Q,t) with the actual measured
        # S(Q,t) for the sample
    
        self.sqwfitmodel = np.zeros((self.NQ,self.Nomegasym))
        for i in range(self.NQ):
            self.sqwfitmodel[i] = sqwfitmodelFFT[i][InverseIndexFFT]
            if (i+1) % 2:
                fig3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(15,5),
                        gridspec_kw={'height_ratios':[4,1], 'hspace':0.03}, sharex=True)
                ax3[0,0].errorbar(self.omegasym,self.sqwsymcorrnorm[i],yerr=self.sqwerrorsymcorrnorm[i] ,fmt='.',capsize=3,color='r',alpha=0.4, label = 'QENS Data')
                ax3[0,0].errorbar(self.omegasym,self.sqwfitmodel[i],lw=4,color='k', label = 'Fitted model',zorder=1)
                ax3[0,0].set(title='Q='+str(self.Q[i])+r'$Å^{-1}$',xlabel=r'$\hbar\omega$',ylabel=r'log(S($Q,\omega$))', yscale=yscale,xlim=xlim,ylim=ylim)
                ax3[0,0].legend(loc=2,fontsize=12)
                residuals = self.sqwfitmodel[i]-self.sqwsymcorrnorm[i]
                ax3[1,0].errorbar(self.omegasym,residuals,yerr = self.sqwerrorsymcorrnorm[i],
                      fmt='.',capsize=3,color='r',alpha=0.9,label='Residuals');  
                ax3[1,0].plot(self.omegasym,np.zeros_like(self.omegasym),'k',lw=3);  
                ax3[1,0].set(xlabel=r'$\hbar\omega$',ylabel=r'Residual',xlim=xlim)
                ax3[1,0].legend(loc=4,fontsize=14)
                if xlim == False:
                    chi2_red, ndof, pval = calc_chi2(self.sqwsymcorrnorm[i],self.sqwfitmodel[i],
                                           self.sqwerrorsymcorrnorm[i] )
                else: 
                    ind1 = int(np.where(self.omegasym>xlim[0])[0][0]-1)
                    ind2 = int(np.where(self.omegasym>xlim[1])[0][0])
                    chi2_red, ndof, pval = calc_chi2(self.sqwsymcorrnorm[i][ind1:ind2],self.sqwfitmodel[i][ind1:ind2],
                                           self.sqwerrorsymcorrnorm[i][ind1:ind2] )
                d = {r"$\chi^2$=":"{:.2f}".format(chi2_red),
                     }
                text = nice_string_output(d, extra_spacing=0, decimals=3)
                add_text_to_ax(0.05, 0.7, text, ax3[0,0], fontsize=16)
        
            else:
                ax3[0,1].errorbar(self.omegasym,self.sqwsymcorrnorm[i],yerr= self.sqwerrorsymcorrnorm[i], fmt='.',capsize=3,color='r',alpha=0.4, label = 'QENS Data')
                ax3[0,1].errorbar(self.omegasym,self.sqwfitmodel[i],lw=4,color='k', label= 'Fitted model',zorder=1)
                ax3[0,1].set(title='Q='+str(self.Q[i])+r'$Å^{-1}$',xlabel=r'$\hbar\omega$',ylabel=r'log(S($Q,\omega$))', yscale=yscale,xlim=xlim,ylim=ylim)
                ax3[0,1].legend(loc=2,fontsize=12)
                residuals = self.sqwfitmodel[i]-self.sqwsymcorrnorm[i]
                ax3[1,1].errorbar(self.omegasym,residuals,yerr = self.sqwerrorsymcorrnorm[i],
                        fmt='.',capsize=3,color='r',alpha=0.9,label='Residuals');  
                ax3[1,1].plot(self.omegasym,np.zeros_like(self.omegasym),'k',lw=3);  
                ax3[1,1].set(xlabel=r'$\hbar\omega$',ylabel=r'Residual')
                ax3[1,1].legend(loc=4,fontsize=14)

                chi2_red, ndof, pval = calc_chi2(self.sqwsymcorrnorm[i],self.sqwfitmodel[i],
                                            self.sqwerrorsymcorrnorm[i] )
                d = {r"$\chi^2$=":"{:.2f}".format(chi2_red),
                     }
                text = nice_string_output(d, extra_spacing=0, decimals=3)
                add_text_to_ax(0.05, 0.7, text, ax3[0,1], fontsize=16)

            if not showplot:
                plt.close(fig3)
            if saveplot and i % 2:
                fig3.savefig('./figure/'+self.filename+str(i)+'_fittedmodelcomparision.png', dpi=600) 
