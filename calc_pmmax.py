import os,sys

def print_usage():
    print("################################ Usage ################################")
    print("# Usage:                                                              #")
    print("#    -GenSamples: convert mass measurements and their uncertainties   #")
    print("#        to data points and store in the mass_sample.csv file         #")
    print("#    -RunHyperModel argv: constrain the model parameters with the     #")
    print("#        data points; argv determines which model and which group of  #")
    print("#        data points to use                                           #")
    print("#    -PlotFigure: generate the corner plot for model parameters and   #")
    print("#        reconstruct the mass distribution of neutron stars           #")
    print("#    -h or --help: show this message                                  #")
    print("# Note: 1. please modify the outdir to your path                      #")
    print("#       2. you may need to install bilby package via:                 #")
    print("#          conda install -c conda-forge bilby                         #")
    print("#######################################################################")
    exit()
try:
    if sys.argv[1]=='-h' or sys.argv[1]=='--help':
        print_usage()
    elif sys.argv[1]=='-GenSamples':
        generate_sample = 1
        run_hyper_model = 0
        plot_figures = 0
    elif sys.argv[1]=='-RunHyperModel':
        generate_sample = 0
        run_hyper_model = 1
        plot_figures = 0
        work_id = int(sys.argv[2])
    elif sys.argv[1]=='-PlotFigure':
        generate_sample = 0
        run_hyper_model = 0
        plot_figures = 1
    else:
        print("Invalid arguments! Please use -h for more information!")
        exit()
except IndexError:
    print("Invalid arguments! Please use -h for more information!")
    exit()

import numpy as np
use_server = 0
if use_server:
    outdir = '/data/home/tangsp/nsmass/'
    pool_size = 16
else:
    outdir = '/Users/tangsp/Code/Result/nsmass/'
    pool_size = 1

load_rawdata = 1
if load_rawdata:
    f_mt_data = [('J0024-7204H', 0.001927, 1.665, 0.007), ('J1748-2021B', 0.0002266235, 2.69, 0.071), \
                 ('J1748-2446I', 0.003658, 2.17, 0.02), ('J1748-2446J', 0.013066, 2.20, 0.04), \
                 ('J1824-2452C', 0.006553, 1.616, 0.007), ('J2140-2311B', 0.2067, 2.53, 0.08), \
                 ('J1750-37A', 0.0518649, 1.97, 0.15), ('J1411+2551', 0.1223898, 2.538, 0.022), \
                 ('J1518+4904', 0.115988, 2.7183, 0.0007), ('J1811-1736', 0.128121, 2.57, 0.10), \
                 ('J1930-1852', 0.34690765, 2.54, 0.03), ('J1946+2052', 0.268184, 2.50, 0.04), \
                 ('J1325-6253', 0.1415168, 2.57183, 0.06), ('J1759+5036', 0.081768, 2.62, 0.03), \
                 ('J1823-3021G', 0.0123, 2.65, 0.07), ('J1748-2446an', 0.0242, 2.97, 0.52), \
                 ('J1018-1523', 0.238062, 2.3, 0.3)]

    f_q_data = [('J1740-5340', 0.002644, 5.85, 0.13), ('J1816+4510', 0.0017607, 9.54, 0.21), \
                ('J1048+2339', 0.01, 5.15, 0.19), ('J1431-4715', 0.000885, 10.42, 0.11), \
                ('J1622-0315', 0.000436, 14.29, 0.20), ('J1628-3205', 0.00171, 8.33, 0.21)]

    m_lu_data = [('4U 1700-377', 1.96, 0.19, 0.19), ('Cyg X-2', 1.71, 0.21, 0.21), \
                 ('SMC X-1', 1.21, 0.12, 0.12), ('Cen X-3', 1.57, 0.16, 0.16), \
                 ('XTE J2123-058', 1.53, 0.42, 0.42), ('X 1822-371', 1.96, 0.36, 0.36), \
                 ('OAO 1657-415', 1.74, 0.3, 0.3), ('4U1702-429', 1.9, 0.3, 0.3), \
                 ('J1913+1102B', 1.27, 0.03, 0.03), ('4U 1538-522', 1.02, 0.17, 0.17), \
                 ('LMC X-4', 1.57, 0.11, 0.11), ('Her X-1', 1.07, 0.36, 0.36), \
                 ('2S 0921-630', 1.44, 0.1, 0.1), ('EXO 1722-363', 1.91, 0.45, 0.45), \
                 ('SAX J1802.7-2017', 1.57, 0.25, 0.25), ('XTE J1855-026', 1.41, 0.24, 0.24), \
                 ('J0453+1559A', 1.559, 0.005, 0.005), ('J0453+1559B', 1.174, 0.004, 0.004), \
                 ('J1906+0746A', 1.291, 0.011, 0.011), ('J1906+0746B', 1.322, 0.011, 0.011), \
                 ('B1534+12B', 1.3330, 0.0002, 0.0002), ('B1534+12A', 1.3455, 0.0002, 0.0002), \
                 ('B1913+16A', 1.4398, 0.0002, 0.0002), ('B1913+16B', 1.3886, 0.0002, 0.0002), \
                 ('B2127+11CA', 1.358, 0.01, 0.01), ('B2127+11CB', 1.354, 0.01, 0.01), \
                 ('J0737-3039A', 1.338185, 0.000014, 0.000012), ('J0737-3039B', 1.248868, 0.000011, 0.000013), \
                 ('J1756-2251A', 1.341, 0.007, 0.007), ('J1756-2251B', 1.230, 0.007, 0.007), \
                 ('J1807-2500BA', 1.3655, 0.0021, 0.0021), ('J1807-2500BB', 1.2064, 0.0020, 0.0020), \
                 ('J2045+3633', 1.251, 0.021, 0.021), ('J2053+4650', 1.4, 0.18, 0.21), \
                 ('J1713+0747', 1.35, 0.07, 0.07), ('B1855+09', 1.37, 0.10, 0.13), \
                 ('J0751+1807', 1.64, 0.15, 0.15), ('J1141-6545', 1.27, 0.01, 0.01), \
                 ('J1738+0333', 1.47, 0.06, 0.07), ('J1614-2230', 1.908, 0.016, 0.016), \
                 ('J0348+0432', 2.01, 0.04, 0.04), ('J2222-0137', 1.831, 0.01, 0.01), \
                 ('J2234+0611', 1.353, 0.017, 0.014), ('J1949+3106', 1.34, 0.15, 0.17), \
                 ('J1012+5307', 1.72, 0.16, 0.16), ('J0437-4715', 1.44, 0.07, 0.07), \
                 ('J1909-3744', 1.492, 0.014, 0.14), ('J1802-2124', 1.24, 0.11, 0.11), \
                 ('J1911-5958A', 1.34, 0.08, 0.08), ('J2043+1711', 1.38, 0.13, 0.12), \
                 ('J0337+1715', 1.4359, 0.0003, 0.0003), ('J1946+3417', 1.828, 0.022, 0.022), \
                 ('J1918-0642', 1.29, 0.09, 0.1), ('J1829+2456A', 1.306, 0.007, 0.007), \
                 ('J0045-7319', 1.58, 0.34, 0.34), ('J1023+0038', 1.65, 0.16, 0.19), \
                 ('J1903+0327', 1.666, 0.012, 0.01), ('J2129-0429', 1.74, 0.18, 0.18), \
                 ('J2339-0533', 1.47, 0.09, 0.09), ('J0514-4002AA', 1.25, 0.06, 0.05), \
                 ('J0514-4002AB', 1.22, 0.05, 0.06), ('J0621+1002', 1.53, 0.2, 0.1), \
                 ('J1748-2446am', 1.649, 0.11, 0.037), ('J0509+3801B', 1.34, 0.08, 0.08), \
                 ('J0509+3801A', 1.46, 0.08, 0.08), ('J1757-1854B', 1.3406, 0.0005, 0.0005), \
                 ('J1757-1854A', 1.3922, 0.0005, 0.0005), ('J1950+2414', 1.496, 0.023, 0.023), \
                 ('4U 1608-52', 1.57, 0.29, 0.3), ('4U 1724-207', 1.81, 0.37, 0.25), \
                 ('KS 1731-260', 1.61, 0.37, 0.35), ('EXO 1745-248', 1.65, 0.31, 0.21), \
                 ('SAX J1748.9-2021', 1.81, 0.37, 0.25), ('4U 1820-30', 1.77, 0.28, 0.25), \
                 ('B2303+46', 1.30, 0.46, 0.13), ('B1802-07', 1.26, 0.17, 0.08), \
                 ('J1829+2456B', 1.299, 0.007, 0.007), ('J1913+1102A', 1.62, 0.03, 0.03), \
                 ('J1555-2908', 1.67, 0.05, 0.07), ('J1723-2837', 1.22, 0.2, 0.26), \
                 ('J1741+1351', 1.14, 0.25, 0.43), ('J0030+0451', 1.34, 0.16, 0.15), \
                 ('GW170817A', 1.47, 0.07, 0.09), ('GW170817B', 1.27, 0.07, 0.06), \
                 ('GW190425A', 1.4, 0.21, 0.16), ('GW190425B', 1.95, 0.22, 0.37), \
                 ('GW191219', 1.17, 0.06, 0.07), ('4FGL J2039.5-5617', 1.3, 0.1, 0.155), \
                 ('GW200105', 1.91, 0.24, 0.33), ('GW200115', 1.44, 0.29, 0.85), \
                 ('1FGL J1417.7-4407', 1.62, 0.17, 0.43), ('2FGL J0846.0+2820', 1.96, 0.41, 0.41), \
                 ('3FGL J0212.1+5320', 1.85, 0.26, 0.32), ('3FGL J0427.9-6704', 1.86, 0.10, 0.11), \
                 ('J013236.7+303228', 2.0, 0.4, 0.4), ('J1811-2405', 2.0, 0.5, 0.8), \
                 ('J0955-6150', 1.71, 0.02, 0.02), ('J1518+0204B', 2.08, 0.19, 0.19), \
                 ('Vela X-1', 2.12, 0.16, 0.16), ('J1600-3053', 2.3, 0.6, 0.7), \
                 ('3FGL J2039.6-5618', 2.04, 0.25, 0.37), ('J1959+2048', 1.81, 0.07, 0.07), \
                 ('PSR J0740+6620', 2.0505, 0.066, 0.067), ('J1311-3430', 2.22, 0.10, 0.10), \
                 ('J1653-0158', 2.15, 0.16, 0.16), ('J1810+1744', 2.11, 0.04, 0.04), \
                 ('J2215+5135', 2.25638, 0.09, 0.10), ('PSR J0952-0607', 2.28465122, 0.17, 0.17), \
                 ('J1017-7156', 2.0, 0.8, 0.8), ('J1022+1001', 1.44, 0.44, 0.44), \
                 ('J1125-6014', 1.5, 0.2, 0.2), ('J1528-3146', 1.61, 0.13, 0.14), \
                 ('J1301+0833', 1.6, 0.25, 0.22)]

    if False:
        source_name_array = np.empty(shape=(0), dtype=str)
        for data_type,source_info in enumerate([f_mt_data, f_q_data, m_lu_data]):
            for _,(source_name, median, l, u) in enumerate(source_info):
                source_name = source_name.replace(".", "")
                source_name = source_name.replace(" ", "")
                source_name_array = np.append(source_name_array, source_name)
        unique_name_array, counts = np.unique(source_name_array, return_counts=True)
        if len(source_name_array)!=len(unique_name_array):
            print(unique_name_array[counts!=1])
            exit()

if generate_sample:
    from pandas.core.frame import DataFrame
    columns_name, sample_size = [], np.inf
    def loglike_mass(mp, x, fhat, mu, sig, m_type):
        if m_type==0:
            mt, mtmu, mtsig = x, mu, sig
            if mp>=mt:
                return np.nan_to_num(-np.inf)
            inroot = 1-fhat**(2./3.)*mt**(4./3.)/(mt-mp)**2
            if inroot>0:
                return -((mt-mtmu)/mtsig)**2/2. - (np.log(3.) + 2*np.log1p(-mp/mt) + 1./3.*np.log(fhat*mt**2) + 1./2.*np.log(inroot))
            else:
                return np.nan_to_num(-np.inf)
        else:
            # see https://github.com/farr/AlsingNSMassReplication
            q, qmu, qsig = x, mu, sig
            opq43_q2 = (1+q)**(4./3.)/q**2
            inroot = 1-(fhat/mp)**(2./3.)*opq43_q2
            if inroot>0:
                return -((q-qmu)/qsig)**2/2. + np.log(opq43_q2) - (np.log(3.) + 1./3.*np.log(fhat) + 2./3.*np.log(mp) + 1./2.*np.log(inroot))
            else:
                return np.nan_to_num(-np.inf)

    from bilby.core.likelihood import Likelihood   
    class NSMassLikelihood(Likelihood):
        def __init__(self):
            Likelihood.__init__(self, parameters={'mp':None, 'x':None, 'fhat':None, 'mu':None, 'sig':None, 'm_type':None})
        def log_likelihood(self):
            return loglike_mass(self.parameters['mp'], self.parameters['x'], self.parameters['fhat'], \
                self.parameters['mu'], self.parameters['sig'], self.parameters['m_type'])
    likelihood = NSMassLikelihood()

    from bilby.core.sampler import run_sampler
    from bilby.core.prior import Uniform, TruncatedGaussian, Constraint, PriorDict
    def constraint_pars(params):
        params['constraint'] = np.sign(params['x']-params['mp'])
        return params
    for data_type,source_info in enumerate([f_mt_data, f_q_data]):
        for _,(source_name,fhat,mu,sigma) in enumerate(source_info):
            priors = PriorDict(conversion_function=constraint_pars)
            priors['m_type'] = data_type
            if data_type==0:
                dlogz = 0.05
                priors['mp'] = Uniform(0.9, 2.9, unit='$M_{\\odot}$', latex_label='$M_{\\rm p}$')
                priors['x'] = TruncatedGaussian(mu=mu, sigma=sigma*2, minimum=0.9, maximum=np.inf, unit='$M_{\\odot}$', latex_label='$M_{\\rm T}$')
                priors['constraint'] = Constraint(minimum=0.9, maximum=1.1)
            else:
                dlogz = 1e-4
                priors['mp'] = Uniform(0.9, 2.9, unit='$M_{\\odot}$', latex_label='$M_{\\rm p}$')
                priors['x'] = TruncatedGaussian(mu=mu, sigma=sigma*2, minimum=0, maximum=np.inf, latex_label='$q$')
                priors['constraint'] = Constraint(minimum=-np.inf, maximum=np.inf)
            priors['fhat'] = fhat
            priors['mu'] = mu
            priors['sig'] = sigma
            
            source_name = source_name.replace(".", "")
            source_name = source_name.replace(" ", "")
            columns_name.append(source_name)
            if os.path.exists(outdir+'sample_run/'+source_name+'_result.json'):
                continue
            results = run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', check_point=False, check_point_plot=False, \
                use_ratio=False, queue_size=pool_size, dlogz=dlogz, npoints=1000, outdir=outdir+'sample_run', label=source_name)
            results.plot_corner()

    from bilby.core.result import read_in_result as rr
    for data_type,source_info in enumerate([f_mt_data, f_q_data]):
        for _,(source_name,_,_,_) in enumerate(source_info):
            source_name = source_name.replace(".", "")
            source_name = source_name.replace(" ", "")
            fp = rr(outdir+'sample_run/'+source_name+'_result.json')
            print(source_name, len(fp.posterior['mp']))
            sample_size = min(len(fp.posterior['mp']), sample_size)
    mass_sample = np.empty(shape=(0,sample_size))
    for data_type,source_info in enumerate([f_mt_data, f_q_data]):
        for _,(source_name,_,_,_) in enumerate(source_info):
            source_name = source_name.replace(".", "")
            source_name = source_name.replace(" ", "")
            fp = rr(outdir+'sample_run/'+source_name+'_result.json')
            resampled_mass = np.random.choice(fp.posterior['mp'], size=sample_size, replace=False)
            mass_sample = np.vstack((mass_sample, resampled_mass))

    from scipy.stats import norm
    from scipy.integrate import quad
    from scipy.optimize import ridder
    from bilby.core.prior import Interped
    def asym_norm(c, d, x):
        scale = 2./(d*(c+1./c))
        return (x<=0)*scale*norm.pdf(c*x/d)+(x>0)*scale*norm.pdf(x/(d*c))
    def int_asymnorm(c, d, lb, ub):
        return quad(lambda x: asym_norm(c, d, x), lb, ub)[0]

    for source_info in m_lu_data:
        source_name = source_info[0].replace(".", "")
        columns_name.append(source_name.replace(" ", ""))
        left_bd = max(0.9, source_info[1]-5*source_info[2])
        right_bd = min(2.9, source_info[1]+5*source_info[3])
        ns_mass = np.linspace(left_bd, right_bd, 10000)
        if left_bd!=0.9:
            ns_mass = np.insert(ns_mass, 0, 0.9)
        if right_bd!=2.9:
            ns_mass = np.append(ns_mass, 2.9)
        par_c = np.sqrt(source_info[3]/source_info[2])
        par_d = ridder(lambda d: int_asymnorm(par_c, d, -source_info[2], source_info[3])-0.683, \
            min(source_info[2],source_info[3])/5., max(source_info[2], source_info[3])*5.)
        mass_model = Interped(xx=ns_mass, yy=asym_norm(par_c, par_d, ns_mass-source_info[1]), minimum=0.9, maximum=2.9)
        mass_sample = np.vstack((mass_sample, mass_model.sample(sample_size)))
    df = DataFrame(mass_sample.T, columns=columns_name)
    df.to_csv(outdir+'mass_sample.csv')

if run_hyper_model:
    label = ['mns_hyper_all_rot_doublegau', 'mns_hyper_notwice_rot_doublegau', 'mns_hyper_nohigh_rot_doublegau', \
        'mns_hyper_all_rot_gaucauchy', 'mns_hyper_notwice_rot_gaucauchy', 'mns_hyper_nohigh_rot_gaucauchy'][work_id]
    method_choice = label.split('_')[1]
    data_choice = label.split('_')[2]
    model_choice = label.split('_')[-1]
    if data_choice in ['all']:
        exclude_sources = []
    elif data_choice in ['notwice']:
        exclude_sources = ['J0030+0451', 'PSRJ0740+6620', 'GW170817A', 'GW170817B']
    elif data_choice in ['nohigh']:
        exclude_sources = ['J2215+5135', 'PSRJ0952-0607', 'J1311-3430', 'J1748-2021B']
    else:
        exit()
    import pandas as pd
    from pandas.core.frame import DataFrame
    samples = []
    df = pd.read_csv(outdir+'mass_sample.csv')
    for data_type,source_info in enumerate([f_mt_data, f_q_data, m_lu_data]):
        for _,(source_name,_,_,_) in enumerate(source_info):
            source_name = source_name.replace(".", "")
            source_name = source_name.replace(" ", "")
            if source_name in exclude_sources:
                print('exclude ', source_name)
                continue
            #if data_choice in ['nohigh']:
            #    u_mass = np.percentile(df[source_name], 84.135)
            #    if u_mass>2.3:
            #        print('exclude ', source_name)
            #        continue
            samples.append(DataFrame({'mass':df[source_name], 'prior':1./(2.9-0.9)}, columns=['mass', 'prior']))
    log_evidences = [0]*len(samples)
    from scipy.stats import norm, cauchy
    from bilby.core.sampler import run_sampler
    from bilby.core.prior import Uniform, Constraint, PriorDict
    def constraint_pars(params):
        params['constraint'] = \
            np.sign(params['mu2']-params['mu1'])+\
            np.sign(params['mmax']-params['mu2'])+\
            np.sign(params['sig2']-params['sig1'])
        return params
    priors = PriorDict(conversion_function=constraint_pars)
    priors['mmin'] = 0.9
    priors['mu1'] = Uniform(0.9, 2.9, unit='$M_{\\odot}$', latex_label='$\\mu_1$')
    priors['mu2'] = Uniform(0.9, 2.9, unit='$M_{\\odot}$', latex_label='$\\mu_2$')
    priors['sig1'] = Uniform(0.01, 2., unit='$M_{\\odot}$', latex_label='$\\sigma_1$')
    priors['sig2'] = Uniform(0.01, 2., unit='$M_{\\odot}$', latex_label='$\\sigma_2$')
    priors['r'] = Uniform(0.1, 0.9, latex_label='$r$')
    priors['mmax'] = Uniform(1.9, 2.9, unit='$M_{\\odot}$', latex_label='$M_{\\rm max}$')
    priors['constraint'] = Constraint(minimum=2.9, maximum=3.1)

    from bilby.hyper.likelihood import HyperparameterLikelihood
    if model_choice in ['doublegau']:
        def hyper_prior(data, mmin, mu1, mu2, sig1, sig2, r, mmax):
            g1 = norm(loc=mu1, scale=sig1)
            g2 = norm(loc=mu2, scale=sig2)
            return (r*g1.pdf(data['mass'])/(g1.cdf(mmax)-g1.cdf(mmin))+\
                (1-r)*g2.pdf(data['mass'])/(g2.cdf(mmax)-g2.cdf(mmin)))*\
                (data['mass']>mmin)*(data['mass']<mmax)
    else:
        def hyper_prior(data, mmin, mu1, mu2, sig1, sig2, r, mmax):
            g1 = norm(loc=mu1, scale=sig1)
            g2 = cauchy(loc=mu2, scale=sig2)
            return (r*g1.pdf(data['mass'])/(g1.cdf(mmax)-g1.cdf(mmin))+\
                (1-r)*g2.pdf(data['mass'])/(g2.cdf(mmax)-g2.cdf(mmin)))*\
                (data['mass']>mmin)*(data['mass']<mmax)
    likelihood = HyperparameterLikelihood(posteriors=samples, hyper_prior=hyper_prior, \
        log_evidences=log_evidences, max_samples=1000)

    results = run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', \
        use_ratio=False, queue_size=pool_size, dlogz=0.05, npoints=1000, outdir=outdir, label=label)
    results.plot_corner()

if plot_figures:
    from scipy.stats import norm, cauchy
    def handle_json(fpath, sort_keys=None, return_best=False):
        from bilby.core.result import read_in_result
        fp = read_in_result(fpath)
        llh = fp.log_likelihood_evaluations
        idx = np.where(llh==max(llh))[0][0]
        if sort_keys is None:
            sample = fp.samples
            sort_keys = fp.search_parameter_keys
        else:
            sorted_sample = np.empty(shape=(0, len(llh)))
            for key in sort_keys:
                if key=='likelihood':
                    sorted_sample = np.vstack((sorted_sample, llh))
                else:
                    sorted_sample = np.vstack((sorted_sample, fp.posterior[key]))
            sample = sorted_sample.T
        print('maximum likelihood: {}-th, {}'.format(idx, llh[idx]))
        print('corresponding pars: ', dict(zip(sort_keys, sample[idx])))
        if return_best:
            return sample, idx
        return sample

    def plot_corner(data, injection_values=None, labels=None, colors=None, show_orders=None, keys=None, \
        precisions=None, percentiles=[5, 50, 95], levels=None, title_size=3.6, lw=0.45, smooth=0.9, scale_ys=None, \
        plot_densities=None, range_var=None, plt_rcParams=None, show_vlines=True, truth_color='gray', figsize=None, \
        textloc=[0.8, 0.95, 0.00, 0.035], legend_text=True, filename=None, **corner_kwargs):
        """Compare multi data with package corner.

        Parameters
        ----------
        data: 3D array like
            Samples to compare, must take the form: [ds1, ..., dsN], where dsN: [var1, ..., varN].
        labels: list of strings of size data.shape[0], optional
            Labels of different datasets.
        colors: list of colors of size data.shape[0], optional
            Colors of different datasets.
        show_orders: list of int of size data.shape[0], optional
            The orders of data to plot.
        keys: list of strings of size data.shape[1], optional
            Keys of different variables.
        precisions: list of int of size data.shape[1], optional
            Precision of titles of variables.
        percentiles: list of float of the form: [lower, median, upper], optional
            Percentiles to show in the 1D histograms and the titles.
        levels: list of float, optional
            Percentile regions to show.
        title_size: float, optional
            Size of the titles, labels, and ticks.
        lw: float, optional
            Line width in the 1D hist plots.
        smooth: float, optional
            Smooth level that will be directly transfered to the corner.
        scale_ys: list of float of size data.shape[1], optional
            To scale y_lim for each variable.
        plot_densities: list of bool of size data.shape[0], optional
            Whether to plot the density of each data set.
        plt_rcParams: dict, optional
            Key ward arguments that will be updated to the pyplot.rcParams.
        show_vlines: bool, optional
            Whether to show the vertical lines in the 1D hist plots.
        figsize: (float, float), optional
            The size of the figure.
        textloc: list of float of the form: [x_begin, y_begin, x_shift, y_shift], optional
            Location of the legend, useful only if the legend_text is set to be True.
        legend_text: bool, optional
            Whether to set legend to annotation form or the original pyplot.legend() form.
        filename: str
            Save to path 'filename' if given, show it directly if it is default value.
        corner_kwargs: kwargs that will be directly transfered to the corner.corner().
        """
        import corner
        from matplotlib import lines as mpllines

        # set rcParams
        default_rcParams = {'lines.linewidth': lw+0.6, 'axes.linewidth': lw, 'font.size': 5.0, \
                            'font.sans-serif':['DejaVu Sans'], 'grid.linewidth': lw-0.2}
        if plt_rcParams is not None:
            default_rcParams = default_rcParams.update(plt_rcParams)
        plt.rcParams.update(default_rcParams)
        # initialize parameters
        da_shape = [len(data), len(data[0])]
        colors = ['y', 'm', 'c', 'r', 'g', 'b', 'k'][:da_shape[1]] if colors is None else colors
        show_orders = [i for i in range(da_shape[0])] if show_orders is None else show_orders
        keys = ['x{}'.format(i+1) for i in range(da_shape[1])] if keys is None else keys
        precisions = [2]*da_shape[1] if precisions is None else precisions
        quantiles = [percentiles[i]/100. for i in [0, 2]] if show_vlines else []
        levels = [0.6827, 0.9545, 0.9973] if levels is None else levels
        scale_ys = [1 for i in range(da_shape[1])] if scale_ys is None else scale_ys
        plot_densities = [False for i in range(da_shape[0])] if plot_densities is None else plot_densities

        inner_key = [chr(letter).lower() for letter in range(65, 91)]
        tot_groups = len(data)
        dims = len(data[0])
        select_key = inner_key[0:dims]
        group_samples = [dict(zip(select_key, subdata)) for subdata in data]

        if range_var is None:
            temp_mins = {key:min([min(samples[key]) for samples in group_samples]) for key in select_key}
            temp_maxs = {key:max([max(samples[key]) for samples in group_samples]) for key in select_key}
            shifts = {key:(temp_maxs[key]-temp_mins[key])/100. for key in select_key}
            mins = [temp_mins[key]-shifts[key] for key in select_key]
            maxs = [temp_maxs[key]+shifts[key] for key in select_key]
            #minn = [min([min(data[j][i]) for j in range(da_shape[0])]) for i in range(da_shape[1])]
            #maxx = [max([max(data[j][i]) for j in range(da_shape[0])]) for i in range(da_shape[1])]
            range_var = [(it1, it2) for it1, it2 in zip(mins, maxs)]
        figsize = (da_shape[1], da_shape[1]) if figsize is None else figsize
        kwargs_init = dict(labels=keys, smooth=smooth, bins=25, levels=levels, show_titles=False, \
            quantiles=quantiles, label_kwargs=dict(fontsize=title_size+2), range=range_var, \
            max_n_ticks=3, plot_datapoints=False, fill_contours=False)
        kwargs_init.update(corner_kwargs)
        # create figure
        fig = plt.figure(figsize=figsize)
        for i,(da, color, pld) in enumerate([(data[i], colors[i], plot_densities[i]) for i in show_orders]):
            kwargs_init.update({'hist_kwargs':{'density':True, 'color':color, 'lw':lw}, \
                'plot_density':pld, 'truths':None, 'truth_color':truth_color})
            corner.corner(np.array(da).T, color=color, fig=fig, labelpad=title_size*0.01, **kwargs_init)
            #corner.overplot_points(fig, injection_values[i], marker="s", size=0.5)
            if injection_values is not None:
                axes_i = np.array(fig.axes).reshape((dims, dims))
                for j in range(dims):
                    ax_j = axes_i[j, j]
                    ax_j.axvline(injection_values[i][j], color=truth_color, lw=lw-0.2)
                for yi in range(dims):
                    for xi in range(yi):
                        ax_yx = axes_i[yi, xi]
                        ax_yx.axvline(injection_values[i][xi], color=truth_color, lw=lw-0.2)
                        ax_yx.axhline(injection_values[i][yi], color=truth_color, lw=lw-0.2)
                        ax_yx.scatter([injection_values[i][xi]], [injection_values[i][yi]], color=truth_color, marker='s', s=15)

        axes = fig.get_axes()
        ndim = int(np.sqrt(len(axes)))
        # set title
        title_axes = [axes[i*ndim+i] for i in range(ndim)]
        tot_groups = len(data)
        if tot_groups%2==0:
            locs = ['left', 'right']
            pad_size = 2
        else:
            locs = [['left', 'center', 'right'], ['center', 'left', 'right'], ['left', 'right', 'center']]
            locs = locs[tot_groups%3]
            pad_size = 3
        for j, da in enumerate(data):
            for i, (dd, precs, ax) in enumerate(zip(da, precisions, title_axes)):
                temp_ax = ax.twiny()
                probs = np.percentile(dd, percentiles)
                if precs<=0:
                    format_string = r"${:.0f}_{{-{:.0f}}}^{{+{:.0f}}}$"
                else:
                    format_string = r"${:."+str(precs)+r"f}_{{-{:."+str(precs)+r"f}}}^{{+{:."+str(precs)+r"f}}}$"
                title_str = format_string.format(np.round(probs[1], precs), np.round(probs[1]-probs[0], precs), \
                                                 np.round(probs[2]-probs[1], precs))
                tloc = 'center' #locs[j%2] if tot_groups==1 or tot_groups%2==0 else locs[j%3]

                temp_ax.set_title(title_str, fontsize=title_size, color=colors[j], pad=4+j*(title_size+3), loc=tloc)
                temp_ax.set_xticks([])
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(y_min, y_max*np.power(scale_ys[i], 1./da_shape[0]))
            if legend_text and labels is not None:
                plt.annotate(labels[j], xy=(textloc[0]-textloc[2]*j, textloc[1]-textloc[3]*j), \
                             xycoords='figure fraction', color=colors[j], fontsize=title_size+4)
        for ax in axes:
            ax.tick_params(direction='in', labelsize=title_size, length=2., width=lw-0.2)#, rotation=0)
        # plot legend
        axes[0].set_ylabel('PDF', size=title_size+2)
        if not legend_text and labels is not None:
            lines = [mpllines.Line2D([0], [0], lw=0.05, color=color) for color in colors]
            axes[ndim - 1].legend(lines, labels, fontsize=title_size+2)
        if filename is None:
            return fig, axes
        else:
            plt.savefig(filename, dpi=300)

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    label_list = ['mns_hyper_all_rot_doublegau', 'mns_hyper_notwice_rot_doublegau', 'mns_hyper_nohigh_rot_doublegau', \
        'mns_hyper_all_rot_gaucauchy', 'mns_hyper_notwice_rot_gaucauchy', 'mns_hyper_nohigh_rot_gaucauchy']
    corner_content = ['mns_hyper_notwice_rot_doublegau', 'mns_hyper_nohigh_rot_doublegau', \
        'mns_hyper_notwice_rot_gaucauchy', 'mns_hyper_nohigh_rot_gaucauchy']
    addtext = ['Data: Exclude GW+NICER sources, Model: Gaussian+Gaussian', 'Data: Exclude High-mass NSs, Model: Gaussian+Gaussian', \
               'Data: Exclude GW+NICER sources, Model: Gaussian+Cauchy-Lorentz', 'Data: Exclude High-mass NSs, Model: Gaussian+Cauchy-Lorentz']
    corner_data = []
    for j,label in enumerate(label_list):
        method_choice = label.split('_')[1]
        data_choice = label.split('_')[2]
        model_choice = label.split('_')[-1]
        if model_choice in ['doublegau']:
            def hyper_prior(data, mmin, mu1, mu2, sig1, sig2, r, mmax):
                g1 = norm(loc=mu1, scale=sig1)
                g2 = norm(loc=mu2, scale=sig2)
                return (r*g1.pdf(data)/(g1.cdf(mmax)-g1.cdf(mmin))+\
                    (1-r)*g2.pdf(data)/(g2.cdf(mmax)-g2.cdf(mmin)))
        else:
            def hyper_prior(data, mmin, mu1, mu2, sig1, sig2, r, mmax):
                g1 = norm(loc=mu1, scale=sig1)
                g2 = cauchy(loc=mu2, scale=sig2)
                return (r*g1.pdf(data)/(g1.cdf(mmax)-g1.cdf(mmin))+\
                    (1-r)*g2.pdf(data)/(g2.cdf(mmax)-g2.cdf(mmin)))
        fig = plt.figure()
        axis = fig.add_subplot(111)
        cutoff_min, cutoff_max = 0.9, 2.8
        m_min, mu_1, mu_2, sig_1, sig_2, r1, m_max, llk = \
            handle_json(outdir+label+'_result.json', \
            ['mmin', 'mu1', 'mu2', 'sig1', 'sig2', 'r', 'mmax', 'likelihood']).T
        if label in corner_content:
            corner_data.append([mu_1, mu_2, sig_1, sig_2, r1, m_max])

        plot_size = 1000
        take_index = np.random.choice(range(len(llk)), plot_size, replace=False)
        best_index = np.where(llk==np.sort(llk)[-1])[0][0]
        take_index = np.append(take_index, best_index)
        for i,index in enumerate(take_index):
            mm = np.linspace(m_min[index], m_max[index], 100, endpoint=False)
            pp = hyper_prior(mm, m_min[index], mu_1[index], mu_2[index], sig_1[index], sig_2[index], r1[index], m_max[index])
            mm = np.append(np.linspace(cutoff_min, m_min[index], 10), mm)
            pp = np.append(np.zeros(10), pp)
            mm = np.append(mm, np.linspace(m_max[index], cutoff_max, 10))
            pp = np.append(pp, np.zeros(10))
            if i==plot_size:
                axis.plot(mm, pp, lw=1.5, c='red', alpha=1, label='Best Fit')
            else:
                axis.plot(mm, pp, lw=0.5, c='gray', alpha=0.1)
        axis.set_xlim(cutoff_min, cutoff_max)
        axis.set_ylim(-0.05, 5)
        axis.set_xlabel(r'$M_{\rm NS}\ (M_\odot)$')
        axis.set_ylabel('PDF')
        
        inside_axis = fig.add_axes([0.55, 0.575, 0.30, 0.25])
        mlist = np.linspace(1.9,2.9, 100, endpoint=True)
        pct = np.percentile(m_max, [[15.865, 50, 84.135],[5, 50, 95]][0])
        inside_axis.axvline(pct[0], c='g', ls=':')
        inside_axis.axvline(pct[2], c='g', ls=':')
        mmax_kde = gaussian_kde(m_max)
        pdf_of_mmax = mmax_kde(mlist)
        inside_axis.plot(mlist, pdf_of_mmax, c='g')
        inside_axis.fill_between(mlist, 0, pdf_of_mmax, alpha=0.4, edgecolor='g', hatch='', facecolor='g')
        inside_axis.set_xlabel(r'$M_{\rm max}\,(M_\odot)$', fontsize=10)
        inside_axis.set_ylabel('PDF', fontsize=10)
        inside_axis.tick_params(direction='in', which='both', labelsize=8)
        inside_axis.set_xlim(1.9,2.9)
        inside_axis.set_ylim(0)
        inside_axis.set_title(r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(pct[1], \
            pct[1]-pct[0], pct[2]-pct[1]), fontsize=8, color='g', pad=0, loc='center')

        axis.tick_params(direction='in', which='both', labelsize=12)
        axis.legend(loc='upper left', fontsize=12)
        plt.savefig(outdir+'mass_dist_{}.pdf'.format(j))

    textloc = [0.3, 0.95, 0.00, 0.035]
    if len(corner_data)>1:
        show_density, show_vlines = [False]*len(corner_data), False
    else:
        show_density, show_vlines = [True], True
    colors, scale_ys, range_var = ['black', 'blue', 'orange', 'purple', 'pink', 'red'], None, None
    scale_ys = [1, 1, 1, 1, 1, 1]
    keys = [r'$\mu_1/M_\odot$', r'$\mu_2/M_\odot$', r'$\sigma_1/M_\odot$', r'$\sigma_2/M_\odot$', r'$r$', r'$M_{\rm max}/M_\odot$']
    fig, axes = plot_corner(corner_data, injection_values=None, labels=addtext, \
        smooth=None, plot_densities=show_density, keys=keys, show_vlines=show_vlines, figsize=(8,8), \
        percentiles=[[15.865, 50, 84.135], [5, 50, 95]][1], levels=[0.6827, 0.9][1:], colors=colors, \
        title_size=7, lw=0.9, scale_ys=scale_ys, textloc=textloc, range_var=range_var, filename=None)
    plt.savefig(outdir+'meta_corner.pdf')
