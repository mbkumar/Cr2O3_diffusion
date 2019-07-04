#!/usr/bin/env python

__author__ = "Geoffroy Hautier, Bharat Medasani, Danny Broberg"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Bharat Medasani"
__email__ = "mbkumar@gmail.com"
__status__ = "Development"
__date__ = "December 1, 2016"

from collections import defaultdict
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('agg')

from pymatgen.util.plotting import pretty_plot

from finite_temp_defects_analyzer import IntrinsicCarrier

class DefectPlotter(object):
    """
    Class performing all the typical plots from a defects study
    """

    def __init__(self, analyzer):
        """
        Args:
            analyzer: DefectsAnalyzer object 
        """

        self._analyzer = analyzer

    def get_plot_form_energy(self, xlim=None, ylim=None):
        """
        Formation energy vs Fermi energy plot
        Args:
            xlim:
                Tuple (min,max) giving the range of the x (fermi energy) axis
            ylim:
                Tuple (min,max) giving the range for the formation energy axis
        Returns:
            a matplotlib object

        """
        if xlim is None:
            xlim = (-0.5, self._analyzer._band_gap+1.5)
        max_lim = xlim[1]
        min_lim = xlim[0]
        nb_steps = 10000
        step = (max_lim-min_lim) / nb_steps
        x = [min_lim+step*i for i in range(nb_steps)]
        x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/nb_steps)
        y = {}
        trans_level_pt = {}
        for t in self._analyzer._get_all_defect_types():
            y_tmp = []
            trans_level = []
            prev_min_q, cur_min_q = None, None
            for x_step in x:
                min = 10000
                for i, dfct in enumerate(self._analyzer._defects):
                    if dfct.name == t:
                        val = self._analyzer._formation_energies[i] + \
                                dfct.charge*x_step
                        if val < min:
                            min = val
                            cur_min_q = dfct.charge
                if prev_min_q is not None:
                    if cur_min_q != prev_min_q:
                        trans_level.append((x_step, min))
                prev_min_q = cur_min_q
                y_tmp.append(min)
            trans_level_pt[t] =  trans_level
            y[t] = y_tmp

        y_vals = np.array(y.values())
        y_min = y_vals.min()
        y_max = y_vals.max()

        width, height = 12, 8
        import matplotlib.pyplot as plt
        plt.clf()
        import matplotlib.cm as cm
        colors=cm.Dark2(np.linspace(0, 1, len(y)))
        for cnt, c in enumerate(y):
            plt.plot(x, y[c], linewidth=3, color=colors[cnt])
        plt.plot([min_lim, max_lim], [0, 0], 'k-')

        def get_legends(types):
            legends = []
            for name in types:
                for dfct in self._analyzer._defects:
                    if name == dfct.name:
                        sub_str = '_{'+dfct.site.species_string+'}$'
                        if 'vac' in name:
                            base = '$Vac'
                        elif 'inter' in name:
                            flds = name.split('_')
                            base = '$'+flds[2]
                            sub_str = '_{i_{'+','.join(flds[3:5])+'}}$'
                        else:
                            base = '$'+name.split('_')[2] 
                        legend = base + sub_str
                        break
                legends.append(legend)
            return legends

        if len(y.keys())<5:
            plt.legend(get_legends(y.keys()), fontsize=1.8*width, loc=8)
        else: #note to Bharat -> so far I have been having to adjust the bbox_to_anchor based on number of defects
            #would like to be able to have this be automagic enough that legend won't interfere with plot when lots
            # of defects exist....
            plt.legend(get_legends(y.keys()), fontsize=1.8*width, ncol=3,
                       loc='lower center', bbox_to_anchor=(.5,-.6))
        for cnt, c in enumerate(y):
        #for c in y:
           # plt.plot(x, y[c], next(linecycler), linewidth=6, color=colors[cnt])
            x_trans = [pt[0] for pt in trans_level_pt[c]]
            y_trans = [pt[1] for pt in trans_level_pt[c]]
            plt.plot(x_trans, y_trans,  marker='*', color=colors[cnt], markersize=12, fillstyle='full')
        plt.axvline(x=0.0, linestyle='--', color='k', linewidth=3)
        plt.axvline(x=self._analyzer._band_gap, linestyle='--', color='k',
                    linewidth=3)
        if ylim is not None:
            plt.ylim(ylim)
        else:
            plt.ylim((y_min-0.1, y_max+0.1))
        plt.xlabel("Fermi energy (eV)", size=2*width)
        plt.ylabel("Defect Formation Energy (eV)", size=2*width)
        return plt

    def plot_conc_temp(self, me=[1.0, 1.0, 1.0], mh=[1.0, 1.0, 1.0]):
        """
        plot the concentration of carriers vs temperature both in eq and non-eq after quenching at 300K
        Args:
            me:
                the effective mass for the electrons as a list of 3 eigenvalues
            mh:
                the effective mass for the holes as a list of 3 eigenvalues
        Returns;
            a matplotlib object

        """
        temps = [i*100 for i in range(3,20)]
        qi = []
        qi_non_eq = []
        for t in temps:
            qi.append(self._analyzer.get_eq_Ef(t, me, mh)['Qi']*1e-6)
            qi_non_eq.append(
                    self._analyzer.get_non_eq_Ef(t, 300, me, mh)['Qi']*1e-6)

        plt = pretty_plot(12, 8)
        plt.xlabel("temperature (K)")
        plt.ylabel("carrier concentration (cm$^{-3}$)")
        plt.semilogy(temps, qi, linewidth=3.0)
        plt.semilogy(temps, qi_non_eq, linewidth=3)
        plt.legend(['eq','non-eq'])
        return plt

    def plot_carriers_ef(self, temp=300, me=[1.0, 1.0, 1.0], mh=[1.0, 1.0, 1.0]):
        """
        plot carrier concentration in function of the fermi energy
        Args:
            temp:
                temperature
            me:
                the effective mass for the electrons as a list of 3 eigenvalues
            mh:
                the effective mass for the holes as a list of 3 eigenvalues
        Returns:
            a matplotlib object
        """
        plt = get_publication_quality_plot(12, 8)
        qi = []
        efs = []
        for ef in [x * 0.01 for x in range(0, 100)]:
            efs.append(ef)
            qi.append(self._analyzer.get_Qi(ef, temp, me, mh)*1e-6)
        plt.ylim([1e14, 1e22])
        return plt.semilogy(efs, qi)

class FiniteTempDefectPlotter(object):
    """
    Class performing the typical finite temperature plots from a
    defects study
    Args:
        analyzer: FiniteTempDefectsAnalyzer object
        bulk_dos: Dos object of unit cell
        exp_gap: Experimental band gap
    """

    def __init__(self, analyzer, bulk_dos, exp_gap):
        self._analyzer = deepcopy(analyzer)
        self._dos = bulk_dos
        self._gap = exp_gap

    def get_fermi_vs_pressure_plot(self, temps=[300], pressure_limit=(1e-5, 1)):
        """
        E_F vs p_O2 plot for different temperatures
        Args:
            temps: Temperatures in K as list
                Default is 300 K
            pressure_limit: Tuple with lower and upper limits on pressure.
                Defaults are 1e-5 and 1 (Note these are very high values)
        Returns:
            matplotlib plot object containg E_F vs p_O2 plot
        """
        po2s = []
        pressure = pressure_limit[0]
        while pressure < pressure_limit[1]:
            po2s.append(pressure)
            pressure *= 10
        fermi_levels = defaultdict(list)
        for T in temps:
            self._analyzer.set_T(T)
            for po2 in po2s:
                self._analyzer.set_gas_pressure(po2)
                self._analyzer.solve_for_fermi_energy(self._dos, self._gap)
                ef = self._analyzer.fermi_energy
                fermi_levels[T].append(ef)

        width, height = 12, 8
        import matplotlib.pyplot as plt
        plt.clf()
        import matplotlib.cm as cm
        colors=cm.Dark2(np.linspace(0, 1, len(temps)))
        for i, temp in enumerate(temps):
            plt.semilogx(po2s, fermi_levels[temp], linewidth=2, color=colors[i])

        plt.legend(list(map(lambda x: str(x) + ' K', temps)), fontsize=0.8*width,
                   loc='best')
        plt.xlabel("$p_{O_2}$ (atm)", size=width)
        plt.ylabel("$E_F$ (eV)", size=width)
        #plt.title("Varition of Fermi level with T and $p_{O_2}$")
        return plt

    def get_stoichiometry_vs_pressure_plot(self, specie, temps=[300],
                                           pressure_limit=(1e-5, 1)):
        """
        stoichiometry deviation vs p_O2 plot for different temperatures
        Args:
            temps: Temperatures in K as list
                Default is 300 K
            pressure_limit: Tuple with lower and upper limits on pressure.
                Defaults are 1e-5 and 1 (Note these are very high values)
        Returns:
            matplotlib plot object containg E_F vs p_O2 plot
        """
        po2s = []
        pressure = pressure_limit[0]
        while pressure < pressure_limit[1]:
            po2s.append(pressure)
            pressure *= 10
        del_x = defaultdict(list)
        for T in temps:
            self._analyzer.set_T(T)
            for po2 in po2s:
                self._analyzer.set_gas_pressure(po2)
                self._analyzer.solve_for_fermi_energy(self._dos, self._gap)
                del_x[T].append(
                    self._analyzer.get_stoichiometry_deviations()[specie])

        width, height = 12, 8
        import matplotlib.pyplot as plt
        plt.clf()
        import matplotlib.cm as cm
        colors=cm.Dark2(np.linspace(0, 1, len(temps)))
        for i, temp in enumerate(temps):
            plt.semilogx(po2s, del_x[temp], linewidth=2, color=colors[i])

        plt.legend(list(map(lambda x: str(x) + ' K', temps)), fontsize=.8*width,
                   loc='best')
        plt.xlabel("$p_{O_2}$ (atm)", size=width)
        plt.ylabel("$\Delta X$ of {}".format(specie), size=width)
        #plt.title("Deviation from stoichiometry")
        return plt

    def get_concentration_vs_pressure_plot(self, temp,
                                           pressure_limit=(1e-5, 1)):
        """
        Defect concentration vs p_O2 plot for different temperatures
        Args:
            temps: Temperatures in K as list
                Default is 300 K
            pressure_limit: Tuple with lower and upper limits on pressure.
                Defaults are 1e-5 and 1 (Note these are very high values)
        Returns:
            matplotlib plot object containg E_F vs p_O2 plot
        """
        po2s = []
        pressure = pressure_limit[0]
        while pressure < pressure_limit[1]:
            po2s.append(pressure)
            pressure *= 10
        concs = defaultdict(list)
        self._analyzer.set_T(temp)
        ic = IntrinsicCarrier(self._dos, self._gap)
        for po2 in po2s:
            self._analyzer.set_gas_pressure(po2)
            self._analyzer.solve_for_fermi_energy(self._dos, self._gap)
            ef = self._analyzer.fermi_energy
            def_conc = self._analyzer.get_defects_concentration(temp=temp,
                                                                ef=ef,
                                                                unitcell=True)
            specie_def_conc = defaultdict(int)
            for item in def_conc:
                specie_def_conc[item['name']] += item['conc']
            for key, value in specie_def_conc.items():
                concs[key].append(value)
            nden = ic.get_n_density(ef, temp)
            concs['n'].append(nden)
            pden = ic.get_p_density(ef, temp)
            concs['h'].append(pden)

        width, height = 12, 8
        import matplotlib.pyplot as plt
        plt.clf()
        import matplotlib.cm as cm
        colors=cm.Dark2(np.linspace(0, 1, len(concs.keys())+1))

        def get_legend(name):
            flds = name.split('_')
            if 'vac' in name:
                base = '$V'
                sub_str = '_{'+flds[2]+'}$'
            elif 'inter' in name:
                base = '$'+flds[2]
                sub_str = '_{i_{'+','.join(flds[3:5])+'}}$'
            elif 'sub' in name or 'antisite' in name or 'as' in name:
                base = '$'+flds[2]
                sub_str = '_{'+flds[4]+'}$'
            else:
                base = name
                sub_str = ''
            return base + sub_str

        legends = []
        for defect_name in concs:
            conc = concs[defect_name]
            legends.append(get_legend(defect_name))
            plt.loglog(po2s, conc, linewidth=2)#, color=colors[i])


        #plt.legend(get_legends(concs), fontsize=1.8*width, loc='best')
        plt.legend(legends, fontsize=width, loc='best')
        #plt.legend(map(lambda x: str(x) + ' K', temps), fontsize=1.8*width,
        #           loc='best')
        plt.xlabel("$p_{O_2}$ (atm)", size=width)
        plt.ylabel("Defect concentration", size=width)
        plt.title(str(temp) + "K")
        return plt

class FiniteTempDiffPlotter(object):
    """
    Class performing the finite temperature plots for diffusion 
    Args:
        analyzer: FiniteTempDefectsAnalyzer object
        bulk_dos: Dos object of unit cell
        exp_gap: Experimental band gap
    """

    def __init__(self, analyzer, bulk_dos, exp_gap, debye_freq):
        self._analyzer = deepcopy(analyzer)
        self._dos = bulk_dos
        self._gap = exp_gap
        self._debye_freq = debye_freq

    def get_fermi_level(self, temp, pressure):
        """
        E_F vs p_O2 plot for different temperatures
        Args:
            temps: Temperatures in K as list
                Default is 300 K
            pressure_limit: Tuple with lower and upper limits on pressure.
                Defaults are 1e-5 and 1 (Note these are very high values)
        Returns:
            matplotlib plot object containg E_F vs p_O2 plot
        """
        self._analyzer.set_T(temp)
        self._analyzer.set_gas_pressure(pressure)
        self._analyzer.solve_for_fermi_energy(self._dos, self._gap)
        ef = self._analyzer.fermi_energy
        return ef

    def get_diff_vs_temp_plot(self, pressures=[1e-5], 
                              temps_limit=(300,1200)):
        """
        stoichiometry deviation vs p_O2 plot for different temperatures
        Args:
            temps: Temperatures in K as list
                Default is 300 K
            pressure_limit: Tuple with lower and upper limits on pressure.
                Defaults are 1e-5 and 1 (Note these are very high values)
        Returns:
            matplotlib plot object containg E_F vs p_O2 plot
        """
        temps = []
        T = temps_limit[0]
        while T < temps_limit[1]:
            temps.append(T)
            T += 50
        del_x = defaultdict(list)
        for pO2 in pressures:
            for T in temps:
                #D = get_diffusion_coefficient(fermi,  
                del_x[pO2].append(
                    self._analyzer.get_stoichiometry_deviations()[specie])

        width, height = 12, 8
        import matplotlib.pyplot as plt
        plt.clf()
        import matplotlib.cm as cm
        colors=cm.Dark2(np.linspace(0, 1, len(temps)))
        for i, temp in enumerate(temps):
            plt.semilogx(po2s, del_x[temp], linewidth=3, color=colors[i])

        plt.legend(map(lambda x: str(x) + ' K', temps), fontsize=1.8*width,
                   loc='best')
        plt.xlabel("$p_{O_2}$ (atm)", size=2*width)
        plt.ylabel("$\Delta X$ of {}".format(specie), size=2*width)
        plt.title("Deviation from stoichiometry")
        return plt

    def get_concentrations(self, temp, pressure):
        """
        Defect concentration 
        Args:
            temps: Temperatures in K as list
                Default is 300 K
            pressure_limit: Tuple with lower and upper limits on pressure.
                Defaults are 1e-5 and 1 (Note these are very high values)
        Returns:
            matplotlib plot object containg E_F vs p_O2 plot
        """
        ef = self.get_fermi_level(temp, pressure)
        return self._analyzer.get_defects_concentration_per_site(temp=temp, 
                                                                 ef=ef)

