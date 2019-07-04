import sys
import os
import math
import itertools
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from monty.serialization import loadfn, dumpfn
from monty.json import MontyDecoder, MontyEncoder
from pycdt.core.finite_temp_defects_analyzer import ConstrainedGCAnalyzer, \
    IntrinsicCarrier
from pycdt.utils.units import kb, conv, hbar, joule_per_mole_to_ev
from plotter import FiniteTempDiffPlotter

ang_to_cm = 1e-8
da = loadfn('defect_analyzer_Cr2O3_hse_O2-rich.json', cls=MontyDecoder)

#energy_O2 = -9.574  (O2 at T0, P0)
#energy_O2 = -9.647   (equilibrium E_O2 at Cr2O3/CrO2 phase boundary)

gc_analyzer = ConstrainedGCAnalyzer(
    entry_bulk=da._entry_bulk, e_vbm=da._e_vbm, band_gap = 3.328,
    mu_elts=None,  energy_O2=-11.6711984312) # Does energy_O2 still the same? Where is this value coming from?
for defect in da._defects:
    gc_analyzer.add_computed_defect(defect)

#bulk_dos = loadfn('Cr2O3_hse_dos.json')
bulk_dos = loadfn('Cr2O3_hse_dos_new.json')
pO2s = [1e-12,  1e-6,  1e-3]
temps_limits = (300, 2000)
debye_freq = 1.428e13

temps = []
T = temps_limits[0]
while T < temps_limits[1]:
    temps.append(T)
    T += 100


temp_invs = [10000./T for T in temps]

plotter = FiniteTempDiffPlotter(gc_analyzer, bulk_dos, 3.383, debye_freq)
#for press in pO2s:
press  = 1e-0
#temp = 1300
diff_coeff = {}
for temp in temps:
    diff_coeff[temp] = {}
    concs = plotter.get_concentrations(temp, press)
    conc_dict = defaultdict(lambda: defaultdict(float))
    for defect in concs:
        conc_dict[defect['name']][defect['charge']] = defect['conc']
    #print '------------concs-----------------'
    #for dfct_name in conc_dict:
    #    for q in conc_dict[dfct_name]:
    #        print dfct_name, q, conc_dict[dfct_name][q]
    #print '------------end of concs-----------------\n\n\n'

    beta = 1.0/(kb*temp)
    #diff_coeff = {}
    diff_params = loadfn('barrier.yaml')
    for dfct_name in diff_params:
        diff_coeff[temp][dfct_name] = {}
        for q, params in diff_params[dfct_name].items():
            print(dfct_name, q)
            diff_perp = 0
            diff_par = 0
            for path in params:
                dperp = path['d_perp']
                dpar = path['d_par']
                mult = path['multiplicity']
                barr = path['energy']
                freq = path.get('freq')
                if not freq:
                    freq = debye_freq
                jump_freq = freq * math.exp(-beta*barr)
                conc = conc_dict[dfct_name][q] 
                #print jump_freq, conc, dperp*ang_to_cm
                #print (debye_freq*dperp*ang_to_cm * dperp*ang_to_cm , math.exp(-beta*barr))
                diff_perp += mult*jump_freq*dperp*dperp*ang_to_cm*ang_to_cm*conc
                diff_par += mult*jump_freq*dpar*dpar*ang_to_cm*ang_to_cm*conc
            #diff_coeff[temp][dfct_name][q] = {'D_perp': diff_perp, 'D_par': diff_par}
            diff_coeff[temp][dfct_name][q] = diff_perp + diff_par
        

#print '------------diff params-----------------'
#for dfct_name in diff_params:
#    for q in diff_params[dfct_name]:
#        print dfct_name, q, diff_coeff[dfct_name][q]
#print '------------end of diff params-----------------\n\n\n'

#sys.exit()
dc = defaultdict(lambda: defaultdict(list))
for T in sorted(diff_coeff.keys()):
    for dfct_name in diff_coeff[T]:
        for q in diff_coeff[T][dfct_name]:
            dc[dfct_name][q].append(diff_coeff[T][dfct_name][q])

dumpfn(dc, 'diff_coeff_P-{}.json'.format(press), cls=MontyEncoder, indent=2)

width, height = 12, 8
#plt.clf()
colors=itertools.cycle(cm.Dark2(np.linspace(0, 1, 8)))
#for color in colors:
#    print ("printing color")
#    print(color)
#sys.exit()
#colors=cm.Dark2(np.linspace(0, 1, 4))
#for i, press in enumerate(pO2s):
def get_legend(dfct_name):
    fields = dfct_name.split('_')
    if fields[0] == 'vac':
        base = 'V'
        sub = fields[2]
    elif fields[0] == 'inter':
        base = fields[2]
        sub = 'i'
    legend = "$"+base+"_{"+sub+"}$"
    return legend
ls = {'Cr': '-', 
      'O': '--',} #| 'None' | ' ' | ''}
legends = []
diff_coeff_el = {}
for dfct_name in dc:
    print ('defect_name', dfct_name)
    diff_coeff = np.zeros_like(dc[dfct_name][0])
    for q in dc[dfct_name]:
        print ('q', q)
        #print ('diff_coeff', dc[dfct_name][q])
        diff_coeff += np.array(dc[dfct_name][q])

    if 'Cr' in dfct_name:
        el = 'Cr'
    elif 'O' in dfct_name:
        el = 'O'

    if el in diff_coeff_el:
        diff_coeff_el[el] += diff_coeff
    else:
        diff_coeff_el[el] = diff_coeff

for el in diff_coeff_el:
    plt.plot(temp_invs, np.log10(diff_coeff_el[el]), lw=2, ls=ls[el], color=next(colors))
    #plt.plot(temp_invs, np.log10(dc[dfct_name][q]), lw=2, ls=ls[dfct_name], color=colors[abs(q)])
    legends.append(el)

#plt.legend(map(lambda x: str(x) + ' atm', pO2s), fontsize=1.8*width,
#           loc='best')
plt.legend(legends)#, fontsize=1.8*width, loc='best')

plt.xlabel("10${}^4$/T (K$^{-1}$)", size=width)
plt.xlim([5.5,14])
plt.ylim([-50,1])
plt.ylabel("log D (cm${}^2$.s)", size=width)
plt.title("$p_{O_2}$  %0.1e atm"%press)

#plt = plotter.get_stoichiometry_vs_pressure_plot('Cr', temps=temps, pressure_limit=pO2_lims)
plt.savefig('Cr2O3_diff_vs_T_log_def_el_P-{}.eps'.format(press))
plt.close()
#plt = plotter.get_fermi_vs_pressure_plot(temps=temps1, pressure_limit=pO2_lims)
#plt.savefig('Cr2O3_fermi.png')
    #gcmc_analyzer.set_gas_pressure(gas_pressure)
    #gcmc_analyzer.solve_for_fermi_energy(bulk_dos=bulk_dos, gap=3.4)
    #print gas_pressure, gcmc_analyzer.fermi_energy
    #print gcmc_analyzer.get_stoichiometry_deviations()
    #def_con = gcmc_analyzer.get_defects_concentration(temp=gcmc_analyzer.T,
    #                                                  ef=gcmc_analyzer.fermi_energy,
    #                                                  unitcell=True)
    #for conc in def_con:
    #    print conc

    #print 'electron density', ic.get_n_density(gcmc_analyzer.fermi_energy, temp)
    #print 'hole density', ic.get_p_density(gcmc_analyzer.fermi_energy, temp)
