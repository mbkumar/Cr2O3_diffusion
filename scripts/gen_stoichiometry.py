from monty.serialization import loadfn, dumpfn
from monty.json import MontyDecoder

from pycdt.core.finite_temp_defects_analyzer import ConstrainedGCAnalyzer, \
    IntrinsicCarrier
from plotter import FiniteTempDefectPlotter

da = loadfn('defect_analyzer_Cr2O3_O2-rich.json', cls=MontyDecoder)

gc_analyzer = ConstrainedGCAnalyzer(
    entry_bulk=da._entry_bulk, e_vbm=da._e_vbm, band_gap = 3.384,
    mu_elts=None,  energy_O2=-11.6711984312)   # (equilibrium E_O2 at Cr2O3/CrO2 phase boundary)
for defect in da._defects:
    gc_analyzer.add_computed_defect(defect)

bulk_dos = loadfn('Cr2O3_dos.json')
pO2s = [1e-16,1e-15,1e-14,1e-13,1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
pO2_lims = (1e-16,  1e-2)
temps1 = [300, 400, 600, 800, 1000, 1200]
temps = [ 300, 800, 1200]

plotter = FiniteTempDefectPlotter(gc_analyzer, bulk_dos, 3.384)
plt = plotter.get_stoichiometry_vs_pressure_plot('Cr', temps=temps, pressure_limit=pO2_lims)
plt.savefig('Cr2O3_Cr_delta_X.eps')
plt = plotter.get_fermi_vs_pressure_plot(temps=temps1, pressure_limit=pO2_lims)
plt.savefig('Cr2O3_fermi.eps')
