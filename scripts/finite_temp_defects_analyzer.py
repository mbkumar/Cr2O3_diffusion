#!/usr/bin/env python

__author__ = "Bharat Medasani"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Bharat Medasani"
__email__ = "mbkumar@gmail.com"
__status__ = "Development"
__date__ = "Aug 22, 2016"

import os
import csv
from math import log

import numpy as np
from scipy import integrate, interpolate

from pymatgen.core.periodic_table import get_el_sp
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pycdt.utils.units import kb, conv, hbar, joule_per_mole_to_ev
from defects_analyzer import ComputedDefect, DefectsAnalyzer

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class OxygenChemPotNISTable(object):
    """
    Oxygen chemical potential evaluation based on O2 partial
    pressure and Temperature
    """
    def __init__(self):
        with open(os.path.join(MODULE_DIR, 'nist_janaf_O2.csv')) as f:
            mu_o2_reader = csv.reader(f, delimiter=',')
            T = []
            C_p = []
            S = []
            G_min_H_by_T = []
            delta_H = []
            for row in mu_o2_reader:
                T.append(float(row[0]))
                C_p.append(float(row[1]))
                S.append(float(row[2]))
                G_min_H_by_T.append(float(row[3]))
                delta_H.append(float(row[4]))

            self.C_p_interp = interpolate.interp1d(T, C_p, kind='cubic')
            self.S_interp = interpolate.interp1d(T, S, kind='cubic')
            self.G_min_H_by_T_interp = interpolate.interp1d(T, G_min_H_by_T,
                                                            kind='cubic')
            self.delta_H_interp = interpolate.interp1d(T, delta_H, kind='cubic')

    def o2_mu(self, T, p_O2):
        """
        Return the temperature and p_O2 dependent oxygen chem pot
        Args:
            T: Temperature in Kelvin
            p_O2: Oxygen partial pressure (in atm)
        """
        press_contrib = kb*T*log(p_O2)
        temp_contrib = (self.delta_H_interp(T)+8.683)*1000 - T*self.S_interp(T)
        temp_contrib *= joule_per_mole_to_ev
        print('diff in ev', 8.683*1000*joule_per_mole_to_ev)
        return press_contrib + temp_contrib


class IntrinsicCarrier(object):
    def __init__(self, dos, exp_gap=None):
        self.dos = dos
        self.exp_gap = exp_gap

    def get_n_density(self, ef, T, ref='VBM'):
        """
        Obtain the electron concentration as a function of Fermi level and
        temperature using Fermi-Dirac statistics and conduction band DOS
        Args:
            ef: Fermi energy
            dos: Dos object of pymatgen
            ef_ref: Options 'VBM' or 'CBM'
                The second option is useful when ef > DFT gap but <
                experimental gap. In that case, ef close to CBM can be
                given as -ve value w.r.t. CBM.
        Returns:
            Electron density per unit volume
        """
        dos_gap = self.dos.get_gap()
        cbm, vbm = self.dos.get_cbm_vbm()
        gap = self.exp_gap if self.exp_gap else dos_gap

        if abs(ef) > gap+2e-6:
            raise ValueError("Fermi level is greater than bandgap")
        if ref == 'CBM':
            ef += cbm
        else:
            ef += vbm + dos_gap - gap

        energies = self.dos.energies
        densities = self.dos.get_densities()
        if energies[-1] - ef < 3.0:
            print ("The upper limit of energy is within 3 eV of Fermi level. "
                   "Check for the convergence of electron concentration")
        i = np.searchsorted(energies, cbm)
        fd_stat = 1./(1 + np.exp((energies[i:] - ef) / (kb*T)))
        y = fd_stat * densities[i:]
        den = integrate.trapz(y, energies[i:])
        #print(T, den)
        return den

    def get_p_density(self, ef, T, ref='VBM'):
        """
        Obtain the hole concentration as a function of Fermi level and
        temperature using Fermi-Dirac statistics and conduction band DOS
        Args:
            ef: Fermi energy
            dos: Dos object of pymatgen
            ref: Options 'VBM' or 'CBM'
                    The second option is useful when ef > DFT gap
                    but < experimental gap
        Returns:
            Hole density per unit volume
        """
        dos_gap = self.dos.get_gap()
        cbm, vbm = self.dos.get_cbm_vbm()
        gap = self.exp_gap if self.exp_gap else dos_gap

        if abs(ef) > gap+2e-6:
            raise ValueError("Fermi level is greater than gap")
        if ref == 'CBM':   # Assumption is ef is -ve if ref=='cbm'
            ef += cbm + gap - dos_gap
        else:
            ef += vbm

        energies = self.dos.energies
        densities = self.dos.get_densities()
        if ef - energies[0] < 3.0:
            print ("The lower limit of energy is within 3 eV of Fermi level. "
                   "Check for the convergence of hole concentration")
        i = np.searchsorted(energies, vbm) + 1
        fd_stat = 1./(1 + np.exp((ef - energies[:i]) / (kb*T)))
        y = fd_stat*densities[:i]
        den = integrate.trapz(y, energies[:i])
        return  den


class ConstrainedGCAnalyzer(DefectsAnalyzer):
    """
    Accounts for finite temperature and oxygen partial pressure
    Args:
        T: Temperature
        phonon_energies: Phonon contribution to free energy
            (Note: phonon energies themselves are function of T)
        gas_pressure: Partial pressure of the gas (At present only
            partial pressure of oxygen is accepted)
        kwargs: Arguments to DefectsAnalyzer as keyword pair
    """
    def __init__(self, T=298.15, gas_pressure=1.0, phonon_energies=None,
                 energy_O2=0, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.T = T
        self.gas_pressure = gas_pressure
        self.phonon_energies = phonon_energies
        self.energy_O2 = energy_O2
        self._ef = None
        self._compute_0K_chem_pots()

    def as_dict(self):
        d = super(self.__class__, self).as_dict()
        d.update({'phonon_energies': self.phonon_energies, 'T': self.T,
                  'gas_pressure': self.gas_pressure})
        return d

    @classmethod
    def from_dict(cls, d):
        entry_bulk = ComputedStructureEntry.from_dict(d['entry_bulk'])
        analyzer = ConstrainedGCAnalyzer(
            T=d['T'], phonon_energies=d['phonon_energies'],
            gas_pressure=d['gas_pressure'],
            entry_bulk=entry_bulk, e_vbm=d['e_vbm'],
            mu_elts={el: d['mu_elts'][el] for el in d['mu_elts']},
            band_gap=d['band_gap'])

        for ddict in d['defects']:
            analyzer.add_defect(ComputedDefect.from_dict(ddict))

        return analyzer

    def set_T(self, T):
        self.T = T
        self._compute_form_en()

    def set_phonon_contributions(self, phonon_energies):
        """
        Args:
            phonon_energies; Phonon contribution to the Gibbs free
            energies of defects. Supply them in the order of defects
        """
        self.phonon_energies = phonon_energies
        self._compute_form_en()

    def set_gas_pressure(self, gas_pressure):
        self.gas_pressure = gas_pressure
        self._compute_form_en()

    def get_oxygen_mu_TP(self, p_O2, temp):
        """
        Supplies the temperature and pressure dependent component of
        oxygen chemical potential. This is in addition to the DFT
        computed oxygen molecule energy needed for finite temperature.
        mu_O(T,P=1atm) from M. W. Chase, NIST-JANAF Thermochemical tables
        """
        return OxygenChemPotNISTable().o2_mu(temp, p_O2)

    def _compute_0K_chem_pots(self):
        mu_elts = {}
        mu_O = self.energy_O2/2.0
        bulk_comp = self._entry_bulk.composition
        red_comp = bulk_comp.reduced_composition
        N = sum(bulk_comp.values())
        red_N = sum(red_comp.values())
        energy_formula_unit = self._entry_bulk.energy/N*red_N

        for elt in bulk_comp.elements:
            if elt == get_el_sp('O'):
                mu_elts[elt] = mu_O
            else:
                mu_elts[elt] = \
                    (energy_formula_unit - red_comp['O']*mu_O) / red_comp[elt]
        self._mu_elts = mu_elts

    def _compute_P_T_chem_pots_change(self, p_O2, temp):
        delta_mu_elts = {}
        delta_mu_O = self.get_oxygen_mu_TP(p_O2, temp)/2.0
        bulk_comp = self._entry_bulk.composition
        red_comp = bulk_comp.reduced_composition

        for elt in bulk_comp.elements:
            if elt == get_el_sp('O'):
                delta_mu_elts[elt] = delta_mu_O
            else:
                delta_mu_elts[elt] = -red_comp['O']*delta_mu_O / red_comp[elt]
        return delta_mu_elts

    def _compute_form_en(self):
        """
        compute the formation energies for all defects in the analyzer
        """
        self._formation_energies = []
        #red_comp = self._entry_bulk.composition.reduced_composition
        #red_comp_N = sum(red_comp.values())
        #red_comp_O_weighted_N = red_comp_N/red_comp['O']
        #mu_O2 = self.energy_O2
        delta_mu_elts = self._compute_P_T_chem_pots_change(self.gas_pressure,
                                                           self.T)
        for d in self._defects:
            #compensate each element in defect with the chemical potential
            mu_needed_coeffs = {}

            for elt in d.entry.composition.elements:
                el_def_comp = d.entry.composition[elt]
                el_blk_comp = self._entry_bulk.composition[elt]
                mu_needed_coeffs[elt] = el_blk_comp - el_def_comp

            sum_mus = 0.0
            for elt in mu_needed_coeffs:
                #el = elt.symbol
                sum_mus += mu_needed_coeffs[elt] * \
                           (self._mu_elts[elt] + delta_mu_elts[elt])
            self._formation_energies.append(
                    d.entry.energy - self._entry_bulk.energy + \
                            sum_mus + d.charge*self._e_vbm + \
                            d.charge_correction + d.other_correction)

    def _compute_gibbs_form_en(self):
        """
        compute the formation energies for all defects in the analyzer
        """
        if not self._formation_energies:
            self._compute_form_en()

        self._gibbs_formation_energies = []
        if self._formation_energies:
            for i in len(self._formation_energies):
                self._gibbs_formation_energies[i] = \
                    self._formation_energies[i] + self.phonon_energies[i]

    def solve_for_fermi_energy(self, bulk_dos, gap=None):
        """
        Solve for the Fermi energy self-consistently as a function of T
        and p_O2
        Observations are Defect concentrations, electron and hole conc
        Args:
            bulk_dos: bulk system dos (pymatgen Dos object)
            gap: Can be used to specify experimental gap.
                Will be useful if the self consistent Fermi level
                is > DFT gap
        Returns:
            Fermi energy
        """
        if not gap:
            gap = self._band_gap
        from scipy.optimize import bisect
        self._ef = bisect(lambda e: self._get_total_q(e, gap, bulk_dos), -1e-6, gap)

    @property
    def fermi_energy(self):
        return self._ef

    def _get_stoichiometry_deviation(self, specie):
        """
        Get the deviation in stoichiometry for given specie at given
        T and pressure
        Args:
            specie: pymatgen specie object
        Returns:
            Deviation from stoichiometry (delta x) of specie
        """
        bulk_comp = self._entry_bulk.composition
        red_comp = bulk_comp.reduced_composition
        del_x = 0
        multiplicity = 0
        for i, d in enumerate(self._defects):
            def_conc = self._get_concentration(i, self.T, self._ef,
                                               unitcell=True)
            if d.site.specie == specie:
                if 'vac' in d.name:
                    del_x -= def_conc
                    multiplicity += d.multiplicity
                elif 'inter' in d.name:
                    del_x += def_conc
                elif 'as' in d.name or 'antisite' in d.name:
                    del_x -= def_conc
                elif 'sub' in d.name:
                    del_x -= def_conc
            if 'as' in d.name or 'antisite' in d.name or 'sub' in d.name:
                nm_fields = d.name.split('_')
                sub_specie = get_el_sp(nm_fields[2])
                if sub_specie == specie:
                    del_x += def_conc

        factor = red_comp[get_el_sp(specie)] / multiplicity
        return del_x * factor

    def get_stoichiometry_deviations(self):
        delta_x = {}
        for specie in self._entry_bulk.composition.keys():
            deviation = self._get_stoichiometry_deviation(specie)
            delta_x[specie.symbol] = deviation
        return delta_x


    def _get_total_q(self, ef, gap, bulk_dos):
        qd_tot = 0
        for d in self.get_defects_concentration(self.T, ef, unitcell=True):
            qd_tot += d['charge'] * d['conc']
            #print d['name'], d['conc']
        q_h_cont = IntrinsicCarrier(bulk_dos, exp_gap=gap)
        ef_ref_cbm = ef - gap
        nden = q_h_cont.get_n_density(ef_ref_cbm, self.T, ref='CBM')
        #print 'n_den', nden
        pden = q_h_cont.get_p_density(ef, self.T)
        #print 'p_den', pden
        qd_tot += pden - nden
        return qd_tot

    #def get_concentrations(self):
    #    ef = self.

