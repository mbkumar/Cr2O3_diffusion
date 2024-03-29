#!/usr/bin/env python

__author__ = "Geoffroy Hautier, Bharat Medasani,  Danny Broberg"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Geoffroy Hautier, Bharat Medasani"
__email__ = "geoffroy@uclouvain.be, mbkumar@gmail.com"
__status__ = "Development"
__date__ = "November 4, 2012"

from math import sqrt, pi, exp, log
from collections import defaultdict 
from itertools import combinations

import numpy as np

from pymatgen.core.structure import PeriodicSite
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pycdt.utils.units import kb, conv, hbar, joule_per_mole_to_ev

class ComputedDefect(object):
    """
    Holds all the info concerning a defect computation: 
    composition+structure, energy, correction on energy and name
    """
    def __init__(self, entry_defect, site_in_bulk, multiplicity=None,
                 supercell_size=[1, 1, 1], charge=0.0,
                 charge_correction=0.0, other_correction=0.0, name=None):
        """
        Args:
            entry_defect: 
                An ComputedStructureEntry object corresponding to the 
                defect supercell
            site_in_bulk: 
                Site of the defect in bulk supercell. Defect positions
                are often required to perform posteriori corrections
            multiplicity:
                Multiplicity of defect site in a cell. Useful to 
                evaluate defect concentrations
            supercell_size: 
                Size of the defect supercell in terms of unit cell
            charge: 
                The charge of the defect
            charge_correction: 
                Correction to the energy due to charge
            other_correction:
                Correction to the energy due to other factors
            name: 
                The name of the defect
        """

        self.entry = entry_defect
        self.site = site_in_bulk
        self.multiplicity = multiplicity
        self.supercell_size = supercell_size
        self.charge = charge
        self.charge_correction = charge_correction # Can be added after initialization
        self.other_correction = other_correction
        self.name = name
        self.full_name = self.name + "_" + str(charge)

    def as_dict(self):
        return {'entry': self.entry.as_dict(),
                'site': self.site.as_dict(),
                'multiplicity': self.multiplicity,
                'charge': self.charge,
                'charge_correction': self.charge_correction,
                'other_correction': self.other_correction,
                'supercell_size': self.supercell_size,
                'name': self.name,
                'full_name': self.full_name,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d):
        return cls(
                ComputedStructureEntry.from_dict(d['entry']), 
                PeriodicSite.from_dict(d['site']),
                multiplicity=d.get('multiplicity', None),
                supercell_size=d.get('supercell_size',[1,1,1]),
                charge=d.get('charge', 0.0),
                charge_correction=d.get('charge_correction', 0.0),
                other_correction=d.get('other_correction', 0.0),
                name=d.get('name', None))


class DefectsAnalyzer(object):
    """
    a class aimed at performing standard analysis of defects
    """
    def __init__(self, entry_bulk, e_vbm, band_gap, mu_elts=None):
        """
        Args:
            entry_bulk:
                the bulk data as an Entry
            e_vbm:
                the energy of the vbm (in eV)
            mu_elts:
                a dictionnary of {Element:value} giving the chemical
                potential of each element
            band_gap:
                the band gap (in eV)
        """
        self._entry_bulk = entry_bulk
        self._e_vbm = e_vbm
        self._mu_elts = mu_elts
        self._band_gap = band_gap
        self._defects = []
        self._formation_energies = []

    def as_dict(self):
        d = {'entry_bulk': self._entry_bulk.as_dict(),
             'e_vbm': self._e_vbm,
             'mu_elts': self._mu_elts,
             'band_gap': self._band_gap,
             'defects': [d.as_dict() for d in self._defects],
             'formation_energies': self._formation_energies,
             "@module": self.__class__.__module__,
             "@class": self.__class__.__name__}
        return d

    @classmethod
    def from_dict(cls, d):
        entry_bulk = ComputedStructureEntry.from_dict(d['entry_bulk'])
        analyzer = DefectsAnalyzer(
                entry_bulk, d['e_vbm'], d.get('band_gap', 0),
                {el: d['mu_elts'][el] for el in d['mu_elts']})
        for ddict in d['defects']:
            analyzer.add_computed_defect(ComputedDefect.from_dict(ddict))
        return analyzer

    def add_computed_defect(self, defect):
        """
        add a parsed defect to the analyzer
        Args:
            defect:
                a Defect object
        """
        self._defects.append(defect)
        self._compute_form_en()

    def change_charge_correction(self, i, correction):
        """
        Change the charge correction for defect at index i
        Args:
            i:
                Index of defects
            correction:
                New correction to be applied for defect
        """
        self._defects[i].charge_correction = correction
        self._compute_form_en()

    def change_other_correction(self, i, correction):
        """
        Change the charge correction for defect at index i
        Args:
            i:
                Index of defects
            correction:
                New correction to be applied for defect
        """
        self._defects[i].other_correction = correction
        self._compute_form_en()

    def _get_all_defect_types(self):
        to_return = []
        for d in self._defects:
            if d.name not in to_return: to_return.append(d.name)
        return to_return

    def _compute_form_en(self):
        """
        compute the formation energies for all defects in the analyzer
        """
        self._formation_energies = []
        for d in self._defects:
            #compensate each element in defect with the chemical potential
            mu_needed_coeffs = {}
            for elt in d.entry.composition.elements:
                el_def_comp = d.entry.composition[elt] 
                el_blk_comp = self._entry_bulk.composition[elt]
                mu_needed_coeffs[elt] = el_blk_comp - el_def_comp

            sum_mus = 0.0
            for elt in mu_needed_coeffs:
                el = elt.symbol
                sum_mus += mu_needed_coeffs[elt] * self._mu_elts[el]

            self._formation_energies.append(
                    d.entry.energy - self._entry_bulk.energy + \
                            sum_mus + d.charge*self._e_vbm + \
                            d.charge_correction + d.other_correction)

    def correct_bg_simple(self, vbm_correct, cbm_correct):
        """
        correct the band gap in the analyzer.
        We assume the defects level remain the same when moving the 
        band edges
        Args:
            vbm_correct:
                The correction on the vbm as a positive number. e.g., 
                if the VBM goes 0.1 eV down vbm_correct=0.1
            cbm_correct:
                The correction on the cbm as a positive number. e.g., 
                if the CBM goes 0.1 eV up cbm_correct=0.1

        """
        self._band_gap = self._band_gap + cbm_correct + vbm_correct
        self._e_vbm = self._e_vbm - vbm_correct
        self._compute_form_en()

    def get_transition_levels(self):
        """
        Charge transition levels are computed
        :return: Transition levels for each pair of the defects.
        If any pair is missing, the transition level for that pair is
        not within the band gap.
        """
        xlim = (-0.5, self._band_gap+1.5)
        nb_steps = 1000
        x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/nb_steps)
 
        y = defaultdict(defaultdict)
        for i, dfct in enumerate(self._defects):
            yval = self._formation_energies[i] + dfct.charge*x
            y[dfct.name][dfct.charge] = yval

        transit_levels = defaultdict(defaultdict)
        for dfct_name in y:
            q_ys = y[dfct_name]
            for qpair in combinations(q_ys.keys(), 2):
                y_absdiff = abs(q_ys[qpair[1]] - q_ys[qpair[0]])
                if y_absdiff.min() < 0.4: # Forgot why 0.4 check was added
                    transit_levels[dfct_name][qpair] = x[np.argmin(y_absdiff)]
        return transit_levels

    def correct_bg(self, dict_levels, vbm_correct, cbm_correct):
        """
        correct the band gap in the analyzer and make sure the levels move
        accordingly.
        There are two types of defects vbm_like and cbm_like and we need
        to provide a formal oxidation state
        The vbm-like will follow the vbm and the cbm_like the cbm. If nothing
        is specified the defect transition level does not move
        Args:
            dict_levels: a dictionary of type {defect_name:
            {'type':type_of_defect,'q*':formal_ox_state}}
            Where type_of_defect is a string: 'vbm_like' or 'cbm_like'
        """
        self._band_gap = self._band_gap + cbm_correct + vbm_correct
        self._e_vbm = self._e_vbm - vbm_correct
        self._compute_form_en()
        for i in range(len(self._defects)):
            name = self._defects[i].name
            if not name in dict_levels:
                continue

            if dict_levels[name]['type'] == 'vbm_like':
                z = self._defects[i].charge - dict_levels[name]['q*']
                self._formation_energies[i] += z * vbm_correct
            if dict_levels[name]['type'] == 'cbm_like':
                z = dict_levels[name]['q*'] - self._defects[i].charge
                self._formation_energies[i] +=  z * cbm_correct

    def get_defect_occupancies(self):
        """
        Defect occupancies with respect to defect charges are computed
        The assumption is that the highest charge of defect numerically
        has zero occuoancy:
        Ex: In Cr2O3, V_O has 0 occupancy for +2 q and 2 occupancy for 0 q.
            V_{Cr} has 0 occupancy for 0 q and 3 occupancy for -3 q
        Caution: Didn't checkk for semiconductor with large # of defect q's.
        Returns: 
            Defect occupancies as a nested dict
        """
        charges = defaultdict(list)
        for dfct in self._defects:
            charges[dfct.name].append(dfct.charge)

        occupancies = defaultdict(lambda: defaultdict(int))
        for dfct_name in charges:
            for i, q in enumerate(sorted(charges[dfct_name], reverse=True)):
                occupancies[dfct_name][q] = i
            occupancies[dfct_name]['0_occupancy'] = \
                    sorted(charges[dfct_name], reverse=True)[0]
        return occupancies

    def _get_form_energy(self, ef, i):
        return self._formation_energies[i] + self._defects[i].charge*ef

    def get_formation_energies(self, ef=0.0):
        """
        Get the defect formation energies for a given Fermi level
        Args:
            ef:
                the fermi level in eV (with respect to the VBM)
        Returns:
            a list of dict of {'name': defect name, 'charge': defect charge 
                               'energy': defect formation energy in eV}
        """
        energies = []
        i = 0
        for i, d in enumerate(self._defects):
            energies.append({
                'name': d.name, 
                'charge': d.charge, 
                'energy': self._get_form_energy(ef, i)
                })
        return energies

    def _get_concentration(self, d_index, temp, ef, unitcell=False):
        """
        Get the concentration of defect i for a temperature and Fermi level.
        Args:
            temp: the temperature in K
            Ef: the fermi level in eV (with respect to the VBM)
            d_index: Index of the defect
        Returns:
            Defect concentration
        """
        struct = self._entry_bulk.structure
        d = self._defects[d_index]
        cell_multiplier = np.prod(d.supercell_size)
        if unitcell:
            n = d.multiplicity
        else:
            n = d.multiplicity * cell_multiplier * 1e30 / struct.volume
        return n*exp(-self._get_form_energy(ef, d_index)/(kb*temp))

    def get_defects_concentration(self, temp=300, ef=0.0, unitcell=False):
        """
        Get the defect concentration for a temperature and Fermi level.
        Note: This method has an approximation and can fail
        [Ref: PRB 86, 144109 (2012)]
        Args:
            temp:
                the temperature in K
            Ef:
                the fermi level in eV (with respect to the VBM)
        Returns:
            A list of dict of {'name': defect name, 'charge': defect charge
                               'conc': defects concentration in m-3}
        """
        conc = []
        for i, d in enumerate(self._defects):
            conc.append({'name': d.name, 'charge': d.charge,
                         'conc': self._get_concentration(i, temp, ef,
                                                         unitcell=unitcell)})
        return conc

    def get_defects_concentration_per_site(self, temp=300, ef=0.0): 
        """
        Get the defect concentration for a temperature and Fermi level.
        Note: This method has an approximation and can fail
        [Ref: PRB 86, 144109 (2012)]
        Args:
            temp:
                the temperature in K
            Ef:
                the fermi level in eV (with respect to the VBM)
        Returns:
            A list of dict of {'name': defect name, 'charge': defect charge
                               'conc': defects concentration in m-3}
        """
        conc = []
        for i, d in enumerate(self._defects):
            conc.append({'name': d.name, 'charge': d.charge,
                         'conc': exp(-self._get_form_energy(ef, i)/(kb*temp))})
        return conc

    def get_defects_concentration_old(self, temp=300, ef=0.0):
        """
        get the defect concentration for a temperature and Fermi level
        Written when the site multiplicity is not supplied with ComputedDefect
        Args:
            temp:
                the temperature in K
            Ef:
                the fermi level in eV (with respect to the VBM)
        Returns:
            a list of dict of {'name': defect name, 'charge': defect charge
                               'conc': defects concentration in m-3}
        """
        conc=[]
        spga = SpacegroupAnalyzer(self._entry_bulk.structure, symprec=1e-1)
        struct = spga.get_symmetrized_structure()
        i = 0
        for d in self._defects:
            df_coords = d.site.frac_coords
            target_site=None
            #TODO make a better check this large tol. is weird
            #TODO: Get the equivalent sites from the defect class, because
            #TODO: this method is not designed with interstitials in mind.
            for s in struct.sites:
                sf_coords = s.frac_coords
                if abs(s.frac_coords[0]-df_coords[0]) < 0.1 \
                        and abs(s.frac_coords[1]-df_coords[1]) < 0.1 \
                        and abs(s.frac_coords[2]-df_coords[2]) < 0.1:
                    target_site=s
                    break
            equiv_site_no = len(struct.find_equivalent_sites(target_site))
            n = equiv_site_no * 1e30 / struct.volume
            conc.append({'name': d.name, 'charge': d.charge,
                         'conc': n*exp(
                             -self._get_form_energy(ef, i)/(kb*temp))})
            i += 1
        return conc

    def _get_dos(self, e, m1, m2, m3, e_ext):
        return sqrt(2) / (pi**2*hbar**3) * sqrt(m1*m2*m3) * sqrt(e-e_ext)

    def _get_dos_fd_elec(self, e, ef, t, m1, m2, m3):
        return conv * (2.0/(exp((e-ef)/(kb*t))+1)) * \
               (sqrt(2)/(pi**2)) * sqrt(m1*m2*m3) * \
               sqrt(e-self._band_gap)

    def _get_dos_fd_hole(self, e, ef, t, m1, m2, m3):
        return conv * (exp((e-ef)/(kb*t))/(exp((e-ef)/(kb*t))+1)) * \
               (2.0 * sqrt(2)/(pi**2)) * sqrt(m1*m2*m3) * \
               sqrt(-e)

    def _get_qd(self, ef, t):
        summation = 0.0
        for d in self.get_defects_concentration(t, ef):
            summation += d['charge'] * d['conc']
        return summation

    def _get_qi(self, ef, t, m_elec, m_hole):
        from scipy import integrate as intgrl

        elec_den_fn = lambda e: self._get_dos_fd_elec(
                e, ef, t, m_elec[0], m_elec[1], m_elec[2])
        hole_den_fn = lambda e: self._get_dos_fd_hole(
                e, ef, t, m_hole[0], m_hole[1], m_hole[2])

        bg = self._band_gap
        elec_count = -intgrl.quad(elec_den_fn, bg, bg+5)[0]
        hole_count = intgrl.quad(hole_den_fn, -5, 0.0)[0]

        return elec_count + hole_count

    def _get_qtot(self, ef, t, m_elec, m_hole):
        return self._get_qd(ef, t) + self._get_qi(ef, t, m_elec, m_hole)

    def get_eq_ef(self, t, m_elec, m_hole):
        """
        access to equilibrium values of Fermi level and concentrations 
        in defects and carriers obtained by self-consistent solution of 
        charge balance + defect and carriers concentrations
        Args:
            t: temperature in K
            m_elec: electron effective mass as a 3 value list 
                    (3 eigenvalues for the tensor)
            m_hole:: hole effective mass as a 3 value list 
                    (3 eigenvalues for the tensor)
        Returns:
            a dict with {
                'ef':eq fermi level,
                'Qi': the concentration of carriers
                      (positive for holes, negative for e-) in m^-3,
                'conc': the concentration of defects as a list of dicts
                }
        """
        from scipy.optimize import bisect
        e_vbm = self._e_vbm
        e_cbm = self._e_vbm+self._band_gap
        ef = bisect(lambda e:self._get_qtot(e,t,m_elec,m_hole), 0, 
                self._band_gap)
        return {'ef': ef, 'Qi': self._get_qi(ef, t, m_elec, m_hole),
                'QD': self._get_qd(ef,t), 
                'conc': self.get_defects_concentration(t, ef)}

    def get_non_eq_ef(self, tsyn, teq, m_elec, m_hole):
        """
        access to the non-equilibrium values of Fermi level and 
        concentrations in defects and carriers obtained by 
        self-consistent solution of charge balance + defect and 
        carriers concentrations

        Implemented following Sun, R., Chan, M. K. Y., Kang, S., 
        and Ceder, G. (2011). doi:10.1103/PhysRevB.84.035212

        Args:
            tsyn: the synthesis temperature in K
            teq: the temperature of use in K
            m_elec: electron effective mass as a 3 value list 
                    (3 eigenvalues for the tensor)
            m_hole: hole effective mass as a 3 value list 
                    (3 eigenvalues for the tensor)
        Returns:
            a dict with {
                'ef':eq fermi level,
                'Qi': the concentration of carriers
                      (positive for holes, negative for e-) in m^-3,
                'conc': the concentration of defects as a list of dict
                }
        """
        from scipy.optimize import bisect
        eqsyn = self.get_eq_ef(tsyn, m_elec, m_hole)
        cd = {}
        for c in eqsyn['conc']:
            if c['name'] in cd:
                cd[c['name']] += c['conc']
            else:
                cd[c['name']] = c['conc']
        ef = bisect(lambda e:self._get_non_eq_qtot(cd, e, teq, m_elec, m_hole),
                    -1.0, self._band_gap+1.0)
        return {'ef':ef, 'Qi':self._get_qi(ef, teq, m_elec, m_hole),
                'conc_syn':eqsyn['conc'],
                'conc':self._get_non_eq_conc(cd, ef, teq)}

    def _get_non_eq_qd(self, cd, ef, t):
        sum_tot = 0.0
        for n in cd:
            sum_d = 0.0
            sum_q = 0.0
            i = 0
            for d in self._defects:
                if d.name == n:
                    sum_d += exp(-self._get_form_energy(ef, i)/(kb*t))
                    sum_q += d.charge * exp(
                            -self._get_form_energy(ef, i)/(kb*t))
                i += 1
            sum_tot += cd[n]*sum_q/sum_d
        return sum_tot

    def _get_non_eq_conc(self, cd, ef, t):
        sum_tot = 0.0
        res=[]
        for n in cd:
            sum_tot = 0
            i = 0
            for d in self._defects:
                if d.name == n:
                    sum_tot += exp(-self._get_form_energy(ef,i)/(kb*t))
                i += 1
            i=0
            for d in self._defects:
                if d.name == n:
                    res.append({'name':d.name,'charge':d.charge,
                                'conc':cd[n]*exp(-self._get_form_energy(
                                    ef,i)/(kb*t))/sum_tot})
                i += 1
        return res

    def _get_non_eq_qtot(self, cd, ef, t, m_elec, m_hole):
        return self._get_non_eq_qd(cd, ef, t) + \
               self._get_qi(ef, t, m_elec, m_hole)


