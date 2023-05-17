from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from mendeleev import element as men_ele
import numpy as np
import pandas as pd

class ElementalProperties():

    def __init__(self, molecule, element):
        self.molecule = molecule
        self.element = element
    
    @property
    def count(self):
        ele_count_dict = self.molecule.get_el_amt_dict()
        ele = self.element.symbol
        return ele_count_dict[ele]
    
    @property
    def atomic_weight(self):
        return self.element.atomic_mass
    
    @property
    def atomic_radius_cal(self):
        return self.element.atomic_radius_calculated
    
    @property
    def atomic_radius_vdw(self):
        return self.element.van_der_waals_radius
    
    @property
    def electronegativity(self):
        return self.element.X
    
    @property
    def first_ionization(self):
        return self.element.ionization_energies[0]
    @property
    def second_ionization(self):
        ionization_energies = self.element.ionization_energies
        if (len(ionization_energies)>1):
            return ionization_energies[1]
        return None
    
    @property
    def spdf_orbital_electrons(self):
        s, p, d, f = 0, 0, 0, 0
        elec_config = self.element.full_electronic_structure
        for tup in elec_config:
            if tup[1]=='s':
                s = s + tup[2]
            if tup[1]=='p':
                p = p + tup[2]
            if tup[1]=='d':
                d = d + tup[2]
            if tup[1]=='f':
                f = f + tup[2]
        return [s, p, d, f]
    
    @property
    def mendeleev_no(self):
        return self.element.mendeleev_no
    
    @property
    def boiling_point(self):
        return self.element.boiling_point
    
    @property
    def melting_point(self):
        return self.element.melting_point
    
    @property
    def periodic_group(self):
        return self.element.group
    
    @property
    def valence_electrons(self):
        ele = men_ele(self.element.symbol)
        return ele.nvalence()
    
    @property
    def electron_affinity(self):
        return self.element.electron_affinity
    
    @property
    def heat_of_formation(self):
        ele = men_ele(self.element.symbol)
        return ele.heat_of_formation
    
    @property
    def density(self):
        ele = men_ele(self.element.symbol)
        return ele.density
    

feature_method_mapping = {
    'count': 'count',
    'weight': 'atomic_weight',
    'radius_cal': 'atomic_radius_cal',
    'radius_vdw': 'atomic_radius_vdw',
    'electro_neg': 'electronegativity',
    'mendeleev': 'mendeleev_no',
    'boiling_pt': 'boiling_point',
    'melting_pt': 'melting_point',
    'ion_1': 'first_ionization',
    'ion_2': 'second_ionization',
    's_elec': ['spdf_orbital_electrons', 0],
    'p_elec': ['spdf_orbital_electrons', 1],
    'd_elec': ['spdf_orbital_electrons', 2],
    'f_elec': ['spdf_orbital_electrons', 3],
    'periodic_grp': 'periodic_group',
    'valence_elec': 'valence_electrons',
    'electron_affin': 'electron_affinity',
    'heat_of_form': 'heat_of_formation',
    'density': 'density'
}



def rearrange_elements(ele_list):
    num = len(ele_list)
    output = [0]*num
    for i in ele_list:
        if i.symbol in [ 'Sc', 'Mn', 'Ti', 'V', 'Cr', 'Y', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']:
            output[0] = i
        elif i.symbol in ['C', 'N']:
            output[1] = i
        elif i.symbol in ['O', 'H', 'F', 'Cl']:
            if i.symbol=='H':
                output[-1]=i
            if i.symbol=='O':
                output[2]=i
            else:
                output[2]=i
        else:
            print('***Some element not placed in the order ***')
    return output
        

def input_feature(mxene):
    input_vector = [0]*77

    molecule = Composition(mxene)
    num_ele = len(molecule.elements)
    input_vector[0] = num_ele
    atoms = rearrange_elements(molecule.elements)
    count = 1
    for j in range(num_ele):
        element = atoms[j]
        properties = ElementalProperties(molecule, element)
        for k in feature_method_mapping.keys():
            if (type(feature_method_mapping[k])==str):
                property_val = getattr(properties, feature_method_mapping[k])
            else:
                property_ind = feature_method_mapping[k][1]
                property_val = getattr(properties, feature_method_mapping[k][0])[property_ind]
            input_vector[count]=property_val
            count+=1
    input_vector = pd.Series(input_vector).fillna(0)
    return input_vector

#result = input_feature('C2O2Sc3')
#print(result)
            