import os
import sys
import json
import time
from tqdm import tqdm
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.optimize import minimize
import numpy as np
from scqubits import Fluxonium


class fitting():
    """
    A class representing a fitting task. It shoulds be initialized by a quantum object that contains
    the extracted experimental points.
    
    """

    def __init__(self, quantum_object, *args, **kwargs):
        self.quantum_object = quantum_object

    def cost_function(self, parameters, cost_type='absolute', **kwargs):    
        cost = 0
        flux_bias = self.quantum_object.extracted_flux_bias
        extracted_points = self.quantum_object.extracted_points
        model = self.quantum_object.model

        for f in flux_bias:
            transitions = list(extracted_points[c].keys())
            data_to_fit = np.array([extracted_points[c][t] for t in transitions])
            fit = model(f, parameters, transitions,)

            if cost_type=='absolute':
                cost += np.sum(np.abs(data - fit))
            elif ost_type=='relative':
                cost += np.sum(np.abs(data - fit))/data

        return cost

    def rescale_parameter_array(parameters_list):
        return np.array(parameters_list*np.array([1e-9, 1e-9, 1e-9, 1e6, 1e6, 1e-9]))

    def unrescale_parameter_array(parameters_list):
        return np.array(parameters_list*np.array([1e9, 1e9, 1e9, 1e-6, 1e-6, 1e9]))

class quantum_object():
    """
    General class for representing a quantum object with its default parameters and extracted points.

    """

    def __init__(self, *args, **kwargs):
        self.quantum_system = None
        self.extracted_points = dict()
        self.extracted_flux_bias = []

    def model(self, flux_bias, parameters, transitions):
        pass

    def current_to_phiext(current_value, parameters):
        return (current_value - parameters['current_integer_flux'])/(parameters['Amp_per_phi0'])

    def phiext_to_current(phiext, parameters):
        return phiext * parameters['Amp_per_phi0'] + parameters['current_integer_flux']

    def from_parameter_array_to_parameter_dict(parameter_array):
        keys_list = ['EJ', 'EC', 'EL', 'Amp_per_phi0', 'current_integer_flux', 'g']
        parameters = {k:parameter_array[i] for i,k in enumerate(keys_list)}
        return parameters

class fluxonium(quantum_object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_EJ = kwargs.get("EJ", 4)
        self.initial_EC = kwargs.get("EC", 1)
        self.initial_EL = kwargs.get("EL", 1)
        self.quantum_system = Fluxonium(EJ = self.initial_EJ, EC = 2.5, EL = 0.5, flux = 0.33, cutoff = 110)


    def model(self, flux_bias, parameters, transitions):
        
        fluxonium = Fluxonium(EJ=parameters['EJ']/1e9, EC=parameters['EC']/1e9, EL=parameters['EL']/1e9, flux=current_to_phiext(current, parameters), cutoff=cutoff,)


# class fluxonium_and_cavity(quantum_object):

# class fluxonium_and_cavities(quantum_object):


class dataset():
    """
    Class for importing and formatting properly the data.

    The elements of file_list are either just a string with the path or a dict of the form
        {
            'path': string containing the path to the file,
            'label_for_I' (optional): string with the name of the dataset for I quadrature,
            'label_for_Q' (optional): string with the name of the dataset for Q quadrature,
            'label_quantity_to_plot' (optional): string in the list 'I', 'Q', 'I_rotated', 'Q_rotated', 'Mag', 'LogMag' 'Phase',
            'center' (optional): boolean, default True,
            'rescale' (optional): boolean, default False,
        }
        
    """

    def __init__(self, file_list=[], **kwargs) -> None:
        self.default_dict = kwargs.get('default_dict',
        {
            'label_for_I': 'I',
            'label_for_Q': 'Q',
            'label_quantity_to_plot': 'Mag',
            'center': True,
            'rescale': False,
        }
        )
        self.file_list = file_list.copy()
        self.file_list = complete_file_dictionnary(self.file_list)
        self.process_data()

    def complete_file_dictionnary(self, file_list):
        output_list = []

        for f in file_list:
            if type(f)==str:
                output_list.append({'path': f})
            elif type(f)==dict:
                if 'path' in f.keys():
                    output_list.append(f)
            else:
                pass
        
        for f in output_list:
            for k in self.default_dict.keys():
                if k not in f.keys():
                    f.update({k: default_dict[k]})

        return output_list

    def process_data(self):
        self.create_file_object()
        self.load_data()
        self.calculate_quantity_to_plot()

    def create_file_object(self):
        for f in self.file_list:
            f.update({'file_object':h5py.File(f['path'])})

    def load_data(self):
        for f in self.file_list:
            f.update({'I':h5py.File(f['label_for_I'])})
            f.update({'Q':h5py.File(f['label_for_Q'])})

    def calculate_quantity_to_plot(self):
        for f in self.file_list:
            if f['label_quantity_to_plot'] = 'I':
                quantity_to_plot = f['I']
            f.update({'label_quantity_to_plot':quantity_to_plot})
        
    def center(self, dataset, apply=True):
        if apply:
            return np.array([m - np.mean(m) for m in dataset])
        else:
            return dataset

    def rotate(self, complex_dataset, apply=True):
        if apply:
            return np.array([m - np.mean(m) for m in dataset])
        else:
            return dataset


class plotting():
    """
    Class for handling the plotting of experimental data, theory lines, extracted points, etc.
    
    """

    def __init__(self, quantum_object, *args, **kwargs):
        self.quantum_object = quantum_object
        self.initialise_plot()

    def initialise_plot(self):
        plt.close()
        self.fig = plt.figure(num = 1)
        self.ax = self.fig.axes[0]

    def plot_extracted_points(self):
        for f in self.quantum_object.extracted_flux_values:
            list_transitions = list(self.quantum_object.extracted_points[f].keys())
            for t in list_transitions:
                self.ax.plot(f, self.quantum_object.extracted_points[f][t], marker='x', markersize=8, color="red")
        self.fig.canvas.draw_idle()

    def remove_lines(self):
        try:
            self.ax.get_legend().remove()
        except:
            pass
        for i,l in enumerate(self.ax.lines):
            l.remove()
        self.fig.canvas.draw_idle()

    def plot_theory_lines(self, transitions=["01","02","12","03"], **kwargs):

        if 'flux_values' in kwargs:
            flux_values = kwargs.get('flux_values')
        else:
            flux_values = np.linspace(-1.0, 1.0, 101)
            
        if 'cutoff' in kwargs:
            cutoff = kwargs.get('cutoff')
        else:
            cutoff = 110


    
# def plot_theory_lines(fig, parameters, transitions=["01","02","03","04","05","06","12","13","14","15","16","23","24","25","26","34","35","36","45","46",], **kwargs):
def plot_theory_lines(fig, parameters, transitions=["03","05","13","23"], **kwargs):
    ### Get parameters from kwargs
    if 'flux_values' in kwargs:
        flux_values = kwargs.get('flux_values')
    else:
        flux_values = np.linspace(-1.0, 1.0, 101)
        
    if 'cutoff' in kwargs:
        cutoff = kwargs.get('cutoff')
    else:
        cutoff = 110
        
    ### calculate the lines
    lines_toplot = {}
    for t in transitions:
        lines_toplot[t] = []
    for f in flux_values:
        fluxonium = Fluxonium(EJ=parameters['Ej']/1e9, EC=parameters['Ec']/1e9, EL=parameters['El']/1e9, flux=f, cutoff=cutoff,)
        osc = Oscillator(3.85, truncated_dim = 21)
        # osc = Oscillator(2.993, truncated_dim = 21)
        hilbertspace = HilbertSpace([fluxonium, osc])
        g1 = parameters['g']/1e9  # coupling resonator-tmon1 (without charge matrix elements)
        operator1 = fluxonium.n_operator()
        operator2 = osc.creation_operator() + osc.annihilation_operator()
        hilbertspace.add_interaction(
        g=g1,
        op1=(operator1, fluxonium),
        op2=(operator2, osc)
        )
        dressed_hamiltonian = hilbertspace.hamiltonian()
        # Edn.append(dressed_hamiltonian.eigenenergies(0)[:6])
        # eigenenergies = fluxonium.diago()[0]
        # eigenenergies = fluxonium.eigenvals()
        eigenenergies = dressed_hamiltonian.eigenenergies(0)
        for t in transitions:
            #lines_toplot[t].append(1e9*np.array([eigenenergies[int(t[1])] - eigenenergies[int(t[0])] for t in transitions]))
            lines_toplot[t].append(1e9*np.array([eigenenergies[int(t[1])] - eigenenergies[int(t[0])]]))
    
    ### Plotting the lines
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    current = phiext_to_current(flux_values, parameters)
    for t in transitions:
        ax.plot(current, lines_toplot[t], label=t)
        # ax.plot(flux_values, np.array(lines_toplot[t])/1e9, label=t)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='upper right')
    ax.set_title('Ej = {:.2f} GHz, Ec = {:.2f} GHz, El = {:.2f} GHz, g = {:.2f} MHz'.format(parameters['Ej']/1e9, parameters['Ec']/1e9, parameters['El']/1e9, parameters['g']/1e6))
    fig.canvas.draw_idle()






def model(current, parameters, transitions, **kwargs):
    if 'cutoff' in kwargs:
        cutoff = kwargs.get('cutoff')
    else:
        cutoff = 110
    # fluxonium = Fluxonium(EJ=parameters['Ej']/1e9, EC=parameters['Ec']/1e9, EL=parameters['El']/1e9, flux=current_to_phiext(current, parameters), cutoff=cutoff,)
    # eigenenergies = fluxonium.eigenvals()
    # eigenenergies = fluxonium.diago()[0]
    fluxonium = Fluxonium(EJ=parameters['Ej']/1e9, EC=parameters['Ec']/1e9, EL=parameters['El']/1e9, flux=current_to_phiext(current, parameters), cutoff=cutoff,)
    osc = Oscillator(3.85, truncated_dim = 21)
    # osc = Oscillator(2.993, truncated_dim = 21)
    hilbertspace = HilbertSpace([fluxonium, osc])
    g1 = parameters['g']/1e9  # coupling resonator-tmon1 (without charge matrix elements)
    operator1 = fluxonium.n_operator()
    operator2 = osc.creation_operator() + osc.annihilation_operator()
    hilbertspace.add_interaction(
    g=g1,
    op1=(operator1, fluxonium),
    op2=(operator2, osc)
    )
    dressed_hamiltonian = hilbertspace.hamiltonian()
    # Edn.append(dressed_hamiltonian.eigenenergies(0)[:6])
    # eigenenergies = fluxonium.diago()[0]
    # eigenenergies = fluxonium.eigenvals()
    eigenenergies = dressed_hamiltonian.eigenenergies(0)
    return np.array([eigenenergies[int(t[1])] - eigenenergies[int(t[0])] for t in transitions])

