import os
from ase import Atoms
from ase.io import read, write
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor


class XYZParser():
    def __init__(self, file_path):
        """
        Initializes the parser with the given file path.

        Args:
            file_path (str): The path to the XYZ file to be parsed.

        Attributes:
            n_atoms (int): The number of atoms in the molecule.
            params (dict): Additional parameters from the XYZ file.
            elements (list): List of atomic elements.
            positions (list): List of atomic positions.
            charges (list): List of atomic charges.
            atoms (list): List of atom objects created from the parsed data.
        """
        self.n_atoms, self.params, self.elements, self.positions, self.charges = self.read_xyz(file_path)
        self.atoms = self.create_atoms(self.elements, self.positions, self.params, self.charges)

    def read_xyz(self, file_path):
        """
        Reads an XYZ file and extracts molecular information.
        Parameters:
        file_path (str): The path to the XYZ file.
        Returns:
        tuple: A tuple containing:
            - n_atoms (int): The number of atoms in the molecule.
            - params (dict): A dictionary containing various molecular parameters.
            - elements (list): A list of element symbols for each atom.
            - positions (list): A list of atomic positions (x, y, z) for each atom.
            - charges (numpy.ndarray): An array of atomic charges.
        """
        params_keys = ["db", "id", "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"]
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        params_values = lines[1].split()
        elements = []
        positions = []
        charges = []
        for i in range(2, n_atoms+2):
            lines[i] = lines[i].replace("\t", " ")
            elements.append(lines[i].split()[0])
            positions.append(lines[i].split()[1:4])
            charges.append(lines[i].split()[4])

        params = dict(zip(params_keys, params_values))
        for prop in ["homo", "lumo", "gap", "zpve", "U0", "U", "H", "G"]:
            params[prop] = float(params[prop]) * 27.2114 # convert hartree to eV
        return n_atoms, params, elements, positions, np.array(charges, dtype="float64")

    def create_atoms(self, elements, positions, params, charges):
        """
        Create an Atoms object with specified elements, positions, parameters, and charges.

        Args:
            elements (list): List of element symbols.
            positions (list): List of atomic positions.
            params (dict): Dictionary of additional parameters to be stored in the Atoms object.
            charges (list): List of atomic charges.

        Returns:
            Atoms: An Atoms object with the specified elements, positions, parameters, and charges.
        """
        atoms = Atoms(elements, positions=positions)
        atoms.info = params
        atoms.set_array("charges", charges)
        return atoms


class QM9Parser(XYZParser):
    def __init__(self, file_paths):
        self.list_atoms = []

        for file_path in file_paths:
            super().__init__(file_path)
            self.atoms = self.create_atoms(self.elements, self.positions, self.params, self.charges)
            self.list_atoms.append(self.atoms)

def process_file(file_path):
    # Instantiate QM9Parser with a single file path.
    parser = QM9Parser([file_path])
    return parser.atoms


# Example usage:
if __name__ == '__main__':
    file_pattern = 'QM9/*.xyz'
    xyz_files = glob(file_pattern)
    with ProcessPoolExecutor() as executor:
        list_atoms = list(executor.map(process_file, xyz_files))
        
    parsed_files_dir = 'QM9_parsed'
    if not os.path.exists(parsed_files_dir):
        os.makedirs(parsed_files_dir)

    for i, atoms in enumerate(list_atoms):
        atoms_idx = int(atoms.info["id"])
        write(f"QM9_parsed/mol_{atoms_idx:06}.xyz", atoms)
        print(f"Parsed file {atoms_idx:06}")



