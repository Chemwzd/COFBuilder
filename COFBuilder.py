import numpy as np
import math
import os
from fractions import Fraction

from chembondpy import bonder
from itertools import combinations
from scipy.spatial.distance import cdist


def operation(Matrix, op_all):
    """
        ---The original code without modification.---
        source: https://github.com/zhangxiangyu6/COF

    Perform operations on the matrix according to the given operation list op_all,
    which includes translations and rotations.
    """
    MT = []
    for op in op_all:
        ops = op.replace(' ', '').split(',')
        for no in ['x', 'y', 'z']:
            if no in ops[0]:
                X = ops[0].lstrip().split(no)
        for no in ['x', 'y', 'z']:
            if no in ops[1]:
                Y = ops[1].lstrip().split(no)
        for no in ['x', 'y', 'z']:
            if no in ops[2]:
                Z = ops[2].lstrip().split(no)
        MM = Matrix.copy()
        matrix1 = Matrix.copy()
        if ops[0].replace('-', '')[0] == 'x':
            if X[0] == '-':
                if X[1] != '':
                    matrix1[:, 0] = -matrix1[:, 0] + Fraction(X[1])
                else:
                    matrix1[:, 0] = -matrix1[:, 0]
            else:
                if X[1] != '':
                    matrix1[:, 0] = matrix1[:, 0] + Fraction(X[1])
                else:
                    matrix1[:, 0] = matrix1[:, 0]

        if ops[0].replace('-', '')[0] == 'y':
            if X[0] == '-':
                if X[1] != '':
                    matrix1[:, 0] = -matrix1[:, 1] + Fraction(X[1])
                else:
                    matrix1[:, 0] = -matrix1[:, 1]

            else:
                if X[1] != '':
                    matrix1[:, 0] = matrix1[:, 1] + Fraction(X[1])
                else:
                    matrix1[:, 0] = matrix1[:, 1]

        if ops[0].replace('-', '')[0] == 'z':
            if X[0] == '-':
                if X[1] != '':
                    matrix1[:, 0] = -matrix1[:, 2] + Fraction(X[1])
                else:
                    matrix1[:, 0] = -matrix1[:, 2]
            else:
                if X[1] != '':
                    matrix1[:, 0] = matrix1[:, 2] + Fraction(X[1])
                else:
                    matrix1[:, 0] = matrix1[:, 2]
        MM[:, 0] = matrix1[:, 0]
        matrix1 = Matrix.copy()
        if ops[1].replace('-', '')[0] == 'y':

            if Y[0] == '-':
                if Y[1] != '':
                    matrix1[:, 1] = -matrix1[:, 1] + Fraction(Y[1])
                else:
                    matrix1[:, 1] = -matrix1[:, 1]
            else:
                if Y[1] != '':
                    matrix1[:, 1] = matrix1[:, 1] + Fraction(Y[1])
                else:
                    matrix1[:, 1] = matrix1[:, 1]

        if ops[1].replace('-', '')[0] == 'x':
            if Y[0] == '-':
                if Y[1] != '':
                    matrix1[:, 1] = -matrix1[:, 0] + Fraction(Y[1])
                else:
                    matrix1[:, 1] = -matrix1[:, 0]
            else:
                if Y[1] != '':
                    matrix1[:, 1] = matrix1[:, 0] + Fraction(Y[1])
                else:
                    matrix1[:, 1] = matrix1[:, 0]

        if ops[1].replace('-', '')[0] == 'z':
            if Y[0] == '-':
                if Y[1] != '':
                    matrix1[:, 1] = -matrix1[:, 2] + Fraction(Y[1])
                else:
                    matrix1[:, 1] = -matrix1[:, 2]
            else:
                if Y[1] != '':
                    matrix1[:, 1] = matrix1[:, 2] + Fraction(Y[1])
                else:
                    matrix1[:, 1] = matrix1[:, 2]

        MM[:, 1] = matrix1[:, 1]
        matrix1 = Matrix.copy()

        if ops[2].replace('-', '')[0] == 'z':
            if Z[0] == '-':
                if Z[1] != '':
                    matrix1[:, 2] = -matrix1[:, 2] + Fraction(Z[1])
                else:
                    matrix1[:, 2] = -matrix1[:, 2]
            else:
                if Z[1] != '':
                    matrix1[:, 2] = matrix1[:, 2] + Fraction(Z[1])
                else:
                    matrix1[:, 2] = matrix1[:, 2]

        if ops[2].replace('-', '')[0] == 'x':
            if Z[0] == '-':
                if Z[1] != '':
                    matrix1[:, 2] = -matrix1[:, 0] + Fraction(Z[1])
                else:
                    matrix1[:, 2] = -matrix1[:, 0]
            else:
                if Z[1] != '':
                    matrix1[:, 2] = matrix1[:, 0] + Fraction(Z[1])
                else:
                    matrix1[:, 2] = matrix1[:, 0]

        if ops[2].replace('-', '')[0] == 'y':
            if Z[0] == '-':
                if Z[1] != '':
                    matrix1[:, 2] = -matrix1[:, 1] + Fraction(Z[1])
                else:
                    matrix1[:, 2] = -matrix1[:, 1]
            else:
                if Z[1] != '':
                    matrix1[:, 2] = matrix1[:, 1] + Fraction(Z[1])
                else:
                    matrix1[:, 2] = matrix1[:, 1]

        MM[:, 2] = matrix1[:, 2]
        MT.append(MM.round(4).tolist())
    return MT


def appendseg(initialcoor, latercoor, trancoor):
    """
        ---The original code without modification.---
        source: https://github.com/zhangxiangyu6/COF
    """
    initialvec1 = np.array(initialcoor[1]) - np.array(initialcoor[0]) + np.array(initialcoor[2]) - np.array(
        initialcoor[0])
    initialvec2 = np.cross(np.array(initialcoor[1]) - np.array(initialcoor[0]),
                           np.array(initialcoor[2]) - np.array(initialcoor[0])) \
                  / np.linalg.norm(np.cross(np.array(initialcoor[1]) - np.array(initialcoor[0]),
                                            np.array(initialcoor[2]) - np.array(initialcoor[0]))) \
                  * (np.linalg.norm(np.array(initialcoor[1]) - np.array(initialcoor[0])) +
                     np.linalg.norm(np.array(initialcoor[2]) - np.array(initialcoor[0])))
    initialvec3 = np.cross(initialvec1, initialvec2) / np.linalg.norm(np.cross(initialvec1, initialvec2)) \
                  * (np.linalg.norm(initialvec1) + np.linalg.norm(initialvec2))
    initialm = np.matrix(np.vstack((initialvec1, initialvec2, initialvec3))).T

    latervec1 = np.array(latercoor[1]) - np.array(latercoor[0]) + np.array(latercoor[2]) - np.array(latercoor[0])
    latervec2 = np.cross(np.array(latercoor[1]) - np.array(latercoor[0]),
                         np.array(latercoor[2]) - np.array(latercoor[0])) \
                / np.linalg.norm(
        np.cross(np.array(latercoor[1]) - np.array(latercoor[0]), np.array(latercoor[2]) - np.array(latercoor[0]))) \
                * (np.linalg.norm(np.array(latercoor[1]) - np.array(latercoor[0])) + np.linalg.norm(
        np.array(latercoor[2]) - np.array(latercoor[0])))
    latervec3 = np.cross(latervec1, latervec2) / np.linalg.norm(np.cross(latervec1, latervec2)) \
                * (np.linalg.norm(latervec1) + np.linalg.norm(latervec2))
    laterm = np.matrix(np.vstack((latervec1, latervec2, latervec3))).T

    tranm = [np.array(trancoor[i]) - np.array(initialcoor[0]) for i in range(len(trancoor))]
    tranm = np.matrix(tranm).T

    m = np.linalg.inv(initialm).dot(tranm)

    new = np.array((laterm.dot(m)).T)
    newnew = [(new[i] + np.array(latercoor[0])).tolist() for i in range(len(trancoor))]
    return newnew


def reflect_frage(new_xyz_c_mat, original_linker):
    """
        ---The original code without modification.---
        source: https://github.com/zhangxiangyu6/COF
    """
    linker_in_mof = np.array(new_xyz_c_mat)
    old_linker = np.array(original_linker)
    lin = 0
    vectors = 0
    while vectors == 0:
        one_length = np.around(
            (linker_in_mof[lin, :] - linker_in_mof[lin + 1, :]).dot(linker_in_mof[lin, :] - linker_in_mof[lin + 1, :]) - \
            (old_linker[lin, :] - old_linker[lin + 1, :]).dot(old_linker[lin, :] - old_linker[lin + 1, :]), 1)
        two_length = np.around((linker_in_mof[lin + 2, :] - linker_in_mof[lin + 1, :]).dot(
            linker_in_mof[lin + 2, :] - linker_in_mof[lin + 1, :]) - \
                               (old_linker[lin + 2, :] - old_linker[lin + 1, :]).dot(
                                   old_linker[lin + 2, :] - old_linker[lin + 1, :]), 1)
        if one_length == 0.0 and two_length == 0.0:
            vectors = 1
            initialcoor = [original_linker[lin, :], original_linker[lin + 1, :], original_linker[lin + 2, :]]
            latercoor = [new_xyz_c_mat[lin, :], new_xyz_c_mat[lin + 1, :], new_xyz_c_mat[lin + 2, :]]
            trancoor = original_linker
            new_xyz_c_mat_maped = np.array(appendseg(initialcoor, latercoor, trancoor))
        else:
            lin = lin + 1
            if lin > linker_in_mof.shape[0] - 3:
                break
    return new_xyz_c_mat_maped


def rotation(Angle, new_xyz_c_mat0, center, VV):
    """
        ---The original code without modification.---
        source: https://github.com/zhangxiangyu6/COF
    """
    VV = np.array(VV)
    new_xyz_c_mat = new_xyz_c_mat0.T
    a, b, c = center[0], center[1], center[2]
    length = (VV.dot(VV)) ** 0.5
    u, v, w = VV[0] / length, VV[1] / length, VV[2] / length
    angle = math.radians(Angle)

    matrix = np.matrix([[u * u + (v * v + w * w) * math.cos(angle), u * v * (1 - math.cos(angle)) - w * math.sin(angle),
                         u * w * (1 - math.cos(angle)) + v * math.sin(angle),
                         (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - math.cos(angle)) + (
                                 b * w - c * v) * math.sin(angle)], \
                        [u * v * (1 - math.cos(angle)) + w * math.sin(angle), v * v + (u * u + w * w) * math.cos(angle),
                         v * w * (1 - math.cos(angle)) - u * math.sin(angle),
                         (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - math.cos(angle)) + (
                                 c * u - a * w) * math.sin(angle)], \
                        [u * w * (1 - math.cos(angle)) - v * math.sin(angle),
                         v * w * (1 - math.cos(angle)) + u * math.sin(angle), w * w + (u * u + v * v) * math.cos(angle),
                         (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - math.cos(angle)) + (
                                 a * v - b * u) * math.sin(angle)], \
                        [0, 0, 0, 1]])
    add_line = np.ones(new_xyz_c_mat.shape[1])
    result = matrix.dot(np.array(np.insert(new_xyz_c_mat, 3, values=add_line, axis=0)))
    res = result.round(6)
    return res.T[:, 0:3]


def convert_frac_to_cartesian(fractional_coords, unitcell):
    """
    Convert fractional coordinates into Cartesian coordinates.

    Parameters:
    ----------
        fractional_coords:
        unitcell:

    Returns:

    """
    a, b, c, alpha, beta, gamma = unitcell[0], unitcell[1], unitcell[2], unitcell[3], unitcell[4], unitcell[5]
    cell_matrix = calculate_cell_matrix(a, b, c, alpha, beta, gamma)
    cartesian_coord = np.dot(cell_matrix, np.array(fractional_coords).T).round(4).T
    return cartesian_coord


def convert_cartesian_to_frac(cartesian_coords, unitcell):
    """
    Convert Cartesian coordinates into fractional coordinates.

    Parameters:
    ----------
        cartesian_coords:
        unitcell:

    Returns:

    """
    a, b, c, alpha, beta, gamma = unitcell[0], unitcell[1], unitcell[2], unitcell[3], unitcell[4], unitcell[5]

    cell_matrix = calculate_cell_matrix(a, b, c, alpha, beta, gamma)
    inverse_cell_matrix = np.linalg.inv(cell_matrix)
    fractional_coord = np.dot(inverse_cell_matrix, cartesian_coords.T).T.round(4)
    return fractional_coord


def check_atom_clash_in_cell(cif_file, unitcell):
    """
    Check if the distances between atoms in the crystal cell meet the requirements.
    For example, whether the atomic distances are within 0 to 0.8 angstroms.

    Returns:
        bool: Whether the distances between atoms in the crystal cell meet the requirements.
    """
    # Load data from CIF file
    cif_data = np.loadtxt(cif_file, skiprows=19, usecols=[2, 3, 4])

    # Convert fractional coordinates to Cartesian coordinates
    cartesian_coords = convert_frac_to_cartesian(cif_data, unitcell)
    is_clashed = False
    for i in range(cartesian_coords.shape[0]):
        for j in range(i):
            # Calculate distance between points
            distance = np.linalg.norm(cartesian_coords[i, :] - cartesian_coords[j, :])
            if 0 < distance < 0.8:
                is_clashed = True
                break
        break
    return is_clashed


def calculate_cell_volume(a, b, c, alpha, beta, gamma):
    """
    Calculate the volume of the cell.

    Notes:
        alpha, beta and gamma should be in radians.
    """
    volume = a * b * c * np.sqrt(
        1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(
            gamma))
    return volume


def calculate_cell_matrix(a, b, c, alpha, beta, gamma):
    """
    Calculate the matrix of the cell.

    Notes:
        alpha, beta and gamma should be in radians.
    """
    alpha_radian = np.radians(alpha)
    beta_radian = np.radians(beta)
    gamma_radian = np.radians(gamma)
    cell_volume = calculate_cell_volume(a, b, c, alpha_radian, beta_radian, gamma_radian)
    cell_matrix = np.array([
        [a, b * np.cos(gamma_radian), c * np.cos(beta_radian)],
        [0, b * np.sin(gamma_radian),
         c * (np.cos(alpha_radian) - np.cos(beta_radian) * np.cos(gamma_radian)) / np.sin(gamma_radian)],
        [0, 0, cell_volume / (a * b * np.sin(gamma_radian))]
    ])
    return cell_matrix


def calculate_fractional_distance(coord1, coord2, cell_matrix):
    """
    Calculate the Euclidean distance between two given fractional coordinates.

    Parameters:
    ----------
        coord1:
        coord2:
        cell_matrix: numpy.ndarray
            Can be calculated through function :calculate_cell_matrix
            by giving the unitcell parameters.

    Returns:

    """
    coord_vector = np.array(coord1) - np.array(coord2)
    for axis in range(3):
        while abs(coord_vector[axis]) > 0.5:
            if coord_vector[axis] < 0:
                coord_vector[axis] = coord_vector[axis] + 1
            else:
                coord_vector[axis] = coord_vector[axis] - 1
    cartesian_coord = np.dot(cell_matrix, coord_vector)
    distance = np.linalg.norm(cartesian_coord)
    return distance


def count_error_bonds(cif_file, unitcell, num_node_atoms=146, num_linker_atoms=14, num_nodes=4, num_linkers=16,
                      bonded_elements=('N', 'C'), bonded_num=2, distance_threshold=(0, 0.8)):
    """
    Check the number of incorrect bond connections in the constructed COF structure.

    Parameters:
    ----------
        cif_file: str
            The CIF file generated by our code.
        unitcell: list
            The cell parameters of the unit cell. The angle parameters of the unit cell should be in degrees.
        num_node_atoms: int
            The number of atoms for the node molecule. (default to be 45 for COF-300)
        num_linker_atoms: int
            The number of atoms for the linker molecule. (default to be 14 for COF-300)
        num_nodes: int
            The number of node molecule for constructing the COF structure. (default to be 4 for COF-300)
        num_linkers: int
            The number of linker molecule for constructing the COF structure. (default to be 8 for COF-300)
        bonded_elements: tuple, in the form of [{node_bonded_element}, {linker_bonded_element}]
            The bonded elements during constructing of the COF structure. (default to be ['N', 'C'] for COF-300)
            For example, if an imine-bond is built, the bonded atom for node is 'N' and 'C' for linker, the parameter
            :bonded_elements should be ['N', 'C']
        bonded_num: int
            The number of connected bonds during constructing the COF structure. (default to be 2 for COF-300)
        distance_threshold: tuple
            Range for determining bond connectivity

    Returns:
        int, the number of incorrect bond connections in the constructed COF structure
    """
    # the input unicell parameters should be in degree
    a, b, c, alpha, beta, gamma = (unitcell[0], unitcell[1], unitcell[2],
                                   unitcell[3], unitcell[4], unitcell[5])
    cell_matrix = calculate_cell_matrix(a, b, c, alpha, beta, gamma)

    atom_coords = np.loadtxt(cif_file, skiprows=19, usecols=[2, 3, 4])
    total_node_atoms = num_node_atoms * num_nodes

    node_coords = atom_coords[:total_node_atoms]
    linker_coords = atom_coords[total_node_atoms:total_node_atoms + num_linker_atoms]

    with open(cif_file) as f:
        cif_file_contents = f.readlines()
    atom_labels = [i.split()[0] for i in cif_file_contents[19:]]

    error_bonded = 0  # Count the error bond of the bridge atoms between
    # node and linker molecules with the bonded_elements.

    wrong_bonds = 0  # Count the error bond for other atoms between
    # node and linker molecules.

    for node_atom_index, node_atom_coord in enumerate(node_coords):
        for linker_atom_index, linker_atom_coord in enumerate(linker_coords):
            distance = calculate_fractional_distance(node_atom_coord, linker_atom_coord, cell_matrix)

            if distance_threshold[0] < distance < distance_threshold[1]:
                wrong_bonds += 1
            else:
                node_atom_symbol = atom_labels[node_atom_index]
                # The index of linker_atom should be added the number of linker atoms.
                linker_atom_symbol = atom_labels[linker_atom_index + total_node_atoms]
                is_bond = bonder([node_atom_symbol, linker_atom_symbol], distance, [])
                if is_bond == 1 and (node_atom_symbol, linker_atom_symbol) != bonded_elements:
                    wrong_bonds += 1
                if is_bond == 1 and (node_atom_symbol, linker_atom_symbol) == bonded_elements:
                    error_bonded += 1

    # calculate the number of wrong bonds between each linker-linker pair.
    all_linker_coords = atom_coords[total_node_atoms:]
    split_linker_coords = [all_linker_coords[i:i + len(all_linker_coords) // num_linkers] for i in
                         range(0, len(all_linker_coords), len(all_linker_coords) // num_linkers)]
    linker_atom_labels = atom_labels[total_node_atoms:total_node_atoms+num_linker_atoms]
    linker_coords_combinations = list(combinations(split_linker_coords, 2))
    wrong_linker_bonds = 0
    for combination in linker_coords_combinations:
        wrong_linker_bonds += sum([bonder(
            [linker_atom_labels[i], linker_atom_labels[j]],
            calculate_fractional_distance(combination[0][i], combination[1][j], cell_matrix), []
        )
            for i in range(num_linker_atoms) for j in range(num_linker_atoms)])

    # calculate the number of wrong bonds between each node-node pair.
    split_node_coords = [node_coords[i:i + len(node_coords) // num_nodes] for i in
                         range(0, len(node_coords), len(node_coords) // num_nodes)]
    node_atom_labels = atom_labels[:num_node_atoms]
    node_coords_combinations = list(combinations(split_node_coords, 2))
    wrong_node_bonds = 0
    for combination in node_coords_combinations:
        wrong_node_bonds += sum([bonder(
            [node_atom_labels[i], node_atom_labels[j]],
            calculate_fractional_distance(combination[0][i], combination[1][j], cell_matrix), []
        )
            for i in range(num_node_atoms) for j in range(num_node_atoms)])

    return wrong_bonds + abs(error_bonded - bonded_num) + wrong_linker_bonds + wrong_node_bonds


def write_node_cif(out_filename, coordinates, atom_labels, unitcell, num_nodes=8):
    """
    Write the header and node sections of the CIF file.

    Parameters:
    ----------
        out_filename:
        coordinates:
        atom_labels:
        unitcell:
        num_nodes:

    Returns:
        None
    """
    cif_contents = ['data_\n', f'_cell_length_a    {str(unitcell[0])}\n', f'_cell_length_b    {str(unitcell[1])}\n',
                    f'_cell_length_c    {str(unitcell[2])}\n', f'_cell_angle_alpha    {str(unitcell[3])}\n',
                    f'_cell_angle_beta    {str(unitcell[4])}\n', f'_cell_angle_gamma    {str(unitcell[5])}\n',
                    "_symmetry_space_group_name_H-M		'P1'\n", '_symmetry_Int_Tables_number		1\n',
                    '_symmetry_cell_setting		Monoclinic\n', 'loop_\n', '_symmetry_equiv_pos_as_xyz\n',
                    '+x,+y,+z\n', 'loop_\n', '_atom_site_label\n', '_atom_site_type_symbol\n', '_atom_site_fract_x\n',
                    '_atom_site_fract_y\n', '_atom_site_fract_z\n']

    for coord in coordinates[0:num_nodes]:
        for i in range(len(atom_labels)):
            cif_contents.append(
                f'{atom_labels[i]} {atom_labels[i]} {str(coord[i][0])} {str(coord[i][1])} {str(coord[i][2])}\n')

    with open(out_filename, 'w') as f:
        f.writelines(cif_contents)

    return None


def write_linker_cif(out_filename, matrixs, atom_labels, num_linkers=8):
    """
    Write the linker sections of the CIF file.

    Parameters:
    ----------
        out_filename:
        matrixs:
        atom_labels:
        unitcell:

    Returns:
        None
    """
    linker_cif_contents = []
    for matrix in matrixs[0:num_linkers]:
        for i in range(len(atom_labels)):
            linker_cif_contents.append(
                f'{atom_labels[i]} {atom_labels[i]} {str(matrix[i][0])} {str(matrix[i][1])} {str(matrix[i][2])}\n')

    with open(out_filename, 'a') as f:
        f.writelines(linker_cif_contents)

    return None


def check_bond_atom_dis(bond_atom_coords, unitcell, num_bond_atoms=8, distance_tolerance=2):
    """
    Note:
        The input is fractional coordinates.

    Parameters:
    ----------
        bond_atom_coords:
        unitcell:
        num_bond_atoms:
        distance_tolerance:

    Returns:

    """
    is_wrong_distance = False
    min_distance_matrix = []
    for i in range(num_bond_atoms):
        min_dis, _, _ = calculate_min_distance(bond_atom_coords[i], bond_atom_coords[num_bond_atoms:], unitcell)
        min_distance_matrix.append(min_dis)
    if min(min_distance_matrix) < distance_tolerance:
        is_wrong_distance = True
    return is_wrong_distance


def calculate_min_distance(coordinate_a, coordinates_set, unitcell):
    """
    Given the coordinates of a point A in fractional coordinate, calculate its distance relationship with a set of coordinates B.

    Parameters:
    ----------
        coordinate_a:
        coordinates_set:
        unitcell:

    Returns:
        1. the minimum distance dis_min between A and set B
        2. the vector between A and its nearest point
        3. the coordinates of the point closest to A.
    """
    a, b, c, alpha, beta, gamma = unitcell[0], unitcell[1], unitcell[2], unitcell[3], unitcell[4], unitcell[5]
    cell_matrix = calculate_cell_matrix(a, b, c, alpha, beta, gamma)

    distance_matrix = []
    all_vectors = []
    for coord in coordinates_set:
        distance = calculate_fractional_distance(coordinate_a, coord, cell_matrix)
        vector = np.array(coordinate_a) - np.array(coord)
        distance_matrix.append(distance)
        all_vectors.append(np.dot(cell_matrix, vector))

    distance_min = min(distance_matrix)
    min_index = distance_matrix.index(distance_min)

    return distance_min, all_vectors[min_index], coordinates_set[min_index]


def check_node_overate(fraction_coords, unitcell, num_symmetry_operation=16, num_bond_atoms=8):
    """
    Note:
        The first n lines of the XYZ file should correspond to the coordinates of the n atoms used to connect the COF.

    Parameters:
    ----------
        fraction_coords:
        unitcell:
        num_symmetry_operation:
        num_bond_atoms:

    Returns:

    """
    is_overate = False

    # For example, in imine-linked COFs, it's the N atoms.
    bond_atom_coords = []
    for i in range(num_symmetry_operation):
        for j in range(num_bond_atoms):
            bond_atom_coords.append(fraction_coords[i][j])

    # Coordinates of atoms connected to the bond_atom.
    connected_atoms = []
    for i in range(num_symmetry_operation):
        for j in range(num_bond_atoms, num_bond_atoms * 2):
            connected_atoms.append(fraction_coords[i][j])

    distance_between_nodes = []
    for i in range(num_bond_atoms):
        distance_min, _, _ = calculate_min_distance(bond_atom_coords[i], bond_atom_coords[num_bond_atoms:], unitcell)
        distance_between_nodes.append(distance_min)

    if sum(distance_between_nodes) / len(distance_between_nodes) < 0.5:
        is_overate = True

    return bond_atom_coords, connected_atoms, is_overate, sum(distance_between_nodes) / len(distance_between_nodes)


def calculate_angle_between_vectors(vector1, vector2):
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)

    cos_angle = np.dot(vector1, vector2) / (length1 * length2)

    # It's possible that the cosine of the angle may slightly exceed 1
    # due to floating-point arithmetic errors.
    cos_angle = min(cos_angle, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def check_linker_bonds(linker_bond_atom_coords, node_bond_atom_coords, unitcell, linker_coords, node_coords,
                       ref_bond_length=1.273, ref_linker_side_angle=122.439, ref_node_side_angle=120.414):
    """
    Note:
        Input is fractional Coordinates
    Parameters:
    ----------
        linker_bond_atom_coords: numpy.ndarray
            The atomic coordinates of the linker used for bonding, for example, in COF-300,
            it would be the coordinates of the two C atoms at both ends of the linker.
        node_bond_atom_coords: numpy.ndarray
            The atomic coordinates of the node used for bonding, for example, in COF-300,
            it would be the coordinates of the four N atoms at both ends of the linker.
        unitcell: list or tuple
            The cell parameters.
        linker_coords: numpy.ndarray
            Coordinates of all non-hydrogen atoms in the linker.
        node_coords: numpy.ndarray
            Coordinates of all non-hydrogen atoms in the node.
        ref_bond_length: float
            The average bond length. For COF-300 with imine bonds, the average length is 1.273.
        ref_linker_side_angle: float
            The average bond angle of the linker side. For COF-300, the value is 122.439.
        ref_node_side_angle: float
            The average bond angle of the node side. For COF-300, the value is 120.414.

    Returns:

    """
    if_write = False
    bond_length_deviation = []
    linker_side_angles = []
    node_side_angles = []

    # Note that all the input coordinates are in fractional coordinate.
    for i, linker_atom_coord in enumerate(linker_bond_atom_coords):
        # Obtain the node atom B closest to this linker atom A,
        # the vector between the two atoms, and the position of atom B.
        # Note that the {bond_vector} {node_bond_atom_coord} are in the Cartesian coordinate.
        a_b_bond_dis, a_b_bond_vector, b_coord = calculate_min_distance(linker_atom_coord, node_bond_atom_coords,
                                                                        unitcell)

        # Obtain the non-hydrogen atom C closest to atom B in the node molecule,
        # which is the atom connected to this end-point atom, for subsequent calculation of bond angles.
        # For example, in COF-300, this will be used to calculate the angle of -C-C=N-.
        b_c_bond_dis, b_c_bond_vector, c_coord = calculate_min_distance(b_coord, node_coords, unitcell)

        # Obtain the non-hydrogen atom D closest to this linker atom A in the linker molecule,
        # which is the atom connected to this end-point atom, for subsequent calculation of bond angles.
        a_d_bond_dis, a_d_bond_vector, d_coord = calculate_min_distance(linker_atom_coord, linker_coords, unitcell)

        # For COF-300, it's -C-C-N- angle on the linker side.
        angle_d_a_b = calculate_angle_between_vectors(np.array(a_b_bond_vector), np.array(a_d_bond_vector))

        # For COF-300, it's -C-C-N- angle on the node side.
        angle_a_b_c = 180 - calculate_angle_between_vectors(np.array(a_b_bond_vector), np.array(b_c_bond_vector))

        linker_side_angles.append(angle_d_a_b)
        node_side_angles.append(angle_a_b_c)

        bond_length_deviation.append(abs(a_b_bond_dis - ref_bond_length))

    if max(bond_length_deviation) <= 0.15:
        if_write = True

    linker_side_error = abs(
        sum(linker_side_angles) / len(linker_side_angles) - ref_linker_side_angle) / ref_linker_side_angle
    node_side_error = abs(sum(node_side_angles) / len(node_side_angles) - ref_node_side_angle) / ref_node_side_angle

    return if_write, sum(bond_length_deviation) / len(bond_length_deviation), linker_side_error, node_side_error


def check_fractional_coords(frac_coords):
    return np.array(frac_coords) % 1


def get_node_matrix(node_coords, node_x_rot, node_y_trans, node_z_trans, unitcell, ref_coord, operations):
    """
    Note:
        Input is Cartesian Coordinates
    Parameters:
    ----------
        node_coords:
        node_x_rot:
        node_y_trans:
        node_z_trans:
        unitcell:
        ref_coord:
        operations:

    Returns:

    """
    center = [np.sum(node_coords[:, i]) / node_coords.shape[0] for i in range(node_coords.shape[1])]
    rotated_coords = rotation(node_x_rot, node_coords, center, [1, 0, 0])

    rotated_matrix = convert_cartesian_to_frac(rotated_coords, unitcell)

    # Restore the node to its initial position after rotation.
    rotated_center = [np.sum(rotated_matrix[:, i]) / rotated_matrix.shape[0] for i in range(rotated_matrix.shape[1])]
    x0, y0, z0 = ref_coord[0], ref_coord[1], ref_coord[2]
    mx = x0 - rotated_center[0]
    my = y0 - rotated_center[1]
    mz = z0 - rotated_center[2]
    rotated_matrix += np.array([mx, my, mz])
    translated_matrix = rotated_matrix + np.array([0, node_y_trans, node_z_trans])
    operated_matrix = operation(translated_matrix, operations)

    return operated_matrix


def get_linker_matrix(linker_coords, linker_rot_angle, linker_rot_axis, ref_coord, unitcell, operations):
    """
    Note:
        Input is Cartesian Coordinates
    Parameters:
    ----------
        linker_coords:
        linker_rot_angle:
        linker_rot_axis:
        ref_coord: The center of N-N atoms.
        unitcell:
        operations:

    Returns:

    """
    # Translate the linker to the center of N-N atoms.
    center = [np.sum(linker_coords[:, i]) / linker_coords.shape[0] for i in range(linker_coords.shape[1])]
    translation_vector = np.array(ref_coord) - center
    translated_coords = linker_coords + translation_vector

    rotated_coords = rotation(linker_rot_angle, translated_coords, ref_coord, linker_rot_axis)
    rotated_matrix = convert_cartesian_to_frac(rotated_coords, unitcell)

    operated_matrix = operation(rotated_matrix, operations)

    return operated_matrix


def is_proper(bond_atom_coords, unitcell, num_nodes=4, num_bond_atoms=8, ref_distance=7.46, threshold=1.):
    """
    Determine if the distance between the two N atoms is suitable for placing the linker.

    Note:
        Input is fractional Coordinates
    Parameters:
    ----------
        bond_atom_coords:
        unitcell:
        num_nodes:
        num_bond_atoms:
        ref_distance:
        threshold:

    Returns:

    """
    sliced_bond_atom_coords = [convert_frac_to_cartesian(bond_atom_coords[i: i + 8], unitcell)
                               for i in range(0, num_nodes * num_bond_atoms, num_bond_atoms)]
    comb = list(combinations(range(num_nodes), 2))

    proper_dict = {}

    for i, j in comb:
        distance_matrix = cdist(sliced_bond_atom_coords[i], sliced_bond_atom_coords[j])
        min_distance = np.min(np.abs(distance_matrix - ref_distance))
        min_idx = np.unravel_index(np.argmin(np.abs(distance_matrix - 2)), distance_matrix.shape)
        min_coords = (sliced_bond_atom_coords[i][min_idx[0]], sliced_bond_atom_coords[j][min_idx[1]])

        if min_distance <= threshold:
            proper_dict[(i, j)] = {'proper_distance': min_distance,
                                   'proper_coords': min_coords,
                                   'proper_node_coords': (sliced_bond_atom_coords[i], sliced_bond_atom_coords[j])
                                   }

    return proper_dict


def align_vectors(coords, ref_vectors, angle=0):
    """
    Align the two vectors together at an angle of {angle}.

    """
    ref_start = ref_vectors[0]
    ref_end = ref_vectors[1]
    v_ref = ref_end - ref_start
    point1 = coords[0]
    point2 = coords[1]
    v_original = point2 - point1
    axis = np.cross(v_original, v_ref)
    axis = axis / np.linalg.norm(axis)

    cos_theta = np.dot(v_original, v_ref) / (np.linalg.norm(v_original) * np.linalg.norm(v_ref))
    theta = np.arccos(cos_theta)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    transformed_coords = np.dot(coords - point1, R.T) + ref_start
    theta = np.radians(angle)

    v_trans = transformed_coords[0] - transformed_coords[1]
    v_original_norm = v_trans / np.linalg.norm(v_trans)
    v_ref_norm = v_ref / np.linalg.norm(v_ref)
    axis = np.cross(v_original_norm, v_ref_norm)
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = cos_theta * I + sin_theta * K + (1 - cos_theta) * np.outer(axis, axis)
    centroid = np.mean(transformed_coords, axis=0)
    coords_centered = transformed_coords - centroid
    coords_rotated = np.dot(coords_centered, R.T)
    coords_transformed = coords_rotated + centroid

    return coords_transformed


def is_same_bond_atom(coords1, coords2):
    is_same_bond = False
    distance_matrix = cdist(coords2, coords1)
    if np.min(distance_matrix) <= 0.1:
        is_same_bond = True
    return is_same_bond


def _construct_cof(node_file, linker_file, out_file, workdir, node_ref_coord,
                   node_x_rot, node_y_trans, node_z_trans, linker_rot_angle,
                   unitcell, operations, num_nodes=4, num_linkers=16, num_bond_atoms=8):
    os.chdir(workdir)

    with open(node_file) as f:
        node_file_contents = f.readlines()[2:]

    node_atom_coords = np.loadtxt(node_file, skiprows=2, usecols=[1, 2, 3])
    node_atom_labels = [content.split()[0] for content in node_file_contents]

    # first_atom can also be determined out of the pso algorithm.
    # node_matrix.shape is {number of operations}*{number of node atoms}*3
    node_matrix = get_node_matrix(node_atom_coords, node_x_rot, node_y_trans, node_z_trans, unitcell,
                                  ref_coord=node_ref_coord, operations=operations)
    # For imine-linked COFs, the {node_bond_atom_coords} is coordinates of N atom in node.
    # {connected_atoms} is the atom connected with the bond atom.
    node_bond_atom_coords, connected_atoms, is_overate, average_distance = (
        check_node_overate(node_matrix, unitcell, num_symmetry_operation=len(operations),
                           num_bond_atoms=num_bond_atoms))

    write_node_cif(out_file, node_matrix, node_atom_labels, unitcell, num_nodes=num_nodes)

    # {is_wrong_distance} is used to evaluate whether the N distance used for bonding linkers between nodes is too close.
    # If it is too close, it will not be able to form an imine bond with the linker.
    # is_wrong_node_distance = check_bond_atom_dis(node_bond_atom_coords, unitcell, num_bond_atoms=num_bond_atoms,
    #                                              distance_tolerance=2)

    # Obtain the coordinates of the two node N atoms which are most suitable for the linker.
    proper_dict = (
        is_proper(node_bond_atom_coords, unitcell, num_nodes=num_nodes, num_bond_atoms=num_bond_atoms))
    num_proper_positions = len(proper_dict)

    with open('all_records.txt', 'a') as records:
        records.write(f'{out_file} {str(num_proper_positions)} {str(is_overate)}\n')

    proper_coords_list = [i['proper_coords'] for i in proper_dict.values()]

    if is_overate and num_proper_positions == 2 and not is_same_bond_atom(proper_coords_list[0], proper_coords_list[1]):
        linker_atom_coords = np.loadtxt(linker_file, skiprows=2, usecols=[1, 2, 3])
        with open(linker_file) as f:
            linker_file_contents = f.readlines()[2:]
        linker_atom_labels = [i.split()[0] for i in linker_file_contents]

        deviations = 0

        for i, values in enumerate(proper_dict.values()):
            proper_coords = values['proper_coords']
            linker_ref_coord = np.mean(np.array(proper_coords), axis=0)
            linker_rot_axis = proper_coords[1] - proper_coords[0]

            aligned_linker_coords = align_vectors(linker_atom_coords, proper_coords, 15)

            linker_matrix = get_linker_matrix(aligned_linker_coords, linker_rot_angle, linker_rot_axis,
                                              linker_ref_coord,
                                              unitcell, operations)

            linker_bond_atom_coords = linker_matrix[0][0:2]
            linker_connect_atom_coords = linker_matrix[0][2:4]

            # Calculate the error of two linked nodes.
            _, ave_dis_deviation1, linker_side_error1, node_side_error1 = check_linker_bonds(linker_bond_atom_coords,
                                                                                             node_bond_atom_coords,
                                                                                             unitcell,
                                                                                             linker_connect_atom_coords,
                                                                                             convert_cartesian_to_frac(
                                                                                                 proper_coords[0],
                                                                                                 unitcell))

            _, ave_dis_deviation2, linker_side_error2, node_side_error2 = check_linker_bonds(linker_bond_atom_coords,
                                                                                             node_bond_atom_coords,
                                                                                             unitcell,
                                                                                             linker_connect_atom_coords,
                                                                                             convert_cartesian_to_frac(
                                                                                                 proper_coords[1],
                                                                                                 unitcell))
            write_linker_cif(out_file, linker_matrix, linker_atom_labels,
                             num_linkers=int(num_linkers / num_proper_positions))
            deviations += (ave_dis_deviation1 + ave_dis_deviation2 + linker_side_error1 +
                           linker_side_error2 + node_side_error1 + node_side_error2)
        rewards = 1 - 1 / (deviations + 1)
        check_bond = count_error_bonds(out_file, unitcell)
        r_value = rewards + check_bond
        # print(f'r_value: {round(r_value, 4)}, rewards: {round(10 * rewards, 4)}, error_bonds: {round(check_bond, 4)}'
        #       f' Parameters: {round(node_x_rot, 4), round(node_y_trans, 4), round(node_z_trans, 4), round(linker_rot_angle, 4)}')

        with open('result.txt', 'a') as results:
            results.write(
                f'{r_value} {rewards} {check_bond} {deviations}\n')
    else:
        r_value = 999
    return round(r_value, 2)


def construct_cof(node_file, linker_file, node_ref_coord,
                  workdir, node_x_rot, node_y_trans, node_z_trans,
                  linker_rot_angle,
                  unitcell, operations, num_nodes=4, num_linkers=16, num_bond_atoms=8):
    """
    Construct a COF structure from the given node, linker, and unitcell parameters.

    Parameters:
    ----------
        node_file: str
            XYZ coordinate file for the node molecule.
        linker_file: str
            XYZ coordinate file for the linker molecule.
        workdir: str
            Working directory for constructing the COF structures.
        unitcell: tuple or list
            Unitcell parameters for the COF structure.
        *args:
            Parameters to be optimized. For COF-300,
            there will be six rotation variables that need to be optimized.

    Returns:
        fitness_value: float
            The value of the fitness function.

    """
    out_file = '_'.join(
        map(str, [round(node_x_rot, 4), round(node_y_trans, 4), round(node_z_trans, 4),
                  round(linker_rot_angle, 4)])) + '.cif'

    fitness_value = _construct_cof(node_file, linker_file, out_file, workdir, node_ref_coord,
                                   node_x_rot, node_y_trans, node_z_trans,
                                   linker_rot_angle, unitcell, operations,
                                   num_nodes=num_nodes, num_linkers=num_linkers, num_bond_atoms=num_bond_atoms)
    return fitness_value

