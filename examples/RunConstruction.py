import sys 
sys.path.append("..") 
from COFBuilder import construct_cof
from sko.PSO import PSO
import os
import random
import numpy as np
import shutil
from multiprocessing import Pool

def run_test(running_time):
    node_file = '/path/to/your/node.xyz'
    linker_file = '/path/to/your/linker.xyz'
    
    workdir = os.path.join(basedir, str(running_time))
    os.makedirs(workdir, exist_ok=True)

    unitcell = [25.4720, 37.1170, 35.5780, 90.00, 90.00, 90.00]
    operations = [' x, y, z', '-x+1/2, y+1/2, z+1/2', 'x+1/2, y+1, -z+1/2', 'x, y+1/2, -z',
                  '-x+1/2, y+1, -z+1/2', ' x+1/2, y+1/2, z+1/2', '-x, y, z', '-x, y+1/2, -z',
                  'x, -y, -z', '-x, -y, -z', 'x, -y+1/2, z', '-x+1/2, -y+1, z+1/2', 'x+1/2, -y+1/2, -z+1/2',
                  '-x+1/2, -y+1/2, -z+1/2', 'x+1/2, -y+1, z+1/2', '-x, -y+1/2, z']
    num_nodes = 4
    num_linkers = 16
    num_bond_atoms = 8
    node_ref_coord = np.array([0, random.random(), random.random()])
    print(f'node_ref_coord: {node_ref_coord}')

    def get_fitness_value(node_x_rot, node_y_trans, node_z_trans, linker_rot_angle):
        fitness_value = construct_cof(node_file, linker_file, node_ref_coord,
                                      workdir,
                                      node_x_rot, node_y_trans, node_z_trans, linker_rot_angle,
                                      unitcell, operations, num_nodes, num_linkers, num_bond_atoms)
        return fitness_value

    pso = PSO(func=get_fitness_value, n_dim=4, pop=1000, max_iter=100,
              lb=[0, 0, 0, 0], ub=[360, 1, 1, 360], verbose=True)

    fitness = pso.run()

    with open('inf.txt', 'a') as inf:
        for i in pso.gbest_y_hist:
            inf.write(str(i) + '\n')

    print("The Best Parameters：", pso.gbest_x)
    float_values = np.array(pso.gbest_x).astype(float)
    best_cif_name = '_'.join(map(str, [round(i, 4) for i in float_values])) + '.cif'

    with open('best_parameter.txt', 'a') as bp:
        bp.write(best_cif_name)
    print(best_cif_name)

    shutil.copy(best_cif_name, os.path.join(best_struc_dir, f'{running_time}_{best_cif_name}'))

if __name__ == '__main__':
    random.seed(1000)

    running_times = 10

    basedir = f'/path/to/save/PSO/generated/structures'
    best_struc_dir = '/path/to/save/best/structure'

    os.makedirs(basedir, exist_ok=True)

    with Pool(processes=10) as pool:
        pool.map(run_test, range(running_times))
