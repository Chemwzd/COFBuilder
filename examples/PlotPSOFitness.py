import matplotlib.pyplot as plt
import os

basedir = '/path/to/your/dir'

running_times = 10

plt.figure(figsize=(8, 6))

for i in range(running_times):
    os.chdir(os.path.join(basedir, f'{i}'))
    results_txt = 'inf.txt'

    if os.path.exists(results_txt):
        with open(results_txt, 'r') as f:
            results_content = f.readlines()
        results = [float(i[1:-2]) for i in results_content]

        plt.plot(range(len(results)), results, label=f'Cycle {i}')

    else:
        print(f'{i} is not finished.')

plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.legend()
plt.savefig('/path/to/save/fig', dpi=400)
plt.show()
