---

# COFBuilder
Build possible COF structures from the given unitcell parameters, node and linker.

---
# 1.Installation

```bash
git clone https://github.com/Chemwzd/COFBuilder.git
cd COFBuilder
pip install -r requirements.txt
```

# 2.How to Build a COF Structure
## Terminal
Note: Modify the paths and PSO parameters in `RunConstruction.py` (the paper uses `popsize=1000`, `iter=100`, which will take about 24 hours). For testing purposes, the `popsize` can be set to 10-100.
```bash
cd ./COFBuilder/examples
python RunConstruction.py
```
All structures generated during the PSO optimization will be saved in `basedir`, and the best structure from each run will be saved in `best_struc_dir`.

# 3.How to Plot the Fitness Values
## Terminal
```bash
cd ./COFBuilder/examples
python PlotPSOFitness.py
```
The fitness values from `n` independent runs will be saved to an image.
# Authors
## Maintainer

 - Zidi Wang (wangzd@shanghaitech.edu.cn)
## Code Contributors
 - Xiangyu Zhang (zhangxy6@alumni.shanghaitech.edu.cn): Contributed the source code for building sturcture of COF-300, https://github.com/zhangxiangyu6/COF
 - Zidi Wang (wangzd@shanghaitech.edu.cn): Rewrite most of the functions in the code and enable the construction of new COF structures.

