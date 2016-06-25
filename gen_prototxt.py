''' genrate my solver'''
from utils import CaffeSolver
import cfgs
import os

# gen solver prototxt
solver=CaffeSolver(debug=cfgs.debug)
solver.sp=cfgs.sp.copy()
solver.write(os.path.join(cfgs.pt_folder,cfgs.solver_pt))
