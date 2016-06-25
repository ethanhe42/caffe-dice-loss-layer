''' genrate my solver'''
from utils import CaffeSolver
import cfgs
import os

# gen solver prototxt
solver=CaffeSolver(debug=cfgs.debug)
solver.sp=cfgs.sp.copy()
solver.write(cfgs.solver_pt)
