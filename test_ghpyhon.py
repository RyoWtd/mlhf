"""Provides a scripting component.
    Inputs:
        x: The x script variable
        y: The y script variable
    Output:
        a: The a output variable"""

__author__ = "RyoWATADA"
__version__ = "2022.02.17"

import rhinoscriptsyntax as rs
import ghpythonlib.treehelpers as th
import Rhino.Geometry as rg
import math

import System
from Grasshopper.Kernel.Data import GH_Path
from Grasshopper import DataTree

ids = [i for i in range(5)]
points = [rg.Point3d(i,0,0) for i in range(4)]
lines = [rg.Line(rg.Point3d(i,0,0),rg.Point3d(i+1,1,0)) for i in range(3)]
lines2 = [rs.AddLine(rg.Point3d(i,0,0),rg.Point3d(i+1,1,0)) for i in range(3)]

data = {}

data["ids"] = ids
data["points"] = points
data["lines"] = lines
data["lines2"] = lines2

class result:
    pass

result.data = data

print(result)