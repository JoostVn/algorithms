"""
Contains functions not directly used in testing geometry, but used by functions
that do. Thes module is split into the following sub-modules:
    
    - getinfo: Contains functions that are used to fetch information or
    restrucutre data from geometries, such as 
        
        
    -tests: Contains preliminary tests that are cheaper than most geometry 
    tests, such as a domain overlap check.



"""

from . import getinfo
from . import test
from . import transform