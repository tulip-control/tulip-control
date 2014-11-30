import polytope
import numpy
from poly2str import matrix2str, polytope2str


def pppexport(ppp, filename="pppdata"):
    """Export a proposition preserving partition to a C file

    Restrictions:
        - Each region is assumed to be a convex polytope.
        - If not, then only the first polytope inside the region is
          exported. Hence, the MPC may be infeasible

    @param ppp: tulip.abstract.prop2partition.PropPreservingPartition
    @param filename: a string of the export filename
    @rtype: None
    """
    numregions = len(ppp.regions)
    # generate the .c file
    f = open("sources/"+filename+".c", "w")

    f.write('#include "'+filename+'.h"\n\n')

    f.write("idxint nRegion = "+str(numregions)+";\n")
    f.write("Polytope* regions["+str(numregions)+"];\n")

    for index in range(0, numregions):
        region = ppp.regions[index]
        p = region.list_poly[0]
        s = polytope2str(p, 'p'+str(index))
        f.write("/****Polytope "+str(index) + " ****/\n")
        f.write(s+"\n")

    f.write("void init_region()\n{\n")
    for index in range(0, numregions):
        f.write("   regions[" + str(index) + "]=create_poly(p" + str(index) +
                "k,p" + str(index) + "l,p" + str(index) + "A,p" + str(index) +
                "b,p" + str(index) + "center);\n")

    f.write("}\n\n")

    f.write("""\
void free_region()\n{
    int i;
    for (i=0;i<nRegion;i++)
    {
        free(regions[i]);
    }
}""")

    f.close()
