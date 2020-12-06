/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::traverseOctet(Vec3_t<float> &pos, const ParametricScalarField &field, size_t cubeGrid) {
    unsigned totalTriangles = 0;
    const unsigned X_MASK = 4;// 100
    const unsigned Y_MASK = 2;// 010
    const unsigned Z_MASK = 1;// 001
    const float x = pos.x, y = pos.y, z = pos.z;

    if (cubeGrid > 1) // cut-off
    {
        unsigned halfCubeGrid = cubeGrid/2;
        float edgeLen = cubeGrid*mGridResolution;
        Vec3_t<float> midOfBlock((x+halfCubeGrid)*mGridResolution,
                                (y+halfCubeGrid)*mGridResolution,
                                (z+halfCubeGrid)*mGridResolution);
        if (evaluateFieldAt(midOfBlock, field) > (field.getIsoLevel()+sqrt(3)/2.0*edgeLen))
        {
            return 0;
        }
        for (size_t i=0; i < 8; i++)
        {
            pos.x = x + (X_MASK & i) * halfCubeGrid;
            pos.y = y + (Y_MASK & i) * halfCubeGrid;
            pos.z = z + (Z_MASK & i) * halfCubeGrid;
            totalTriangles += traverseOctet(pos, field, halfCubeGrid);
        }
    }
    else // lowest level
    {
        totalTriangles += buildCube(pos, field);
    }

    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    unsigned totalTriangles = 0;

    Vec3_t<float> cubeOffset(0, 0, 0);
    #pragma omp parallel schedule(guided, 32)
    totalTriangles = traverseOctet(cubeOffset, field, mGridSize);

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    mTriangles.push_back(triangle);
}
