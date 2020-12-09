/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Juraj Holub <xholub40@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    December 2020
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
            //std::cout << ((X_MASK & i)>>2) << ((Y_MASK & i)>>1) << (Z_MASK & i) << "\n";
            pos.x = x + ((X_MASK & i)>>2) * halfCubeGrid;
            pos.y = y + ((Y_MASK & i)>>1) * halfCubeGrid;
            pos.z = z + (Z_MASK & i) * halfCubeGrid;

            #pragma omp task shared(totalTriangles)
            {
                unsigned taskRes = traverseOctet(pos, field, halfCubeGrid);
                #pragma omp atomic
                totalTriangles += taskRes;
            }
        }
    }
    else // lowest level
    {
        unsigned buildCubeRes = buildCube(pos, field);
        #pragma omp atomic
        totalTriangles += buildCubeRes;
    }

    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;

    Vec3_t<float> cubeOffset(0, 0, 0);

    #pragma omp parallel
    {
        #pragma omp master
        {
            totalTriangles = traverseOctet(cubeOffset, field, mGridSize);
        }
    }

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical(TREE_BUILDER_LOCK)
    {
        mTriangles.push_back(triangle);
    }
}
