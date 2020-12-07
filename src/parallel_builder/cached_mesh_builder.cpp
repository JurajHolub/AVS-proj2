/**
 * @file    cached_mesh_builder.cpp
 *
 * @author  Juraj Holub <xholub40@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using pre-computed field
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "cached_mesh_builder.h"

CachedMeshBuilder::CachedMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Cached")
{

}

unsigned CachedMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    dist = new float[(mGridSize+1)*(mGridSize+1)*(mGridSize+1)];

    #pragma omp parallel for collapse(3)
    for (int x = 0; x < mGridSize+1; ++x) {
        for (int y = 0; y < mGridSize+1; ++y) {
            for (int z = 0; z < mGridSize+1; ++z) {
                float value = std::numeric_limits<float>::max();
                for(unsigned j = 0; j < count; ++j)
                {
                    float distanceSquared  = (x*mGridResolution - pPoints[j].x) * (x*mGridResolution - pPoints[j].x);
                    distanceSquared       += (y*mGridResolution - pPoints[j].y) * (y*mGridResolution - pPoints[j].y);
                    distanceSquared       += (z*mGridResolution - pPoints[j].z) * (z*mGridResolution - pPoints[j].z);
                    value = std::min(value, distanceSquared);
                }
                dist[x*(mGridSize+1)*(mGridSize+1)+y*(mGridSize+1)+z] = sqrt(value);
            }
        }
    }

    size_t totalCubesCount = mGridSize*mGridSize*mGridSize;
    unsigned totalTriangles = 0;

    #pragma omp parallel for schedule(static, 32) reduction(+:totalTriangles)
    for(size_t i = 0; i < totalCubesCount; ++i)
    {
        Vec3_t<float> cubeOffset( i % mGridSize,
                                  (i / mGridSize) % mGridSize,
                                  i / (mGridSize*mGridSize));
        totalTriangles += buildCube(cubeOffset, field);
    }

    delete dist;

    return totalTriangles;
}

float CachedMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    unsigned x = pos.x/mGridResolution+0.5;
    unsigned y = pos.y/mGridResolution+0.5;
    unsigned z = pos.z/mGridResolution+0.5;

    return dist[x*(mGridSize+1)*(mGridSize+1)+y*(mGridSize+1)+z];
}

void CachedMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical(CACHED_BUILDER_LOCK)
    {
        mTriangles.push_back(triangle);
    }
}
