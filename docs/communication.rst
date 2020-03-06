=============
Communication
=============

As mentioned when discussing State_, each process or "rank" in fv3gfs is responsible
for a subset of the cubed sphere grid. In order to operate, the model needs to know
how to partition that cubed sphere into parts for each rank, and to be able to
communicate data between those ranks.

Partitioning is managed by so-called "Partitioner" objects. The
:py:class:`fv3gfs.CubedSpherePartitoner` manages the entire cubed sphere, while the
:py:class:`fv3gfs.TilePartitioner` manages one of the six faces of the cube, or a
region on one of those faces. For communication, we similarly have
:py:class:`fv3gfs.CubedSphereCommunicator` and :py:class:`fv3gfs.TileCommunicator`.
