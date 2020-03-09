.. _state-overview:
=====
State
=====

Getting/setting state
---------------------

In addition to just running the model the same as in Fortran, the Python wrapper allows you to interact
with the model state while it is still running.
This is done through :py:func:`fv3gfs.get_state` and :py:func:`fv3gfs.set_state`.
The getters will return Quantity objects which nominally have units information, but these aren't guaranteed
to be accurate and in some cases are indicated as missing.

An example which gets and sets the model state to damp moisture is present in the examples folder of this repo.

You should also keep in mind that the arrays used by :py:func:`fv3gfs.get_state` and :py:func:`fv3gfs.set_state`
are domain-decomposed fields (as opposed to global fields).

Quantity
--------

Data in ``fv3gfs-python`` is managed using a container type called :py:class:`fv3gfs.Quantity`.
This stores metadata such as dimensions and units (in ``quantity.dims`` and ``quantity.units``),
and manages the "computational domain" of the data.

When running a model on multiple
processors ("ranks"), each process is responsible for a subset of the domain, called its
"computational domain". Arrays may contain additional data in a "halo" of "ghost cells",
which a different rank is responsible for updating, but need to be used as inputs for
the local rank. Depending on optimization choices, it may also make sense to include
"filler" data which serves only to align the computational domain into blocks within
memory.

If all of that sounded confusing, we agree! That's why :py:class:`fv3gfs.Quantity`
abstracts away as much of this information as possible. If you perform indexing on the
``view`` attribute of quantity, the index will be applied within the computational
domain::

    quantity.view[:] = 0.  # set all data this rank is responsible for to 0
    quantity.view[1:-1, :] = 1.0  # set data not on the first dimension edge to 1
    array = quantity.view[:]  # gives an array accessing just the compute domain
    new_array = np.ascontiguousarray(quantity.view[:])  # gives a *copy* of the compute domain

If you actually do want to access data in ghost cells, instead of ``.view`` you should
access ``.data``, which is the underlying ``ndarray``-like object used by the ``Quantity``::

    quantity.data[:] = 0.  # set all data this rank has, including ghost cells, to zero
    quantity.data[quantity.origin[0]-3:quantity.origin[0]] == 1.  # set the left three ghost cells to 1
    array = quantity.data[quantity.origin[0]:quantity.origin[0]+quantity.extent[0]]  # same as quantity.view[:] for a 1D quantity

``data`` may be a numpy array or a cupy array. Both provide the same interface and
can be used identically. If you would like to use the appropriate "numpy" package
to manipulate your data, you can use ``quantity.np``. For example, the following
will give you the mean of your array, regardless of whether the data is on CPU or GPU,
and regardless of whether halo values are present::

    quantity.np.mean(quantity.view[:])
