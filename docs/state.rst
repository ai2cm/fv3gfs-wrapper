State
=====

Getting/setting state
---------------------

In addition to just running the model the same as in Fortran, the Python wrapper allows you to interact
with the model state while it is still running.
This is done through :py:func:`fv3gfs.wrapper.get_state` and :py:func:`fv3gfs.wrapper.set_state`.
The getters will return Quantity objects which nominally have units information, but these aren't guaranteed
to be accurate and in some cases are indicated as missing.

An example which gets and sets the model state to damp moisture is present in the examples folder of this repo.

You should also keep in mind that the arrays used by :py:func:`fv3gfs.wrapper.get_state` and :py:func:`fv3gfs.wrapper.set_state`
are domain-decomposed fields (as opposed to global fields).

Quantity
--------

Data in ``fv3gfs-wrapper`` is managed using a container type called :py:class:`pace.util.Quantity`.
This stores metadata such as dimensions and units (in ``quantity.dims`` and ``quantity.units``),
and manages the "computational domain" of the data. See the `pace-util`_ documentation for
full details about this object.

.. _`pace-util`: https://pace-util.readthedocs.io/en/latest/
