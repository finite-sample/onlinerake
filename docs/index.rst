onlinerake: Streaming Survey Raking
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   notebooks
   diagnostics
   api_reference
   software-audit

Raking adjusts weights so that weighted shares of groups match known
population shares. The standard algorithm, iterative proportional fitting
(IPF), needs every row seen so far. ``StreamingRaker`` rakes a stream without
storing it: its state is ``p + 5`` running totals for ``p`` margins, each
arriving row gets its weight in closed form, and the cost per row does not grow
with the stream.

Which Tool
----------

- **Rows can be stored** and the stream is survey-sized: refit ``BatchIPF``
  periodically. It is exact at each refit.
- **Rows cannot or should not be stored**, memory is fixed, or the stream is
  long: ``StreamingRaker``.
- ``fit_penalized_raking`` minimizes the streaming objective over all stored
  rows at once.
- ``OnlineRakingSGD`` and ``OnlineRakingMWU`` store every row and re-optimize
  all weights at each arrival.

Quick Start
-----------

.. code-block:: python

   from onlinerake import StreamingRaker, Targets

   raker = StreamingRaker(Targets(female=0.5), outcomes=["spend"])
   for female, spend in [(1, 10.0), (1, 12.0), (1, 11.0), (0, 20.0)] * 50:
       raker.partial_fit({"female": female}, outcomes={"spend": spend})

   print(round(raker.margins["female"], 2))  # 0.52
   print(round(raker.estimate("spend"), 1))  # 15.3

The manuscript in ``ms/onlinerake.pdf`` derives the update and evaluates it
against full-data and periodic IPF.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`