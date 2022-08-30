Tutorial
============
See the tutorial jupyter notebook for a worked example.

**Minimum needs**: a folder containing files representing X-ray diffraction patterns from a *in situ* cycling experiment.

**Optional extra**: an electrochemical cycling data file. Currently biologic .mpt files are supported.

.. note::
    Aside from XRD patterns, other 1-D data such as XAS (IR, Raman...) spectra can also be used. The correlation function is agnostic to measurement type.

    Aside from battery data, other cycling experiments such as thermal cycling can also be used.

For each *in situ* experiment, create a ``cycad`` object.

.. code-block:: python

   from cycad import cycad
   run = cycad()

Use the :meth:`cycad.cycad.read_folder()` method to read a list of data files from a folder and specify the file type to be read. Use :meth:`cycad.cycad.read_data()` to read all data files from the folder into the cycad.df dataframe.

.. code-block:: python

   run.read_folder('data/', 'csv')

If electrochemical cycling data is available, use :meth:`cycad.cycad.read_echem_mpt()` to read echem data from a mpt file.

.. code-block:: python

   run.read_echem_mpt('echem.mpt')

Use :meth:`cycad.cycad.autocorrelate()` and :meth:`cycad.cycad.autocorrelate_ec()` to calculate the autocorrelation matrices for the *in situ* XRD and cycling voltage data respectively.

.. code-block:: python

   run.autocorrelate_ec()
   run.autocorrelate()

Finally use :meth:`cycad.cycad.plot()` to plot the autocorrelation matrices. If you have loaded echem data, set the ``echem`` parameter to ``True`` to plot the echem autocorrelation matrix.

.. code-block:: python

   run.plot(echem=True)