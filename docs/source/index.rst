.. cycad documentation master file, created by
   sphinx-quickstart on Fri Aug 12 13:08:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cycad overview
=================================
CYCling Autocorrelation Dataviz

Cycad is a library for visualizing the autocorrelation from *in situ* battery cycling meaurements such as X-ray diffraction (XRD) or X-ray absorption spectroscopy. In this documentation, XRD data will be assumed unless otherwise specified.

The motivation for cycad to clearly show the relationship between battery cycling data at different stages. It does this by calculating the correlation between each pair of measurements in an *in situ* dataset and plotting it as a heatmap.

Reading cycad plots
===================

.. image:: _static/example3.png
   :align: center
   
* Plots (1) and (2) show heatmaps of *in situ* XRD data in two different orientations.
* Plots (3) and (4) show plots of the cycling voltage in two different orientations.
* The central plot (5) shows two plots. The underlying plot in grayscale show the Pearson correlation between each pair of measurements in the dataset, with darker regions showing greater correlation (similarity between patterns). The overlaid plot shows a translucent red region that corresponds to pairs of measurements for which the voltage is within a certain range.
* Interpretation: for a material with stable cycling, similarity between patters should correspond to similar voltages, so the red region should overlay the darker regions below.


Installation
============
Install the library using pip (preferably in a new virtual environment):

.. code-block:: console

   $ conda create -n cycad
   $ conda activate cycad
   (cycad) $ pip install cycad

Tutorial
============
For each *in situ* experiment, create a ``cycad`` object.

.. code-block:: python

   >>> import cycad
   >>> model = cycad.cycad()

Use the ``read_folder()`` method to read a list of data files from a folder and specify the file type to be read. Use ``read_data()`` to read all data files from the folder into the cycad.df dataframe.

.. code-block:: python

   >>> model.read_folder('data/', 'csv')
   >>> model.read_data()

If electrochemical cycling data is available, use ``read_echem_mpt(filename)`` to read echem data from a mpt file.

.. code-block:: python

   >>> model.read_echem_mpt('echem.mpt')

Use ``autocorrelate()`` and ``generate_distance_matrix()`` to calculate the autocorrelation matrices for the *in situ* XRD and cycling voltage data respectively.

.. code-block:: python

   >>> model.autocorrelate()
   >>> model.generate_distance_matrix()

Finally use ``plot()`` to plot the autocorrelation matrices.

.. code-block:: python

   >>> model.plot(echem=True)



Advanced usage
===============





.. toctree::
   :maxdepth: 2
   :caption: Contents:







Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
