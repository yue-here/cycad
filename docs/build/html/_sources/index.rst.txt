.. cycad documentation master file, created by
   sphinx-quickstart on Fri Aug 12 13:08:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cycad overview
=================================
CYCling Autocorrelation Dataviz

Cycad is a library for visualizing the autocorrelation from *in situ* battery cycling meaurements such as X-ray diffraction (XRD) or X-ray absorption spectroscopy. In this documentation, XRD data will be assumed unless otherwise specified.

The motivation for cycad to clearly show the relationship between battery cycling data at different stages. It does this by calculating the correlation between each pair of measurements in an *in situ* dataset and plotting it as a heatmap.

:doc:`installation`
   How to install cycad

:doc:`tutorial`
   Learn how to use cycad

Reading cycad plots
===================

.. image:: _static/example3.png
   :align: center
   
* Subplots (1) and (2) show heatmaps of *in situ* XRD data in two different orientations.
* Subplots (3) and (4) show plots of the cycling voltage in two different orientations.
* The subplot (5) shows two datasets. The underlying plot in grayscale show the Pearson correlation between each pair of measurements in the dataset, with darker regions showing greater correlation (similarity between patterns). The overlaid plot shows a translucent red region that corresponds to pairs of measurements for which the voltage is within a certain range.
* Interpretation: for a material with stable cycling, similarity between patters should correspond to similar voltages, so the red region should overlay the darker regions below.


Contents
========

.. toctree::
   :maxdepth: 2

   installation
   tutorial
   Advanced usage<advanced>







Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

