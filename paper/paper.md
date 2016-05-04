---
title: 'mst_clustering: Clustering via Euclidean Minimum Spanning Trees'
tags:
  - machine learning
  - clustering
authors:
 - name: Jake VanderPlas
   orcid: 0000-0002-9623-3401
   affiliation: University of Washington eScience Institute
date: 04 May 2016
bibliography: paper.bib
---

# Summary

This package contains a Python implementation of a clustering algorithm based
on an efficiently-constructed approximate Euclidean minimum spanning tree
(described in [@ivezic2014]). The method produces a Hierarchical clustering of
input data, and is quite similar to single-linkage Agglomerative clustering.
The advantage of this implementation is the ability to find significant clusters
even in the presence of background noise.

The code makes use of tools within SciPy [@scipy] and scikit-learn [@scikit-learn],
and is designed for compatibility with the scikit-learn API [@scikit-learn-api].

-![Simple Clustering Example](mst_example.png)

# References
