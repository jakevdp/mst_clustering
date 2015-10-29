test:
	nosetests mst_clustering

doctest:
	nosetests --with-doctest mst_clustering

test-coverage:
	nosetests --with-coverage --cover-package=mst_clustering

test-coverage-html:
	nosetests --with-coverage --cover-html --cover-package=mst_clustering
