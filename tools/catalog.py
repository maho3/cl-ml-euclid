"""
This is a module to define the Cluster and Catalog classes. These classes make
it easier and more efficient to store mock observation data.
"""

# ~~~~~ IMPORTS ~~~~~~
import pickle
import numpy as np
from numpy import ndarray
import pandas as pd


# ~~~~~ CLASS DEFINITIONS ~~~~~~
class Catalog(object):
    """A container to store mock observation catalogs. The container is designed
    to store catalog metadata, properties of clusters, and their galaxies in a
    storage-efficient manner.

    Attributes
    ----------
    par : dictionary
        Dictionary-like object containing metadata about the mock observation
        catalog (e.g. name, redshift, aperture, vcut, etc.)
    prop : pandas Dataframe
        Dataframe containing properites of host clusters. Columns contain
        cluster properties, rows correspond to individual clusters in the
        catalog
    gal :  list of numpy structured arrays
        List of numpy arrays with datafields corresponding to galaxy
        observables (e.g. vlos, Rproj, stellar mass, etc.). This is separate
        from prop because the length of each array (number of galaxies in
        clusters) is variable. Entries in the list match by index with the
        clusters in prop. len(gal)==len(prop)
    """

    def __init__(self, par=None, prop=None, gal=None):
        self.par = par
        self.prop = prop
        self.gal = gal

    def __getitem__(self, key):
        """Returns cluster or series of clusters as a new Catalog object """
        if (isinstance(key, int) | isinstance(key, list) |
                isinstance(key, ndarray)):
            return Catalog(par=self.par,
                           prop=self.prop.iloc[key].reset_index(drop=True),
                           gal=self.gal[key])
        else:
            raise Exception("Unknown key type: " + str(type(key)))

    def __add__(self, catalog):
        """Concatenates two Catalog objects into a new Catalog"""
        return Catalog(par=self.par,
                       prop=self.prop.append(catalog.prop).reset_index(
                        drop=True),
                       gal=np.append(self.gal, catalog.gal))

    def __len__(self):
        """Returns number of clusters in Catalog object"""
        return len(self.prop)

    def save(self, filename, protocol=4):
        """Saves Cluster object as a pickle file"""
        print('Pickle dumping to %s with protocol %s' % (filename, protocol))
        with open(filename, 'wb') as out_file:
            pickle.dump(self, out_file, protocol=protocol)

    def load(self, filename):
        """Loads Cluster object from a pickle file"""
        print('Loading catalog from: %s' % filename)
        with open(filename, 'rb') as in_file:
            new_cat = pickle.load(in_file)

        self.par = new_cat.par
        self.prop = new_cat.prop
        self.gal = new_cat.gal

        return self

    def save_npy(self, filename):
        """Converts prop and gal attributes to a single structured numpy array
        and saves this to disk. Galaxy attributes are stored as vectors of
        fixed length, corresponding to the maximum number of galaxies for a
        single cluster in the catalog. Clusters with less galaxies than this
        are padded with 0's to fill the fixed length property vectors. par
        metadata attributes are not stored. This is useful for loading Catalogs
        in Python 2, where pickle loading may run into issues."""

        print('Saving as npy: ' + filename)
        max_ngal = self.prop['Ngal'].max()

        dtype = [(x, 'float64') for x in self.prop.columns.values]
        dtype += [('gal_'+x[0], x[1], int(max_ngal))
                  for x in self.gal[0].dtype.descr]

        out = np.zeros(shape=(len(self),), dtype=dtype)

        print('Loading prop')
        for x in self.prop.columns.values:
            out[x] = self.prop[x].values

        print('Loading gal')
        for i in range(len(self)):
            for f in self.gal[0].dtype.names:
                out[i]['gal_'+f][:int(out[i]['Ngal'])] = self.gal[i][f]

        print('Parameters not transferred:')
        print(self.par)

        np.save(filename, out)

    def load_npy(self, filename, par={}):
        """Loads Catalog object from a npy file via the format described in
        save_npy"""

        print('Loading as npy: ' + filename)
        x = np.load(filename)
        gal_descr = [i for i in x.dtype.descr if i[0][:4] == 'gal_']
        dtype = np.dtype([(a[4:], b) for a, b, _ in gal_descr])
        gals = [np.zeros(shape=(int(x['Ngal'][i]),), dtype=dtype)
                for i in range(len(x))]
        for i in range(len(x)):
            for field in dtype.names:
                gals[i][field] = x[i]['gal_' + field][:int(x['Ngal'][i])]

        self.par = par
        self.prop = pd.DataFrame(x[[i for i in x.dtype.names
                                    if i[:4] != 'gal_']])
        self.gal = np.array(gals)

        return self
