__author__ = 'fabian'
import sys
sys.path.append('../')
import utils
import filter_feature_selection
import numpy as np
import feast
import mutual_information

X, Y = utils.load_digits()
X = mutual_information.normalize_data_for_MI(X)
X = X.astype("float64")
selector = filter_feature_selection.FilterFeatureSelection(X, Y, "CIFE")

num_feat = 10

print "CIFE"
print "ours\t", selector.run_selection(num_feat)
print "feast\t", np.array(feast.CIFE(X, Y, num_feat)).astype("int")
print "\n"

print "JMI"
selector.change_method("JMI")
print "ours\t", selector.run_selection(num_feat)
print "feast\t", np.array(feast.JMI(X, Y, num_feat)).astype("int")
print "\n"

print "ICAP"
selector.change_method("ICAP")
print "ours\t", selector.run_selection(num_feat)
print "feast\t", np.array(feast.ICAP(X, Y, num_feat)).astype("int")
print "\n"

print "CMIM"
selector.change_method("CMIM")
print "ours\t", selector.run_selection(num_feat)
print "feast\t", np.array(feast.CMIM(X, Y, num_feat)).astype("int")
print "\n"