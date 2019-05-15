# -AnomalyDetection
...
## autoencoder
when I was searching about methods used in unsupervised learning, I've seen many methods that categorized unsupervised, just because they assume a very very basic hypothesis. maybe these hypotheses are acceptable in most fields and projects, but I couldn't accept them easily in the network connections data. (giving some example)
However, the autoencoders learning have no hypothesis. in a better word, the hypothesis is the architecture of our neural network. so I was encouraged to use autoencoders for this project. the anomaly detection using autoencoder can be defined like this: we learn our model using whole data, then again show each of the data to the model. the idea is that the loss of reconstruction shows how much a data is further from the others. so we can sort the reconstruction losses and consider data in the top of the list as anomalies. (use a figure)
this method seems good, but have some problems. the reconstruction loss is too dependent on learning data. so it is not an accurate and suitable way to use for which data is not in the training set. 

## variational autoencoder
a variational autoencoder sees all input data and map (or represent) them on an m-dimension space; just like the autoencoders. but the representation is on an input distribution(for example Gaussian distribution).
the idea is that: show data to a variational autoencoder and learn that by Normal distribution, and then find point anomalies by one the definitions of point anomaly(which data is more far than 3 little-sigma the center of distribution are point anomaly). (use a complete of idea figure for better understanding)
but the problem is that: the learning process just certify that the all points in the representaion belongs to input distribution, but it's not necessary to certify all properties of the distribution; as the point anomaly measurment.
so I couldn't use variational autoencoder directly, but I found a newer type of them,which named disentengled variational autoencoder. (show a Counterexample)

## disentangled variational autoencoder
