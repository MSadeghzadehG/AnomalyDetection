# -AnomalyDetection
...
## variational autoencoder
a variational autoencoder sees all input data and map (or represent) them on an m-dimension space; just like the autoencoders. but the representation is on an input distribution(for example Gaussian distribution).
the idea is that: show data to a variational autoencoder and learn that by Normal distribution, and then find point anomalies by one the definitions of point anomaly(which data is more far than 3 little-sigma the center of distribution are point anomaly).
but the problem is that: the learning process just certify that the all points in the representaion belongs to input distribution, but it's not necessary to certify all properties of the distribution; as the point anomaly measurment.
so I couldn't use variational autoencoder directly, but I found a newer type of them,which named disentengled variational autoencoder. 

## disentangled variational autoencoder
