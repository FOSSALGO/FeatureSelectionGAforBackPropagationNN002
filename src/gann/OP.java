package gann;

public class OP {

    public static double[] summation(double[] X, double[][] W) {
        double[] net = null;
        if (X != null && W != null && X.length == W.length) {
            net = new double[W[0].length];
            for (int j = 0; j < net.length; j++) {
                double sum = 0;
                for (int i = 0; i < X.length; i++) {
                    sum += (X[i] * W[i][j]);
                }
                net[j] = sum;
            }
        }
        return net;
    }

    public static double[] activation(double[] net, ActivationFunction operation) {
        double[] fnet = null;
        if (net != null) {
            if (operation == ActivationFunction.SIGMOID_BINER) {
                //aktivasi menggunakan SIGMOID_BINER
                fnet = new double[net.length];
                for (int i = 0; i < fnet.length; i++) {
                    fnet[i] = 1.0 / (1.0 + Math.exp(-net[i]));
                }
            } else if (operation == ActivationFunction.SIGMOID_BIPOLAR) {
                //aktivasi menggunakan SIGMOID_BIPOLAR
                fnet = new double[net.length];
                for (int i = 0; i < fnet.length; i++) {
                    fnet[i] = (1.0 - Math.exp(-net[i])) / (1.0 + Math.exp(-net[i]));
                }
            } else if (operation == ActivationFunction.TANH) {
                //aktivasi menggunakan SIGMOID_TANH
                fnet = new double[net.length];
                for (int i = 0; i < fnet.length; i++) {
                    fnet[i] = (Math.exp(net[i]) - Math.exp(-net[i])) / (Math.exp(net[i]) + Math.exp(-net[i]));
                    //fnet[i] = (2.0 / (1+Math.exp(-net[i])))-1;
                }
            } else if (operation == ActivationFunction.SOFTMAX) {
                //aktivasi menggunakan SOFTMAX
                fnet = new double[net.length];
                double[] exp_net = new double[net.length];
                double sum = 0;
                for (int i = 0; i < fnet.length; i++) {
                    exp_net[i] = Math.exp(net[i]);
                    sum += exp_net[i];
                }
                for (int i = 0; i < fnet.length; i++) {
                    fnet[i] = exp_net[i] / sum;
                }
            }
        }
        return fnet;
    }

    public static double[] hitungFaktorError(double[] error, double[] fnet, ActivationFunction operation) {
        double[] faktorError = null;
        if (error != null && fnet != null) {
            if (operation == ActivationFunction.SIGMOID_BINER) {
                //hitung faktor error menggunakan SIGMOID_BINER
                faktorError = new double[fnet.length];
                for (int k = 0; k < faktorError.length; k++) {
                    faktorError[k] = error[k] * fnet[k] * (1 - fnet[k]);
                }
            } else if (operation == ActivationFunction.SIGMOID_BIPOLAR) {
                //hitung faktor error menggunakan SIGMOID_BIPOLAR
                faktorError = new double[fnet.length];
                for (int k = 0; k < faktorError.length; k++) {
                    faktorError[k] = error[k] * (1.0 / 2.0) * (1 + fnet[k]) * (1 - fnet[k]);
                }
            } else if (operation == ActivationFunction.TANH) {
                //hitung faktor error menggunakan SIGMOID_TANH
                faktorError = new double[fnet.length];
                for (int k = 0; k < faktorError.length; k++) {
                    faktorError[k] = error[k] * (1.0 / 2.0) * (1 + fnet[k]) * (1 - fnet[k]);
                }
            } else if (operation == ActivationFunction.SOFTMAX) {
                //aktivasi menggunakan SOFTMAX
                faktorError = new double[fnet.length];

            }
        }
        return faktorError;
    }

    public static double[][] hitungDeltaWeight(double alpha, double[] faktorError, double[] X) {
        double[][] deltaWeight = null;
        if (faktorError != null && X != null) {
            deltaWeight = new double[X.length][faktorError.length];
            for (int j = 0; j < deltaWeight.length; j++) {
                for (int k = 0; k < deltaWeight[j].length; k++) {
                    deltaWeight[j][k] = alpha * faktorError[k] * X[j];
                }
            }
        }
        return deltaWeight;
    }

    public static double[] summationError(double[] faktorError, double[][] W) {
        double[] netDoY = null;
        if (faktorError != null && W != null) {
            netDoY = new double[W.length];
            for (int j = 0; j < netDoY.length; j++) {
                double sum = 0;
                for (int k = 0; k < W[0].length; k++) {
                    double doYkWjk = faktorError[k] * W[j][k];
                    sum += doYkWjk;
                }
                netDoY[j] = sum;
            }
        } else {
            System.out.println(" tidak sama panjang: " + faktorError.length + ", " + W[0].length);
        }
        return netDoY;
    }

}
