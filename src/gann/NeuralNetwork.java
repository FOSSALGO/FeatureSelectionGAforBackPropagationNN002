package gann;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetwork {

    public Model training(int[] selectedFeatures, Data data, double alpha, int[] numHiddenNeurons, ActivationFunction[] activationFunction, int MAX_EPOCH, double targetError) {
        Model model = null;
        if (selectedFeatures != null && data != null && data.training != null) {
            model = new Model();
            model.alpha = alpha;
            model.numHiddenNeurons = numHiddenNeurons;
            model.MAX_EPOCH = MAX_EPOCH;
            model.targetError = targetError;
            double[][] dataTraining = data.training;
            int numCols = 0;
            for (int i = 0; i < selectedFeatures.length; i++) {
                int alele = selectedFeatures[i];
                if (alele == 1) {
                    if (data.typesOfData[i].equalsIgnoreCase(Data.NUM)) {
                        numCols++;
                    } else if (data.typesOfData[i].equalsIgnoreCase(Data.CAT)) {
                        int nCategories = data.kategori[i].length;
                        numCols += nCategories;
                    }
                }
            }

            int numKelas = data.kategori[data.indeksKelas].length;//banyaknya kelas

            //List array bobot
            ArrayList<double[][]> weights = new ArrayList<>();
            //inisialisasi array bobot
            int numInput = numCols + 1;//numInput = banyaknya kolom header dikurang 1 karena kolom terakhir adalah target dan ditambah satu lagi karena menggunakan bias
            int numOutput = numKelas;
            // RANDOM BOBOT AWAL
            Random r = new Random();
            int s0 = numInput;
            for (int i = 0; i < numHiddenNeurons.length; i++) {
                int s1 = numHiddenNeurons[i];
                weights.add(new double[s0][s1]);
                s0 = s1 + 1;//tambah satu neuron untuk bias
            }
            weights.add(new double[s0][numOutput]);
            //random nilai bobot untuk semua bobot di array list
            for (int k = 0; k < weights.size(); k++) {
                double[][] weight = weights.get(k);
                for (int i = 0; i < weight.length; i++) {
                    for (int j = 0; j < weight[i].length; j++) {
                        weight[i][j] = r.nextDouble();// random uniform betwee 0 to 1
                    }
                }
            }//inisialisasi bobot selesai

            int epochReached = -1;
            double error = Double.MAX_VALUE;//akan menggunakan persen error
            int epoch = 1;
            while (epoch <= MAX_EPOCH && error > targetError) {//iterasi epoch
                int numError = 0;
                for (int row = 0; row < dataTraining.length; row++) {
                    //INISIALISASI NEURON INPUT DAN NEURON TARGET---------------
                    //neuron input
                    double[] X = new double[numInput];//sudah termasuk bias
                    int indexInput = 0;
                    for (int f = 0; f < selectedFeatures.length; f++) {
                        int alele = selectedFeatures[f];
                        double value = dataTraining[row][f];
                        if (alele == 1) {
                            if (data.typesOfData[f].equalsIgnoreCase(Data.NUM)) {
                                X[indexInput] = value;
                                indexInput++;
                            } else if (data.typesOfData[f].equalsIgnoreCase(Data.CAT)) {
                                int sizeCategory = data.kategori[f].length;
                                int index = (int) value;
                                int[] oneHotEncoding = new int[sizeCategory];
                                oneHotEncoding[index] = 1;
                                for (int i = 0; i < oneHotEncoding.length; i++) {
                                    X[indexInput] = oneHotEncoding[i];
                                    indexInput++;
                                }
                            }
                        }
                    }
                    X[X.length - 1] = 1;//neuron yang terhubung dengan bias ke hidden layer

                    //neuron target
                    double[] T = new double[numOutput];
                    int valueKelas = (int) dataTraining[row][data.indeksKelas];
                    T[valueKelas] = 1;// one Hot Encoding untuk Neuron Target
                    //INISIALISASI NEURON INPUT DAN NEURON TARGET SELESAI-------

                    //INISIALISASI NEURONS--------------------------------------
                    double[][] neurons = new double[numHiddenNeurons.length + 2][];//jumlah layer keseluruhan adalah sebanyak: 1 layer input + jumlah hidden layer + 1 layer output
                    neurons[0] = X;//neurons[0] adalah layer pertama dan akan berisi neuron-neuron dari input / X
                    for (int i = 0; i < numHiddenNeurons.length; i++) {
                        int nHidden = numHiddenNeurons[i];
                        neurons[i + 1] = new double[nHidden + 1];//termasuk neuron bias
                        neurons[i + 1][nHidden] = 1;//neuron yang berpasangan dengan bobot bias
                    }
                    neurons[neurons.length - 1] = new double[numOutput];//neurons pada layer terakhir akan berisi neuron-neuron output
                    //INISIALISASI NEURONS SELESAI------------------------------

                    //FEEDFORWARD-----------------------------------------------                    
                    for (int i = 0; i < weights.size(); i++) {
                        double[] input = neurons[i];
                        double[][] weight = weights.get(i);
                        double[] net = OP.summation(input, weight);//hasil summation. input x weight
                        double[] fnet = OP.activation(net, activationFunction[i]);//hasil activation. fnet = y
                        //set fnet ke neuron berikutnya
                        for (int j = 0; j < fnet.length; j++) {
                            neurons[1 + i][j] = fnet[j];
                        }
                    }
                    //END OF FEEDFORWARD----------------------------------------

                    //TRANSFORMASI ke Y dan hitung error------------------------
                    double[] output = neurons[neurons.length - 1];
                    double max = Double.MIN_VALUE;
                    int imax = -1;
                    double[] Y = new double[numOutput];
                    double[] netError = new double[numOutput];//do_net
                    for (int i = 0; i < Y.length; i++) {
                        Y[i] = 0;
                        if (output[i] > max) {
                            max = output[i];
                            imax = i;
                        }
                        //hitung net error di output
                        netError[i] = T[i] - output[i];

                    }
                    Y[imax] = 1;
                    //index Y dengan nilai aktivasi terbesar akan diberi nilai 1

                    //cek apakah Y = T
                    if (convertOutputToNumeric(Y) != convertOutputToNumeric(T)) {
                        numError++;
                    }
                    //System.out.println("Y: " + Arrays.toString(Y));
                    //End of Hitung Error---------------------------------------

                    //BACKPROPAGATION-------------------------------------------
                    ArrayList<double[][]> listDeltaW = new ArrayList<>();
                    for (int i = weights.size() - 1; i >= 0; i--) {
                        double[] neuronsOut = neurons[i + 1];//
                        double[] neuronsIn = neurons[i];//
                        double[] faktorError = OP.hitungFaktorError(netError, neuronsOut, activationFunction[i]);//do 
                        double[][] deltaW = OP.hitungDeltaWeight(alpha, faktorError, neuronsIn);
                        listDeltaW.add(deltaW);
                        if (i > 0) {
                            netError = OP.summationError(faktorError, weights.get(i));
                        }
                    }
                    //END OF BACKPROPAGATION------------------------------------

                    //UPDATE WEIGHTS--------------------------------------------
                    for (int i = 0; i < weights.size(); i++) {
                        double[][] weight = weights.get(i);
                        double[][] deltaW = listDeltaW.get(listDeltaW.size() - 1 - i);
                        for (int j = 0; j < weight.length; j++) {
                            for (int k = 0; k < weight[j].length; k++) {
                                weight[j][k] = weight[j][k] + deltaW[j][k];
                            }
                        }
                    }
                    //END OF UPDATE WEIGHTS-------------------------------------
                }//end of for (int row = 0; row < dataTraining.length; row++)

                //Evaluasi Error
                error = (double) numError / (double) dataTraining.length;
                if (error <= targetError) {
                    System.out.println("TARGET ERROR REACHED in epoch-" + epoch);
                    epochReached = epoch;
                    break;//akhiri epoch   
                }

                epoch++;

            }//end of while (epoch <= MAX_EPOCH && error > targetError)

            //SET MODEL---------------------------------------------------------
            model.epochReached = epochReached;
            model.error = error;
            model.weights = weights;

        }
        return model;
    }

    public Result testing(int[] selectedFeatures, Data data, Model model, ActivationFunction[] activationFunction) {
        Result testResult = null;
        if (selectedFeatures != null && data != null && model != null && data.testing != null) {
            double[][] dataTesting = data.testing;
            int numCols = 0;
            for (int i = 0; i < selectedFeatures.length; i++) {
                int alele = selectedFeatures[i];
                if (alele == 1) {
                    if (data.typesOfData[i].equalsIgnoreCase(Data.NUM)) {
                        numCols++;
                    } else if (data.typesOfData[i].equalsIgnoreCase(Data.CAT)) {
                        int nCategories = data.kategori[i].length;
                        numCols += nCategories;
                    }
                }
            }

            int numKelas = data.kategori[data.indeksKelas].length;//banyaknya kelas

            //CONFUSION MATRIX
            int[][] confusionMatrix = new int[numKelas][numKelas];//PREDIKSI <-> AKTUAL
            //Save hasil klasifikasi untuk setiap row data testing
            int[][] classifications = new int[dataTesting.length][2];// classisfication(Y) | actual(T)
            //List array bobot
            ArrayList<double[][]> weights = model.weights;//inisialisasi array bobot dari model
            int numInput = numCols + 1;//numInput = banyaknya kolom header dikurang 1 karena kolom terakhir adalah target dan ditambah satu lagi karena menggunakan bias
            int numOutput = numKelas;
            int[] numHiddenNeurons = model.numHiddenNeurons;// dibaca dari model yang dihasilkan dari proses training

            for (int row = 0; row < dataTesting.length; row++) {
                //INISIALISASI NEURON INPUT DAN NEURON TARGET---------------
                //neuron input
                double[] X = new double[numInput];//sudah termasuk bias
                int indexInput = 0;
                for (int f = 0; f < selectedFeatures.length; f++) {
                    int alele = selectedFeatures[f];
                    if (alele == 1) {
                        double value = dataTesting[row][f];
                        if (data.typesOfData[f].equalsIgnoreCase(Data.NUM)) {
                            X[indexInput] = value;
                            indexInput++;
                        } else if (data.typesOfData[f].equalsIgnoreCase(Data.CAT)) {
                            int sizeCategory = data.kategori[f].length;
                            int index = (int) value;
                            int[] oneHotEncoding = new int[sizeCategory];
                            oneHotEncoding[index] = 1;
                            for (int i = 0; i < oneHotEncoding.length; i++) {
                                X[indexInput] = oneHotEncoding[i];
                                indexInput++;
                            }
                        }
                    }
                }
                X[X.length - 1] = 1;//neuron yang terhubung dengan bias ke hidden layer

                //neuron target
                double[] T = new double[numOutput];
                int valueKelas = (int) dataTesting[row][data.indeksKelas];
                T[valueKelas] = 1;// one Hot Encoding untuk Neuron Target
                //INISIALISASI NEURON INPUT DAN NEURON TARGET SELESAI-------

                //INISIALISASI NEURONS--------------------------------------
                double[][] neurons = new double[numHiddenNeurons.length + 2][];//jumlah layer keseluruhan adalah sebanyak: 1 layer input + jumlah hidden layer + 1 layer output
                neurons[0] = X;//neurons[0] adalah layer pertama dan akan berisi neuron-neuron dari input / X
                for (int i = 0; i < numHiddenNeurons.length; i++) {
                    int nHidden = numHiddenNeurons[i];
                    neurons[i + 1] = new double[nHidden + 1];//termasuk neuron bias
                    neurons[i + 1][nHidden] = 1;//neuron yang berpasangan dengan bobot bias
                }
                neurons[neurons.length - 1] = new double[numOutput];//neurons pada layer terakhir akan berisi neuron-neuron output
                //INISIALISASI NEURONS SELESAI------------------------------

                //FEEDFORWARD-----------------------------------------------                    
                for (int i = 0; i < weights.size(); i++) {
                    double[] input = neurons[i];
                    double[][] weight = weights.get(i);
                    double[] net = OP.summation(input, weight);//hasil summation. input x weight
                    double[] fnet = OP.activation(net, activationFunction[i]);//hasil activation. fnet = y
                    //set fnet ke neuron berikutnya
                    for (int j = 0; j < fnet.length; j++) {
                        neurons[1 + i][j] = fnet[j];
                    }
                }
                //END OF FEEDFORWARD----------------------------------------

                //TRANSFORMASI ke Y dan hitung error------------------------
                double[] output = neurons[neurons.length - 1];
                double max = Double.MIN_VALUE;
                int imax = -1;
                double[] Y = new double[numOutput];
                double[] netError = new double[numOutput];//do_net
                for (int i = 0; i < Y.length; i++) {
                    Y[i] = 0;
                    if (output[i] > max) {
                        max = output[i];
                        imax = i;
                    }
                    //hitung net error di output
                    netError[i] = T[i] - output[i];

                }
                Y[imax] = 1;
                //index Y dengan nilai aktivasi terbesar akan diberi nilai 1
                int indexY = convertOutputToNumeric(Y);//prediksi
                int indexT = convertOutputToNumeric(T);//aktual
                confusionMatrix[indexY][indexT]++;//increment confusion matrix
                //save hasil classification
                classifications[row][0] = indexY;
                classifications[row][1] = indexT;

            }// for (int row = 0; row < dataTesting.length; row++) {

            //SET OUTPUT--------------------------------------------------------
            testResult = new Result();
            testResult.data = data;
            testResult.model = model;
            testResult.classifications = classifications;
            testResult.confusionMatrix = confusionMatrix;
            //hitung accuracy
            double sumAccuracy = 0;
            double[] precision = new double[confusionMatrix.length];
            double[] recall = new double[confusionMatrix.length];
            double accuracy;

            //HORIZONTAL <-- PRECISION
            double denominator = 0;
            for (int i = 0; i < confusionMatrix.length; i++) {
                double sumPrecision = 0;
                sumAccuracy += confusionMatrix[i][i];
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    sumPrecision += confusionMatrix[i][j];
                    denominator += confusionMatrix[i][j];
                }
                precision[i] = 100 * ((double) confusionMatrix[i][i] / (double) sumPrecision);
            }
            accuracy = 100 * (sumAccuracy / denominator);

            //VERTICAL <-- RECALL
            for (int j = 0; j < confusionMatrix.length; j++) {
                double sumRecall = 0;
                for (int i = 0; i < confusionMatrix.length; i++) {
                    sumRecall += confusionMatrix[i][j];
                }
                recall[j] = 100 * ((double) confusionMatrix[j][j] / (double) sumRecall);
            }

            //SET TEST RESULT
            testResult.precision = precision;
            testResult.recall = recall;
            testResult.accuracy = accuracy;

        }// end of if (selectedFeatures != null && data != null && model != null && data.testing != null)
        return testResult;
    }

    int convertOutputToNumeric(double[] T) {
        int index = -1;
        for (int i = 0; i < T.length; i++) {
            if (T[i] == 1) {
                index = i;
                break;
            }
        }
        return index;
    }

}
