package test;

import gann.ActivationFunction;
import gann.Data;
import gann.Individu;

public class TestNN {

    public static void main(String[] args) {
        // DATA ----------------------------------------------------------------     
        String filename = "src/test/Invistico_Airline.csv";
        String separator = ",";
        int indeksKelas = 0;
        String[] typesOfData = {Data.CAT, Data.CAT, Data.CAT, Data.NUM, Data.CAT, Data.CAT, Data.NUM, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.CAT, Data.NUM, Data.NUM};
        int[] ignored = {};
        double percentTraining = 80;
        double percentTesting = 20;
        // baca dataset
        Data data = new Data(filename, separator, indeksKelas, typesOfData, ignored);
        data.splitData(percentTraining, percentTesting);
        // ---------------------------------------------------------------------

        // PARAMETER Neural Network --------------------------------------------
        ActivationFunction[] activationFunction = {ActivationFunction.SIGMOID_BINER, ActivationFunction.SIGMOID_BINER};//tipe fungsi aktivasi
        double alpha = 0.1;
        int[] numHiddenNeurons = {4};
        int MAX_EPOCH = 10;
        double targetError = 0.1;
        int[] gen = {1, 0, 1};
        // ---------------------------------------------------------------------       

        // RUN GANN ------------------------------------------------------------
        Individu individuNN = new Individu(gen, data, alpha, numHiddenNeurons, activationFunction, MAX_EPOCH, targetError);
        individuNN.calculateFitnessValue();
        individuNN.writeToFile("src/test/result.nn");

    }
}
