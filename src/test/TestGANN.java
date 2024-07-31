package test;

import gann.ActivationFunction;
import gann.Data;
import gann.GeneticAlgorithm;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestGANN {
    
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
        // ---------------------------------------------------------------------       

        // PARAMETER GA --------------------------------------------------------
        int populationSize = 8;
        int MAX_GENERATION = 4;
        int numIndividuTerseleksi = 5;
        double mutationRate = 0.5;
        // ---------------------------------------------------------------------

        // RUN GANN ------------------------------------------------------------
        GeneticAlgorithm ga = new GeneticAlgorithm(populationSize, numIndividuTerseleksi, mutationRate, MAX_GENERATION, data, alpha, numHiddenNeurons, activationFunction, MAX_EPOCH, targetError);
        ga.run();
        ga.elitism.writeToFile("src/test/result.gann");
        ga.printBestIndividu();
        System.out.println("RESULT of Best Individu ditulis ke file");
        //System.out.println(ga.elitism.result);

    }
}
