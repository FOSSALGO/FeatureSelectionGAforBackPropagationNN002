package gann;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import test.TestGANN;

public class Individu {

    //PARAMETER GENETIC ALGORITHM
    public int[] gen;
    public int min_feature = 2;
    public double fitnessValue;

    //PARAMETER NEURAL NETWORK
    private Data data;
    private double alpha;
    private int[] numHiddenNeurons;
    private ActivationFunction[] activationFunction;
    private int MAX_EPOCH;
    private double targetError;
    public Result result = null;

    public Individu() {
        //
    }

    public Individu(Data data, double alpha, int[] numHiddenNeurons, ActivationFunction[] activationFunction, int MAX_EPOCH, double targetError) {
        this.data = data;
        this.alpha = alpha;
        this.numHiddenNeurons = numHiddenNeurons;
        this.activationFunction = activationFunction;
        this.MAX_EPOCH = MAX_EPOCH;
        this.targetError = targetError;
        int length = this.data.header.length - 1;
        initialization(length);
    }

    public Individu(int[] gen, Data data, double alpha, int[] numHiddenNeurons, ActivationFunction[] activationFunction, int MAX_EPOCH, double targetError) {
        this.gen = gen;
        this.data = data;
        this.alpha = alpha;
        this.numHiddenNeurons = numHiddenNeurons;
        this.activationFunction = activationFunction;
        this.MAX_EPOCH = MAX_EPOCH;
        this.targetError = targetError;
    }

    public Individu clone() {
        //PARAMETER GENETIC ALGORITHM
        Individu newIndividu = new Individu();
        newIndividu.gen = new int[this.gen.length];
        //array copy
        for (int i = 0; i < this.gen.length; i++) {
            newIndividu.gen[i] = this.gen[i];
        }
        newIndividu.min_feature = this.min_feature;
        newIndividu.fitnessValue = this.fitnessValue;

        //PARAMETER NEURAL NETWORK
        newIndividu.data = this.data;
        newIndividu.alpha = this.alpha;
        newIndividu.numHiddenNeurons = this.numHiddenNeurons;
        newIndividu.activationFunction = this.activationFunction;
        newIndividu.MAX_EPOCH = this.MAX_EPOCH;
        newIndividu.targetError = this.targetError;
        newIndividu.result = this.result;
        return newIndividu;
    }

    public void initialization(int length) {
        gen = new int[length];
        for (int i = 0; i < length; i++) {
            int alele = Utility.randomBiner();
            gen[i] = alele;
        }
        featureValidation();

    }

    public boolean featureValidation() {
        boolean status = false;
        if (gen != null) {
            int nSatu = 0;
            for (int i = 0; i < gen.length; i++) {
                int alele = gen[i];
                if (alele == 1) {
                    nSatu++;
                }
            }
            if (nSatu < min_feature) {
                boolean valid = false;
                while (!valid && nSatu < min_feature) {
                    int r = Utility.randomBetween(0, gen.length - 1);
                    if (gen[r] == 0) {
                        gen[r] = 1;
                        nSatu++;
                        valid = true;
                    }
                }
            }
            status = true;
        }
        return status;
    }

    public double calculateFitnessValue() {
        fitnessValue = 0;
        if (gen != null) {
            NeuralNetwork nn = new NeuralNetwork();
            int[] selectedFeatures = gen.clone();
            Model model = nn.training(selectedFeatures, data, alpha, numHiddenNeurons, activationFunction, MAX_EPOCH, targetError);
            result = nn.testing(selectedFeatures, data, model, activationFunction);
            double accuracy = result.accuracy / 100.0;
            double numFeature = 0;
            for (int i = 0; i < gen.length; i++) {
                if (gen[i] == 1) {
                    numFeature++;
                }
            }
            double epsilon = 0.0000000000001;
            double rasioFitur = numFeature / (double) gen.length;
            double fitness = (1.0 / (rasioFitur + epsilon)) * accuracy;
            fitnessValue = fitness;// Sesuaikan fungsi fitness berdasarkan objective
        }
        return fitnessValue;
    }

    public String toString() {
        return "Individu: " + Arrays.toString(gen) + "";
    }

    public void writeToFile(String filename) {
        try {
            File fileOut = new File(filename);
            Path path = fileOut.toPath();
            String info = "FITUR-----------------------------------------------\n";
            info += Arrays.toString(gen);
            info += "\n";
            info += fitnessValue;
            info += "\n";
            info += result.toString();
            Files.writeString(path, info);
        } catch (IOException ex) {
            Logger.getLogger(TestGANN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
