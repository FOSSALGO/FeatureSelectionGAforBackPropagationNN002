package gann;

import java.util.Arrays;

public class GeneticAlgorithm {

    private int genSize;
    private int populationSize;
    private double mutationRate;
    private int MAX_GENERATION;
    private int numIndividuTerseleksi;
    public Individu elitism = null;

    private Data data;
    private double alpha;
    private int[] numHiddenNeurons;
    private ActivationFunction[] activationFunction;
    private int MAX_EPOCH;
    private double targetError;

    public GeneticAlgorithm(int populationSize, int numIndividuTerseleksi, double mutationRate, int MAX_GENERATION, Data data, double alpha, int[] numHiddenNeurons, ActivationFunction[] activationFunction, int MAX_EPOCH, double targetError) {
        this.populationSize = populationSize;
        this.numIndividuTerseleksi = numIndividuTerseleksi;
        this.mutationRate = mutationRate;
        this.MAX_GENERATION = MAX_GENERATION;
        this.data = data;
        this.alpha = alpha;
        this.numHiddenNeurons = numHiddenNeurons;
        this.activationFunction = activationFunction;
        this.MAX_EPOCH = MAX_EPOCH;
        this.targetError = targetError;
        this.genSize = data.header.length - 1;
    }

    public boolean isValid() {
        if (populationSize > 0
                && genSize > 0) {
            return true;
        } else {
            return false;
        }
    }

    public void run() {
        if (isValid()) {
            // INITIALIZATION
            Individu[] population_0 = new Individu[populationSize];
            // GENERATE POPULASI AWAL dan CALCULATE FITNESS VALUE
            double[][] fitness = new double[populationSize][2];
            for (int i = 0; i < populationSize; i++) {
                population_0[i] = new Individu(data, alpha, numHiddenNeurons, activationFunction, MAX_EPOCH, targetError);
                population_0[i].calculateFitnessValue();
                fitness[i][0] = population_0[i].fitnessValue;
                fitness[i][1] = i;
            }
            // SORT BASE ON FITNESS VALUE
            for (int i = 0; i < fitness.length - 1; i++) {
                double MAX = fitness[i][0];
                int iMAX = i;
                for (int j = 1 + i; j < fitness.length; j++) {
                    if (fitness[j][0] > MAX) {
                        MAX = fitness[j][0];
                        iMAX = j;
                    }
                }
                //SWAP
                if (iMAX > i) {
                    double temp_0 = fitness[i][0];
                    double temp_1 = fitness[i][1];
                    fitness[i][0] = fitness[iMAX][0];
                    fitness[i][1] = fitness[iMAX][1];
                    fitness[iMAX][0] = temp_0;
                    fitness[iMAX][1] = temp_1;
                }
            }
            // simpan hasil sorting
            Individu[] population = new Individu[populationSize];
            // GENERATE POPULASI AWAL dan CALCULATE FITNESS VALUE
            for (int i = 0; i < fitness.length; i++) {
                int index = (int) fitness[i][1];
                population[i] = population_0[index].clone();
            }

            // ELITISM
            elitism = population[0].clone();

            // PROSES EVOLUSI
            for (int g = 1; g <= MAX_GENERATION; g++) {
                // PROSES SELEKSI TURNAMEN--------------------------------------
                Individu[] newPopulation = new Individu[populationSize];
                for (int i = 0; i < numIndividuTerseleksi; i++) {
                    newPopulation[i] = population[i].clone();//cloning
                }
                // PROSES CROSSOVER---------------------------------------------
                int batasCrossover = (int) Math.ceil((double) (populationSize - numIndividuTerseleksi) / (double) 2);
                int k = numIndividuTerseleksi;
                for (int i = 0; i < batasCrossover; i++) {
                    //random index parent
                    int indexParent1 = Utility.randomBetween(0, numIndividuTerseleksi - 1);
                    int indexParent2 = indexParent1;
                    while (indexParent1 == indexParent2) {
                        indexParent2 = Utility.randomBetween(0, numIndividuTerseleksi - 1);
                    }
                    //random titik mutasi
                    int crossoverPoint = Utility.randomBetween(0, genSize - 2);
                    int[] parent1 = population[indexParent1].gen;
                    int[] parent2 = population[indexParent2].gen;
                    int[] child1 = new int[parent1.length];
                    int[] child2 = new int[parent2.length];
                    for (int j = 0; j <= crossoverPoint; j++) {
                        child1[j] = parent1[j];
                        child2[j] = parent2[j];
                    }
                    for (int j = crossoverPoint + 1; j < genSize; j++) {
                        child1[j] = parent2[j];
                        child2[j] = parent1[j];
                    }
                    if (k < populationSize) {
                        //set child1 ke new population
                        newPopulation[k] = new Individu(child1, data, alpha, numHiddenNeurons, activationFunction, MAX_EPOCH, targetError);
                        k++;
                    }
                    if (k < populationSize) {
                        //set child2 ke new population
                        newPopulation[k] = new Individu(child2, data, alpha, numHiddenNeurons, activationFunction, MAX_EPOCH, targetError);
                        k++;
                    }
                }
                // PROSES MUTASI------------------------------------------------                
                for (int i = 0; i < populationSize; i++) {
                    double r = Utility.randomMutation();
                    if (r < mutationRate) {
                        //dimutasi
                        int titikMutasi = Utility.randomBetween(0, genSize - 1);
                        //flip mutation
                        if (newPopulation[i].gen[titikMutasi] == 1) {
                            newPopulation[i].gen[titikMutasi] = 0;
                        } else if (newPopulation[i].gen[titikMutasi] == 0) {
                            newPopulation[i].gen[titikMutasi] = 1;
                        }
                    }

                    // VALIDASI INDIVIDU
                    newPopulation[i].featureValidation();
                }
                // EVALUASI FITNESS---------------------------------------------
                fitness = new double[populationSize][2];
                for (int i = 0; i < populationSize; i++) {
                    newPopulation[i].calculateFitnessValue();
                    fitness[i][0] = newPopulation[i].fitnessValue;
                    fitness[i][1] = i;
                }
                // SORTING berdasarkan nilai fitness
                for (int i = 0; i < fitness.length - 1; i++) {
                    double MAX = fitness[i][0];
                    int iMAX = i;
                    for (int j = 1 + i; j < fitness.length; j++) {
                        if (fitness[j][0] > MAX) {
                            MAX = fitness[j][0];
                            iMAX = j;
                        }
                    }
                    //SWAP
                    if (iMAX > i) {
                        double temp_0 = fitness[i][0];
                        double temp_1 = fitness[i][1];
                        fitness[i][0] = fitness[iMAX][0];
                        fitness[i][1] = fitness[iMAX][1];
                        fitness[iMAX][0] = temp_0;
                        fitness[iMAX][1] = temp_1;
                    }
                }
                // UPDATE POPULASI ---------------------------------------------
                population = new Individu[populationSize];
                // GENERATE POPULASI AWAL dan CALCULATE FITNESS VALUE
                for (int i = 0; i < fitness.length; i++) {
                    int index = (int) fitness[i][1];
                    population[i] = newPopulation[index].clone();
                }

                // ELITISM------------------------------------------------------
                elitism = population[0].clone();
            }// END OF PROSES EVOLUSI

        }
    }

    public void printBestIndividu() {
        if (elitism != null) {
            System.out.println("BEST INDIVIDU = " + Arrays.toString(elitism.gen));
            System.out.println("BEST FITNESS  = " + elitism.fitnessValue);
        } else {
            System.out.println("ELITISM = NULL");
        }
    }
}
