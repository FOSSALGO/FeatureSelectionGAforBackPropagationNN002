package gann;

import java.util.ArrayList;

public class Model {

    //TRAINING PARAMETERS
    public double alpha;
    public int[] numHiddenNeurons;
    public int MAX_EPOCH;
    public double targetError;
    public int epochReached = -1;
    public double error;

    //ArrayList bobot
    public ArrayList<double[][]> weights = new ArrayList<>();
}
