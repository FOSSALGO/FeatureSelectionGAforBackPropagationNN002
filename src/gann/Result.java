package gann;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public class Result {

    public Data data;
    public Model model;
    public int[][] classifications;
    public int[][] confusionMatrix;
    public double accuracy;
    public double[] precision;
    public double[] recall;

    public void saveToFile(String filename) {
        String content = this.toString();
        File file = new File(filename);
        Path path = file.toPath();
        try {
            Files.writeString(path, content);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public String toString() {
        String[] label = data.kategori[data.indeksKelas];
        StringBuilder sb = new StringBuilder();
        sb.append("====================================\n");
        sb.append("T R A I N I N G\n");
        sb.append("====================================\n");
        sb.append("Model                  : " + model.numHiddenNeurons.length + " Hidden Layers\n");
        sb.append("Hidden Neurons         : " + Arrays.toString(model.numHiddenNeurons) + "\n");
        sb.append("Learning Rate          : " + model.alpha + "\n");
        sb.append("Number of Data         : " + data.numRows + "\n");
        sb.append("Number of Training Data: " + data.training.length + "\n");
        sb.append("Number of Testing Data : " + data.testing.length + "\n");
        sb.append("Error Tolerance        : " + model.targetError + "\n");
        sb.append("Error                  : " + model.error + "\n");
        sb.append("MAX_EPOCH              : " + model.MAX_EPOCH + "\n");
        sb.append("Epoch Reached          : " + model.epochReached + "\n");
        sb.append("====================================\n");
        sb.append("T E S T I N G\n");
        sb.append("====================================\n");
        sb.append("H A S I L    K L A S S I F I K A S I\n");
        sb.append("------------------------------------\n");
        sb.append("\t  CONFUSION MATRIX\n");
        sb.append("\t      Actual\n");
        sb.append("Predict\t");
        for (int i = 0; i < label.length; i++) {
            sb.append(label[i] + "\t");
        }
        sb.append("\n");
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(label[i] + "\t");
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                sb.append(confusionMatrix[i][j] + "\t");
            }
            sb.append("\n");
        }
        sb.append("------------------------------------\n");
        sb.append("ACCURACY \t: " + String.format("%.2f", accuracy) + "%\n");
        sb.append("------------------------------------\n");
        sb.append("PRECISION\n");
        for (int i = 0; i < precision.length; i++) {
            sb.append("Preci " + label[i] + "\t: " + String.format("%.2f", precision[i]) + "%\n");
        }
        sb.append("------------------------------------\n");
        sb.append("RECALL\n");
        for (int i = 0; i < recall.length; i++) {
            sb.append("Recall " + label[i] + "\t: " + String.format("%.2f", recall[i]) + "%\n");
        }
        sb.append("------------------------------------\n");
        if (data != null && classifications != null) {
            sb.append("DATA (String Value)\n");
            for (int i = 0; i < data.stringValues.length; i++) {
                sb.append("[" + (1 + i) + "]" + Arrays.toString(data.stringValues[i]) + "\n");
            }
            sb.append("------------------------------------\n");
            sb.append("Klasifikasi\n");
            for (int i = 0; i < classifications.length; i++) {
                sb.append("Test-" + (1 + i) + " Classification[" + label[classifications[i][0]] + "] <--> [" + label[classifications[i][1]] + "]Classification\n");
            }
            sb.append("------------------------------------\n");
            sb.append("Data Testing\n");
            sb.append("[Test] " + Arrays.toString(data.header) + " --- [Classification, Actual] --- [Classification] <-- [Actual]\n");
            for (int i = 0; i < data.testing.length; i++) {
                sb.append("[" + (1 + i) + "]" + Arrays.toString(data.testing[i]) + " --- " + Arrays.toString(classifications[i]) + " [" + label[classifications[i][0]] + "] <-- [" + label[classifications[i][1]] + "]\n");
            }
            sb.append("------------------------------------\n");
            sb.append("Data Training\n");
            sb.append("[No.] " + Arrays.toString(data.header) + "\n");
            for (int i = 0; i < data.training.length; i++) {
                sb.append("[" + (1 + i) + "]" + Arrays.toString(data.training[i]) + "\n");
            }
            sb.append("------------------------------------\n");
            sb.append("Data\n");
            sb.append("[No.] " + Arrays.toString(data.header) + "\n");
            for (int i = 0; i < data.training.length; i++) {
                sb.append("[" + (1 + i) + "]" + Arrays.toString(data.stringValues[i]) + "\n");
            }
            sb.append("------------------------------------\n");
        }

        if (model != null) {
            sb.append("====================================\n");
            sb.append("M O D E L\n");
            sb.append("====================================\n");
            for (int i = 0; i < model.weights.size(); i++) {
                sb.append("W" + i + ":\n");
                double[][] w = model.weights.get(i);
                for (int j = 0; j < w.length; j++) {
                    sb.append(Arrays.toString(w[j]) + "\n");
                }
                sb.append("------------------------------------\n");
            }
        }

        return sb.toString();
    }

}
