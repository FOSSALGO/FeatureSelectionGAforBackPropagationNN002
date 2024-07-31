package gann;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Data {

    //KONSTANTA
    //tipe data numerical or categorical
    public static final String NUM = "NUMERICAL";
    public static final String CAT = "CATEGORICAL";

    //VARIABEL
    public String filename = null;
    public String separator = ",";
    public int indeksKelas;
    public String[] typesOfData;

    //OUTPUT    
    public String[] header;
    public String[][] stringValues;
    public String[][] kategori;
    public double[][] normalValues;
    public double[][] training;
    public double[][] testing;
    public int numRows = 0;

    public Data(String filename, String separator, int indeksKelas, String[] typesOfData, int[] ignored) {
        this.filename = filename;
        this.separator = separator;
        this.indeksKelas = indeksKelas;
        this.typesOfData = typesOfData;
        readData(filename, separator, indeksKelas, typesOfData, ignored);
    }

    private boolean readData(String filename, String separator, int indeksKelas, String[] typesOfData, int[] ignored) {
        boolean status = false;
        if (filename != null && separator != null) {
            File file = new File(filename);
            try {
                Scanner sc = new Scanner(file);
                // BACA HEADER
                String record = sc.nextLine();
                String[] tempHeader = record.split(separator);
                header = rearrange(tempHeader, indeksKelas, ignored);
                this.typesOfData = rearrange(typesOfData, indeksKelas, ignored);
                //BACA DATA
                ArrayList<String[]> tempData = new ArrayList<>();
                while (sc.hasNextLine()) {
                    record = sc.nextLine();
                    String[] rowData = record.split(separator);

                    // TRIM
                    for (int i = 0; i < rowData.length; i++) {
                        rowData[i] = rowData[i].trim();
                    }

                    // IF !MISSING VALUE
                    if (tempHeader.length == rowData.length) {
                        rowData = rearrange(rowData, indeksKelas, ignored);
                        tempData.add(rowData);
                    }
                }//end of while (sc.hasNextLine)

                //KONVERSI ke ARRAY bertipe String
                stringValues = new String[tempData.size()][];
                for (int i = 0; i < tempData.size(); i++) {
                    stringValues[i] = new String[header.length];
                    stringValues[i] = tempData.get(i);

                }

                //SET INDEX KELAS ke Kolom TErakhir
                indeksKelas = header.length - 1;
                this.indeksKelas = indeksKelas;

                // Hitung Kategori
                // BACA KATEGORI DATA
                ArrayList<String>[] kategori = new ArrayList[header.length];
                for (int j = 0; j < this.typesOfData.length; j++) {
                    if (this.typesOfData[j].equalsIgnoreCase(CAT)) {
                        kategori[j] = new ArrayList<>();
                        for (int i = 0; i < stringValues.length; i++) {
                            String key = stringValues[i][j];
                            if (!kategori[j].contains(key)) {
                                kategori[j].add(key);
                            }
                        }
                    } else {
                        kategori[j] = null;
                    }
                }

                // SAVE KATEGORI
                this.kategori = new String[header.length][];
                for (int j = 0; j < header.length; j++) {
                    if (kategori[j] != null) {
                        this.kategori[j] = new String[kategori[j].size()];
                        for (int m = 0; m < this.kategori[j].length; m++) {
                            this.kategori[j][m] = kategori[j].get(m);
                        }
                    } else {
                        this.kategori[j] = null;
                    }
                }

                // NORMALISASI   
                double[][] doubleValues = new double[stringValues.length][header.length];
                normalValues = new double[stringValues.length][header.length];
                double[] MIN = new double[header.length];
                double[] MAX = new double[header.length];
                for (int j = 0; j < header.length; j++) {
                    if (this.typesOfData[j].equalsIgnoreCase(NUM)) {
                        MIN[j] = Double.MAX_VALUE;
                        MAX[j] = Double.MIN_VALUE;
                    } else {
                        MIN[j] = 0;
                        MAX[j] = 0;
                    }
                }

                for (int i = 0; i < stringValues.length; i++) {
                    for (int j = 0; j < header.length; j++) {
                        String value = stringValues[i][j].trim();
                        if (this.typesOfData[j].equalsIgnoreCase(NUM)) {
                            doubleValues[i][j] = Double.parseDouble(value);
                            if (MIN[j] > doubleValues[i][j]) {
                                MIN[j] = doubleValues[i][j];
                            }
                            if (MAX[j] < doubleValues[i][j]) {
                                MAX[j] = doubleValues[i][j];
                            }
                        } else {
                            String[] kategori_j = this.kategori[j];
                            for (int m = 0; m < kategori_j.length; m++) {
                                if (value.equalsIgnoreCase(kategori_j[m])) {
                                    doubleValues[i][j] = m;// index kategori yang disimpan ke doubleValue
                                    break;
                                }
                            }
                        }
                    }
                }

                for (int i = 0; i < normalValues.length; i++) {
                    for (int j = 0; j < header.length; j++) {
                        double value = doubleValues[i][j];
                        if (this.typesOfData[j].equalsIgnoreCase(NUM)) {
                            // NORMALISASI MIN-MAX
                            normalValues[i][j] = (value - MIN[j]) / (MAX[j] - MIN[j]);
                        } else {
                            normalValues[i][j] = value;
                        }
                    }
                }

                // SET NORMAL VALUE
                numRows = normalValues.length;

                status = true;// SET STATUS = true jika berhasil membaca data sampai normalisasi

            } catch (FileNotFoundException ex) {
                Logger.getLogger(Data.class.getName()).log(Level.SEVERE, null, ex);
            }

        }//end of if (filename != null && separator != null)
        return status;
    }

    public static String[] rearrange(String[] rowData, int indeksKelas, int[] ignored) {
        String[] result = new String[rowData.length - ignored.length];
        int k = 0;
        for (int i = 0; i < rowData.length; i++) {
            if (i == indeksKelas) {
                result[result.length - 1] = rowData[i];
            } else {
                boolean status = false;
                for (int j = 0; j < ignored.length; j++) {
                    if (i == ignored[j]) {
                        status = true;
                        break;
                    }
                }

                if (status == false) {
                    result[k] = rowData[i];
                    k++;
                }
            }
        }
        return result;
    }

    public void splitData(double percentTraining, double percentTesting) {
        if ((percentTraining + percentTesting) <= 100) {
            int numTraining = (int) Math.floor(numRows * percentTraining / 100.0);
            int numTesting = numRows - numTraining;//(int) Math.floor(numRows * percentTesting / 100.0);
            training = new double[numTraining][];
            testing = new double[numTesting][];
            //distribusikan hasil split data
            for (int i = 0; i < numTraining; i++) {
                training[i] = normalValues[i];
            }
            for (int i = 0; i < numTesting; i++) {
                testing[i] = normalValues[i + numTraining];
            }
        }
    }
}
