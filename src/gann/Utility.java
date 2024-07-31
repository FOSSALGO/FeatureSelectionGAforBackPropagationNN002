package gann;

import java.util.Random;

public class Utility {

    private static Random random = new Random();

    public static int randomBiner() {
        double r = random.nextDouble();
        double threshold = 0.5;
        if (r < threshold) {
            return 0;
        } else {
            return 1;
        }
    }

    public static int randomBetween(int lower, int upper) {
        if (lower > upper) {
            int temp = lower;
            lower = upper;
            upper = temp;
        }
        return lower + random.nextInt(upper - lower + 1);
    }

    public static double randomMutation() {
        return random.nextDouble();
    }

}
