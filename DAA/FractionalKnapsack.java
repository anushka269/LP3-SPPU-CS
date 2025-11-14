import java.util.*;

class Item {
    int value, weight;
    Item(int value, int weight) {
        this.value = value;
        this.weight = weight;
    }
}

public class FractionalKnapsack {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        Item[] items = new Item[n];

        System.out.println("Enter value and weight of each item:");
        for (int i = 0; i < n; i++) {
            int v = sc.nextInt();
            int w = sc.nextInt();
            items[i] = new Item(v, w);
        }

        System.out.print("Enter capacity of knapsack: ");
        int capacity = sc.nextInt();

        // Sort items by value/weight ratio in descending order
        Arrays.sort(items, (a, b) ->
                Double.compare((double)b.value / b.weight,
                               (double)a.value / a.weight));

        double totalValue = 0;

        for (Item item : items) {
            if (capacity == 0)
                break;

            if (item.weight <= capacity) {
                // take full item
                totalValue += item.value;
                capacity -= item.weight;
            } else {
                // take fraction
                double fraction = (double) capacity / item.weight;
                totalValue += item.value * fraction;
                capacity = 0;
            }
        }

        System.out.println("Maximum value in Knapsack = " + totalValue);
    }
}

//Time Complexity = O(n log n) due to sorting
//Space Complexity = O(1) (Constant)
