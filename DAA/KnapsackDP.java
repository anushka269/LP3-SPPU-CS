import java.util.Scanner;

public class KnapsackDP {

    public static int knapsack(int W, int wt[], int val[], int n) {

        int dp[][] = new int[n + 1][W + 1];

        // Build DP table
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {

                // If item's weight is less than current capacity
                if (wt[i - 1] <= w) {
                    dp[i][w] = Math.max(
                            val[i - 1] + dp[i - 1][w - wt[i - 1]],
                            dp[i - 1][w]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];  // skip item
                }
            }
        }

        return dp[n][W]; // final answer
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        int[] val = new int[n];
        int[] wt = new int[n];

        System.out.println("Enter value and weight of each item:");
        for (int i = 0; i < n; i++) {
            val[i] = sc.nextInt();
            wt[i] = sc.nextInt();
        }

        System.out.print("Enter capacity of knapsack: ");
        int W = sc.nextInt();

        int result = knapsack(W, wt, val, n);

        System.out.println("Maximum value in Knapsack = " + result);
    }
}

//Time Complexity = O(n*W) (Polynomial)
//Space Complexity = O(n*W) (Polynomial)