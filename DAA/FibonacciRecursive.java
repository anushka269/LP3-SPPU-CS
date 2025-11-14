import java.util.Scanner;

public class FibonacciRecursive {

    // Recursive function to return nth Fibonacci number
    static int fibonacci(int n) {
        if (n <= 1) {
            return n;   // Base case
        }
        return fibonacci(n - 1) + fibonacci(n - 2); // Recursive case
    }

    public static void main(String[] args) {
          Scanner sc=new Scanner(System.in);
          System.out.println("Enter the number:");
            int n = sc.nextInt();
        System.out.println("Fibonacci of " + n + " = " + fibonacci(n));

        sc.close();
    }
}

//Time Complexity = O(2^n) (Exponential)
//Space Complexity = O(n) (Linear) due to recursion stack
