import java.util.Scanner;

public class FibonacciIterative {

    static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }

        int a = 0, b = 1, c = 0;

        for (int i = 2; i <= n; i++) {
            c = a + b;  // next Fibonacci number
            a = b;      // shift forward
            b = c;
        }

        return c;
    }

    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter the number:");
        int n = sc.nextInt();
        System.out.println("Fibonacci of " + n + " = " + fibonacci(n));

        sc.close();
    }
}

//Time Complexity = O(n) (Linear)
//Space Complexity = O(1) (Constant)