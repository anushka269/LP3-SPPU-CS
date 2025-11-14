import java.util.Scanner;

public class NQueens {

    static int N;
    static int[][] board;

    // Check if placing a queen at board[row][col] is safe
    static boolean isSafe(int row, int col) {

        // Check column
        for (int i = 0; i < row; i++)
            if (board[i][col] == 1)
                return false;

        // Check left diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1)
                return false;

        // Check right diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < N; i--, j++)
            if (board[i][j] == 1)
                return false;

        return true;
    }

    // Backtracking function
    static boolean solveNQueens(int row) {

        if (row == N)       // all queens placed
            return true;

        // If queen already placed in this row (first queen row)
        for (int col = 0; col < N; col++) {
            if (board[row][col] == 1) {
                if (isSafe(row, col))
                    return solveNQueens(row + 1);
                else
                    return false;
            }
        }

        // Try all columns
        for (int col = 0; col < N; col++) {

            if (isSafe(row, col)) {
                board[row][col] = 1;       // place queen

                if (solveNQueens(row + 1)) // recursive step
                    return true;

                board[row][col] = 0;       // backtrack
            }
        }

        return false;
    }

    // Print solution matrix
    static void printBoard() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter value of N: ");
        N = sc.nextInt();

        board = new int[N][N];

        System.out.print("Enter position of first queen (row col): ");
        int r = sc.nextInt();
        int c = sc.nextInt();

        board[r][c] = 1; // First queen placed by user

        if (solveNQueens(0)) {
            System.out.println("Solution:");
            printBoard();
        } else {
            System.out.println("No solution exists!");
        }
    }
}
