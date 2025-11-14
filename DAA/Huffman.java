import java.util.PriorityQueue;
import java.util.Scanner;

class Node {
    char ch;
    int freq;
    Node left, right;

    Node(char ch, int freq) {
        this.ch = ch;
        this.freq = freq;
    }

    Node(int freq, Node left, Node right) {
        this.ch = '-';
        this.freq = freq;
        this.left = left;
        this.right = right;
    }
}

public class Huffman{

    // Print Huffman codes
    static void printCodes(Node root, String code) {
        if (root == null) return;

        // If leaf node
        if (root.left == null && root.right == null) {
            System.out.println(root.ch + ": " + code);
        }

        printCodes(root.left, code + "0");
        printCodes(root.right, code + "1");
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of characters: ");
        int n = sc.nextInt();

        char[] chars = new char[n];
        int[] freq = new int[n];

        System.out.println("Enter characters:");
        for (int i = 0; i < n; i++) {
            chars[i] = sc.next().charAt(0);
        }

        System.out.println("Enter frequencies:");
        for (int i = 0; i < n; i++) {
            freq[i] = sc.nextInt();
        }

        // Min-heap priority queue
        PriorityQueue<Node> pq = new PriorityQueue<>(
                (a, b) -> a.freq - b.freq);

        // Insert nodes
        for (int i = 0; i < n; i++) {
            pq.add(new Node(chars[i], freq[i]));
        }

        // Build the Huffman tree
        while (pq.size() > 1) {
            Node left = pq.poll();
            Node right = pq.poll();

            Node newNode = new Node(left.freq + right.freq, left, right);
            pq.add(newNode);
        }

        Node root = pq.poll();

        System.out.println("\nHuffman Codes:");
        printCodes(root, "");
    }
}
//Time Complexity = O(n log n) due to priority queue operations
//Space Complexity = O(n) for storing the nodes in the priority queue