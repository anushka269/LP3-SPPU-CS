**DAA - 1**
Write a program **recursive program** to calculate **Fibonacci numbers** and analyze their time and space complexity

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
---------------------------------------------------------2----------------------------------------------------------------------
**DAA-2**
Write a program **non-recursive program** to calculate **Fibonacci numbers** and analyze their time and space complexity. 

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
-----------------------------------------------3----------------------------------------------------------------
**DAA - 3**
Write a program to implement **Huffman Encoding** using a greedy strategy

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
-----------------------------------------4---------------------------------------------------------
**DAA - 4**
Write a program to solve a **fractional Knapsack** problem using a greedy method

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
--------------------------------------5------------------------------------------------------------------
**DAA-5**
Write a program to solve a **0-1 Knapsack problem** using dynamic programming or branch and bound strategy.

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
----------------------------------------6----------------------------------------------------------------
**DAA-6**
Design **n-Queens** matrix having first Queen placed. Use backtracking to place remaining Queens to generate the final nqueen‘s matrix.

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

------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------
**ML -1**
Price of Uber

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
#We do not want to see warnings
warnings.filterwarnings("ignore") 

#import data
data = pd.read_csv("uber.csv")
#Create a data copy
df = data.copy()
#Print data
df.head()

#Get Info
df.info()

#pickup_datetime is not in required data format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df.info()

#Statistics of data
df.describe()

#Number of missing values
df.isnull().sum()

#Correlation
df.select_dtypes(include=[np.number]).corr()

print(df.columns)

#Drop the rows with missing values
df.dropna(inplace=True)
plt.boxplot(df['fare_amount'])

#Remove Outliers
q_low = df["fare_amount"].quantile(0.01)
q_hi  = df["fare_amount"].quantile(0.99)

df = df[(df["fare_amount"] < q_hi) & (df["fare_amount"] > q_low)]
#Check the missing values now
df.isnull().sum()

#Time to apply learning models
from sklearn.model_selection import train_test_split
#Take x as predictor variable
x = df.drop("fare_amount", axis = 1)
#And y as target variable
y = df['fare_amount']

#Necessary to apply model
x['pickup_datetime'] = pd.to_numeric(pd.to_datetime(x['pickup_datetime']))
x = x.loc[:, x.columns.str.contains('^Unnamed')]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

#Prediction
predict = lrmodel.predict(x_test)

# evaluation
from sklearn.metrics import mean_squared_error, r2_score
lr_rmse = np.sqrt(mean_squared_error(y_test, predict))
lr_r2 = r2_score(y_test, predict)
print("Linear Regression → RMSE:", lr_rmse, "R²:", lr_r2)

#Let's Apply Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(n_estimators = 100, random_state = 101)
#Fit the Forest
rfrmodel.fit(x_train, y_train)
rfrmodel_pred = rfrmodel.predict(x_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfrmodel_pred))
rfr_r2 = r2_score(y_test, rfrmodel_pred)

print("Random Forest → RMSE:", rfr_rmse, "R²:", rfr_r2)

------------------------------------------------2------------------------------------------------------
**ML - 2** ->Email Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load dataset
df = pd.read_csv("emails.csv")
df.head()


df.isnull().sum()

X = df.iloc[:,1:3001]  # word frequency features
X

Y = df.iloc[:,-1].values # 1 = spam, 0 = not spam
Y

# Visualize outliers
import matplotlib.pyplot as plt
import seaborn as sns

# compute IQR outlier counts (you already had this)
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outlier_mask = ((df_numeric < lower) | (df_numeric > upper))
outlier_counts = outlier_mask.sum().sort_values(ascending=False)

# pick top N features
topN = 12
top_features = outlier_counts.head(topN).index.tolist()

plt.figure(figsize=(16,6))
sns.boxplot(data=df_numeric[top_features])
plt.title(f"Boxplots for top {topN} features by outlier count")
plt.xticks(rotation=45, ha='right')
plt.show()

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
from sklearn.metrics import classification_report, confusion_matrix

# -------- Support Vector Machine --------
svc = SVC(C=1.0, kernel='rbf', gamma='auto')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svc_pred))
print("SVM Classification Report:\n", classification_report(y_test, svc_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svc_pred))

# -------- K-Nearest Neighbors --------
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:", knn.score(X_test, y_test))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
 
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
ks = [1, 3, 5] 
results = {}
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_s, y_train)              # X_train_s must be scaled features
    y_pred = knn.predict(X_test_s)          # X_test_s must be scaled feature
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    results[k] = acc
    print(f"\nK = {k}:")
    print(f"  Accuracy = {acc:.4f}")
    print("  Confusion Matrix:")
    print(cm)
    print("  Classification Report:")
    print(report)
------------------------------------3--------------------------------------
**Ml - 3** -> Gradient Descent Algorithm

import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x):
    return (x + 3)**2
    
# Define its derivative (gradient)
def grad_f(x):
    return 2 * (x + 3)
    
# Gradient Descent Implementation
def gradient_descent(start_x=2, learning_rate=0.1, max_iter=50, tol=1e-6):
    x = start_x
    x_history = [x]
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        x_history.append(x_new)
        if abs(x_new - x) < tol:  # convergence check
            break
        x = x_new
    return x, f(x), x_history
# Run Gradient Descent
min_x, min_y, x_steps = gradient_descent()
print("Local minima at x =", min_x)
print("Minimum value y =", min_y)

# Visualization
x_vals = np.linspace(-6, 2, 100)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label='y = (x+3)^2')
plt.scatter(x_steps, [f(x) for x in x_steps], color='red', label='Gradient Descent Steps')
plt.title("Gradient Descent to find Local Minima")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

-------------------------------4-------------------------------
**Ml - 4** -> KNN on Diabetes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Load dataset
data = pd.read_csv("diabetes.csv")
print(data.head())

#Check for null or missing values
data.isnull().sum()

# Replace zeros with mean for selected columns
cols_to_replace = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in cols_to_replace:
    data[column].replace(0, np.nan, inplace=True)
    data[column].fillna(round(data[column].mean(skipna=True)), inplace=True)

# Features and target
X = data.iloc[:, :8]   # first 8 columns are features
Y = data['Outcome']    # target column

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
import matplotlib.pyplot as plt

# Visualize outliers using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data)
plt.title("Outlier Detection using Boxplots")
plt.show()

# Identify outliers using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Display count of outliers per column
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
print("\nNumber of Outliers per Feature:\n", outliers)

# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=5)  # you can change k
knn.fit(X_train, Y_train)

# Predictions
knn_pred = knn.predict(X_test)

# Metrics
cm = confusion_matrix(Y_test, knn_pred)
accuracy = accuracy_score(Y_test, knn_pred)
error_rate = 1 - accuracy
precision = precision_score(Y_test, knn_pred)
recall = recall_score(Y_test, knn_pred)
f1 = f1_score(Y_test, knn_pred)

# Print results
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy)
print("Error Rate:", error_rate)
print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)

accuracy_scores = []

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    knn_pred = knn.predict(X_test)
    acc = accuracy_score(Y_test, knn_pred)
    accuracy_scores.append(acc)
    print(f"K = {k} → Accuracy = {acc * 100:.2f}%")

plt.plot([3, 5, 7], accuracy_scores, marker='o')
plt.title("KNN Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

-----------------------------------5---------------------------------------------------------------------------------------
**ML - 5** -> KNN on Sales Data Sample

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# Load dataset
df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')
df.head()
df.info()

# Drop unnecessary columns
to_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATE', 'POSTALCODE', 'PHONE']
df = df.drop(to_drop, axis=1)
#Check for null values
df.isnull().sum()

df.dtypes

# Select numeric columns only
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Visualize outliers
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_numeric)
plt.title("Outlier Detection using Boxplots (Numeric Columns Only)")
plt.show()

# Identify outliers using IQR
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

# Count outliers per column
outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()
print("\nNumber of Outliers per Numeric Feature:\n", outliers)

#normilization data
from sklearn.preprocessing import StandardScaler

df_capped = df_numeric.copy() 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_capped)
print(" Data normalized using StandardScaler.")

df_normalized = pd.DataFrame(X_scaled, columns=df_capped.columns)

print("\nSample of Normalized Data:")
display(df_normalized.head())

print("\nMean of each feature after normalization:\n", df_normalized.mean())
print("\nStandard deviation of each feature after normalization:\n", df_normalized.std())

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

for k in range (2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    print(f"K = {k} → Silhouette Score = {sil_score:.4f}")

#Visualize Clusters for K = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df_capped['SALES'],
    y=df_capped['MSRP'],
    hue=labels,
    palette='Set2'
)
plt.title("K-Means Clustering Visualization (K = 3)")
plt.xlabel("Sales")
plt.ylabel("MSRP")
plt.legend(title='Cluster')
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------
**BLT -1** -> Student Data

//Practical no 1

pragma solidity ^0.8.19;

contract StudentManager {
    struct Student {
        uint256 id;
        string name;
        uint8 age;
        string course;
    }
    Student[] private students;
    mapping(uint256 => uint256) private idx;
    uint256 public studentCount;

    uint256 public depositsCount;
    uint256 public lastDepositAmount;
    address public lastSender;

    event StudentAdded(uint256 indexed id, string name);
    event StudentUpdated(uint256 indexed id);
    event StudentRemoved(uint256 indexed id);
    event Received(address indexed from, uint256 amount);
    event FallbackCalled(address indexed from, uint256 amount, bytes data);

    constructor() {
        studentCount = 0;
    }

    function addStudent(string calldata _name, uint8 _age, string calldata _course) external {
        studentCount += 1;
        uint256 newId = studentCount;
        students.push(Student({id: newId, name: _name, age: _age, course: _course}));
        idx[newId] = students.length;
        emit StudentAdded(newId, _name);
    }

    function getStudent(uint256 _id) public view returns (Student memory) {
        uint256 i = idx[_id];
        require(i != 0, "Student not found");
        return students[i - 1];
    }

    function getAllStudents() external view returns (Student[] memory) {
        return students;
    }

    function updateStudent(uint256 _id, string calldata _name, uint8 _age, string calldata _course) external {
        uint256 i = idx[_id];
        require(i != 0, "Student not found");
        Student storage s = students[i - 1];
        s.name = _name;
        s.age = _age;
        s.course = _course;
        emit StudentUpdated(_id);
    }

    function removeStudent(uint256 _id) external {
        uint256 i = idx[_id];
        require(i != 0, "Student not found");
        uint256 arrayIndex = i - 1;
        uint256 lastIndex = students.length - 1;
        if (arrayIndex != lastIndex) {
            Student memory lastStudent = students[lastIndex];
            students[arrayIndex] = lastStudent;
            idx[lastStudent.id] = arrayIndex + 1;
        }
        students.pop();
        idx[_id] = 0;
        emit StudentRemoved(_id);
    }

    function deposit() external payable {
        require(msg.value > 0, "Send ETH");
        depositsCount += 1;
        lastDepositAmount = msg.value;
        lastSender = msg.sender;
        emit Received(msg.sender, msg.value);
    }

    receive() external payable {
        depositsCount += 1;
        lastDepositAmount = msg.value;
        lastSender = msg.sender;
        emit Received(msg.sender, msg.value);
    }

    fallback() external payable {
        depositsCount += 1;
        lastDepositAmount = msg.value;
        lastSender = msg.sender;
        emit FallbackCalled(msg.sender, msg.value, msg.data);
    }

    function getStudentsLength() external view returns (uint256) {
        return students.length;
    }
}

----------------------------------------------2---------------------------------------------------------------------------
**BLT - 2** -> Bank Account

//Bank Account - Practical No 2

pragma solidity ^0.8.19;

contract BankAccount {
    mapping(address => uint256) private balances;
    mapping(address => bool) private isUser;

    event AccountCreated(address user, uint256 amount);
    event Deposit(address user, uint256 amount);
    event Withdraw(address user, uint256 amount);

    function createAccount() public payable {
        require(!isUser[msg.sender], "Account exists");
        isUser[msg.sender] = true;
        balances[msg.sender] = msg.value;
        emit AccountCreated(msg.sender, msg.value);
    }

    function deposit() public payable {
        require(isUser[msg.sender], "No account");
        require(msg.value > 0, "Amount > 0");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) public {
        require(isUser[msg.sender], "No account");
        require(balances[msg.sender] >= amount, "Low balance");
        balances[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
        emit Withdraw(msg.sender, amount);
    }

    function showBalance() public view returns (uint256) {
        require(isUser[msg.sender], "No account");
        return balances[msg.sender];
    }
}
