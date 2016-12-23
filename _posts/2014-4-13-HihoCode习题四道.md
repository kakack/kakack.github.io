---

layout: post
categories: [ACM]
tags: [ACM,MicroSoft,Algorithm]

---

录四个之前做MS家Online test的题，AC掉了两个，之后有空把剩下的两个做了


## Q1

```
Time Limit: 10000ms
Case Time Limit: 1000ms
Memory Limit: 256MB
```


Description


For this question, your program is required to process an input string containing only ASCII characters between ‘0’ and ‘9’, or between ‘a’ and ‘z’ (including ‘0’, ‘9’, ‘a’, ‘z’). 

Your program should reorder and split all input string characters into multiple segments, and output all segments as one concatenated string. The following requirements should also be met,
1. Characters in each segment should be in strictly increasing order. For ordering, ‘9’ is larger than ‘0’, ‘a’ is larger than ‘9’, and ‘z’ is larger than ‘a’ (basically following ASCII character order).
2. Characters in the second segment must be the same as or a subset of the first segment; and every following segment must be the same as or a subset of its previous segment. 

Your program should output string “<invalid input string>” when the input contains any invalid characters (i.e., outside the '0'-'9' and 'a'-'z' range).



### Input

```
Input consists of multiple cases, one case per line. Each case is one string consisting of ASCII characters.
```


### Output

```
For each case, print exactly one line with the reordered string based on the criteria above.
```


### Sample In

```
aabbccdd
007799aabbccddeeff113355zz
1234.89898
abcdefabcdefabcdefaaaaaaaaaaaaaabbbbbbbddddddee
```

### Sample Out

```
abcdabcd
013579abcdefz013579abcdefz
<invalid input string>
abcdefabcdefabcdefabdeabdeabdabdabdabdabaaaaaaa
```

---

## Q2

```
Time Limit: 10000ms
Case Time Limit: 1000ms
Memory Limit: 256MB
```


Description

Consider a string set that each of them consists of {0, 1} only. All strings in the set have the same number of 0s and 1s. Write a program to find and output the K-th string according to the dictionary order. If s​uch a string doesn’t exist, or the input is not valid, please output “Impossible”. For example, if we have two ‘0’s and two ‘1’s, we will have a set with 6 different strings, {0011, 0101, 0110, 1001, 1010, 1100}, and the 4th string is 1001.


### Input

```
The first line of the input file contains a single integer t (1 ≤ t ≤ 10000), the number of test cases, followed by the input data for each test case.
Each test case is 3 integers separated by blank space: N, M(2 <= N + M <= 33 and N , M >= 0), K(1 <= K <= 1000000000). N stands for the number of ‘0’s, M stands for the number of ‘1’s, and K stands for the K-th of string in the set that needs to be printed as output.
```

### Output

```
For each case, print exactly one line. If the string exists, please print it, otherwise print “Impossible”. 
```

### Sample In

```
3
2 2 2
2 2 7
4 7 47
```

### Sample Out

```
0101
Impossible
01010111011
```

---
## Q3

```
Time Limit: 10000ms
Case Time Limit: 1000ms
Memory Limit: 256MB
```

Description

Find a pair in an integer array that swapping them would maximally decrease the inversion count of the array. If such a pair exists, return the new inversion count; otherwise returns the original inversion count.

Definition of Inversion: Let (A[0], A[1] ... A[n]) be a sequence of n numbers. If i < j and A[i] > A[j], then the pair (i, j) is called inversion of A.

Example:
Count(Inversion({3, 1, 2})) = Count({3, 1}, {3, 2}) = 2
InversionCountOfSwap({3, 1, 2})=>
{
  InversionCount({1, 3, 2}) = 1 <-- swapping 1 with 3, decreases inversion count by 1
  InversionCount({2, 1, 3}) = 1 <-- swapping 2 with 3, decreases inversion count by 1
  InversionCount({3, 2, 1}) = 3 <-- swapping 1 with 2 , increases inversion count by 1
}


### Input

```
Input consists of multiple cases, one case per line.Each case consists of a sequence of integers separated by comma.
```

### Output

```
For each case, print exactly one line with the new inversion count or the original inversion count if it cannot be reduced.
```

### Sample In

```
3,1,2
1,2,3,4,5
```

### Sample Out

```
1
0
```

---

```
Time Limit: 10000ms
Case Time Limit: 3000ms
Memory Limit: 256MB
```

Description

In a running system, there're many logs produced within a short period of time, we'd like to know the count of the most frequent logs.
Logs are produced by a few non-empty format strings, the number of logs is N(1=N=20000), the maximum length of each log is 256.
Here we consider a log same with another when their edit distance (see note) is = 5.
Also we have a) logs are all the same with each other produced by a certain format string b) format strings have edit distance  5 of each other.
Your program will be dealing with lots of logs, so please try to keep the time cost close to O(nl), where n is the number of logs, and l is the average log length.
Note edit distance is the minimum number of operations (insertdeletereplace a character) required to transform one string into the other, please refer to httpen.wikipedia.orgwikiEdit_distance for more details.

### Input

```
Multiple lines of non-empty strings.
```

### Output

```
The count of the most frequent logs.
```

### Sample In

```
Logging started for id:1
Module ABC has completed its job
Module XYZ has completed its job
Logging started for id:10
Module ? has completed its job
```

### Sample Out

```
3
```