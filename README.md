# FIZZBUZZ-WITH-NN
Solving the Fizzbuzz problem with NN approach, using a binary and Prime factors representations<br>
This excersize was part of a DL course:<br>
Instructions:<br>
1)	Take the FizzBuzz example and make it work.<br>
2)	Add to the code a function that analyzes the results.<br>
a.	Print the accuracy of the classifier.<br>
b.	Generate a confusion matrix.<br>
3)	Run it once and put the results in a word file.<br>
4)	Rerun the algorithm 10 times and see if there are differences in the accuracy.<br>
Generate a graph showing the differences in the results.<br>
5)	Change the representation from binary to prime based. That means each number is coded by how many time each prime appears in the product. For large primes you can put them all in one bucket.<br>
Example:  24 is coded as 2^3* 3^1 -   [3,1,0,0,0,0…]<br>
Which of the algorithms (first and second) will work better? Run them and write the results and compare them.<br>
6)	Find the smallest network for the prime representation that will yield perfect results<br>
7)	Write a function that is given as input the prime representation and it classifies the number. It does not have to be a neural net classifier but works on the representation without any additional knowledge or data.<br>
8)	Take the smallest classifier you built and instead of training it, hardcode the weights into it and show that it yields perfect results like in 7 & 8.<br>
9)	Implement in Python a Neuron class. It should not be based on a well-known system like TF. Just pure code. Here is the main that uses it:<br>

n = Neuron(10,0.1)   # init assumption is that the activation function is sigmoid<br>
x = np.random.normal(0,0.1,10)  <br>
res = n.forward(x)   #given x compute the result and maintain results in the object<br>
db,dw,dx = n.backward(0.3) # Given the error compute the derivatives<br>
print(res,db,dw,dx)<br>
x = np.ones(10) <br>
eta = 1.0e-6 <br>
n.ModifyWb(-eta*dw,-eta*db)  # change w and b by adding to them the gradients.<br>
res = n.forward(x)<br>
db,dw,dx = n.backward(0.3)<br>
print(res,db,dw,dx)<br>
