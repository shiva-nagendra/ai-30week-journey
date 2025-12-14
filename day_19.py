#Recursion Mastery

def countdown(n):
    if n == 0:          # base case
        print("Done")
        return
    print(n)
    countdown(n-1) # recursive case

countdown(9)


def sum_list(arr):
    if len(arr) == 0:
        return 0
    return arr[0] + sum_list(arr[1:])

sum_l= sum_list([2, 4, 6])
print("sum is: ", sum_l)



