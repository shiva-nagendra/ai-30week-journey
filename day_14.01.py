#practice 

for i in range(1, 6):
    for j in range(1, i+1):
        print("%", end="-")
    print()

# Selection sort logic:
# 	1.	Look at the whole list
# 	2.	Find the smallest number
# 	3.	Put it at position 0
# 	4.	Look at the remaining list
# 	5.	Find the next smallest
# 	6.	Put it at position 1
# 	7.	Repeat…

def sort_list(arr):
    n = len(arr)

    for i in range(n):
        min_num = i

        for j in range(i+1, n):
            if arr[j]<arr[min_num]:
                min_num=j
        arr[i],arr[min_num]=arr[min_num],arr[i]
    return arr
num=[6,1,4,2,0]
print(sort_list(num))


def bubble_sort(arr):
    n = len(arr)
    for x in range(n):
        for j in range(n-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr
num = [2,5,3,2,1]
print(bubble_sort(num))



def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swap = False
        for j in range(n-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
                swap = True
        if not swap:
            break
    return arr
num = [2,5,3,2,1,4,1,1,2,3,3]
print(bubble_sort(num))



def insertion_sort(arr):
    n = len(arr)
    for i in range(n):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j-=1
        arr[j+1]=key
    return arr
num = [2,5,3,2,1,4,1,1,2,3,3]
print(bubble_sort(num))



def insertion_sort(arr):
    n = len(arr)
    for i in range(n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j]>key:
            arr[j+1]=arr[j]
            j-=1
        arr[j+1]= key
    return arr
num = [2,5,3,2,1,4,1,1,2,3,3]
print(bubble_sort(num))


            
    




                
    



    

