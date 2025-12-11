#merge sort #merge function

#merge function

comparisons = 0
def merge(left, right):
    global comparisons
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] < right[j]:
            result.append(left[i])
            i+=1
        else:
            result.append(right[j])
            j+=1
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

#merge sort

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2  # Split into halves
    left = arr[:mid]
    right = arr[mid:]

    left_sorted = merge_sort(left) # Recursive sort on both halves
    right_sorted = merge_sort(right)

    return merge(left_sorted, right_sorted) # Merge back

words = ["cat", "apple", "banana", "dog"]
print(merge_sort(words))

print("Sorted:", merge_sort)
print("Comparisons:", comparisons)




