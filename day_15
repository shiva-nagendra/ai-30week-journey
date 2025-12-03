#Simple binary search code

def binary_search_simple(arr, target):
    """Iterative simple binary search. Assumes arr is sorted ascending."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

nums = [1, 3, 5, 7, 9, 11, 13]
print(binary_search_simple(nums, 7))   
print(binary_search_simple(nums, 6))   



#Recursive version (conceptual clarity)

def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

print(binary_search_recursive([1,2,3,4,5], 4))  





#Engineering-grade binary search (robust, typed, with helpers)

from typing import List, Optional, Callable, Any

def is_sorted(arr: List[Any]) -> bool:
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def binary_search(
    arr: List[int],
    target: int,
    *,
    find_first: bool = False,
    find_last: bool = False,
    check_sorted: bool = False
) -> Optional[int]:
    """
    Engineering binary search.
    - find_first=True -> returns first index of target (or None)
    - find_last=True  -> returns last index of target (or None)
    - check_sorted=True -> will raise ValueError if arr is not sorted (cheap for small arrays)
    """
    if check_sorted and not is_sorted(arr):
        raise ValueError("Array must be sorted (ascending) for binary_search.")
    left, right = 0, len(arr) - 1
    result: Optional[int] = None

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            if find_first:
                # keep searching left side
                right = mid - 1
            elif find_last:
                # keep searching right side
                left = mid + 1
            else:
                return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result

# examples
arr = [1,2,2,2,3,4,5]
print(binary_search(arr, 2))                # could be 1,2 or 3 (any index of 2)
print(binary_search(arr, 2, find_first=True))  # -> 1
print(binary_search(arr, 2, find_last=True))   # -> 3
print(binary_search(arr, 10))               # -> None
