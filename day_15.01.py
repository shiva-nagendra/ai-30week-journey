#binary code

def binary_search_trace(arr, target):
    left = 0
    right = len(arr) - 1
    step = 1

    while left <= right:
        mid = (left + right) // 2
        print(f"\nSTEP {step}")
        print(f"LEFT={left}, RIGHT={right}, MID={mid}, VALUE={arr[mid]}")

        if arr[mid] == target:
            print("✓ FOUND IT!")
            return mid

        elif target > arr[mid]:
            print(f"{target} is greater than {arr[mid]} → go RIGHT")
            left = mid + 1

        else:
            print(f"{target} is less than {arr[mid]} → go LEFT")
            right = mid - 1

        step += 1

    print("✗ NOT FOUND")
    return -1


words = ["ant", "ball", "bat", "cat", "dog", "egg", "fan"]

binary_search_trace(words, "dog")



def find_insert_position(arr, target):
    left = 0
    right = len(arr) - 1
    pos = len(arr)   # default position = end

    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] >= target:
            pos = mid          # possible insert position
            right = mid - 1    # search left for earlier place
        else:
            left = mid + 1     # go right
    return pos

arr = [2, 4, 6, 10]
print(find_insert_position(arr, 7))  # expected 3
print(find_insert_position(arr, 1))  # expected 0
print(find_insert_position(arr, 20)) # expected 4




def binary_search_trace(arr, target):
    left = 0
    right = len(arr) - 1
    step = 1

    while left <= right:
        mid = (left + right) // 2
        print(f"\nSTEP {step}")
        print(f"LEFT={left}, RIGHT={right}, MID={mid}, VALUE={arr[mid]}")

        if arr[mid] == target:
            print("✓ FOUND IT!")
            return mid

        elif target > arr[mid]:
            print(f"{target} is greater than {arr[mid]} → go RIGHT")
            left = mid + 1

        else:
            print(f"{target} is less than {arr[mid]} → go LEFT")
            right = mid - 1

        step += 1

    print("✗ NOT FOUND")
    return -1


arr = [2, 5, 8, 12, 15, 19, 20, 23]
binary_search_trace(arr, 99)



def find_insert_pos(arr, target):
    left = 0
    right =len(arr)-1
    pos=len(arr)

    while left<= right:
        mid = (left + right)//2
        if arr[mid]>=target:
            pos = mid
            right = mid - 1
            
        else :
            left = mid + 1
            
    return pos

arr=[2,4,7,7,8,8,88,89]
print(find_insert_pos(arr,1))
print(find_insert_pos(arr,8))



