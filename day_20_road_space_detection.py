#road space detection practice

    # 1.	Detect all empty gaps
	# 2.	Measure their length
	# 3.	Store them in a useful structure
	# 4.	Prepare data for decision making

    # When you scan the road:
	# •	Consecutive 0s = one empty gap
	# •	When 1 appears → gap ends
	# •	At road end → gap may still exist

    # We don’t act yet.
    # We observe and record.

def detect_gaps(road):
    gaps = []
    start_index = None

    for i in range (len(road)):
        current_spot = road[i]
        if current_spot == 0 and start_index is None:
            start_index = i
        elif current_spot == 1 and start_index is not None:
            length = i - start_index
            gaps.append((start_index,length))
            start_index = None

    if start_index is not None:
        length_of_finalgap = len(road)-start_index
        gaps.append((start_index,length_of_finalgap))


    def gap_length(gap_tuple):
     return gap_tuple[1]
    

    sorted_gaps = sorted(gaps, key=gap_length, reverse=True)
    return sorted_gaps

def best_gap(road):
    gaps = detect_gaps(road)
    
    if not gaps:
        return None   
    return gaps[0]    


def insert_car(road):
    gap = best_gap(road)
    if not gap:
        print("No gaps on the route")
        return road
    x,y = gap
    road[x]= [1]
    return road
    

           
road = [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        
        
result = detect_gaps(road)
print(f"The road status is: {road}")
print(f"The sorted gaps are: {result}")
print(best_gap(road))

road = insert_car(road)
print("Best route for the car:", road)

            




    




    

        
        
