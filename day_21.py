#System Design Layer


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

def risk_track(gaps):
    predictions = []
    for start, length in gaps:
        if length >= 3:
            risk = 'low'
        elif length == 2:
            risk = 'medium'
        else:
            risk = 'high'
        predictions.append({
            'start':start,
            'length':length,
            'risk':risk
        })
    return predictions

def decide_and_insert(road):
    gaps = detect_gaps(road)

    if not gaps:
        print("\nNo space available")
        return road

    predictions = risk_track(gaps)
    best = predictions[0]

    if best['risk'] == 'high':
        print("Risk too high — not inserting car")
        return road

    start = best['start']
    road[start] = 1
    print(f"Car inserted at position {start}")

    log_action(
        "car inserted",
        {"position": start, "gap_length":best["length"]}
    )
    return road

history = []
def log_action(action, details):
    history.append({
        "action":action,
        "details": details
        }
    )

           
road = [1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        
        
result = detect_gaps(road)
print(f"\nThe road status is: {road}")
print(f"\nThe sorted gaps are: {result}\n")
print("Best gap: \n\t",best_gap(road))

road = insert_car(road)
print("\nBest spot for the car:\t", road)

gaps = detect_gaps(road)
route=risk_track(gaps)
print("\npredictions:\n\t",route)

best_route = decide_and_insert(road)
print("\nbest route: ",best_route)


