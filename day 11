#dict practice

scores = []
while True:
    
    name = input("Enter name or exit: ").lower()
    if name == "exit":
        break
    marks =  float(input("Enter your marks: "))
    scores.append({"name" : name, "marks" : marks})

    
print("\nAll records = ",scores)

max_scorer = max(scores, key=lambda x: x["marks"])
min_scorer = min(scores, key=lambda x: x["marks"])
total = sum(student["marks"] for student in scores)
average = round(total/ len(scores),2)

print(f"highest scorer is =  {max_scorer['name'].capitalize()}({max_scorer['marks']})")
print(f"least scorer is = , {min_scorer['name'].capitalize()}({min_scorer['marks']})")
print(f"average marks are = , {average}")



       
        
   


