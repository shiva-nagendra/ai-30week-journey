#OOP
#class, attributes, methhods

class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.marks = []

    def introduction (self):
        print(f"Hi i'm {self.name} and i'm {self.age} years old")

    def add_mark(self, score):
        self.marks.append(score)
        print(f"added {score} to {self.name}'s record")

    def average_marks(self):
        if len(self.marks) == 0:
            return"No marks added yet"
        return round(sum(self.marks)/len(self.marks),2)

    
s1 = Student("shiva", 24)
print(s1.name)
print(s1.age)

s2 = Student("Rahul", 30)
print(s2.name)
print(s1.name)

s1.introduction()
s2.introduction()

s1 = Student("shiva", 23)
s1.add_mark(97)
s1.add_mark(88)
print(s1.average_marks())

s2 = Student("SHIV", 22)
s2.add_mark(88)
s2.add_mark(89.3)
print(s2.average_marks())

#class practice

class Employee:
    def __init__(self, name):
        self.name = name
        self.working_days = 0
    def intro(self):
        print(f"hello {name}, welcome to payday.")
    def take_attendancee(self):
        total_working_days= 28
        days = int(input("enter num of days you were attended: "))
        self.working_days=days

        if self.working_days<total_working_days:
            print("you didn't achieve 100% attendance")
        elif self.working_days==total_working_days:
            print("congrats you have achieved 100% attendance")
        else:
            print("YOU ARE WORKING OVERTIME, KEEP IT UP")
  
    def pay_calc(self,per_day_salary):
        total_salary = per_day_salary*self.working_days
        return total_salary

name= input("enter employee name: ")
payday = Employee(name)
payday.intro()
payday.take_attendancee()
per_day_salary = 1500
total_payout = payday.pay_calc(1500)
print(f"{name}, your salary for this month is {total_payout}")
        
    

        


        
    
        

            


    
