#calculator.py
def calculate(num1, operator, num2):
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 != 0:
            return num1 / num2
        else:
           return "cannot be divisible by zero"
    else:
        return "invalid operator"
    
while True:
    num1 = float(input("enter first number: "))
    operator = input("enter operator: +, -,/ , * :  ")
    num2 = float(input("enter second number: "))

    result = round(calculate (num1,operator, num2),2)
    print("your result is: ", result,)
    
    choice = input("do you want to continue(yes/no: ?)").lower()
    if choice == "no":
        print("exiting calc..c u later...")
        break


