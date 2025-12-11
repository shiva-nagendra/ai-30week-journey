# Utility hub 
# Temperature converter and Expense tracker


expenses = {}

def greet_user():
    name = input("enter your name\n")
    print(f"hello {name}, welcome to utility hub\n")

def celsius_to_fahreinheit(c):
    return (c * 9/5) + 32
def fahreinheit_to_celsius(f):
    return (f - 32) * 5/9
def temperature_converter():

    print("""\t\n ===Temperature converter===
             \t1. celc to fahr
             \t2. fahr to celc """)
    choice = input("choose option 1 or 2: ")

    if choice == "1":
        c = float(input("\t\nenter a number in celsius: "))
        f = celsius_to_fahreinheit(c)
        print(f"{c}°C = {round(f, 2)}°F\n")

    elif choice == "2":
        f = float(input("enter a number in fahreinheit: "))
        c = fahreinheit_to_celsius(f)
        print(f"{f}°F = {round(c, 2)}°C\n")

    else:
        print("Invalid choice.\n")

def expense_tracker():
    while True:
        print("""\t\n === EXPENSE TRACKER ===
               \n 1. add expenses
               \n 2. view expenses
               \n 3. view total spent
               \n 4. Back to main menu""")
    
        choice = input("choose ooption from 1 to 4: ")

        if choice == "1":
            category = input("enter category(ex: food, fuel): ").lower()
            amount = float(input("enter amount: "))

# If category already exists, add to previous amount

            if category in expenses:  
               expenses[category] += amount
            else:
                expenses[category] = amount

            print(f"\nAdded ₹{amount} to '{category}'.\n")

        elif choice == "2":
            print("\n___EXPENSES___")
            if not expenses:
                print("no expenses added yet.\n")
            else:
                for category, amount in expenses.items():
                    print(f"{category.capitalize()} : ₹{amount}")
            print()

        elif choice == "3":
            if not expenses:
                print("\nno expenses added yet.\n")
            else:
                total = round(sum(expenses.values()),2)
                print(f"\nTotal spent so far: ₹{total}\n")

        elif choice == "4":
            print("returning to main menu...\n")
            break

        else:
            print("invalid option\n")
            
    
def main():
    while True:
        print("\n1.====Greet me====\n")
        print("2. Temperature converter\n")
        print("3. Expense tracker\n")
        print("4. Exit \n")

        choice = input("CHOOSE AN OPTION FROM 1 TO 4:")
        
        if choice == "1":
            greet_user()
        elif choice == "2":
            temperature_converter()
        elif choice =="3":
            expense_tracker()
        elif choice =="4":
            print("adios amigo!")
            break
        else:
            print("invalid choice\n")
if __name__ == "__main__":
    main()

