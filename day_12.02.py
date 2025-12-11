
#applying the logic in utility hub by including input function


class UtilityHub():
    def __init__(self):
        self.expenses={}
        print("\n===Utility hub initiated===")

    def greet(self):
        name = input("\nEnter your name: ")
        print(f"\nHello {name.capitalize()}, Welcome to utility hub. \n\nChoose from the options below: \n")

    def add_expense(self, category, amount):
        if category in self.expenses:
            self.expenses[category]+= amount
        else:
            self.expenses[category]=amount
        print(f"added rs.{amount} to {category}")

    def show_expenses(self):
        if not self.expenses:
            print("No expenses yet")
        else:
            for cat, amt in self.expenses.items():
                print(f"{cat.capitalize()} : rs.{amt}")

    def show_total_expenses(self):
        if not self.expenses:
            print("no expenses added yet")
        else:
            total = sum(self.expenses.values())
            print(f"total expenses are: rs.{total}")


    def run(self):
        while True:
            print("""\t\n===* UTILITY HUB *===
                  \t 1. Add expenses
                  \t 2. show expenses
                  \t 3. total expenses
                  \t 4. Temperature converter
                  \t 5. exit""")
            choice = input("\nchoose options from 1 to 6: \n")

            if choice == "1":
                cat = input("enter your category = ")
                amt = float(input("enter your amount = "))
                self.add_expense(cat, amt)

            elif choice == "2":
                self.show_expenses()

            elif choice == "3":
                self.show_total_expenses()

            elif choice == "4":
                self.temperature_converter()

            elif choice == "5":
                print("adios amigo")
                break
            else:
                print("invalid input")

    def temperature_converter(self):
        while True:
            print("""===TEMPERATURE CONVERTER===
                  \t 1. celsius to fahrenheit
                  \t 2. fahrenheit to celsius
                  \t 3. go to main menu
                  """)
            choice = input("choose an option: ")
            
            if choice == "1":
                c = float(input("enter number in celcius: "))
                f = (c * 9/5) + 32
                print(f"{c}°C = {round(f, 2)}°F")
            elif choice == "2":
                f = float(input("enter number in fahreinheit: "))
                c = (f - 32) * 5/9
                print(f"{f}°F = {round(c, 2)}°C")
            elif choice == "3":
                break
            else:
                print("invalid choice")

  
if __name__ == "__main__":
    hub = UtilityHub()
    hub.greet()
    hub.run()

 