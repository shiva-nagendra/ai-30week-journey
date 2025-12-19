#customer

#  Responsibility: Identity + authentication
# Attributes:
# 	•	name
# 	•	pin

# Methods:
# 	•	verify_pin(input_pin)

# Account

# Responsibility: Money + operations
# Attributes:
# 	•	customer → Customer object
# 	•	balance

# Methods:
# 	•	deposit(amount)
# 	•	withdraw(amount, pin)
# 	•	check_balance(pin)

# Rules (this is your “POLICY” instinct kicking in)

# We explicitly define rules before code:
# 	•	Deposit:
# 	•	amount must be > 0
# 	•	Withdraw:
# 	•	pin must match
# 	•	amount ≤ balance
# 	•	Check balance:
# 	•	pin must match
# 	•	No direct balance access from outside


class Customer:
    def __init__(self,name,pin):
        self.name = name
        self.pin = pin
        
    def verify_pin(self, input_pin):
        return self.pin == input_pin
    
class Account:
    def __init__(self, customer, balance=0):
        self.customer = customer
        self.balance = balance
        
    def deposit_amount(self,amount):
        if amount <= 0:
            print("please enter positive amount")
            return
        self.balance += amount
        print(f"${amount}successfully deposited.")
        
    def withdraw(self, amount, pin):
        if not self.customer.verify_pin(pin):
            print("invalid pin")
            return
        if amount > self.balance:
            print("insufficient balance")
            return
    
        self.balance -= amount
        print(f"${amount} withdrawn succesfully")

    def check_balance(self, pin):
        if not self.customer.verify_pin(pin):
            print("invalid password")
            return
        print(f"your balance is ${self.balance}")

cust=Customer("shiva", 7006)
acct=Account(cust,1000)

acct.deposit_amount(500)
acct.withdraw(53,7006)
acct.check_balance(7006)
    
            
            

        


        
    







