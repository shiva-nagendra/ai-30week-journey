
#input practice
name = input("what is your name?, ")
food = input("what do u wanna eat?, ")
print(f"hi {name},we don't have {food}!")
      
print("your path is :\n\t-python\n\t-AI ")

print("""your path is :
\t-python
\t-AI """)

#print practice

price_shirt = float(input("enter price of shirts: "))
price_jeans = float(input("enter price of jeans: ")) 

qty_shirt = int(input("enter num of shirt: "))
qty_jeans = int(input("enter num of jeans: "))

total_shirt = price_shirt * qty_shirt
total_jeans = price_jeans * qty_jeans
subtotal = total_shirt + total_jeans
discount_percentage = float(input("enter your discount percenatge: "))
discount_amount = subtotal * discount_percentage/ 100
final_total = subtotal - discount_amount
print("Your payable amount is: ", final_total)